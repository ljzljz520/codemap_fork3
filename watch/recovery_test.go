package watch

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"codemap/scanner"

	"github.com/fsnotify/fsnotify"
)

// newRecoveryTestDaemon starts a daemon over a small Go project. The watcher's
// Errors channel is always replaced with an injectable buffered channel so
// tests can simulate a lost-event watcher failure. When injectEvents is true
// the Events channel is replaced too, so disk mutations produce no loop
// events at all (true event loss); when false the real Events channel stays
// in place, which flush-gate tests need to observe request-file writes while
// recovery is stuck.
func newRecoveryTestDaemon(t *testing.T, injectEvents bool) (*Daemon, string) {
	t.Helper()
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, ".codemap"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, ".codemap", "config.json"), []byte(`{"only":["go"]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "main.go"), []byte("package main\n\nfunc main() {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "removed.go"), []byte("package main\n\nfunc removed() {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	d, err := NewDaemon(root, false)
	if err != nil {
		t.Fatal(err)
	}
	d.watcher.Errors = make(chan error, 8)
	if injectEvents {
		d.watcher.Events = make(chan fsnotify.Event, 32)
	}
	if err := d.Start(); err != nil {
		d.Stop()
		t.Fatal(err)
	}
	t.Cleanup(d.Stop)
	waitForWatchCondition(t, 5*time.Second, func() bool { return ReadState(root) != nil })
	return d, root
}

// stubRecoveryBuild replaces the dependency build with a deterministic graph
// so recovery tests do not depend on language tooling.
func stubRecoveryBuild(t *testing.T, marker string) {
	t.Helper()
	original := buildFileGraph
	t.Cleanup(func() { buildFileGraph = original })
	buildFileGraph = func(context.Context, string, scanner.Filters) (*scanner.FileGraph, error) {
		return &scanner.FileGraph{
			Imports:   map[string][]string{"main.go": {"dep.go"}},
			Importers: map[string][]string{"dep.go": {"main.go"}, "marker.go": {marker}},
		}, nil
	}
}

// assertInventoryMatchesColdScan independently walks disk and proves the
// daemon's tracked files, configured inventory and line cache equal a cold
// scan — the end-state guarantee of watch-failure recovery.
func assertInventoryMatchesColdScan(t *testing.T, d *Daemon) {
	t.Helper()
	cache := scanner.NewGitIgnoreCache(d.root)
	files, err := scanner.ScanFiles(context.Background(), d.root, cache, nil, nil)
	if err != nil {
		t.Fatalf("cold scan failed: %v", err)
	}
	cfg := d.loadConfig()
	wantFiles := make(map[string]struct{})
	wantConfigured := make(map[string]struct{})
	wantLines := make(map[string]int)
	for i := range files {
		f := &files[i]
		wantFiles[f.Path] = struct{}{}
		if lines := countLines(filepath.Join(d.root, f.Path)); lines > 0 {
			wantLines[f.Path] = lines
		}
		slash := filepath.ToSlash(f.Path)
		if slash != ".codemap" && !strings.HasPrefix(slash, ".codemap/") &&
			scanner.MatchesFilters(f.Path, f.Ext, cfg.Only, cfg.Exclude) {
			wantConfigured[f.Path] = struct{}{}
		}
	}

	d.graph.mu.RLock()
	defer d.graph.mu.RUnlock()
	gotFiles := make(map[string]struct{}, len(d.graph.Files))
	for path := range d.graph.Files {
		gotFiles[path] = struct{}{}
	}
	if !reflect.DeepEqual(gotFiles, wantFiles) {
		t.Fatalf("tracked files diverge from cold scan:\n got %v\nwant %v", gotFiles, wantFiles)
	}
	gotConfigured := make(map[string]struct{}, len(d.graph.ConfiguredFiles))
	for path := range d.graph.ConfiguredFiles {
		gotConfigured[path] = struct{}{}
	}
	if !reflect.DeepEqual(gotConfigured, wantConfigured) {
		t.Fatalf("configured inventory diverges from cold scan:\n got %v\nwant %v", gotConfigured, wantConfigured)
	}
	for path, lines := range wantLines {
		state, ok := d.graph.State[path]
		if !ok || state.Lines != lines {
			t.Fatalf("line cache for %q = %+v, want %d lines", path, state, lines)
		}
	}
}

// TestWatchOverflowReconcilesToColdScanWithoutProxyEdits is the end-to-end
// recovery contract: disk changes that never produced an event (queue
// overflow) leave the daemon untrusted until a fresh watcher and a cold scan
// reconcile every structure, and the reconciliation itself is not recorded
// as editing activity.
func TestWatchOverflowReconcilesToColdScanWithoutProxyEdits(t *testing.T) {
	d, root := newRecoveryTestDaemon(t, true)
	stubRecoveryBuild(t, "recovered")
	oldWatcher := d.watcher

	// Hold the recovery cold scan on a gate so the untrusted publication is
	// observable instead of completing inside one polling interval.
	scanGate := make(chan struct{})
	scanStarted := make(chan struct{})
	var blockRecovery atomic.Bool
	originalScan := scanAllFiles
	t.Cleanup(func() { scanAllFiles = originalScan })
	scanAllFiles = func(ctx context.Context, scanRoot string, cache *scanner.GitIgnoreCache, only, exclude []string) ([]scanner.FileInfo, error) {
		if blockRecovery.CompareAndSwap(true, false) {
			close(scanStarted)
			<-scanGate
		}
		return originalScan(ctx, scanRoot, cache, only, exclude)
	}

	// Simulate lost events: mutate disk without sending anything on the
	// watcher channel.
	if err := os.WriteFile(filepath.Join(root, "added.go"), []byte("package main\n\nfunc added() {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(filepath.Join(root, "removed.go")); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "main.go"), []byte("package main\n\nfunc main() {}\n\nfunc extra() {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	// The stale inventory still shows the pre-loss world.
	if d.FileCount() != 2 {
		t.Fatalf("pre-recovery file count = %d, want stale 2", d.FileCount())
	}

	blockRecovery.Store(true)
	d.watcher.Errors <- fsnotify.ErrEventOverflow

	// Untrusted first, with the reconciliation blocked mid-flight.
	select {
	case <-scanStarted:
	case <-time.After(5 * time.Second):
		t.Fatal("recovery cold scan never started")
	}
	waitForWatchCondition(t, 5*time.Second, func() bool { return d.trustStatus() == DataTrustRecovering })
	recoveringState := ReadState(root)
	if recoveringState == nil || recoveringState.Trusted() {
		t.Fatalf("state during recovery = %#v, want untrusted marker", recoveringState)
	}
	close(scanGate)
	waitForWatchCondition(t, 10*time.Second, func() bool { return d.trustStatus() == DataTrustTrusted })

	finalState := ReadState(root)
	if finalState == nil || !finalState.Trusted() || finalState.DataTrust != DataTrustTrusted {
		t.Fatalf("final state = %#v, want trusted", finalState)
	}

	// Watcher subscriptions were rebuilt from scratch. The swap happens-before
	// the trusted publication on the event loop, so graph.mu synchronizes the
	// read here.
	d.graph.mu.RLock()
	rebuilt := d.watcher != oldWatcher
	watchList := 0
	if d.watcher != nil {
		watchList = len(d.watcher.WatchList())
	}
	d.graph.mu.RUnlock()
	if !rebuilt {
		t.Fatal("watcher was not rebuilt after overflow")
	}
	if watchList == 0 {
		t.Fatal("rebuilt watcher carries no directory subscriptions")
	}

	// Every derived structure equals what a cold scan would produce.
	assertInventoryMatchesColdScan(t, d)
	d.graph.mu.RLock()
	hasDeps := d.graph.HasDeps
	status := d.graph.GraphState.Status
	fileGraph := d.graph.FileGraph
	topo := d.graph.Topology
	d.graph.mu.RUnlock()
	if !hasDeps || status != graphLifecycleAvailable || fileGraph == nil {
		t.Fatalf("dependency graph not rebuilt: hasDeps=%t status=%q graph=%#v", hasDeps, status, fileGraph)
	}
	if got := fileGraph.Importers["marker.go"]; len(got) != 1 || got[0] != "recovered" {
		t.Fatalf("stale or wrong dependency graph published: %#v", fileGraph.Importers)
	}
	if topo == nil {
		t.Fatal("module topology was not reconciled")
	}

	// Reconciliation must not look like agent edits: no synthesized events,
	// no working-set touches, no event-log lines for the silently changed
	// paths.
	for _, event := range d.GetEvents(0) {
		if event.Path == "added.go" || event.Path == "removed.go" {
			t.Fatalf("reconciliation fabricated an edit event: %#v", event)
		}
	}
	d.graph.mu.RLock()
	workingSetSize := d.graph.WorkingSet.Size()
	d.graph.mu.RUnlock()
	if workingSetSize != 0 {
		t.Fatalf("reconciliation touched the working set: %d entries", workingSetSize)
	}
	if data, err := os.ReadFile(d.eventLog); err == nil && strings.Contains(string(data), "added.go") {
		t.Fatalf("reconciliation was written to the event log:\n%s", data)
	}

	// The replacement watcher actually delivers events after recovery.
	postFile := filepath.Join(root, "post_recovery.go")
	if err := os.WriteFile(postFile, []byte("package main\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(5 * time.Second)
	observed := false
	for time.Now().Before(deadline) {
		for _, event := range d.GetEvents(20) {
			if event.Path == "post_recovery.go" {
				observed = true
			}
		}
		if observed {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if !observed {
		t.Skip("rebuilt watcher delivered no events; fsnotify may be unavailable in this temp directory")
	}
}

// TestRecoveryFailsFlushUntilTrusted pins the flush gate: requests that
// arrive while the inventory is untrusted receive a watch_recovering failure
// ack, and only requests after recovery completes can confirm trusted state.
func TestRecoveryFailsFlushUntilTrusted(t *testing.T) {
	d, _ := newRecoveryTestDaemon(t, false)
	stubRecoveryBuild(t, "recovered")

	var attempts atomic.Int32
	originalWatcher := newFileWatcher
	t.Cleanup(func() { newFileWatcher = originalWatcher })
	newFileWatcher = func() (*fsnotify.Watcher, error) {
		if attempts.Add(1) <= 2 {
			return nil, errors.New("injected watcher recreation failure")
		}
		return originalWatcher()
	}

	d.watcher.Errors <- fsnotify.ErrEventOverflow
	waitForWatchCondition(t, 5*time.Second, func() bool { return d.trustStatus() == DataTrustRecovering })

	// A flush while recovery is stuck must not be acknowledged as trusted.
	reqDuring := writeRecoveryFlushRequest(t, d, "during")
	ackDuring := waitForFlushAck(t, d, "during", 5*time.Second)
	if ackDuring.Success || ackDuring.ErrorCode != flushErrWatchRecovering {
		t.Fatalf("recovery flush ack = %#v, want failure %q", ackDuring, flushErrWatchRecovering)
	}
	if err := validateFlushAck(reqDuring, ackDuring); err == nil {
		t.Fatal("untrusted flush ack passed validation")
	}

	// Once the injected failures are exhausted, recovery finishes and the
	// next flush confirms trusted state.
	waitForWatchCondition(t, 10*time.Second, func() bool { return d.trustStatus() == DataTrustTrusted })
	if got := attempts.Load(); got < 3 {
		t.Fatalf("watcher creation attempts = %d, want retries after failure", got)
	}
	reqAfter := writeRecoveryFlushRequest(t, d, "after")
	ackAfter := waitForFlushAck(t, d, "after", 5*time.Second)
	if !ackAfter.Success {
		t.Fatalf("post-recovery flush ack = %#v, want success", ackAfter)
	}
	if err := validateFlushAck(reqAfter, ackAfter); err != nil {
		t.Fatalf("post-recovery ack rejected: %v", err)
	}
	assertInventoryMatchesColdScan(t, d)
}

// TestRecoveryScanFailureRetries proves a cold-scan failure keeps trust
// withdrawn and the bounded retry eventually reconciles to a cold scan.
func TestRecoveryScanFailureRetries(t *testing.T) {
	d, _ := newRecoveryTestDaemon(t, false)
	stubRecoveryBuild(t, "recovered")

	var scans atomic.Int32
	originalScan := scanAllFiles
	t.Cleanup(func() { scanAllFiles = originalScan })
	scanAllFiles = func(ctx context.Context, root string, cache *scanner.GitIgnoreCache, only, exclude []string) ([]scanner.FileInfo, error) {
		if scans.Add(1) == 1 {
			return nil, errors.New("injected cold scan failure")
		}
		return originalScan(ctx, root, cache, only, exclude)
	}

	d.watcher.Errors <- fsnotify.ErrEventOverflow
	waitForWatchCondition(t, 5*time.Second, func() bool { return d.trustStatus() == DataTrustRecovering })
	waitForWatchCondition(t, 10*time.Second, func() bool { return d.trustStatus() == DataTrustTrusted })

	if got := scans.Load(); got < 2 {
		t.Fatalf("cold scans = %d, want a retry after the injected failure", got)
	}
	assertInventoryMatchesColdScan(t, d)
}

// TestRecoveryRejectsStaleGenerationBuild verifies the generation gate: a
// dependency build already running when recovery starts (or invalidated by an
// event during recovery) cannot be published over the reconciled state.
func TestRecoveryRejectsStaleGenerationBuild(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, ".codemap"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, ".codemap", "config.json"), []byte(`{"only":["go"]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "main.go"), []byte("package main\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	d := testGraphStateDaemon(root)
	d.configDir = filepath.Join(root, ".codemap")
	d.configPath = filepath.Join(root, ".codemap", "config.json")
	if err := d.ensurePublisher(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		close(d.done)
		if d.dependencyCancel != nil {
			d.dependencyCancel()
		}
		d.dependencyWorkerWG.Wait()
		if d.watcher != nil {
			_ = d.watcher.Close()
		}
	})

	started := make(chan struct{})
	release := make(chan struct{})
	var builds atomic.Int32
	originalBuild := buildFileGraph
	t.Cleanup(func() { buildFileGraph = originalBuild })
	buildFileGraph = func(context.Context, string, scanner.Filters) (*scanner.FileGraph, error) {
		if builds.Add(1) == 1 {
			close(started)
			<-release
			return &scanner.FileGraph{Importers: map[string][]string{"stale.go": {"old.go"}}}, nil
		}
		return &scanner.FileGraph{Importers: map[string][]string{"fresh.go": {"new.go"}}}, nil
	}

	d.beginRecovery()
	if d.trustStatus() != DataTrustRecovering || d.graph.HasDeps || d.graph.GraphState.Status != graphLifecycleStale {
		t.Fatalf("beginRecovery left graph usable: trust=%s state=%#v hasDeps=%t", d.trustStatus(), d.graph.GraphState, d.graph.HasDeps)
	}
	recoveryGen := d.graph.recoveryGeneration
	if recoveryGen == 0 {
		t.Fatal("recovery generation was not advanced")
	}

	// Drive the reconciliation steps the event loop would run, on this single
	// goroutine (matching the other graph-state tests): watcher rebuild, cold
	// scan, topology, then an async generation-checked dependency build.
	if err := d.rebuildWatchSubscriptions(); err != nil {
		t.Fatal(err)
	}
	if err := d.fullScan(); err != nil {
		t.Fatal(err)
	}
	d.computeTopology()
	d.awaitingGraphRecovery = true
	d.refreshDependencies()
	select {
	case <-started:
	case <-time.After(2 * time.Second):
		t.Fatal("recovery dependency build did not start")
	}

	// An event arrives while recovery is in flight: invalidate again.
	d.markGraphLifecycle(graphLifecycleStale)
	close(release)

	// First result belongs to the pre-invalidation generation and must be
	// rejected; recovery stays open and re-kicks the build itself.
	d.handleDependencyGraphResult(<-d.dependencyResults)
	d.graph.mu.RLock()
	firstRejected := !d.graph.HasDeps && d.graph.FileGraph == nil && d.graph.GraphState.Status == graphLifecycleStale
	stillRecovering := d.recovering
	d.graph.mu.RUnlock()
	if !firstRejected || !stillRecovering {
		t.Fatalf("stale generation build was applied: hasDeps=%t recovering=%t state=%#v", d.graph.HasDeps, stillRecovering, d.graph.GraphState)
	}

	// The retried build settles the current generation and restores trust.
	d.handleDependencyGraphResult(<-d.dependencyResults)
	d.graph.mu.RLock()
	got := d.graph.FileGraph.Importers
	settled := d.graph.HasDeps && d.graph.GraphState.Status == graphLifecycleAvailable
	d.graph.mu.RUnlock()
	if !settled {
		t.Fatalf("rebuilt graph did not settle: %#v", d.graph.GraphState)
	}
	if _, ok := got["fresh.go"]; !ok {
		t.Fatalf("reconciled graph missing fresh build result: %#v", got)
	}
	if _, stale := got["stale.go"]; stale {
		t.Fatalf("stale build overwrote the reconciled state: %#v", got)
	}
	if d.trustStatus() != DataTrustTrusted || d.recovering || d.graph.recoveryGeneration != recoveryGen {
		t.Fatalf("recovery did not close cleanly: trust=%s recovering=%t gen=%d", d.trustStatus(), d.recovering, d.graph.recoveryGeneration)
	}
}

// TestColdScanReconciliationDoesNotRecordEdits pins the property at the unit
// level: the cold-scan swap corrects inventory for files whose events were
// lost without appending events or touching the working set.
func TestColdScanReconciliationDoesNotRecordEdits(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, ".codemap"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, ".codemap", "config.json"), []byte(`{"only":["go"]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	tracked := filepath.Join(root, "tracked.go")
	if err := os.WriteFile(tracked, []byte("package main\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	d, err := NewDaemon(root, false)
	if err != nil {
		t.Fatal(err)
	}
	if err := d.fullScan(); err != nil {
		t.Fatal(err)
	}
	// A real edit recorded before the watch failure.
	d.handleEvent(fsnotify.Event{Name: tracked, Op: fsnotify.Write})
	eventsBefore := len(d.GetEvents(0))
	workingBefore := d.graph.WorkingSet.Size()
	if eventsBefore == 0 || workingBefore == 0 {
		t.Fatalf("test setup: events=%d workingSet=%d, want real edit recorded", eventsBefore, workingBefore)
	}

	// Silent disk changes: no events delivered, then a cold-scan reconcile.
	if err := os.WriteFile(filepath.Join(root, "silent_new.go"), []byte("package main\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(tracked); err != nil {
		t.Fatal(err)
	}
	if err := d.fullScan(); err != nil {
		t.Fatal(err)
	}

	if len(d.GetEvents(0)) != eventsBefore {
		t.Fatalf("cold scan changed the event history: %d -> %d", eventsBefore, len(d.GetEvents(0)))
	}
	if d.graph.WorkingSet.Size() != workingBefore {
		t.Fatalf("cold scan changed the working set: %d -> %d", workingBefore, d.graph.WorkingSet.Size())
	}
	for _, event := range d.GetEvents(0) {
		if event.Path == "silent_new.go" {
			t.Fatalf("cold scan fabricated an edit for silent_new.go")
		}
	}
	if _, trackedStill := d.graph.Files["tracked.go"]; trackedStill {
		t.Fatal("cold scan kept a file that no longer exists on disk")
	}
	if _, found := d.graph.Files["silent_new.go"]; !found {
		t.Fatal("cold scan did not pick up the silently created file")
	}
}

func TestStateTrustedMarker(t *testing.T) {
	if (&State{}).Trusted() != true {
		t.Fatal("legacy state without data_trust must default to trusted")
	}
	if (&State{DataTrust: DataTrustTrusted}).Trusted() != true {
		t.Fatal("explicit trusted marker reported untrusted")
	}
	if (&State{DataTrust: DataTrustRecovering}).Trusted() != false {
		t.Fatal("recovering marker reported trusted")
	}
	var nilState *State
	if nilState.Trusted() != true {
		t.Fatal("nil state should be treated as trusted by default")
	}
}

func writeRecoveryFlushRequest(t *testing.T, d *Daemon, nonce string) flushRequest {
	t.Helper()
	// Read identity and generation from the atomically published state rather
	// than publisher-internal fields, which are owned by the event loop.
	state := ReadState(d.root)
	if state == nil {
		t.Fatal("no published state to build a flush request against")
	}
	req := flushRequest{
		Version:            flushProtocolVersion,
		CanonicalRoot:      state.CanonicalRoot,
		DaemonInstance:     state.DaemonInstance,
		Nonce:              nonce,
		ObservedGeneration: state.Generation,
		Timestamp:          time.Now(),
	}
	data, err := json.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(d.publisher.flushDir, "request-"+nonce+".json"), data, 0o600); err != nil {
		t.Fatal(err)
	}
	return req
}

func waitForFlushAck(t *testing.T, d *Daemon, nonce string, timeout time.Duration) flushAck {
	t.Helper()
	deadline := time.Now().Add(timeout)
	path := filepath.Join(d.publisher.flushDir, "ack-"+nonce+".json")
	for time.Now().Before(deadline) {
		data, err := os.ReadFile(path)
		if err == nil {
			var ack flushAck
			if err := json.Unmarshal(data, &ack); err != nil {
				t.Fatal(err)
			}
			return ack
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for flush ack %s", nonce)
	return flushAck{}
}
