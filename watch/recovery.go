package watch

import (
	"fmt"
	"time"

	"codemap/topology"
)

// flushErrWatchRecovering is the flush-ack error code used while the daemon
// is reconciling lost watch events. A flush can only confirm trusted data
// after recovery completes, so waiting requests fail fast and the caller
// retries against the next publication instead of reading stale inventory.
const flushErrWatchRecovering = "watch_recovering"

// Recovery retry backoff. Watcher recreation or the cold scan can fail
// transiently (fd pressure, IO error); the attempt repeats with exponential
// backoff while trust stays withdrawn.
const (
	recoveryRetryMin = 100 * time.Millisecond
	recoveryRetryMax = 2 * time.Second
)

// trustStatus returns the current data-trust marker. The Graph zero value
// (used by unit-test literals and daemons older than this field) is treated
// as trusted.
func (d *Daemon) trustStatus() DataTrust {
	d.graph.mu.RLock()
	defer d.graph.mu.RUnlock()
	if d.graph.trust == "" {
		return DataTrustTrusted
	}
	return d.graph.trust
}

// handleWatchError reacts to a watcher.Errors delivery. An fsnotify error —
// event-queue overflow or a failed kernel subscription — means an unknown
// number of file events were lost, so the in-memory inventory can no longer
// be assumed to match disk. Every derived structure is marked untrusted and
// a cold-scan reconciliation is driven from the event loop.
func (d *Daemon) handleWatchError(err error) {
	if d.verbose {
		fmt.Printf("[watch] Watcher error: %v — marking state untrusted and reconciling\n", err)
	}
	if d.recovering {
		// A second failure while a recovery attempt is in flight (or waiting
		// to retry) must not reset the generation again: keep the current
		// reconciliation and make sure another attempt is scheduled.
		d.scheduleRecoveryRetry()
		return
	}
	d.beginRecovery()
	d.attemptRecovery()
}

// beginRecovery withdraws trust immediately, before any reconciliation work
// starts, so readers cannot keep treating the old inventory and dependency
// graph as authoritative while the gap is being closed.
func (d *Daemon) beginRecovery() {
	d.recovering = true
	d.awaitingGraphRecovery = false
	d.recoveryAttempts = 0

	d.graph.mu.Lock()
	d.graph.trust = DataTrustRecovering
	d.graph.recoveryGeneration++
	cfg := d.loadConfig()
	// markGraphLifecycleLocked bumps graphGeneration and drops the dependency
	// graph: a build already running for the lost-events generation now
	// carries a stale generation and cannot be published over the
	// reconciliation.
	d.markGraphLifecycleLocked(newGraphState(d.root, cfg, graphLifecycleStale, time.Time{}, nil))
	// The module topology is derived from the same possibly-missed events and
	// is rebuilt by the recovery pass.
	d.graph.Topology = nil
	d.graph.TopologyIdentity = topology.CacheIdentity{}
	d.graph.mu.Unlock()

	if d.publisher != nil {
		// Flush requests waiting on pre-loss state must not receive a success
		// acknowledgement from untrusted data. publish() keeps refusing new
		// acknowledgements while trust is withdrawn.
		d.reportPublicationError(d.publisher.failPending(flushErrWatchRecovering))
		d.reportPublicationError(d.writeState())
	}
}

// attemptRecovery runs one reconciliation pass on the event loop: rebuild
// directory subscriptions, then replace every derived structure from a cold
// scan. Any failure leaves trust withdrawn and arms a bounded retry.
func (d *Daemon) attemptRecovery() {
	if !d.recovering {
		return
	}
	select {
	case <-d.done:
		return
	default:
	}

	if err := d.rebuildWatchSubscriptions(); err != nil {
		d.reportRecoveryFailure(err)
		return
	}
	// fullScan replaces Files, ConfiguredFiles and State wholesale. It never
	// appends Events or touches the WorkingSet, so reconciling files that
	// changed while events were lost is not misrecorded as agent edits.
	if err := d.fullScan(); err != nil {
		d.reportRecoveryFailure(err)
		return
	}
	// Reconcile the module topology against the rebuilt subscription set.
	d.computeTopology()

	if !shouldComputeDependencyGraph(d.ConfiguredFileCount()) {
		// Large repos never build a dependency graph; the inventory and
		// topology reconciliation is the whole recovery.
		d.completeRecovery()
		return
	}

	// The dependency rebuild is asynchronous and generation-checked. Trust is
	// only restored once its result settles for the post-scan generation;
	// events that arrive meanwhile bump the generation and force another
	// build instead of letting a stale result win.
	d.awaitingGraphRecovery = true
	d.refreshDependencies()
	d.reportPublicationError(d.writeState())
}

// rebuildWatchSubscriptions creates a fresh watcher, replays every directory
// subscription onto it, and only then swaps it in and closes the old one.
// Events still buffered in the old watcher are deliberately discarded: the
// cold scan that follows establishes disk truth for the whole gap.
func (d *Daemon) rebuildWatchSubscriptions() error {
	watcher, err := newFileWatcher()
	if err != nil {
		return fmt.Errorf("recreate file watcher: %w", err)
	}
	if err := d.addWatchDirsTo(watcher); err != nil {
		_ = watcher.Close()
		return fmt.Errorf("rebuild directory subscriptions: %w", err)
	}
	if err := watcher.Add(d.configDir); err != nil {
		_ = watcher.Close()
		return fmt.Errorf("watch config directory: %w", err)
	}
	if err := ensureControlDirectory(d.publisher.flushDir); err != nil {
		_ = watcher.Close()
		return err
	}
	if err := watcher.Add(d.publisher.flushDir); err != nil {
		_ = watcher.Close()
		return fmt.Errorf("watch flush directory: %w", err)
	}

	previous := d.watcher
	d.watcher = watcher
	d.closeWatcher = watcher.Close
	if previous != nil {
		_ = previous.Close()
	}
	return nil
}

// settleRecoveryAfterGraphResult restores trust once the rebuilt dependency
// graph settles for the post-reconciliation generation. A result rejected for
// a newer generation leaves the graph stale; the invalidation that moved the
// generation already scheduled another build, and recovery stays open until
// one settles.
func (d *Daemon) settleRecoveryAfterGraphResult(pendingRetry bool) {
	if !d.recovering || !d.awaitingGraphRecovery {
		return
	}
	d.graph.mu.RLock()
	status := d.graph.GraphState.Status
	d.graph.mu.RUnlock()
	if status == graphLifecycleStale {
		if !pendingRetry {
			// Defensive: the result could not be published but no invalidation
			// was coalesced while the build ran. Re-kick rather than leave
			// trust withdrawn forever.
			d.refreshDependencies()
		}
		return
	}
	// available and failed both describe an honest outcome for the
	// reconciled inventory; the watch-failure gap itself is closed.
	d.completeRecovery()
}

// completeRecovery marks the in-memory state authoritative again and
// publishes it so flush clients can confirm trusted results.
func (d *Daemon) completeRecovery() {
	d.graph.mu.Lock()
	d.graph.trust = DataTrustTrusted
	d.graph.mu.Unlock()

	d.recovering = false
	d.awaitingGraphRecovery = false
	d.recoveryAttempts = 0
	d.stopRecoveryTimer()

	if d.verbose {
		fmt.Printf("[watch] Recovery complete: %d files reconciled against disk\n", d.FileCount())
	}
	if d.publisher != nil {
		d.reportPublicationError(d.writeState())
	}
}

func (d *Daemon) reportRecoveryFailure(err error) {
	if d.verbose {
		fmt.Printf("[watch] Recovery attempt %d failed: %v (retrying)\n", d.recoveryAttempts+1, err)
	}
	d.scheduleRecoveryRetry()
}

// scheduleRecoveryRetry arms the event-loop timer that re-runs
// attemptRecovery with exponential backoff capped at recoveryRetryMax.
func (d *Daemon) scheduleRecoveryRetry() {
	if d.done != nil {
		select {
		case <-d.done:
			return
		default:
		}
	}
	if d.recoveryTimer == nil {
		d.recoveryTimer = time.NewTimer(time.Hour)
		if !d.recoveryTimer.Stop() {
			<-d.recoveryTimer.C
		}
	}
	delay := recoveryRetryMin << min(d.recoveryAttempts, 4)
	if delay > recoveryRetryMax || delay <= 0 {
		delay = recoveryRetryMax
	}
	d.recoveryAttempts++
	if !d.recoveryTimer.Stop() {
		select {
		case <-d.recoveryTimer.C:
		default:
		}
	}
	d.recoveryTimer.Reset(delay)
	d.recoveryTimerC = d.recoveryTimer.C
}

func (d *Daemon) stopRecoveryTimer() {
	if d.recoveryTimer == nil {
		return
	}
	if !d.recoveryTimer.Stop() {
		select {
		case <-d.recoveryTimer.C:
		default:
		}
	}
	d.recoveryTimerC = nil
}
