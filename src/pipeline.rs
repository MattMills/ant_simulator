//! The decision pipeline: decisions coarse-grained in time and scheduled
//! against a frame budget.
//!
//! An ant on ground that is invariant (the flow through it is what it
//! always was, and runs straight) need not reconsider its heading at
//! every step: its decision is made once for a horizon of steps, as a
//! replayed transit stands in for a node's worth of them. Each decision
//! sets the ant's next: an earliest tick, when its hold on the heading
//! ends, and a deadline, up to a horizon's slack later, by which the
//! next decision must be made. The hold is the expected run of straight
//! choices the decision itself would make, `p / (1 - p)` for the
//! probability `p` it gave to holding the heading, so that a decision
//! is skipped only where it would have been the same; a dial run hot
//! flattens the distribution and shortens the hold of itself, so that
//! the queen orders the colony's decisions by where she spends its
//! entropy. Every tick is a frame with a budget of
//! decisions. Those at their deadline are made whatever the budget, and
//! the rest of the budget goes to the pending ones, earliest deadline
//! first and, among equal deadlines, in the hierarchy's order (the
//! castes the queen put first are served first); an ant not served
//! holds its heading a little longer, within its slack. A hold ends
//! early when the ant's leg changes or the step ahead is not clear.

/// How decisions are coarse-grained and scheduled.
#[derive(Clone, Debug, PartialEq)]
pub struct PipelineConfig {
    /// Decisions a frame may make beyond those at their deadline (zero
    /// for no limit: every decision is made as soon as its hold ends).
    pub budget: usize,
    /// The longest hold, ticks, on fully invariant, straight ground.
    pub max_horizon: u32,
    /// Slack after the hold, as a fraction of the horizon, within which
    /// a decision may wait for the budget.
    pub slack: f64,
    /// Invariance (one minus the flow's relative departure from its
    /// invariant) below which the horizon is a single step.
    pub min_invariance: f64,
    /// Ring positions either side of straight ahead that count as
    /// holding the heading (one: a turn of 22.5° either way, which the
    /// heading's persistence smooths into the direction of travel).
    pub cone: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        PipelineConfig {
            budget: 0,
            max_horizon: 4,
            slack: 1.0,
            min_invariance: 0.5,
            cone: 1,
        }
    }
}

/// The frames' ledger.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FrameStats {
    /// Frames planned.
    pub frames: u64,
    /// Frames whose decisions at their deadline alone exceeded the
    /// budget.
    pub overrun: u64,
    /// Decisions made.
    pub served: u64,
    /// Steps taken under a hold, without a decision.
    pub held: u64,
    /// Steps taken past the hold while waiting for the budget.
    pub deferred: u64,
    /// The horizons set, summed.
    pub horizon_sum: u64,
    /// How many were set.
    pub horizons: u64,
}

impl FrameStats {
    /// Mean horizon of the decisions made.
    pub fn mean_horizon(&self) -> f64 {
        if self.horizons == 0 {
            0.0
        } else {
            self.horizon_sum as f64 / self.horizons as f64
        }
    }

    /// Share of the steps taken without a decision, held or deferred.
    pub fn held_share(&self) -> f64 {
        let steps = self.held + self.deferred + self.served;
        if steps == 0 {
            0.0
        } else {
            (self.held + self.deferred) as f64 / steps as f64
        }
    }

    /// Share of the steps taken while waiting for the budget.
    pub fn deferred_share(&self) -> f64 {
        let steps = self.held + self.deferred + self.served;
        if steps == 0 {
            0.0
        } else {
            self.deferred as f64 / steps as f64
        }
    }
}

/// The horizon of a decision just made: one step on variant or
/// searching ground; on invariant ground the expected run of straight
/// choices, `p / (1 - p)` for the probability `p` the decision gave to
/// holding the heading, up to the longest hold.
pub fn horizon(cfg: &PipelineConfig, invariance: f64, straight: f64, searching: bool) -> u32 {
    if searching || invariance < cfg.min_invariance || straight <= 0.5 {
        return 1;
    }
    let run = straight.min(1.0 - 1e-9) / (1.0 - straight.min(1.0 - 1e-9));
    (run.floor() as u32).clamp(1, cfg.max_horizon.max(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn horizons_follow_invariance_and_the_decision_itself() {
        let cfg = PipelineConfig::default();
        assert_eq!(
            horizon(&cfg, 1.0, 0.99, false),
            4,
            "capped at the longest hold"
        );
        assert_eq!(
            horizon(&cfg, 1.0, 0.99, true),
            1,
            "searching decides every step"
        );
        assert_eq!(horizon(&cfg, 0.2, 0.99, false), 1, "variant ground");
        assert_eq!(horizon(&cfg, 1.0, 0.4, false), 1, "a decision in doubt");
        assert_eq!(
            horizon(&cfg, 1.0, 0.8, false),
            4,
            "the expected run of straight choices"
        );
        assert_eq!(horizon(&cfg, 1.0, 0.75, false), 3);
        assert_eq!(horizon(&cfg, 1.0, 0.7, false), 2);
        assert_eq!(horizon(&cfg, 1.0, 0.6, false), 1);
        let f = FrameStats {
            held: 6,
            deferred: 2,
            served: 2,
            ..FrameStats::default()
        };
        assert!((f.held_share() - 0.8).abs() < 1e-12);
        assert!((f.deferred_share() - 0.2).abs() < 1e-12);
    }
}
