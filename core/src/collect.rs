// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Collection abstraction for list scans.
//!
//! A family's per-list scan kernel decides *which* rows it visits and what
//! value it computes for each. What happens to a visited row is a separate
//! concern, and this trait is the seam between the two, so a kernel is written
//! once rather than copied per consumer. Aligned with Faiss's
//! `InvertedListScanner`, which hands rows to a `ResultHandler&` in
//! `scan_codes`. Faiss splits the range case out into a separately named
//! `scan_codes_range` taking a `RangeQueryResult&`; one collector serves both
//! here, so there is a single kernel rather than a pair to keep in step.
//!
//! `ivfflat_io::ReaderTopKHeap` and [`RangeCollector`] are the two
//! implementations.

use std::io;

use crate::distance::MetricType;
use crate::range::{Bound, DistanceBand};

/// The interface a scan kernel uses to hand candidate rows to a collector.
pub(crate) trait Collector {
    /// Called when the scan kernel abandons a row early because of [`cutoff`].
    ///
    /// An abandoned row never reaches [`push`], so a collector that counts the
    /// rows a scan touched cannot recover that count from its `push` calls --
    /// the kernel has to report the abandonment explicitly. The default empty
    /// body keeps this free for collectors that do not track it, which is why
    /// the top-K heap does not implement it.
    ///
    /// [`cutoff`]: Collector::cutoff
    /// [`push`]: Collector::push
    #[inline]
    fn note_abandoned(&mut self) {}

    /// The admission threshold currently in force. A row may be abandoned as
    /// soon as its partially accumulated distance exceeds this value.
    ///
    /// `f32::INFINITY` means no pruning is possible, and callers must *not*
    /// enter the early-abandon kernel in that case: it would run a full SIMD
    /// pass that can never abandon anything.
    fn cutoff(&self) -> f32;

    /// Delivers one row, with the value the family's scan computed for it. For
    /// IVF-Flat that value is an exact distance.
    ///
    /// Fallible because a collector may own a resource the scan cannot see: the
    /// oversized-list path streams chunks through a callback, and without a
    /// result type here a collector failure would have to panic to escape it.
    fn push(&mut self, id: i64, value: f32) -> io::Result<()>;
}

/// The threshold at which a partially accumulated L2 distance can be abandoned.
///
/// Returns `f32::INFINITY` wherever pruning cannot be justified, which is always
/// safe: it only means no row is abandoned early.
///
/// # Why the raw upper cut will not do
///
/// "partial sum > upper implies the full distance > upper" holds in exact
/// arithmetic only. The early-abandon kernel reduces 128-element blocks into a
/// scalar running total across four accumulators, while the kernel computing the
/// committed distance reduces once. The two disagree, and a row whose committed
/// distance sits just inside `upper` would otherwise be abandoned and silently
/// lost -- precisely the boundary rows that deriving cuts ULP-exactly exists to
/// place correctly. Measured before this was fixed: 1.24% of such rows at
/// d = 128, 1.65% at d = 256, 6.06% at d = 768.
///
/// # The bound
///
/// Two error sources have to be covered **jointly**, which is the step an
/// earlier version got wrong by spending the same headroom on both.
///
/// *Relative.* Both kernels are dot-product shaped and both fuse multiply-add on
/// AArch64, so the portable model is the dot-product bound with `k = d`: each
/// computation is within `g = du/(1 - du)` of exact, `u = EPSILON/2`. The worst
/// ratio between two such computations is `(1 + g)/(1 - g) = 1/(1 - x)` with
/// `x = d * EPSILON`.
///
/// *Additive.* A normal cut does not stop individual terms or per-lane
/// accumulations reaching the subnormal range, where roundings carry absolute
/// error. Call that bound `A`, and take `A = 2d * 2^-150`.
///
/// Why that constant holds, since counting the local errors is not by itself
/// enough -- each is then carried through the later relative roundings:
///
/// * AVX2 has at most `d` independently rounded products that can contribute
///   additive underflow error. Adding non-negative `f32` values whose result is
///   still subnormal is exact, and an addition whose result is normal belongs to
///   the relative term instead.
/// * NEON has at most `d` FMA results that can contribute it, and the
///   horizontal reductions add none when their result is subnormal.
/// * An earlier additive error is amplified by at most `1/(1 - du)` on its way
///   through the rest of the sum. The `x < 0.5` guard gives `du = x/2 < 0.25`,
///   so that factor is below `4/3`, and `d * 2^-150 / (1 - du) < 2d * 2^-150`.
///
/// So `d` local errors, each propagated, still fit inside `2d * 2^-150`.
///
/// Combining them needs care, and two earlier attempts got it wrong. Writing
/// `P` for the pruning sum, `C` for the committed distance and `S` for the
/// exact one:
///
/// ```text
///     P <= (1 + g)S + A          C >= (1 - g)S - A          C < upper
/// ```
///
/// The middle inequality gives `S < (upper + A)/(1 - g)`, and substituting:
///
/// ```text
///     P < R*upper + R*A + A  =  R*upper + (R + 1)*A
/// ```
///
/// So the committed side's additive error is **amplified by `R` as well**. An
/// earlier version used `2A`, short by `(R - 1)A`. At the guard boundary
/// (`x -> 0.5`, so `d -> 2^22` and `R -> 2`) with the smallest accepted cut
/// `f32::MIN_POSITIVE = 2^-126`, that shortfall is `2^-127`: half the cut
/// itself, some four million ULPs, nowhere near something a one-ULP nudge could
/// absorb. Before that, an even earlier version spent the relative headroom
/// twice.
///
/// The threshold is therefore `R*upper + (R + 1)*A`, evaluated in `f64` and
/// rounded up one ULP on the way back to `f32`. Widening is safe in one
/// direction only, which is what makes this the right shape: it can only prune
/// *less*, because membership is still decided by `band.admit` on the exact
/// distance in [`Collector::push`].
fn early_abandon_threshold(band: DistanceBand, dimension: usize) -> f32 {
    // Only an L2 partial sum monotonically lower-bounds the full distance. A
    // partial inner product or cosine accumulation does not bound the final
    // value, because the remaining terms can be either sign.
    if band.metric() != MetricType::L2 {
        return f32::INFINITY;
    }
    let Bound::Finite(upper) = band.upper() else {
        return f32::INFINITY;
    };
    // A subnormal cut's own arithmetic is outside the relative-error model.
    if upper < f32::MIN_POSITIVE {
        return f32::INFINITY;
    }
    let x = dimension as f64 * f64::from(f32::EPSILON);
    // `R = 1/(1 - x)` stays defined up to `x = 1`, so this cap is conservative
    // rather than forced: it holds `R <= 2`, which keeps the widening bounded
    // and the arithmetic away from the pole. No real vector dimension is near
    // it -- `x = 0.5` needs d = 2^22.
    if x >= 0.5 {
        return f32::INFINITY;
    }
    let r = 1.0 / (1.0 - x);
    // `A`: at most `d` roundings per computation can land in the subnormal
    // range, each at most 2^-150, and the factor of two covers propagating them
    // through the rest of the sum -- see the derivation above. Written as a
    // scaled power of two so it stays exact.
    let a = 2.0 * dimension as f64 * f64::powi(2.0, -150);
    // R*upper + (R + 1)*A -- the committed side's additive error is amplified
    // by R too, which is the step the previous version missed.
    let widened = r * f64::from(upper) + (r + 1.0) * a;
    if widened > f64::from(f32::MAX) {
        return f32::INFINITY;
    }
    // Round up one ULP so the f64 -> f32 conversion cannot land below the bound.
    let narrowed = widened as f32;
    let nudged = f32::from_bits(narrowed.to_bits().saturating_add(1));
    if nudged.is_finite() {
        nudged
    } else {
        f32::INFINITY
    }
}

/// Collects the rows falling inside a band. It neither sorts nor truncates:
/// ordering is the caller's business and result caps arrive with later work.
pub(crate) struct RangeCollector {
    band: DistanceBand,
    /// Precomputed early-abandon threshold. Constant for a given band and
    /// dimension, while `cutoff()` is called once per row, so it is derived
    /// here rather than in the scan loop.
    cutoff: f32,
    rows: Vec<(i64, f32)>,
    scanned: usize,
    early_abandoned: usize,
}

impl RangeCollector {
    pub(crate) fn new(band: DistanceBand, dimension: usize) -> Self {
        Self {
            cutoff: early_abandon_threshold(band, dimension),
            band,
            rows: Vec::new(),
            scanned: 0,
            early_abandoned: 0,
        }
    }

    /// Rows read and at least partially evaluated, **including** rows abandoned
    /// early.
    pub(crate) fn scanned(&self) -> usize {
        self.scanned
    }

    pub(crate) fn early_abandoned(&self) -> usize {
        self.early_abandoned
    }

    pub(crate) fn into_rows(self) -> Vec<(i64, f32)> {
        self.rows
    }
}

impl Collector for RangeCollector {
    #[inline]
    fn note_abandoned(&mut self) {
        self.scanned += 1;
        self.early_abandoned += 1;
    }

    #[inline]
    fn cutoff(&self) -> f32 {
        self.cutoff
    }

    #[inline]
    fn push(&mut self, id: i64, value: f32) -> io::Result<()> {
        self.scanned += 1;
        if !value.is_finite() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("non-finite distance {value} computed for row {id}"),
            ));
        }
        if self.band.admit(value) {
            self.rows.push((id, value));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::MetricType;
    use crate::range::{Bound, DistanceBand};

    fn l2_band(lower: f32, upper: f32) -> DistanceBand {
        DistanceBand::new(Bound::Finite(lower), Bound::Finite(upper), MetricType::L2).unwrap()
    }

    /// A dimension below 128, where the two L2 kernels happen to agree exactly,
    /// so the widening margin is negligible but still exercised.
    const TEST_D: usize = 16;

    #[test]
    fn a_range_collector_keeps_only_in_band_rows() {
        let mut collector = RangeCollector::new(l2_band(1.0, 3.0), TEST_D);
        collector.push(10, 0.5).unwrap();
        collector.push(11, 1.0).unwrap();
        collector.push(12, 2.9).unwrap();
        collector.push(13, 3.0).unwrap();
        assert_eq!(collector.into_rows(), vec![(11, 1.0), (12, 2.9)]);
    }

    #[test]
    fn an_exact_l2_collector_reports_a_cutoff_at_or_above_the_upper_cut() {
        // Exact family plus L2: a partial sum above `upper` implies the full
        // distance is above `upper`, so abandoning the row early is sound --
        // but only up to the divergence between the prune kernel and the
        // committed-distance kernel, so the reported cutoff is widened.
        let cutoff = RangeCollector::new(l2_band(1.0, 3.0), TEST_D).cutoff();
        assert!(
            cutoff >= 3.0,
            "the cutoff must never be below the upper cut, or in-band rows are dropped"
        );
        assert!(
            cutoff <= 3.0 * (1.0 + 1e-4),
            "the widening margin must stay negligible, got {cutoff}"
        );
    }

    #[test]
    fn the_widening_margin_dominates_the_summation_error_bound() {
        // The margin must dominate the worst ratio between the two kernels'
        // accumulations. Both fuse multiply-add on AArch64, so the portable
        // conservative model is the dot-product bound with `k = d`, and the
        // requirement then has the closed form `1/(1 - x)` with
        // `x = d * EPSILON`.
        //
        // This is the test the earlier attempts lacked. `1 + d * EPSILON` is the
        // first-order term alone and never dominates, though the shortfall only
        // became visible around d = 4096; every test in place at the time used
        // d <= 768 and stayed green.
        // Sweep several binades: the claim is about arbitrary normal cuts, and
        // fixing one value would leave the final rounding of `upper * margin`
        // untested.
        for upper in [f32::MIN_POSITIVE, 1.0e-20, 0.5, 1.0, 3.0, 1.0e6, 1.0e30] {
            for d in [
                1usize,
                2,
                64,
                128,
                768,
                4095,
                4096,
                8192,
                65536,
                1 << 20,
                1 << 21,
                (1 << 22) - 1,
            ] {
                let cutoff = RangeCollector::new(l2_band(0.0, upper), d).cutoff();
                let x = d as f64 * f64::from(f32::EPSILON);
                // Derived here from the error model rather than copied from the
                // implementation, so the test is not circular:
                //   P <= (1+g)S + A,  C >= (1-g)S - A,  C < upper
                //   => S < (upper + A)/(1-g)
                //   => P < R*upper + (R+1)*A,   R = 1/(1-x)
                // Two earlier versions failed here: one asserted only the
                // relative term, the other used 2A and so missed the R
                // amplification on the committed side.
                let r = 1.0 / (1.0 - x);
                let a = 2.0 * d as f64 * f64::powi(2.0, -150);
                let needed_absolute = r * f64::from(upper) + (r + 1.0) * a;
                // Every pair here is inside both guards, so a non-finite
                // cutoff is a defect rather than a guard firing. Skipping would
                // let an implementation that returned INFINITY for a whole
                // binade pass unnoticed.
                assert!(
                    cutoff.is_finite(),
                    "d={d} upper={upper:e}: expected a finite cutoff inside both guards"
                );
                assert!(
                    f64::from(cutoff) >= needed_absolute,
                    "d={d} upper={upper:e}: threshold {cutoff:e} is below the joint \
                     relative-plus-additive requirement {needed_absolute:e}"
                );
            }
        }
    }

    #[test]
    fn the_margin_declines_to_prune_outside_its_conservative_domain() {
        // `R = 1/(1 - x)` stays defined to `x = 1`, so this cap is conservative
        // rather than forced: it holds `R <= 2` and keeps the widening bounded.
        // Refusing beyond it costs pruning and nothing else.
        let huge = 1usize << 22; // d * EPSILON == 0.5 exactly
        assert_eq!(
            RangeCollector::new(l2_band(0.0, 3.0), huge).cutoff(),
            f32::INFINITY,
            "the guard must trip once d * EPSILON reaches 0.5"
        );
        assert!(
            RangeCollector::new(l2_band(0.0, 3.0), huge - 1)
                .cutoff()
                .is_finite(),
            "just inside the guard a finite margin is still produced"
        );

        // Subnormal cuts: the relative-error model does not survive gradual
        // underflow, and `upper * margin` can round back to `upper`.
        let subnormal = f32::from_bits(1);
        let band = DistanceBand::new(Bound::Finite(0.0), Bound::Finite(subnormal), MetricType::L2)
            .unwrap();
        assert_eq!(
            RangeCollector::new(band, 768).cutoff(),
            f32::INFINITY,
            "a subnormal upper cut must not prune"
        );
    }

    #[test]
    fn the_widening_margin_grows_with_dimension() {
        // The divergence between the two kernels comes from the prune kernel's
        // per-128-block scalar reduction, so it grows with d.
        let narrow = RangeCollector::new(l2_band(0.0, 3.0), 16).cutoff();
        let wide = RangeCollector::new(l2_band(0.0, 3.0), 768).cutoff();
        assert!(
            wide > narrow,
            "a larger d must widen the margin: {narrow} vs {wide}"
        );
        assert!(wide >= 3.0 && narrow >= 3.0);
    }

    #[test]
    fn an_unbounded_upper_reports_no_cutoff() {
        let band = DistanceBand::new(Bound::Finite(1.0), Bound::Unbounded, MetricType::L2).unwrap();
        assert_eq!(RangeCollector::new(band, TEST_D).cutoff(), f32::INFINITY);
    }

    #[test]
    fn an_uncertified_metric_never_prunes_on_a_partial_sum() {
        // A partial cosine or inner-product accumulation does not bound the full
        // value, so the cutoff must stay infinite no matter what the band says.
        for metric in [MetricType::Cosine, MetricType::InnerProduct] {
            let band = DistanceBand::new(Bound::Finite(0.1), Bound::Finite(0.5), metric).unwrap();
            assert_eq!(
                RangeCollector::new(band, TEST_D).cutoff(),
                f32::INFINITY,
                "metric {metric:?} must not expose a finite cutoff"
            );
        }
    }

    #[test]
    fn the_collector_rejects_a_non_finite_computed_value() {
        // Fail loud on every family: under inner product -inf means "extremely
        // similar", so dropping it silently would erase a row that should have
        // matched, and the donor's "exact families may drop it" rule was only
        // ever derived for L2.
        //
        // Scope: this is the *collector's* contract, covering every path that
        // computes a full value. It deliberately does not cover early abandon,
        // which never reaches `push` -- an abandoned row provably satisfies
        // "full distance > upper", so classifying it out of band is the right
        // answer whether or not the computation overflowed.
        let mut collector = RangeCollector::new(l2_band(0.0, 10.0), TEST_D);
        assert!(collector.push(1, f32::NAN).is_err());
        assert!(collector.push(2, f32::INFINITY).is_err());
    }

    #[test]
    fn early_abandoned_rows_are_counted_as_scanned() {
        // rows_scanned means "rows read and at least partially evaluated", which
        // includes abandoned rows. Those never reach push, so the scan kernel
        // has to report them through note_abandoned.
        let mut collector = RangeCollector::new(l2_band(1.0, 2.0), TEST_D);
        collector.note_abandoned();
        collector.push(1, 1.5).unwrap();
        assert_eq!(collector.scanned(), 2);
        assert_eq!(collector.early_abandoned(), 1);
    }

    #[test]
    fn rows_are_counted_even_when_rejected() {
        let mut collector = RangeCollector::new(l2_band(1.0, 2.0), TEST_D);
        collector.push(1, 0.0).unwrap();
        collector.push(2, 1.5).unwrap();
        assert_eq!(collector.scanned(), 2);
    }
}
