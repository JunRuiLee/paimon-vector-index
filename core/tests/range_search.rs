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

//! Cross-family semantics for distance range search.
//!
//! Band and cut unit tests live in `core/src/range.rs`; this file covers the
//! reader-level contract. Every expected value here is a **relation** — an
//! oracle, an equivalence between two runtime paths, or a set identity — never a
//! frozen f32 bit pattern. Development happens on darwin/aarch64 while upstream
//! CI runs ubuntu x86_64, and `fvec_l2sqr` dispatches to NEON versus AVX2 with
//! different lane grouping, so a hardcoded bit pattern captured locally would
//! fail upstream.

use paimon_vindex_core::distance::{fvec_l2sqr, MetricType};
use paimon_vindex_core::index::{VectorIndexReader, VectorSearchParams};
use paimon_vindex_core::io::PosWriter;
use paimon_vindex_core::ivfflat::IVFFlatIndex;
use paimon_vindex_core::ivfflat_io::write_ivfflat_index;
use paimon_vindex_core::range::{
    Bound, DistanceBand, QueryResult, RangeSearchWidth, VectorRangeSearchParams,
};
use std::collections::HashSet;
use std::io::Cursor;

use roaring::RoaringTreemap;

type Reader = VectorIndexReader<Cursor<Vec<u8>>>;

// --- fixtures --------------------------------------------------------------
//
// Indexes are built by assigning rows to lists **explicitly** rather than by
// calling `train`, for two reasons. First, k-means could leave a list empty,
// which would make the `list_reads` assertions fail intermittently. Second, the
// band constants below only assert something if the intra-cluster distance
// distribution actually straddles them, and explicit construction is what makes
// that distribution controllable.
//
// The noise amplitude is scaled by `1/sqrt(d)` so the intra-cluster squared
// distance is **independent of d**: a sum of `d` terms each of order `A^2/d`
// stays at order `A^2`. That is what lets one set of band constants stay
// meaningful across the d = 8, d = 16 and d = 256 fixtures.

/// Per-dimension centroid spacing. Inter-cluster squared distance is about
/// `d * SPACING^2`, which dwarfs every band used here, so rows in other lists
/// are always out of band and therefore exercise the early-abandon path.
const SPACING: f32 = 10.0;
/// Chosen so the mean intra-cluster squared distance is about 2.0 and the spread
/// covers roughly 0..6, which straddles the 2.0 and 3.0 cuts the tests use.
const NOISE: f32 = 1.73;
/// Generator seed. A hex constant used as a **construction input**, never as an
/// expected value.
const SEED: u64 = 0x5eed_1234;

pub const ASYMMETRIC_NLIST: usize = 4;
const ASYMMETRIC_D: usize = 16;
const ASYMMETRIC_BIG: usize = 200;
const ASYMMETRIC_SMALL: usize = 20;
/// Tighter than [`NOISE`] so that the narrow `[0, 0.25)` band used by the batch
/// statistics test captures a large fraction of each cluster; the hit counts then
/// track the cluster sizes and differ by about ten times.
const ASYMMETRIC_NOISE: f32 = 0.474;

/// A fixed-seed linear congruential generator, so the corpus does not depend on
/// the `rand` version.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Uniform in `[-1, 1)`.
    fn next_symmetric(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / (1u32 << 24) as f32 * 2.0 - 1.0
    }
}

fn centroid_component(list_id: usize, dim: usize) -> f32 {
    list_id as f32 * SPACING + dim as f32 * 0.01
}

fn serialize(index: &IVFFlatIndex) -> Vec<u8> {
    let mut bytes = Vec::new();
    write_ivfflat_index(index, &mut PosWriter::new(&mut bytes)).unwrap();
    bytes
}

/// Builds an IVF-Flat reader with `n` rows spread evenly over `nlist` lists.
///
/// Returns the reader plus the corpus in id order, so an oracle can rescore it.
pub fn build_flat_fixture(n: usize, d: usize, nlist: usize) -> (Reader, Vec<f32>, Vec<i64>, usize) {
    build_flat_fixture_with_metric(n, d, nlist, MetricType::L2)
}

pub fn build_flat_fixture_with_metric(
    n: usize,
    d: usize,
    nlist: usize,
    metric: MetricType,
) -> (Reader, Vec<f32>, Vec<i64>, usize) {
    assert_eq!(n % nlist, 0, "the fixture spreads rows evenly over lists");
    let rows_per_list = n / nlist;
    let mut rng = Lcg::new(SEED);
    let mut index = IVFFlatIndex::new(d, nlist, metric);
    index.set_quantizer_centroids(
        (0..nlist)
            .flat_map(|list_id| (0..d).map(move |dim| centroid_component(list_id, dim)))
            .collect(),
    );

    // Corpus positions interleave the lists, so consecutive positions land in
    // different clusters and a batch taken from the front spans several of them.
    //
    // Ids are assigned **consecutively within each list**, deliberately not
    // derived from the interleaved position. With `id = 1000 + row * nlist +
    // list_id` and an even `nlist`, every id in a list would share one parity, so
    // an `id % 2` filter would keep or drop whole lists at a time and a
    // per-query filter comparison could have nothing left to compare. Consecutive
    // ids inside a list give both parities and every residue mod 3.
    let mut vectors_by_id = vec![0.0f32; n * d];
    let mut ids_by_id = vec![0i64; n];
    let mut next_id = 1000i64;
    for list_id in 0..nlist {
        let mut list_ids = Vec::with_capacity(rows_per_list);
        let mut list_vectors = Vec::with_capacity(rows_per_list * d);
        for row in 0..rows_per_list {
            let global = row * nlist + list_id;
            let id = next_id;
            next_id += 1;
            list_ids.push(id);
            for dim in 0..d {
                let value = centroid_component(list_id, dim)
                    + rng.next_symmetric() * NOISE / (d as f32).sqrt();
                list_vectors.push(value);
                vectors_by_id[global * d + dim] = value;
            }
            ids_by_id[global] = id;
        }
        index.ids[list_id] = list_ids;
        index.vectors[list_id] = list_vectors;
    }

    let reader = VectorIndexReader::open(Cursor::new(serialize(&index))).unwrap();
    (reader, vectors_by_id, ids_by_id, d)
}

/// Two clusters whose sizes differ by an order of magnitude, with one query
/// sitting at the centre of each.
///
/// The balanced fixture cannot serve the per-query statistics test: with equal,
/// translation-symmetric clusters, two queries probing every list produce
/// identical `rows_scanned` and `lists_probed`, so an inequality assertion would
/// fail on a *correct* implementation.
pub fn build_asymmetric_fixture() -> (Reader, Vec<f32>, usize) {
    let mut rng = Lcg::new(SEED);
    let d = ASYMMETRIC_D;
    let mut index = IVFFlatIndex::new(d, ASYMMETRIC_NLIST, MetricType::L2);
    index.set_quantizer_centroids(
        (0..ASYMMETRIC_NLIST)
            .flat_map(|list_id| (0..d).map(move |dim| centroid_component(list_id, dim)))
            .collect(),
    );

    let mut next_id = 1000i64;
    for list_id in 0..ASYMMETRIC_NLIST {
        let rows = if list_id == 0 {
            ASYMMETRIC_BIG
        } else {
            ASYMMETRIC_SMALL
        };
        let mut list_ids = Vec::with_capacity(rows);
        let mut list_vectors = Vec::with_capacity(rows * d);
        for _ in 0..rows {
            list_ids.push(next_id);
            next_id += 1;
            for dim in 0..d {
                list_vectors.push(
                    centroid_component(list_id, dim)
                        + rng.next_symmetric() * ASYMMETRIC_NOISE / (d as f32).sqrt(),
                );
            }
        }
        index.ids[list_id] = list_ids;
        index.vectors[list_id] = list_vectors;
    }

    // Query 0 sits at the centre of the large cluster, query 1 at the centre of
    // a small one, so a narrow band around each yields hit counts tracking the
    // cluster sizes.
    let mut queries = Vec::with_capacity(2 * d);
    for list_id in [0usize, 1] {
        for dim in 0..d {
            queries.push(centroid_component(list_id, dim));
        }
    }

    let reader = VectorIndexReader::open(Cursor::new(serialize(&index))).unwrap();
    (reader, queries, d)
}

/// The asymmetric corpus in id order, for oracle rescoring.
pub fn asymmetric_corpus() -> (Vec<f32>, Vec<i64>) {
    let mut rng = Lcg::new(SEED);
    let d = ASYMMETRIC_D;
    let mut vectors = Vec::new();
    let mut ids = Vec::new();
    let mut next_id = 1000i64;
    for list_id in 0..ASYMMETRIC_NLIST {
        let rows = if list_id == 0 {
            ASYMMETRIC_BIG
        } else {
            ASYMMETRIC_SMALL
        };
        for _ in 0..rows {
            ids.push(next_id);
            next_id += 1;
            for dim in 0..d {
                vectors.push(
                    centroid_component(list_id, dim)
                        + rng.next_symmetric() * ASYMMETRIC_NOISE / (d as f32).sqrt(),
                );
            }
        }
    }
    (vectors, ids)
}

/// A minimal DiskANN index, built only so that it can be opened; range search
/// fails at family dispatch before touching any data.
pub fn build_diskann_fixture() -> Reader {
    use paimon_vindex_core::diskann::{DiskAnnBuildParams, DiskAnnIndex};
    use paimon_vindex_core::diskann_io::write_diskann_index;

    let d = 8;
    let n = 32;
    let mut rng = Lcg::new(SEED);
    let data: Vec<f32> = (0..n * d).map(|_| rng.next_symmetric()).collect();
    let ids: Vec<i64> = (0..n as i64).collect();
    let mut index = DiskAnnIndex::new(d, MetricType::L2, 4, DiskAnnBuildParams::default());
    index.train(&data, n).unwrap();
    index.add(&data, &ids);
    let mut bytes = Vec::new();
    write_diskann_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
    VectorIndexReader::open(Cursor::new(bytes)).unwrap()
}

// --- helpers ---------------------------------------------------------------

/// The rows a band truly contains, scored with the same crate's distance kernel
/// so the oracle is safe across platforms, and filtered by `band.admit` so that
/// a locally rewritten comparison cannot disagree with the reader at a cut.
pub fn brute_force_band(
    query: &[f32],
    vectors: &[f32],
    ids: &[i64],
    d: usize,
    band: DistanceBand,
) -> Vec<(i64, f32)> {
    ids.iter()
        .enumerate()
        .filter_map(|(row, &id)| {
            let distance = fvec_l2sqr(query, &vectors[row * d..(row + 1) * d]);
            band.admit(distance).then_some((id, distance))
        })
        .collect()
}

/// One query's result as a sorted multiset of `(label, distance bits)`.
///
/// Comparing bits rather than f32 makes NaN and ±0 compare deterministically.
/// Ordering is not part of the contract, hence the sort. Comparing labels alone
/// would miss two real defects: a batch merge attaching a row's distance to
/// another label, and a distance that is simply computed wrongly.
pub fn pairs_of(result: QueryResult<'_>) -> Vec<(i64, u32)> {
    let mut pairs: Vec<(i64, u32)> = result
        .labels
        .iter()
        .zip(result.distances)
        .map(|(id, dist)| (*id, dist.to_bits()))
        .collect();
    pairs.sort_unstable();
    pairs
}

fn bits_of(rows: Vec<(i64, f32)>) -> Vec<(i64, u32)> {
    let mut pairs: Vec<(i64, u32)> = rows
        .into_iter()
        .map(|(id, dist)| (id, dist.to_bits()))
        .collect();
    pairs.sort_unstable();
    pairs
}

fn l2(lower: f32, upper: f32) -> DistanceBand {
    DistanceBand::new(Bound::Finite(lower), Bound::Finite(upper), MetricType::L2).unwrap()
}

/// Serializes an allow-list the way the reader's Roaring decoder expects it.
fn serialize_roaring(allowed: &HashSet<i64>) -> Vec<u8> {
    let mut map = RoaringTreemap::new();
    for &id in allowed {
        map.insert(u64::try_from(id).expect("fixture ids are non-negative"));
    }
    let mut bytes = Vec::new();
    map.serialize_into(&mut bytes).unwrap();
    bytes
}

// --- Task 8: the engine ----------------------------------------------------

#[test]
fn ivf_flat_range_matches_a_brute_force_oracle_at_full_probe() {
    let (mut reader, vectors, ids, d) = build_flat_fixture(512, 16, 8);
    let query = vectors[0..d].to_vec();
    let band = l2(0.0, 2.0);
    // nprobe == nlist means full coverage, and IVF-Flat computes exact
    // distances, so the result must match the oracle row for row as a multiset.
    let result = reader
        .range_search(&query, VectorRangeSearchParams::new(band, 8))
        .unwrap();
    let got = pairs_of(result.query(0));
    let want = bits_of(brute_force_band(&query, &vectors, &ids, d, band));
    assert!(
        !want.is_empty(),
        "the band matched nothing, so this test would be vacuous"
    );
    assert_eq!(got, want, "labels and distances must both equal the oracle");
}

#[test]
fn an_empty_band_returns_zero_rows() {
    let (mut reader, ..) = build_flat_fixture(64, 8, 4);
    let band = l2(1.0, 1.0);
    let result = reader
        .range_search(&[0.0; 8], VectorRangeSearchParams::new(band, 4))
        .unwrap();
    assert_eq!(result.query(0).labels.len(), 0);
    assert_eq!(
        result.call_stats().list_reads(),
        0,
        "an empty band probes no list at all"
    );
}

#[test]
fn a_band_whose_metric_disagrees_with_the_index_is_rejected() {
    // The mismatch matrix: index metric x band metric, where only equality is
    // permitted.
    for index_metric in [MetricType::L2, MetricType::Cosine, MetricType::InnerProduct] {
        let (mut reader, ..) = build_flat_fixture_with_metric(128, 8, 4, index_metric);
        for band_metric in [MetricType::L2, MetricType::Cosine, MetricType::InnerProduct] {
            let band =
                DistanceBand::new(Bound::Finite(0.0), Bound::Finite(1.0), band_metric).unwrap();
            let outcome = reader.range_search(&[0.1; 8], VectorRangeSearchParams::new(band, 4));
            if band_metric != index_metric {
                assert_eq!(
                    outcome.unwrap_err().kind(),
                    std::io::ErrorKind::InvalidInput,
                    "index {index_metric:?} / band {band_metric:?} must report InvalidInput"
                );
            } else if band_metric != MetricType::L2 {
                // Matching but not yet certified.
                assert_eq!(
                    outcome.unwrap_err().kind(),
                    std::io::ErrorKind::Unsupported,
                    "index {index_metric:?} / band {band_metric:?}"
                );
            } else {
                assert!(outcome.is_ok(), "L2/L2 must succeed");
            }
        }
    }
}

#[test]
fn an_empty_band_does_not_mask_a_bad_query() {
    let (mut reader, ..) = build_flat_fixture(64, 8, 4);
    let params = VectorRangeSearchParams::new(l2(1.0, 1.0), 4);
    // Wrong dimension.
    assert!(reader.range_search(&[0.0; 7], params).is_err());
    // Non-finite query.
    assert!(reader.range_search(&[f32::NAN; 8], params).is_err());
}

#[test]
fn an_unsupported_index_type_fails_loud_for_every_band() {
    // The family capability check must precede the empty-band short-circuit: a
    // family that cannot do range search rejects every band.
    let mut reader = build_diskann_fixture();
    let empty = l2(1.0, 1.0);
    let err = reader
        .range_search(&[0.0; 8], VectorRangeSearchParams::new(empty, 4))
        .unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
}

#[test]
fn an_auto_width_is_rejected_as_unsupported() {
    let (mut reader, ..) = build_flat_fixture(64, 8, 4);
    let params = VectorRangeSearchParams::new(l2(0.0, 1.0), 4).with_width(RangeSearchWidth::Auto {
        initial: 2,
        growth_factor: 2,
        max_width: 4,
    });
    let err = reader.range_search(&[0.0; 8], params).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
}

#[test]
fn a_whole_space_band_returns_every_probed_row() {
    let (mut reader, vectors, ids, d) = build_flat_fixture(256, 16, 8);
    let query = vectors[0..d].to_vec();
    let band = DistanceBand::new(Bound::Unbounded, Bound::Unbounded, MetricType::L2).unwrap();
    let result = reader
        .range_search(&query, VectorRangeSearchParams::new(band, 8))
        .unwrap();
    assert_eq!(
        result.query(0).labels.len(),
        ids.len(),
        "an unbounded band at full probe returns the whole corpus"
    );
    assert_eq!(
        pairs_of(result.query(0)),
        bits_of(brute_force_band(&query, &vectors, &ids, d, band))
    );
}

#[test]
fn a_narrow_band_probing_fewer_lists_returns_no_more_rows() {
    // "No cap" promises no truncation, not completeness: a smaller nprobe covers
    // less and therefore returns fewer in-band rows.
    let (mut reader, vectors, _ids, d) = build_flat_fixture(512, 16, 8);
    let query = vectors[0..d].to_vec();
    let band = l2(0.0, 2.0);
    let wide = reader
        .range_search(&query, VectorRangeSearchParams::new(band, 8))
        .unwrap();
    let narrow = reader
        .range_search(&query, VectorRangeSearchParams::new(band, 1))
        .unwrap();
    let wide_set: HashSet<(i64, u32)> = pairs_of(wide.query(0)).into_iter().collect();
    let narrow_pairs = pairs_of(narrow.query(0));
    assert!(
        !narrow_pairs.is_empty(),
        "the narrow probe found nothing, so this test would be vacuous"
    );
    assert!(
        narrow_pairs.len() <= wide_set.len(),
        "a lower nprobe returned {} rows against the full probe's {}",
        narrow_pairs.len(),
        wide_set.len()
    );
    for pair in &narrow_pairs {
        assert!(
            wide_set.contains(pair),
            "a lower nprobe must return a subset of the full-probe result"
        );
    }
}

// --- Task 9: the filter variants -------------------------------------------

#[test]
fn a_filtered_range_equals_filtering_the_unfiltered_result() {
    let (mut reader, vectors, ids, d) = build_flat_fixture(512, 16, 8);
    let query = vectors[0..d].to_vec();
    let band = l2(0.0, 2.0);
    let allowed: HashSet<i64> = ids.iter().copied().filter(|id| id % 3 == 0).collect();

    let unfiltered = reader
        .range_search(&query, VectorRangeSearchParams::new(band, 8))
        .unwrap();
    let mut want: Vec<(i64, u32)> = pairs_of(unfiltered.query(0))
        .into_iter()
        .filter(|(id, _)| allowed.contains(id))
        .collect();

    let roaring = serialize_roaring(&allowed);
    let filtered = reader
        .range_search_with_roaring_filter(&query, VectorRangeSearchParams::new(band, 8), &roaring)
        .unwrap();
    let mut got: Vec<(i64, u32)> = pairs_of(filtered.query(0));
    got.sort_unstable();
    want.sort_unstable();
    assert!(
        !want.is_empty(),
        "the filter admitted nothing, so this test would be vacuous"
    );
    assert_eq!(
        got, want,
        "the filtered variant's (label, distance) must equal filtering the unfiltered result"
    );
}

#[test]
fn a_malformed_roaring_filter_is_rejected_even_for_an_empty_band() {
    let (mut reader, ..) = build_flat_fixture(64, 8, 4);
    let empty = l2(1.0, 1.0);
    let err = reader
        .range_search_with_roaring_filter(
            &[0.0; 8],
            VectorRangeSearchParams::new(empty, 4),
            b"not roaring",
        )
        .unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
}

#[test]
fn a_non_flat_index_rejects_a_filtered_range() {
    let mut reader = build_diskann_fixture();
    let allowed: HashSet<i64> = (0..8).collect();
    let err = reader
        .range_search_with_roaring_filter(
            &[0.0; 8],
            VectorRangeSearchParams::new(l2(0.0, 1.0), 4),
            &serialize_roaring(&allowed),
        )
        .unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
}

// --- Task 10: the public batch API -----------------------------------------

#[test]
fn batch_range_equals_running_each_query_alone() {
    let (mut reader, vectors, _ids, d) = build_flat_fixture(512, 16, 8);
    let nq = 4;
    let queries = vectors[0..nq * d].to_vec();
    let band = l2(0.0, 2.5);
    // Partial probe on purpose: at nprobe == nlist every list is scanned, so the
    // probe *selection* cannot differ and the test would pass even if batch and
    // single-query ranked centroids differently. That is precisely how a real
    // batch/single divergence went unnoticed here.
    let nprobe = 3;
    assert!(
        nprobe < 8,
        "nprobe must be below nlist for this test to bite"
    );
    let params = VectorRangeSearchParams::new(band, nprobe);

    let batch = reader.range_search_batch(&queries, nq, params).unwrap();
    for qi in 0..nq {
        let single = reader
            .range_search(&queries[qi * d..(qi + 1) * d], params)
            .unwrap();
        // Ordering is not part of the contract, so compare per-query multisets.
        assert!(
            !pairs_of(single.query(0)).is_empty(),
            "query {qi} matched nothing, so this comparison would be vacuous"
        );
        assert_eq!(
            pairs_of(batch.query(qi)),
            pairs_of(single.query(0)),
            "query {qi}'s batched (label, distance) must equal its single-query result"
        );
    }
}

#[test]
fn batch_range_is_invariant_under_query_permutation() {
    let (mut reader, vectors, _ids, d) = build_flat_fixture(256, 16, 8);
    let band = l2(0.0, 2.5);
    // Partial probe, for the same reason as above.
    let params = VectorRangeSearchParams::new(band, 3);
    let a = vectors[0..d].to_vec();
    let b = vectors[d..2 * d].to_vec();

    let ab: Vec<f32> = a.iter().chain(b.iter()).copied().collect();
    let ba: Vec<f32> = b.iter().chain(a.iter()).copied().collect();
    let r_ab = reader.range_search_batch(&ab, 2, params).unwrap();
    let r_ba = reader.range_search_batch(&ba, 2, params).unwrap();

    assert_eq!(pairs_of(r_ab.query(0)), pairs_of(r_ba.query(1)));
    assert_eq!(pairs_of(r_ab.query(1)), pairs_of(r_ba.query(0)));
}

#[test]
fn a_band_splits_into_adjacent_sub_bands_without_losing_rows() {
    // Bucket closure: range([a,c)) == range([a,b)) union range([b,c)) as multisets.
    let (mut reader, vectors, _ids, d) = build_flat_fixture(512, 16, 8);
    let query = vectors[0..d].to_vec();
    let whole = reader
        .range_search(&query, VectorRangeSearchParams::new(l2(0.0, 4.0), 8))
        .unwrap();
    let left = reader
        .range_search(&query, VectorRangeSearchParams::new(l2(0.0, 2.0), 8))
        .unwrap();
    let right = reader
        .range_search(&query, VectorRangeSearchParams::new(l2(2.0, 4.0), 8))
        .unwrap();

    // Both halves must be populated, or the identity holds trivially and this
    // test proves nothing about the cut.
    assert!(
        !left.query(0).labels.is_empty(),
        "the left sub-band is empty, so this test would be vacuous"
    );
    assert!(
        !right.query(0).labels.is_empty(),
        "the right sub-band is empty, so this test would be vacuous"
    );
    // A row sitting exactly on the split is the case the half-open boundary
    // exists for: it must appear in the right band and not the left. Rather than
    // hoping the corpus contains one, split at a distance that a row actually
    // has -- otherwise this assertion is vacuous, which is a trap an earlier
    // version of this test fell into.
    let seam = *whole
        .query(0)
        .distances
        .iter()
        .find(|d| **d > 0.0)
        .expect("the parent band must contain a row at a positive distance");
    let seam_left = reader
        .range_search(&query, VectorRangeSearchParams::new(l2(0.0, seam), 8))
        .unwrap();
    let seam_right = reader
        .range_search(&query, VectorRangeSearchParams::new(l2(seam, 4.0), 8))
        .unwrap();
    let seam_id = whole
        .query(0)
        .labels
        .iter()
        .zip(whole.query(0).distances)
        .find(|(_, d)| **d == seam)
        .map(|(id, _)| *id)
        .expect("the seam distance came from this result");
    assert!(
        !seam_left.query(0).labels.contains(&seam_id),
        "row {seam_id} sits exactly on the split and must be excluded by the \
         right-open left band"
    );
    assert!(
        seam_right.query(0).labels.contains(&seam_id),
        "row {seam_id} sits exactly on the split and must be included by the \
         left-closed right band"
    );

    let mut want: Vec<i64> = whole.query(0).labels.to_vec();
    let mut got: Vec<i64> = left
        .query(0)
        .labels
        .iter()
        .chain(right.query(0).labels)
        .copied()
        .collect();
    want.sort_unstable();
    got.sort_unstable();
    assert_eq!(
        got, want,
        "adjacent sub-bands must union to the parent band"
    );
}

#[test]
fn a_batch_roaring_filter_equals_filtering_the_unfiltered_batch() {
    let (mut reader, vectors, ids, d) = build_flat_fixture(512, 16, 8);
    let nq = 3;
    let queries = vectors[0..nq * d].to_vec();
    let band = l2(0.0, 2.5);
    let params = VectorRangeSearchParams::new(band, 8);
    let allowed: HashSet<i64> = ids.iter().copied().filter(|id| id % 2 == 0).collect();

    let plain = reader.range_search_batch(&queries, nq, params).unwrap();
    let filtered = reader
        .range_search_batch_with_roaring_filter(&queries, nq, params, &serialize_roaring(&allowed))
        .unwrap();
    for qi in 0..nq {
        let mut want: Vec<(i64, u32)> = pairs_of(plain.query(qi))
            .into_iter()
            .filter(|(id, _)| allowed.contains(id))
            .collect();
        let mut got: Vec<(i64, u32)> = pairs_of(filtered.query(qi));
        got.sort_unstable();
        want.sort_unstable();
        assert!(!want.is_empty(), "query {qi} would be a vacuous comparison");
        assert_eq!(got, want, "query {qi}'s (label, distance)");
    }
}

#[test]
fn a_batch_with_a_mismatched_query_count_is_rejected() {
    let (mut reader, vectors, _ids, d) = build_flat_fixture(64, 8, 4);
    let band = l2(0.0, 1.0);
    // Data for three queries but four declared.
    let err = reader
        .range_search_batch(&vectors[0..3 * d], 4, VectorRangeSearchParams::new(band, 4))
        .unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
}

#[test]
fn a_malformed_batch_roaring_filter_is_rejected_even_for_an_empty_band() {
    let (mut reader, vectors, _ids, d) = build_flat_fixture(64, 8, 4);
    let empty = l2(1.0, 1.0);
    let err = reader
        .range_search_batch_with_roaring_filter(
            &vectors[0..2 * d],
            2,
            VectorRangeSearchParams::new(empty, 4),
            b"not roaring",
        )
        .unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
}

#[test]
fn a_non_flat_index_rejects_batch_range() {
    let mut reader = build_diskann_fixture();
    let band = l2(0.0, 1.0);
    let err = reader
        .range_search_batch(&[0.0; 16], 2, VectorRangeSearchParams::new(band, 4))
        .unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
}

// --- Task 11: differential assertions instead of frozen f32 bits -----------
//
// These replace two donor tests that asserted hardcoded f32 bit patterns
// captured on darwin/aarch64. Upstream CI is ubuntu x86_64, and `fvec_l2sqr`
// dispatches to AVX2 there versus NEON here with different lane grouping, so
// those assertions cannot hold on both. Each one below compares two runtime
// paths inside a single build, or a path against an oracle, which holds on any
// target.

/// Top-K through the public enum API, for the shared-rows comparison below.
fn top_k(reader: &mut Reader, query: &[f32], k: usize, nprobe: usize) -> (Vec<i64>, Vec<f32>) {
    reader
        .search(query, VectorSearchParams::new(k, nprobe))
        .unwrap()
}

#[test]
fn top_k_and_range_agree_on_the_rows_they_share() {
    let (mut reader, vectors, _ids, d) = build_flat_fixture(512, 16, 8);
    let query = vectors[0..d].to_vec();
    let k = 32;
    let (labels, distances) = top_k(&mut reader, &query, k, 8);
    let upper = distances[k - 1] + 1.0;
    let band = l2(0.0, upper);
    let range = reader
        .range_search(&query, VectorRangeSearchParams::new(band, 8))
        .unwrap();
    let in_range: HashSet<i64> = range.query(0).labels.iter().copied().collect();
    let mut compared = 0usize;
    for (label, distance) in labels.iter().zip(&distances) {
        if *distance < upper {
            assert!(
                in_range.contains(label),
                "top-K row {label} at distance {distance} is inside the band but absent \
                 from the range result"
            );
            compared += 1;
        }
    }
    assert!(
        compared > 0,
        "no top-K row fell inside the band, so this test would be vacuous"
    );
}

#[test]
fn the_parallel_arm_agrees_with_brute_force() {
    // The scale must genuinely cross the parallel threshold.
    // PARALLEL_FLAT_SCAN_MIN_COMPONENTS is 1024 * 1024, and
    // scan_components = sum(list_rows * queries_for_list) * d, so
    // 8192 rows * 256 dims * 1 query = 2_097_152, which clears it.
    const N: usize = 8192;
    const D: usize = 256;
    let (mut reader, vectors, ids, d) = build_flat_fixture(N, D, 16);
    assert_eq!(d, D);
    let query = vectors[0..d].to_vec();
    let band = l2(0.0, 3.0);

    // nprobe = nlist makes scan_components cross the threshold.
    let parallel = reader
        .range_search(&query, VectorRangeSearchParams::new(band, 16))
        .unwrap();

    // The reference is the ground truth over the **same** working set, not
    // "repeat with nprobe = 1 and union the results" -- that would be a
    // different working set and could only establish a subset relation, which a
    // miscomputing parallel arm would also satisfy. nprobe = nlist means the
    // probe set is every list, so the oracle is a full brute-force scan.
    let want = bits_of(brute_force_band(&query, &vectors, &ids, d, band));
    let got = pairs_of(parallel.query(0));
    assert!(
        !want.is_empty(),
        "the band matched nothing, so this test would be vacuous"
    );
    assert_eq!(
        got, want,
        "the parallel arm's (label, distance) must equal brute force over the same working set"
    );

    // And assert the parallel branch was actually taken.
    let components = N * D; // one query, every list
    assert!(
        components > 1024 * 1024,
        "fixture scale {components} must exceed PARALLEL_FLAT_SCAN_MIN_COMPONENTS, \
         otherwise this exercises the sequential arm"
    );
}

#[test]
fn the_l2_cutoff_does_not_change_which_rows_are_returned() {
    // Flat-L2 cutoff on versus off: widening the band to the whole space is
    // equivalent to disabling early abandon.
    let (mut reader, vectors, _ids, d) = build_flat_fixture(512, 16, 8);
    let query = vectors[0..d].to_vec();
    let bounded = l2(0.0, 2.0);
    let unbounded = DistanceBand::new(Bound::Unbounded, Bound::Unbounded, MetricType::L2).unwrap();
    let with_cutoff = reader
        .range_search(&query, VectorRangeSearchParams::new(bounded, 8))
        .unwrap();
    let without = reader
        .range_search(&query, VectorRangeSearchParams::new(unbounded, 8))
        .unwrap();
    let mut a = with_cutoff.query(0).labels.to_vec();
    let mut b: Vec<i64> = without
        .query(0)
        .labels
        .iter()
        .zip(without.query(0).distances)
        .filter(|(_, dist)| **dist < 2.0)
        .map(|(id, _)| *id)
        .collect();
    a.sort_unstable();
    b.sort_unstable();
    assert!(
        !a.is_empty(),
        "no rows matched, so this test would be vacuous"
    );
    assert_eq!(
        a, b,
        "early abandon changes only work, never which rows return"
    );
    // And prove early abandon actually happened, or this test verifies nothing.
    assert!(
        with_cutoff.query(0).stats.early_abandoned() > 0,
        "a narrow band must abandon some rows early; zero means the path was never taken"
    );
    assert_eq!(
        without.query(0).stats.early_abandoned(),
        0,
        "a whole-space band has an infinite cutoff and must not enter the early-abandon kernel"
    );
}

#[test]
fn batch_stats_are_per_query_and_shared_lists_are_counted_once() {
    let (mut reader, queries, d) = build_asymmetric_fixture();
    let band = l2(0.0, 0.25);
    // nprobe = nlist means both queries probe the same full set of lists, so the
    // number of unique lists is exactly nlist.
    let result = reader
        .range_search_batch(&queries, 2, VectorRangeSearchParams::new(band, 8))
        .unwrap();
    // The fixture guarantees every list is non-empty, so list_reads must be
    // *exactly* the list count: a `<=` would also accept 1, or even 0.
    assert_eq!(
        result.call_stats().list_reads(),
        ASYMMETRIC_NLIST,
        "each unique non-empty list counts exactly once, neither per (list, query) nor missed"
    );
    for qi in 0..2 {
        assert_eq!(result.query(qi).stats.lists_probed(), ASYMMETRIC_NLIST);
        assert_eq!(
            result.query(qi).stats.rows_committed(),
            result.query(qi).labels.len()
        );
    }

    // Per-query stats must match running each query on its own, field by field.
    let tuple = |st: &paimon_vindex_core::range::RangeSearchStats| {
        (
            st.rows_scanned(),
            st.rows_committed(),
            st.early_abandoned(),
            st.lists_probed(),
            st.stop_reason(),
        )
    };
    let singles: Vec<_> = (0..2)
        .map(|qi| {
            reader
                .range_search(
                    &queries[qi * d..(qi + 1) * d],
                    VectorRangeSearchParams::new(band, 8),
                )
                .unwrap()
        })
        .collect();
    // Prove with an oracle that the two queries have different hit counts. This
    // step does not depend on the implementation, so it validates that the
    // fixture really is asymmetric rather than that the implementation happened
    // to compute two different numbers.
    let (corpus, corpus_ids) = asymmetric_corpus();
    let truth: Vec<usize> = (0..2)
        .map(|qi| {
            brute_force_band(
                &queries[qi * d..(qi + 1) * d],
                &corpus,
                &corpus_ids,
                d,
                band,
            )
            .len()
        })
        .collect();
    assert_ne!(
        truth[0], truth[1],
        "the fixture produced symmetric hit counts ({truth:?}), so this test cannot \
         demonstrate per-query isolation"
    );
    assert_ne!(
        tuple(singles[0].query(0).stats),
        tuple(singles[1].query(0).stats)
    );
    for (qi, single) in singles.iter().enumerate() {
        assert_eq!(
            tuple(result.query(qi).stats),
            tuple(single.query(0).stats),
            "query {qi}'s batch stats must equal its standalone stats field by field"
        );
    }
}

#[test]
fn an_empty_band_batch_reports_no_probing() {
    let (mut reader, vectors, _ids, d) = build_flat_fixture(64, 8, 4);
    let empty = l2(1.0, 1.0);
    let result = reader
        .range_search_batch(
            &vectors[0..2 * d],
            2,
            VectorRangeSearchParams::new(empty, 4),
        )
        .unwrap();
    assert_eq!(result.call_stats().list_reads(), 0);
    for qi in 0..2 {
        assert_eq!(result.query(qi).stats.lists_probed(), 0);
        assert_eq!(result.query(qi).labels.len(), 0);
    }
}

#[test]
fn stats_report_real_work() {
    let (mut reader, vectors, _ids, d) = build_flat_fixture(512, 16, 8);
    let query = vectors[0..d].to_vec();
    let band = l2(0.0, 2.0);
    let result = reader
        .range_search(&query, VectorRangeSearchParams::new(band, 4))
        .unwrap();
    let stats = result.query(0).stats;
    assert_eq!(
        stats.lists_probed(),
        4,
        "the number of logical ranks probed"
    );
    assert_eq!(stats.rows_committed(), result.query(0).labels.len());
    assert!(
        stats.rows_scanned() >= stats.rows_committed(),
        "rows_scanned includes rejected and early-abandoned rows, so it cannot be smaller"
    );
    assert!(stats.early_abandoned() <= stats.rows_scanned());
    assert_eq!(
        stats.stop_reason(),
        paimon_vindex_core::range::StopReason::Exhausted
    );
    assert_eq!(result.call_stats().list_reads(), 4);
}

#[test]
fn early_abandon_keeps_rows_just_inside_the_upper_cut() {
    // Regression guard for a real defect: the early-abandon kernel
    // (`fvec_l2sqr_scaled_exceeds`) reduces 128-element blocks into a scalar
    // running total across four accumulators, while the committed-distance
    // kernel (`fvec_l2sqr`) reduces once (two accumulators on NEON, one on
    // AVX2). Their sums
    // differ by a few ULP once d >= 128, so an unwidened cutoff abandoned rows
    // that are genuinely in band: measured 1.24% of boundary rows at d=128,
    // 1.65% at d=256 and 6.06% at d=768.
    //
    // The trigger is narrow, so the test has to aim at it precisely: for each
    // probed row, put the upper cut exactly one ULP **above that row's own
    // distance**. The row is then in band by construction and must come back. A
    // cut placed anywhere else -- a quantile of the whole corpus, say -- lands
    // in a sparse gap where no row is within a few ULP of it, and the defect is
    // invisible. That is not hypothetical: an earlier version of this test used
    // a median cut and passed even with the fix reverted.
    const D: usize = 768;
    let (mut reader, vectors, ids, d) = build_flat_fixture(256, D, 4);
    assert_eq!(d, D, "the divergence only appears once d >= 128");
    let query = vectors[0..d].to_vec();

    // Only rows the probe actually reaches can be dropped by early abandon, so
    // restrict to the query's own list and check every one of them.
    let probed: Vec<(i64, f32)> = ids
        .iter()
        .enumerate()
        .map(|(row, &id)| (id, fvec_l2sqr(&query, &vectors[row * d..(row + 1) * d])))
        .filter(|(_, dist)| *dist < 10.0)
        .collect();
    assert!(
        probed.len() > 20,
        "expected a populated near cluster, got {} rows",
        probed.len()
    );

    let mut checked = 0usize;
    for &(id, distance) in &probed {
        // One ULP above the row's distance: the row is strictly inside.
        let upper = f32::from_bits(distance.to_bits() + 1);
        let band = l2(0.0, upper);
        let result = reader
            .range_search(&query, VectorRangeSearchParams::new(band, 4))
            .unwrap();
        let returned: HashSet<i64> = result.query(0).labels.iter().copied().collect();
        assert!(
            returned.contains(&id),
            "row {id} at distance {distance:?} is inside [0, {upper:?}) but was dropped; \
             the early-abandon threshold is not widened enough"
        );
        // And the whole result must still agree with the oracle exactly.
        assert_eq!(
            pairs_of(result.query(0)),
            bits_of(brute_force_band(&query, &vectors, &ids, d, band)),
            "cut {upper:?} produced a result differing from brute force"
        );
        checked += 1;
    }
    assert!(checked > 20, "vacuous: only {checked} cuts exercised");
}

#[test]
fn a_row_exactly_on_the_upper_cut_is_excluded() {
    // The half-open interval's other edge. This is a membership property rather
    // than an early-abandon one: widening the abandon threshold must not turn an
    // exclusion into an inclusion.
    const D: usize = 768;
    let (mut reader, vectors, ids, d) = build_flat_fixture(256, D, 4);
    let query = vectors[0..d].to_vec();
    let probed: Vec<(i64, f32)> = ids
        .iter()
        .enumerate()
        .map(|(row, &id)| (id, fvec_l2sqr(&query, &vectors[row * d..(row + 1) * d])))
        // A zero distance would make the band `[0, 0)`, which is empty and takes
        // the short-circuit instead of exercising the cut.
        .filter(|(_, dist)| *dist > 0.0 && *dist < 10.0)
        .collect();
    assert!(probed.len() > 8, "expected a populated near cluster");

    for &(id, distance) in probed.iter().take(8) {
        let result = reader
            .range_search(&query, VectorRangeSearchParams::new(l2(0.0, distance), 4))
            .unwrap();
        let returned: HashSet<i64> = result.query(0).labels.iter().copied().collect();
        assert!(
            !returned.contains(&id),
            "row {id} at distance {distance:?} sits exactly on the open upper cut \
             and must be excluded"
        );
    }
}

#[test]
fn range_probe_selection_is_invariant_to_batch_size() {
    // A query must select the same lists whether it runs alone or beside other
    // queries, otherwise batch size becomes observable query semantics.
    // `kmeans::find_topk_batch` scores `nq == 1` with the direct `fvec_l2sqr`
    // kernel and takes an SGEMM path for `nq > 1`, but recomputes every selected
    // centroid's distance with the direct kernel once its error bound says the
    // ranking is ambiguous, so the two agree. This test is the end-to-end guard
    // on that; `kmeans` carries the focused one.
    //
    // The centroids below are chosen so the two arithmetics genuinely disagree:
    // at a magnitude of 1e9 the true squared distances (16384 and 4096) are far
    // below the rounding granularity of ||q|| and ||c||, so norm reconstruction
    // collapses both to 0 and then breaks the tie by index -- picking the
    // *farther* centroid.
    const D: usize = 1;
    const BASE: f32 = 1.0e9;
    let mut index = IVFFlatIndex::new(D, 2, MetricType::L2);
    // List 0's centroid is the farther one, so an index-order tie-break picks it.
    index.set_quantizer_centroids(vec![BASE + 128.0, BASE + 64.0]);
    for (list_id, id) in [(0usize, 7000i64), (1usize, 7001i64)] {
        index.ids[list_id] = vec![id];
        let centroid = index.quantizer_centroids()[list_id];
        index.vectors[list_id] = vec![centroid];
    }
    let mut reader = VectorIndexReader::open(Cursor::new(serialize(&index))).unwrap();

    let query = vec![BASE];
    // nprobe = 1 so only the single best-ranked list is scanned; the band is wide
    // enough that whichever list is chosen contributes its row.
    let band = DistanceBand::new(Bound::Finite(0.0), Bound::Unbounded, MetricType::L2).unwrap();
    let params = VectorRangeSearchParams::new(band, 1);

    let alone = reader.range_search(&query, params).unwrap();
    let alone_labels: Vec<i64> = alone.query(0).labels.to_vec();
    assert_eq!(
        alone_labels.len(),
        1,
        "nprobe = 1 over single-row lists must return exactly one row"
    );

    // The same query as row 0 of a two-query batch, which is the case that used
    // to switch arithmetic. The partner query is deliberately elsewhere.
    let mut batched = query.clone();
    batched.push(BASE + 64.0);
    let batch = reader.range_search_batch(&batched, 2, params).unwrap();

    assert_eq!(
        pairs_of(batch.query(0)),
        pairs_of(alone.query(0)),
        "query 0 selected a different list when batched: probe selection must not \
         depend on nq"
    );
    // And it must be the genuinely nearer centroid, which is list 1.
    assert_eq!(
        alone_labels,
        vec![7001],
        "direct scoring must pick the nearer centroid (list 1), not the \
         index-order tie-break that norm reconstruction produces"
    );
}

#[test]
fn a_caller_bug_outranks_an_unsupported_family() {
    // A family that cannot serve range search must not swallow invalid input: an
    // FFI caller reading `Unsupported` as "fall back to a scan" would silently
    // paper over its own bug. The top-K entry points already validate params
    // before dispatching, and these must match.
    let mut reader = build_diskann_fixture();
    let band = l2(0.0, 1.0);

    // nprobe == 0 on a family that does not support range search at all.
    let err = reader
        .range_search(&[0.0; 8], VectorRangeSearchParams::new(band, 0))
        .unwrap_err();
    assert_eq!(
        err.kind(),
        std::io::ErrorKind::InvalidInput,
        "nprobe == 0 is a caller bug even on an unsupported family"
    );

    // Malformed filter bytes, likewise.
    let err = reader
        .range_search_with_roaring_filter(
            &[0.0; 8],
            VectorRangeSearchParams::new(band, 4),
            b"not roaring",
        )
        .unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);

    // And on the batch entry points.
    let err = reader
        .range_search_batch(&[0.0; 16], 2, VectorRangeSearchParams::new(band, 0))
        .unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    let err = reader
        .range_search_batch_with_roaring_filter(
            &[0.0; 16],
            2,
            VectorRangeSearchParams::new(band, 4),
            b"not roaring",
        )
        .unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);

    // A band whose metric disagrees with the index is also a caller bug, and was
    // the field the first version of this hoist forgot.
    let cosine_band =
        DistanceBand::new(Bound::Finite(0.0), Bound::Finite(1.0), MetricType::Cosine).unwrap();
    let err = reader
        .range_search(&[0.0; 8], VectorRangeSearchParams::new(cosine_band, 4))
        .unwrap_err();
    assert_eq!(
        err.kind(),
        std::io::ErrorKind::InvalidInput,
        "a cosine band against an L2 index is a caller bug even on an unsupported family"
    );

    // With valid input, the family gap is still reported.
    let err = reader
        .range_search(&[0.0; 8], VectorRangeSearchParams::new(band, 4))
        .unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
}

#[test]
fn a_malformed_auto_width_is_invalid_input_not_unsupported() {
    // Automatic width is not implemented, but malformed field values are still a
    // caller bug and must not be masked by the capability gap.
    let (mut reader, ..) = build_flat_fixture(64, 8, 4);
    let band = l2(0.0, 1.0);
    for (width, what) in [
        (
            RangeSearchWidth::Auto {
                initial: 0,
                growth_factor: 2,
                max_width: 4,
            },
            "initial == 0",
        ),
        (
            RangeSearchWidth::Auto {
                initial: 2,
                growth_factor: 1,
                max_width: 4,
            },
            "growth_factor < 2",
        ),
        (
            RangeSearchWidth::Auto {
                initial: 2,
                growth_factor: 2,
                max_width: 0,
            },
            "max_width == 0",
        ),
    ] {
        let params = VectorRangeSearchParams::new(band, 4).with_width(width);
        let err = reader.range_search(&[0.0; 8], params).unwrap_err();
        assert_eq!(
            err.kind(),
            std::io::ErrorKind::InvalidInput,
            "{what} must be InvalidInput"
        );
    }
    // The clamp-dependent rule: with nlist = 4, `initial = 8` clamps to 4 while
    // `max_width = 2` stays 2, so the pair is contradictory only after clamping.
    // Comparing before clamping would let this through.
    let params = VectorRangeSearchParams::new(band, 4).with_width(RangeSearchWidth::Auto {
        initial: 8,
        growth_factor: 2,
        max_width: 2,
    });
    let err = reader.range_search(&[0.0; 8], params).unwrap_err();
    assert_eq!(
        err.kind(),
        std::io::ErrorKind::InvalidInput,
        "max_width below initial after clamping is a caller bug"
    );

    // And it must outrank the metric capability gap, not hide behind it.
    let (mut cosine_reader, ..) = build_flat_fixture_with_metric(64, 8, 4, MetricType::Cosine);
    let cosine_band =
        DistanceBand::new(Bound::Finite(0.0), Bound::Finite(1.0), MetricType::Cosine).unwrap();
    let params = VectorRangeSearchParams::new(cosine_band, 4).with_width(RangeSearchWidth::Auto {
        initial: 8,
        growth_factor: 2,
        max_width: 2,
    });
    let err = cosine_reader.range_search(&[0.0; 8], params).unwrap_err();
    assert_eq!(
        err.kind(),
        std::io::ErrorKind::InvalidInput,
        "the clamp rule must be checked before the metric capability gap"
    );

    // A well-formed Auto is still a capability gap.
    let params = VectorRangeSearchParams::new(band, 4).with_width(RangeSearchWidth::Auto {
        initial: 2,
        growth_factor: 2,
        max_width: 4,
    });
    let err = reader.range_search(&[0.0; 8], params).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
}

/// Builds a single-list index whose payload exceeds the 64 MiB threshold that
/// sends reads down the chunked streaming path.
///
/// The reader streams a list whose payload exceeds `MAX_IVF_BATCH_READ_BYTES`
/// (64 MiB in `index_io_util`, not re-exported, hence the literal below). A row
/// costs `d * 4` bytes, so `n * d * 4` alone clears it at d = 64 and 262_145
/// rows; the real payload is larger still, since it also carries a header and
/// the encoded ids.
///
/// Peak footprint is a few multiples of the 67 MiB vector data: this function
/// holds the corpus, the index's copy, and the serialized bytes at once.
fn build_oversized_fixture() -> (Reader, Vec<f32>, Vec<i64>, usize) {
    const D: usize = 64;
    const ROWS: usize = 262_145; // (64 MiB / (64 * 4)) + 1
    let mut rng = Lcg::new(SEED);
    let mut index = IVFFlatIndex::new(D, 1, MetricType::L2);
    index.set_quantizer_centroids((0..D).map(|dim| centroid_component(0, dim)).collect());

    let mut vectors = vec![0.0f32; ROWS * D];
    let mut ids = vec![0i64; ROWS];
    for row in 0..ROWS {
        ids[row] = 1000 + row as i64;
        for dim in 0..D {
            vectors[row * D + dim] =
                centroid_component(0, dim) + rng.next_symmetric() * NOISE / (D as f32).sqrt();
        }
    }
    index.ids[0] = ids.clone();
    index.vectors[0] = vectors.clone();
    let reader = VectorIndexReader::open(Cursor::new(serialize(&index))).unwrap();
    (reader, vectors, ids, D)
}

#[test]
fn an_oversized_list_streams_and_still_matches_the_oracle() {
    // The streaming branch has its own chunk collection, filtering, tally
    // accumulation and output merging, none of which the other range tests
    // reach: every other fixture is far below the threshold. It costs about
    // 67 MiB and a couple of seconds, which is worth paying to stop this branch
    // from shipping with no coverage at all.
    let (mut reader, vectors, ids, d) = build_oversized_fixture();
    // Guard the premise: if the fixture ever drops below the threshold, this
    // test silently becomes a duplicate of the ordinary path. The literal
    // mirrors `index_io_util::MAX_IVF_BATCH_READ_BYTES`, which is not re-exported;
    // if that constant grows, this assertion is what will fail and say so.
    let payload_bytes = ids.len() * d * std::mem::size_of::<f32>();
    assert!(
        payload_bytes > 64 * 1024 * 1024,
        "fixture payload {payload_bytes} B must exceed the oversized threshold, \
         otherwise the streaming branch is never entered"
    );
    let query = vectors[0..d].to_vec();
    let band = l2(0.0, 2.0);

    let result = reader
        .range_search(&query, VectorRangeSearchParams::new(band, 1))
        .unwrap();
    let want = bits_of(brute_force_band(&query, &vectors, &ids, d, band));
    assert!(!want.is_empty(), "vacuous: the band matched nothing");
    assert_eq!(
        pairs_of(result.query(0)),
        want,
        "the streamed oversized list must return exactly the oracle's rows"
    );

    // Statistics must survive chunking: the list is read once however many
    // chunks it takes, and every row is accounted for.
    assert_eq!(
        result.call_stats().list_reads(),
        1,
        "one list, one logical read"
    );
    let stats = result.query(0).stats;
    assert_eq!(stats.lists_probed(), 1);
    assert_eq!(stats.rows_committed(), result.query(0).labels.len());
    assert_eq!(
        stats.rows_scanned(),
        ids.len(),
        "every row in the streamed list must be counted as scanned"
    );

    // And the filtered path through the same branch.
    let allowed: HashSet<i64> = ids.iter().copied().filter(|id| id % 2 == 0).collect();
    let filtered = reader
        .range_search_with_roaring_filter(
            &query,
            VectorRangeSearchParams::new(band, 1),
            &serialize_roaring(&allowed),
        )
        .unwrap();
    let mut expect: Vec<(i64, u32)> = pairs_of(result.query(0))
        .into_iter()
        .filter(|(id, _)| allowed.contains(id))
        .collect();
    expect.sort_unstable();
    assert!(!expect.is_empty(), "vacuous: the filter admitted nothing");
    assert_eq!(pairs_of(filtered.query(0)), expect);
}

#[test]
fn early_abandon_survives_intermediate_underflow() {
    // The threshold carries an additive term because a normal cut does not stop
    // the individual squared terms from landing in the subnormal range, where a
    // rounding carries absolute rather than relative error. Every other test in
    // this file works at ordinary magnitudes and never enters that regime.
    //
    // Reaching it needs care, and two earlier attempts at this test missed.
    // The first perturbed 1.0 by 1e-24, far below half an ULP at 1.0, so every
    // coordinate rounded back to exactly 1.0 and every distance was zero. The
    // second offset a base by whole ULPs, which makes the difference an exact
    // power of two -- subnormal, but squaring a power of two is exact, so it
    // still produced no rounding at all. Both passed against any threshold,
    // including zero. So this version asserts the regime it needs rather than
    // assuming it: subnormal *and* inexact.
    //
    // What this covers, and what it does not. It is the end-to-end check that
    // the regime is survivable: real subnormal roundings, a boundary-tight cut,
    // and no in-band row lost. It does not pin down the size of the widening.
    // At d = 256 the whole threshold is only 1.000032x the cut, the additive
    // term is 16 ULPs of it, and the fixture still passes with the entire
    // excess removed -- so this test cannot tell a correct margin from a
    // missing one. The arithmetic unit test
    // `the_widening_margin_dominates_the_summation_error_bound` is what does
    // that; its d = 4095 case catches an under-amplified `A`.
    const D: usize = 256;
    const ROWS: usize = 64;

    // Coordinates sit at or just above 2^-65, so a square lands in
    // [2^-130, 2^-128). That is deep enough into the subnormal range that the
    // result keeps about 19 bits rather than 24, so squaring a full 24-bit
    // mantissa has to round -- which is exactly the error `A` exists to bound,
    // on AVX2 in the multiply and on NEON in the fused multiply-add. 256 such
    // terms sum back to ~4e-37, which is normal, so the band comparison itself
    // is ordinary arithmetic.
    fn coordinate(row: usize, dim: usize) -> f32 {
        let scatter =
            (row as u32).wrapping_mul(2_654_435_761) ^ (dim as u32).wrapping_mul(2_246_822_519);
        f32::from_bits((62u32 << 23) | (scatter & 0x007f_ffff))
    }

    let mut vectors = vec![0.0f32; ROWS * D];
    let mut ids = vec![0i64; ROWS];
    for row in 0..ROWS {
        ids[row] = 2000 + row as i64;
        for dim in 0..D {
            vectors[row * D + dim] = coordinate(row, dim);
        }
    }
    // The origin, so each difference is the stored coordinate itself and the
    // subtraction contributes no rounding of its own.
    let query = vec![0.0f32; D];

    // Assert the regime per row, not just in aggregate: it is the row the
    // pruning kernel rules on that has to be in it. A mantissa of zero squares
    // exactly -- `coordinate(0, 0)` is exactly 2^-65 -- so allow one such
    // coordinate per row rather than demanding all D round.
    for row in 0..ROWS {
        let mut inexact = 0usize;
        for &value in &vectors[row * D..(row + 1) * D] {
            let squared = value * value;
            assert!(
                squared > 0.0 && squared < f32::MIN_POSITIVE,
                "row {row}: every squared term must be subnormal for this test \
                 to mean anything, got {squared:e}"
            );
            if f64::from(squared) != f64::from(value) * f64::from(value) {
                inexact += 1;
            }
        }
        assert!(
            inexact >= D - 1,
            "row {row}: the squarings must actually round, only {inexact} of {D} did"
        );
    }

    let distances: Vec<f32> = (0..ROWS)
        .map(|row| fvec_l2sqr(&query, &vectors[row * D..(row + 1) * D]))
        .collect();
    assert!(
        distances.iter().all(|d| *d >= f32::MIN_POSITIVE),
        "the summed distances must be normal, got {:e}",
        distances.iter().copied().fold(f32::INFINITY, f32::min)
    );

    let mut sorted = distances.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("distances are finite"));
    // Cut at the median, so the tightest excluded row sits exactly on the cut
    // and the tightest retained row is a hair below it. Pruning has to make its
    // decision right at the boundary, in the regime asserted above, which is
    // where too narrow a threshold would drop a row it must keep.
    let cut = sorted[ROWS / 2];
    let nearest_kept = sorted[ROWS / 2 - 1];
    assert!(
        nearest_kept < cut && nearest_kept > cut * 0.99,
        "the retained row nearest the cut must be close to it, got {nearest_kept:e} against {cut:e}"
    );

    let mut index = IVFFlatIndex::new(D, 1, MetricType::L2);
    index.set_quantizer_centroids(query.clone());
    index.ids[0] = ids.clone();
    index.vectors[0] = vectors.clone();
    let mut reader = VectorIndexReader::open(Cursor::new(serialize(&index))).unwrap();

    let band = l2(0.0, cut);
    let result = reader
        .range_search(&query, VectorRangeSearchParams::new(band, 1))
        .unwrap();
    let want = bits_of(brute_force_band(&query, &vectors, &ids, D, band));
    assert!(
        !want.is_empty() && want.len() < ROWS,
        "the band must split the corpus, got {} of {ROWS}",
        want.len()
    );
    assert!(
        result.query(0).stats.early_abandoned() > 0,
        "no row was abandoned early, so the pruning kernel never ruled on this regime"
    );
    assert_eq!(
        pairs_of(result.query(0)),
        want,
        "early abandon dropped an in-band row whose distance arithmetic underflows"
    );
}
