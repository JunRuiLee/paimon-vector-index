<!--
  ~ Licensed to the Apache Software Foundation (ASF) under one
  ~ or more contributor license agreements.  See the NOTICE file
  ~ distributed with this work for additional information
  ~ regarding copyright ownership.  The ASF licenses this file
  ~ to you under the Apache License, Version 2.0 (the
  ~ "License"); you may not use this file except in compliance
  ~ with the License.  You may obtain a copy of the License at
  ~
  ~   http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing,
  ~ software distributed under the License is distributed on an
  ~ "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  ~ KIND, either express or implied.  See the License for the
  ~ specific language governing permissions and limitations
  ~ under the License.
-->

# Python distance range search

`VectorIndexReader.range_search` and `range_search_batch` add variable-length
results alongside the unchanged top-K `search` and `search_batch` APIs. IVF-Flat,
IVF-PQ, IVF-RQ, and IVF-SQ support L2, inner product, and cosine. Query
`reader.supports_range_search()` before selecting this API; DiskANN is unsupported.
With an older native library lacking range exports, existing imports and top-K
calls still work: capability returns `False`, and range calls/endpoint conversion
raise a clear error asking for a native-library upgrade. A partially available
range ABI is treated as unavailable rather than risking an unfreeable result.

```python
from paimon_vindex import (
    DistanceBand,
    DistanceEndpoint,
    DistanceEndpointOp,
    RangeSearchParams,
)

band = DistanceBand.from_endpoints(
    "l2", upper=DistanceEndpoint(2.0, DistanceEndpointOp.LE)
)
result = reader.range_search_batch(queries, RangeSearchParams(band, nprobe=8))
labels, distances = result.query(0)
stats = result.stats[0]
```

## Distance contracts

- `DistanceBand(metric, lower=None, upper=None)` describes a half-open
  **internal-distance** interval `[lower, upper)`. Metric names are `l2`,
  `inner_product`, and `cosine`. Internal distances are squared L2, negative inner
  product, and cosine distance, respectively.
- `None` means structurally unbounded, not a finite sentinel. Equal finite cuts
  describe a valid empty band. Core validates finite, ordered, metric-compatible
  cuts during search; invalid bands raise `RuntimeError`.
- `DistanceBand.from_endpoints` accepts optional `DistanceEndpoint(value, op)`
  objects: lower uses `GE`/`GT`, upper uses `LE`/`LT`. Endpoint values are public
  distances (L2 square root, inner product, or cosine distance). Double literals
  pass directly to core, which handles rounding, operator strictness, and inner
  product direction reversal. Python performs no endpoint conversion math.
- Results retain internal distances. Quantized IVF variants return their own
  distance estimates, not an additional exact-vector reranking.

## Queries, filters, and results

Single queries have shape `(dimension,)`; batches have shape
`(query_count, dimension)`. Inputs are converted to aligned, contiguous float32 after
shape and buffer-size checks. Core rejects zero-query batches. `nprobe` is a
positive, platform-sized integer and is independent of top-K parameters.

Both methods accept `roaring_filter=` containing serialized **RoaringTreemap**
bytes, bytearray, or memoryview. `None` selects the unfiltered API; a serialized
empty treemap selects no rows. Empty bytes are not a serialized empty treemap
and are passed to core for validation.

`RangeSearchResult` contains owned NumPy arrays `lims` (`uintp`), `labels`
(`int64`), and `distances` (`float32`), plus an immutable tuple `stats` and the
call-level `list_reads`. `query_count`, `hit_count`, and `query(index)` expose
the CSR shape. The query accessor returns label/distance slices and rejects
negative or out-of-range indices. Arrays and statistics remain valid after
reader closure and native result destruction.

Each `RangeSearchStats` contains `lists_probed`, `rows_scanned`,
`rows_committed`, and `early_abandoned`. These are core's logical counters;
`list_reads` is call-level and should not be summed from per-query counters.
The native result handle is destroyed in `finally`, including conversion and
allocation failures, while the reader's callback-aware lock remains held.

## Verification

```sh
PYTHONPATH=python PAIMON_VINDEX_LIB_PATH=/path/to/native/library \
  python3 -m pytest python/tests
```

The standalone matrix tests run without shared fixtures. To additionally run
the exact cross-language oracle, set `PVI_RANGE_FIXTURES` to the output directory
from `cargo run -p paimon-vindex-ffi --example range_search_fixture -- DIR`.
Oracle tests open a fresh reader for each manifest entry and compare per-query
`(label, float32 bits)` multisets, CSR limits, and every statistics field exactly.
Core does not guarantee row order across calls; the Python wrapper preserves
the order it receives without sorting.
