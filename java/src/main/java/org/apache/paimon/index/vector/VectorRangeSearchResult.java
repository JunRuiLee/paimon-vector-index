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

package org.apache.paimon.index.vector;

import java.util.Arrays;
import java.util.Objects;

/**
 * CSR range-search output in core scan order, with raw distances and no sorting or top-K cap.
 * IVF-Flat distances are exact; SQ, PQ and RQ return their core distance estimates. Each query
 * occupies [lims[query], lims[query + 1]). Public construction and array access make defensive
 * copies; native construction takes exclusive ownership of its arrays.
 */
public final class VectorRangeSearchResult {

    private final long[] labels;
    private final float[] distances;
    private final long[] lims;
    private final long[] listsProbed;
    private final long[] rowsScanned;
    private final long[] rowsCommitted;
    private final long[] earlyAbandoned;
    private final long listReads;

    static VectorRangeSearchResult fromNative(
            long[] labels,
            float[] distances,
            long[] lims,
            long[] listsProbed,
            long[] rowsScanned,
            long[] rowsCommitted,
            long[] earlyAbandoned,
            long listReads) {
        return new VectorRangeSearchResult(
                labels,
                distances,
                lims,
                listsProbed,
                rowsScanned,
                rowsCommitted,
                earlyAbandoned,
                listReads,
                false);
    }

    public VectorRangeSearchResult(
            long[] labels,
            float[] distances,
            long[] lims,
            long[] listsProbed,
            long[] rowsScanned,
            long[] rowsCommitted,
            long[] earlyAbandoned,
            long listReads) {
        this(
                labels,
                distances,
                lims,
                listsProbed,
                rowsScanned,
                rowsCommitted,
                earlyAbandoned,
                listReads,
                true);
    }

    private VectorRangeSearchResult(
            long[] labels,
            float[] distances,
            long[] lims,
            long[] listsProbed,
            long[] rowsScanned,
            long[] rowsCommitted,
            long[] earlyAbandoned,
            long listReads,
            boolean copyArrays) {
        Objects.requireNonNull(labels, "labels");
        Objects.requireNonNull(distances, "distances");
        Objects.requireNonNull(lims, "lims");
        this.labels = copyArrays ? labels.clone() : labels;
        this.distances = copyArrays ? distances.clone() : distances;
        this.lims = copyArrays ? lims.clone() : lims;
        if (this.labels.length != this.distances.length
                || this.lims.length == 0
                || this.lims[0] != 0
                || this.lims[this.lims.length - 1] != this.labels.length) {
            throw new IllegalArgumentException("invalid CSR result shape");
        }
        for (int offset = 1; offset < this.lims.length; offset++) {
            if (this.lims[offset] < this.lims[offset - 1]
                    || this.lims[offset] > this.labels.length) {
                throw new IllegalArgumentException("invalid CSR limits");
            }
        }
        this.listsProbed = validatedCounters(listsProbed, "listsProbed", copyArrays);
        this.rowsScanned = validatedCounters(rowsScanned, "rowsScanned", copyArrays);
        this.rowsCommitted = validatedCounters(rowsCommitted, "rowsCommitted", copyArrays);
        this.earlyAbandoned = validatedCounters(earlyAbandoned, "earlyAbandoned", copyArrays);
        if (listReads < 0) {
            throw new IllegalArgumentException("listReads must be non-negative");
        }
        this.listReads = listReads;
    }

    public int queryCount() {
        return lims.length - 1;
    }

    public long[] labels() {
        return labels.clone();
    }

    public float[] distances() {
        return distances.clone();
    }

    public long[] lims() {
        return lims.clone();
    }

    /** Logical IVF ranks probed per query, including empty lists. */
    public long[] listsProbed() {
        return listsProbed.clone();
    }

    /** Allow-listed rows evaluated per query, including early-abandoned rows. */
    public long[] rowsScanned() {
        return rowsScanned.clone();
    }

    public long[] rowsCommitted() {
        return rowsCommitted.clone();
    }

    /** Core diagnostic count, not an arithmetic work measure. */
    public long[] earlyAbandoned() {
        return earlyAbandoned.clone();
    }

    /** Non-empty unique list reads for the whole call; SQ cache hits do not count. */
    public long listReads() {
        return listReads;
    }

    public long[] labelsForQuery(int queryIndex) {
        checkQueryIndex(queryIndex);
        return Arrays.copyOfRange(
                labels, Math.toIntExact(lims[queryIndex]), Math.toIntExact(lims[queryIndex + 1]));
    }

    public float[] distancesForQuery(int queryIndex) {
        checkQueryIndex(queryIndex);
        return Arrays.copyOfRange(
                distances,
                Math.toIntExact(lims[queryIndex]),
                Math.toIntExact(lims[queryIndex + 1]));
    }

    private void checkQueryIndex(int queryIndex) {
        if (queryIndex < 0 || queryIndex >= queryCount()) {
            throw new IndexOutOfBoundsException("queryIndex " + queryIndex + " out of range");
        }
    }

    private long[] validatedCounters(long[] counters, String name, boolean copyArrays) {
        Objects.requireNonNull(counters, name);
        long[] values = copyArrays ? counters.clone() : counters;
        if (values.length != queryCount()) {
            throw new IllegalArgumentException(name + " length must equal queryCount");
        }
        for (long value : values) {
            if (value < 0) {
                throw new IllegalArgumentException(name + " must be non-negative");
            }
        }
        return values;
    }
}
