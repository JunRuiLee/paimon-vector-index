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

import java.util.Objects;

/**
 * A half-open band [lower, upper) in raw index distance space: squared L2, 1 - cosine, or negative
 * inner product. A null cut is structurally unbounded, not a finite sentinel.
 */
public final class VectorDistanceBand {

    public enum CutOperator {
        GE(0),
        GT(1),
        LE(2),
        LT(3);

        private final int code;

        CutOperator(int code) {
            this.code = code;
        }
    }

    private final String metric;
    private final Float lower;
    private final Float upper;

    public VectorDistanceBand(String metric, Float lower, Float upper) {
        this.metric = Objects.requireNonNull(metric, "metric");
        if (!"l2".equals(metric) && !"cosine".equals(metric) && !"inner_product".equals(metric)) {
            throw new IllegalArgumentException("unknown metric: " + metric);
        }
        validateCut(lower);
        validateCut(upper);
        if (lower != null && upper != null && lower > upper) {
            throw new IllegalArgumentException("inverted distance band");
        }
        this.lower = lower;
        this.upper = upper;
    }

    /**
     * Converts public predicate endpoints through the core's exact f64-to-f32 cut conversion.
     * Public L2 is the f32 square root of stored squared distance; cosine is 1 - cosine; inner
     * product is the positive dot product. Null endpoint/operator pairs are unbounded. Lower
     * accepts GE/GT and upper accepts LE/LT. Unrepresentable L2 cuts fail; out-of-domain linear
     * endpoints produce empty or structurally unbounded bands as defined by core.
     */
    public static VectorDistanceBand fromEndpoints(
            String metric,
            Double lower,
            CutOperator lowerOperator,
            Double upper,
            CutOperator upperOperator) {
        Objects.requireNonNull(metric, "metric");
        if ((lower == null) != (lowerOperator == null)
                || (upper == null) != (upperOperator == null)) {
            throw new IllegalArgumentException(
                    "endpoint value and operator must both be null or set");
        }
        return VectorIndexNative.distanceBandFromEndpoints(
                metric,
                lower,
                lowerOperator == null ? -1 : lowerOperator.code,
                upper,
                upperOperator == null ? -1 : upperOperator.code);
    }

    public String metric() {
        return metric;
    }

    public Float lower() {
        return lower;
    }

    public Float upper() {
        return upper;
    }

    private void validateCut(Float cut) {
        if (cut != null && (!Float.isFinite(cut) || ("l2".equals(metric) && cut < 0))) {
            throw new IllegalArgumentException(
                    "cut must be finite and non-negative for squared L2");
        }
    }
}
