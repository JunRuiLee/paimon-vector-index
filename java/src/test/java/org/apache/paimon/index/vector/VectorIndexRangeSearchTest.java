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

import static org.apache.paimon.index.vector.VectorDistanceBand.CutOperator.GE;
import static org.apache.paimon.index.vector.VectorDistanceBand.CutOperator.GT;
import static org.apache.paimon.index.vector.VectorDistanceBand.CutOperator.LE;
import static org.apache.paimon.index.vector.VectorDistanceBand.CutOperator.LT;

import java.lang.reflect.Field;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class VectorIndexRangeSearchTest {

    private static final long LABEL_BASE = 1L << 33;
    private static final int VECTOR_COUNT = 256;

    public static void main(String[] args) {
        VectorIndexNativeLoaderSmokeTest.configureExternalLibrary(args);
        testValueTypes();
        testNative();
        System.out.println(
                "Range search: value types, 12 IVF/metric combinations, endpoints, filters, validation and callbacks passed");
    }

    static void testValueTypes() {
        testNativeResultOwnership();
        VectorDistanceBand band = new VectorDistanceBand("l2", null, 4.0f);
        check(band.lower() == null && band.upper() == 4.0f, "structural bounds");
        check("l2".equals(band.metric()), "metric");
        check(new VectorRangeSearchParams(band, 2).nprobe() == 2, "nprobe");
        expect(IllegalArgumentException.class, () -> new VectorRangeSearchParams(band, 0));
        expect(NullPointerException.class, () -> new VectorRangeSearchParams(null, 1));
        expect(IllegalArgumentException.class, () -> new VectorDistanceBand("other", null, null));
        expect(IllegalArgumentException.class, () -> new VectorDistanceBand("l2", -1.0f, null));
        expect(IllegalArgumentException.class, () -> new VectorDistanceBand("cosine", 2.0f, 1.0f));
        expect(IllegalArgumentException.class, () -> new VectorDistanceBand("l2", null, Float.NaN));
        expect(
                IllegalArgumentException.class,
                () -> new VectorDistanceBand("cosine", Float.NEGATIVE_INFINITY, null));

        long[] labels = {LABEL_BASE, 7};
        float[] distances = {1, 3};
        long[] lims = {0, 0, 2};
        long[] counters = {0, 2};
        VectorRangeSearchResult result =
                new VectorRangeSearchResult(
                        labels, distances, lims, counters, counters, counters, new long[2], 2);
        labels[0] = -1;
        distances[0] = -1;
        lims[1] = 2;
        counters[1] = -1;
        check(result.queryCount() == 2 && result.listReads() == 2, "result shape");
        check(result.labelsForQuery(0).length == 0, "empty first row");
        check(result.labelsForQuery(1)[0] == LABEL_BASE, "64-bit label and copy");
        check(result.distancesForQuery(1)[0] == 1, "distance copy");
        result.labels()[0] = -2;
        result.distances()[0] = -2;
        result.lims()[1] = 2;
        result.listsProbed()[1] = -2;
        result.rowsScanned()[1] = -2;
        result.rowsCommitted()[1] = -2;
        result.earlyAbandoned()[1] = -2;
        check(result.labels()[0] == LABEL_BASE && result.distances()[0] == 1, "defensive arrays");
        check(result.lims()[1] == 0 && result.listsProbed()[1] == 2, "defensive CSR and stats");
        check(
                result.rowsScanned()[1] == 2
                        && result.rowsCommitted()[1] == 2
                        && result.earlyAbandoned()[1] == 0,
                "defensive counters");
        expect(IndexOutOfBoundsException.class, () -> result.labelsForQuery(-1));
        expect(IndexOutOfBoundsException.class, () -> result.distancesForQuery(2));
        expect(
                IllegalArgumentException.class,
                () ->
                        new VectorRangeSearchResult(
                                new long[0],
                                new float[0],
                                new long[] {0, Long.MAX_VALUE},
                                new long[1],
                                new long[1],
                                new long[1],
                                new long[1],
                                0));
        expect(
                IllegalArgumentException.class,
                () ->
                        new VectorRangeSearchResult(
                                new long[0],
                                new float[0],
                                new long[] {0},
                                new long[1],
                                new long[0],
                                new long[0],
                                new long[0],
                                0));
        expect(
                IllegalArgumentException.class,
                () ->
                        new VectorRangeSearchResult(
                                new long[0],
                                new float[0],
                                new long[] {0},
                                new long[0],
                                new long[0],
                                new long[0],
                                new long[0],
                                -1));

        VectorIndexReader closed = VectorIndexReader.fromNativePointerForTesting(0);
        VectorRangeSearchParams params = new VectorRangeSearchParams(band, 1);
        expect(IllegalStateException.class, closed::supportsRangeSearch);
        expect(IllegalStateException.class, () -> closed.rangeSearch(new float[] {0}, params));
        expect(
                IllegalStateException.class,
                () -> closed.rangeSearch(new float[] {0}, params, new byte[8]));
        expect(IllegalStateException.class, () -> closed.rangeSearchBatch(new float[0], 0, params));
        expect(
                IllegalStateException.class,
                () -> closed.rangeSearchBatch(new float[0], 0, params, new byte[8]));
        expect(NullPointerException.class, () -> closed.rangeSearch(null, params));
        expect(NullPointerException.class, () -> closed.rangeSearch(new float[1], null));
        expect(NullPointerException.class, () -> closed.rangeSearch(new float[1], params, null));
    }

    private static void testNativeResultOwnership() {
        long[] labels = {LABEL_BASE, 7};
        float[] distances = {1, 3};
        long[] lims = {0, 0, 2};
        long[] listsProbed = {0, 1};
        long[] rowsScanned = {0, 3};
        long[] rowsCommitted = {0, 2};
        long[] earlyAbandoned = {0, 1};
        VectorRangeSearchResult result =
                VectorRangeSearchResult.fromNative(
                        labels,
                        distances,
                        lims,
                        listsProbed,
                        rowsScanned,
                        rowsCommitted,
                        earlyAbandoned,
                        1);
        checkOwnedArray(result, "labels", labels);
        checkOwnedArray(result, "distances", distances);
        checkOwnedArray(result, "lims", lims);
        checkOwnedArray(result, "listsProbed", listsProbed);
        checkOwnedArray(result, "rowsScanned", rowsScanned);
        checkOwnedArray(result, "rowsCommitted", rowsCommitted);
        checkOwnedArray(result, "earlyAbandoned", earlyAbandoned);
        result.labels()[0] = -1;
        result.distances()[0] = -1;
        result.lims()[1] = 2;
        result.listsProbed()[1] = -1;
        result.rowsScanned()[1] = -1;
        result.rowsCommitted()[1] = -1;
        result.earlyAbandoned()[1] = -1;
        result.labelsForQuery(1)[0] = -1;
        result.distancesForQuery(1)[0] = -1;
        check(result.queryCount() == 2 && result.listReads() == 1, "owned result shape");
        check(result.labelsForQuery(0).length == 0, "owned empty query");
        check(result.labelsForQuery(1)[0] == LABEL_BASE, "owned labels remain defensive");
        check(result.distancesForQuery(1)[0] == 1, "owned distances remain defensive");
        check(result.lims()[1] == 0 && result.listsProbed()[1] == 1, "owned CSR and stats");
        check(
                result.rowsScanned()[1] == 3
                        && result.rowsCommitted()[1] == 2
                        && result.earlyAbandoned()[1] == 1,
                "owned counters remain defensive");
        expect(
                NullPointerException.class,
                () ->
                        VectorRangeSearchResult.fromNative(
                                null, distances, lims, listsProbed, rowsScanned,
                                rowsCommitted, earlyAbandoned, 1));
        expect(
                IllegalArgumentException.class,
                () ->
                        VectorRangeSearchResult.fromNative(
                                labels, distances, new long[] {0, 3, 2}, listsProbed,
                                rowsScanned, rowsCommitted, earlyAbandoned, 1));
        expect(
                IllegalArgumentException.class,
                () ->
                        VectorRangeSearchResult.fromNative(
                                labels, distances, lims, new long[0], rowsScanned,
                                rowsCommitted, earlyAbandoned, 1));
        expect(
                IllegalArgumentException.class,
                () ->
                        VectorRangeSearchResult.fromNative(
                                labels, distances, lims, listsProbed, new long[] {0, -1},
                                rowsCommitted, earlyAbandoned, 1));
        expect(
                IllegalArgumentException.class,
                () ->
                        VectorRangeSearchResult.fromNative(
                                labels, distances, lims, listsProbed, rowsScanned,
                                rowsCommitted, earlyAbandoned, -1));
    }

    private static void checkOwnedArray(
            VectorRangeSearchResult result, String name, Object expected) {
        try {
            Field field = VectorRangeSearchResult.class.getDeclaredField(name);
            field.setAccessible(true);
            check(field.get(result) == expected, "native result must own " + name + " without copying");
        } catch (ReflectiveOperationException error) {
            throw new AssertionError(error);
        }
    }

    static void testNative() {
        for (String indexType : new String[] {"ivf_flat", "ivf_sq", "ivf_pq", "ivf_rq"}) {
            for (String metric : new String[] {"l2", "cosine", "inner_product"}) {
                testIndex(indexType, metric);
            }
        }
        testExactDistancesAndEndpoints();
        testNativeValidation();
        testCallbacks();
        testUnsupported();
        VectorIndexRangeOracleTest.runIfConfigured();
    }

    private static void testIndex(String indexType, String metric) {
        float[] data = new float[VECTOR_COUNT * 8];
        for (int row = 0; row < VECTOR_COUNT; row++) {
            for (int column = 0; column < 8; column++) {
                data[row * 8 + column] = ((row * 17 + column * 11) % 67 - 33) / 16.0f;
            }
        }
        float[] queries = Arrays.copyOf(data, 16);
        VectorRangeSearchParams all = params(metric, null, null);
        try (VectorIndexReader reader = open(build(indexType, metric, 8, data))) {
            check(reader.supportsRangeSearch(), indexType + " " + metric + " support");
            VectorRangeSearchResult full = reader.rangeSearchBatch(queries, 2, all);
            assertShape(full, 2);
            check(full.labels().length == VECTOR_COUNT * 2, "unbounded includes every row");
            check(full.listReads() <= 2, "shared list reads");
            for (int queryIndex = 0; queryIndex < 2; queryIndex++) {
                float[] query = Arrays.copyOfRange(queries, queryIndex * 8, (queryIndex + 1) * 8);
                assertRows(full, queryIndex, reader.rangeSearch(query, all), 0);
                check(full.listsProbed()[queryIndex] == 2, "lists probed");
                check(full.rowsScanned()[queryIndex] == VECTOR_COUNT, "rows scanned");
            }
            float[] sorted = full.distancesForQuery(0);
            Arrays.sort(sorted);
            Float lower = sorted[VECTOR_COUNT / 4];
            Float upper = sorted[VECTOR_COUNT * 3 / 4];
            VectorRangeSearchParams bounded = params(metric, lower, upper);
            VectorRangeSearchResult selected = reader.rangeSearchBatch(queries, 2, bounded);
            assertSelected(full, selected, lower, upper, false);
            VectorRangeSearchResult filtered =
                    reader.rangeSearchBatch(queries, 2, bounded, filter());
            assertSelected(full, filtered, lower, upper, true);
            for (int queryIndex = 0; queryIndex < 2; queryIndex++) {
                float[] query = Arrays.copyOfRange(queries, queryIndex * 8, (queryIndex + 1) * 8);
                assertRows(selected, queryIndex, reader.rangeSearch(query, bounded), 0);
                assertRows(filtered, queryIndex, reader.rangeSearch(query, bounded, filter()), 0);
            }
            assertSelected(
                    full,
                    reader.rangeSearchBatch(queries, 2, params(metric, null, upper)),
                    null,
                    upper,
                    false);
            assertSelected(
                    full,
                    reader.rangeSearchBatch(queries, 2, params(metric, lower, null)),
                    lower,
                    null,
                    false);
            expectMessage(
                    RuntimeException.class,
                    "query count must be greater than 0",
                    () -> reader.rangeSearchBatch(new float[0], 0, all));
            expectMessage(
                    RuntimeException.class,
                    "query count must be greater than 0",
                    () -> reader.rangeSearchBatch(new float[0], 0, all, filter()));
            VectorRangeSearchResult empty =
                    reader.rangeSearchBatch(queries, 2, params(metric, 0.0f, 0.0f));
            check(empty.labels().length == 0 && empty.listReads() == 0, "empty band");
            check(
                    reader.rangeSearchBatch(queries, 2, all, new byte[8]).labels().length == 0,
                    "empty allow-list");
            expect(
                    RuntimeException.class,
                    () -> reader.rangeSearchBatch(queries, 2, all, new byte[] {1}));
        }
    }

    private static void testExactDistancesAndEndpoints() {
        for (String metric : new String[] {"l2", "cosine", "inner_product"}) {
            try (VectorIndexReader reader =
                    open(build("ivf_flat", metric, 2, new float[] {1, 0, 2, 0, 0, 1, -1, 0}))) {
                float[] query = {1, 0};
                VectorRangeSearchResult raw = reader.rangeSearch(query, params(metric, null, null));
                Map<Long, Float> rows = rows(raw, 0);
                check(
                        rows.get(LABEL_BASE + 1)
                                == ("l2".equals(metric)
                                        ? 1.0f
                                        : "cosine".equals(metric) ? 0.0f : -2.0f),
                        "raw metric distance");
                double endpoint =
                        "inner_product".equals(metric)
                                ? 1.0
                                : "l2".equals(metric) ? Math.sqrt(2.0f) : 1.0;
                if ("l2".equals(metric)) {
                    endpoint = (double) (float) endpoint;
                }
                for (VectorDistanceBand.CutOperator operator :
                        VectorDistanceBand.CutOperator.values()) {
                    boolean lower = operator == GE || operator == GT;
                    for (double literal :
                            new double[] {
                                endpoint, Math.nextUp(endpoint), Math.nextDown(endpoint)
                            }) {
                        VectorDistanceBand band =
                                VectorDistanceBand.fromEndpoints(
                                        metric,
                                        lower ? literal : null,
                                        lower ? operator : null,
                                        lower ? null : literal,
                                        lower ? null : operator);
                        Map<Long, Float> selected =
                                rows(
                                        reader.rangeSearch(
                                                query, new VectorRangeSearchParams(band, 2)),
                                        0);
                        for (Map.Entry<Long, Float> row : rows.entrySet()) {
                            double value =
                                    "l2".equals(metric)
                                            ? (double) (float) Math.sqrt(row.getValue())
                                            : "inner_product".equals(metric)
                                                    ? -row.getValue()
                                                    : row.getValue();
                            boolean admitted =
                                    operator == GE
                                            ? value >= literal
                                            : operator == GT
                                                    ? value > literal
                                                    : operator == LE
                                                            ? value <= literal
                                                            : value < literal;
                            check(
                                    selected.containsKey(row.getKey()) == admitted,
                                    "core endpoint conversion " + metric + " " + operator);
                        }
                    }
                }
            }
        }
        VectorDistanceBand unbounded =
                VectorDistanceBand.fromEndpoints("inner_product", null, null, null, null);
        check(
                unbounded.lower() == null && unbounded.upper() == null,
                "unbounded endpoint conversion");
        VectorDistanceBand outside =
                VectorDistanceBand.fromEndpoints(
                        "cosine", -Double.MAX_VALUE, GE, Double.MAX_VALUE, LE);
        check(
                outside.lower() == -Float.MAX_VALUE && outside.upper() == null,
                "linear out-of-domain endpoints retain core cuts and unbounded upper");
        VectorDistanceBand equal = VectorDistanceBand.fromEndpoints("l2", 1.0, GE, 1.0, LE);
        check(
                equal.lower() <= 1.0f && equal.upper() > 1.0f,
                "inclusive equal endpoints retain equality bucket");
        expect(
                RuntimeException.class,
                () -> VectorDistanceBand.fromEndpoints("l2", 1.0, LT, null, null));
        expect(
                RuntimeException.class,
                () -> VectorDistanceBand.fromEndpoints("cosine", null, null, Double.NaN, LT));
        expect(
                RuntimeException.class,
                () -> VectorDistanceBand.fromEndpoints("l2", null, null, Double.MAX_VALUE, LE));
        expect(
                IllegalArgumentException.class,
                () -> VectorDistanceBand.fromEndpoints("l2", 1.0, null, null, null));
        expect(
                RuntimeException.class,
                () -> VectorIndexNative.distanceBandFromEndpoints("l2", 1.0, 4, null, -1));
    }

    private static void testNativeValidation() {
        VectorRangeSearchParams all = params("l2", null, null);
        expect(RuntimeException.class, () -> VectorIndexNative.supportsRangeSearch(0));
        expect(RuntimeException.class, () -> VectorIndexNative.rangeSearch(0, new float[2], all));
        try (VectorIndexReader reader =
                open(build("ivf_flat", "l2", 2, new float[] {0, 0, 1, 1, 2, 2, 3, 3}))) {
            expect(RuntimeException.class, () -> reader.rangeSearch(new float[1], all));
            expectMessage(
                    RuntimeException.class,
                    "query length",
                    () -> reader.rangeSearch(new float[1], all, new byte[] {1}));
            expect(
                    RuntimeException.class,
                    () -> reader.rangeSearch(new float[] {Float.NaN, 0}, all));
            expect(RuntimeException.class, () -> reader.rangeSearchBatch(new float[2], 2, all));
            expect(RuntimeException.class, () -> reader.rangeSearchBatch(new float[0], -1, all));
            expect(
                    RuntimeException.class,
                    () -> reader.rangeSearchBatch(new float[0], Integer.MAX_VALUE, all));
            expect(
                    RuntimeException.class,
                    () -> reader.rangeSearch(new float[2], params("cosine", null, null)));
            expect(
                    RuntimeException.class,
                    () -> reader.rangeSearchBatch(new float[0], 0, params("cosine", null, null)));
            check(
                    reader.rangeSearch(new float[2], all).labels().length == 4,
                    "reader survives errors");
            check(
                    reader.rangeSearch(
                                            new float[2],
                                            new VectorRangeSearchParams(
                                                    all.band(), Integer.MAX_VALUE))
                                    .listsProbed()[0]
                            == 2,
                    "core clamps nprobe");
        }
    }

    private static void testCallbacks() {
        byte[] bytes = build("ivf_flat", "l2", 2, new float[] {0, 0, 1, 1, 2, 2, 3, 3});
        VectorIndexNativeValidationTest.ByteArraySeekableInputStream delegate =
                new VectorIndexNativeValidationTest.ByteArraySeekableInputStream(bytes);
        VectorIndexReader[] holder = new VectorIndexReader[1];
        int[] callbacks = {0};
        boolean[] fail = {false};
        IllegalStateException callbackFailure = new IllegalStateException("range input failure");
        VectorRangeSearchParams all = params("l2", null, null);
        try (VectorIndexReader reader =
                new VectorIndexReader(
                        (positions, buffers) -> {
                            if (holder[0] != null) {
                                callbacks[0]++;
                                expect(IllegalStateException.class, holder[0]::close);
                                expect(IllegalStateException.class, holder[0]::supportsRangeSearch);
                                expect(
                                        IllegalStateException.class,
                                        () -> holder[0].rangeSearch(new float[2], all));
                                expect(
                                        IllegalStateException.class,
                                        () -> holder[0].rangeSearch(new float[2], all, filter()));
                                expect(
                                        IllegalStateException.class,
                                        () -> holder[0].rangeSearchBatch(new float[2], 1, all));
                                expect(
                                        IllegalStateException.class,
                                        () ->
                                                holder[0].rangeSearchBatch(
                                                        new float[2], 1, all, filter()));
                            }
                            if (fail[0]) {
                                throw callbackFailure;
                            }
                            delegate.pread(positions, buffers);
                        })) {
            holder[0] = reader;
            reader.rangeSearch(new float[2], all);
            reader.rangeSearchBatch(new float[4], 2, all, filter());
            check(callbacks[0] > 0, "range callbacks exercised");
            fail[0] = true;
            check(
                    expect(IllegalStateException.class, () -> reader.rangeSearch(new float[2], all))
                            == callbackFailure,
                    "callback exception identity preserved");
            check(
                    expect(
                                    IllegalStateException.class,
                                    () -> reader.rangeSearchBatch(new float[4], 2, all, filter()))
                            == callbackFailure,
                    "batch callback exception identity preserved");
            fail[0] = false;
            check(
                    reader.rangeSearch(new float[2], all).labels().length == 4,
                    "reader usable after callback failure");
        }
    }

    private static void testUnsupported() {
        float[] data = new float[64 * 8];
        for (int offset = 0; offset < data.length; offset++) {
            data[offset] = (offset * 17 % 67) / 16.0f;
        }
        try (VectorIndexReader reader = open(build("diskann", "l2", 8, data))) {
            check(!reader.supportsRangeSearch(), "DiskANN unsupported");
            VectorRangeSearchParams empty = params("l2", 0.0f, 0.0f);
            expectMessage(
                    RuntimeException.class,
                    "not supported",
                    () -> reader.rangeSearch(new float[8], empty));
            expectMessage(
                    RuntimeException.class,
                    "not supported",
                    () -> reader.rangeSearch(new float[8], empty, filter()));
            expectMessage(
                    RuntimeException.class,
                    "not supported",
                    () -> reader.rangeSearchBatch(new float[16], 2, empty));
            expectMessage(
                    RuntimeException.class,
                    "not supported",
                    () -> reader.rangeSearchBatch(new float[16], 2, empty, filter()));
        }
    }

    private static VectorRangeSearchParams params(String metric, Float lower, Float upper) {
        return new VectorRangeSearchParams(new VectorDistanceBand(metric, lower, upper), 2);
    }

    private static VectorIndexReader open(byte[] bytes) {
        return new VectorIndexReader(
                new VectorIndexNativeValidationTest.ByteArraySeekableInputStream(bytes));
    }

    private static byte[] build(String type, String metric, int dimension, float[] data) {
        Map<String, String> options = new HashMap<String, String>();
        options.put("index.type", type);
        options.put("metric", metric);
        options.put("dimension", Integer.toString(dimension));
        if (!"diskann".equals(type)) {
            options.put("nlist", "2");
        }
        if ("ivf_pq".equals(type) || "diskann".equals(type)) {
            options.put("pq.m", "2");
        }
        if ("ivf_pq".equals(type)) {
            options.put("use-opq", "false");
        }
        if ("ivf_rq".equals(type)) {
            options.put("rq.bits", "5");
        }
        if ("diskann".equals(type)) {
            options.put("pq.bits", "4");
            options.put("diskann.max-degree", "8");
            options.put("diskann.build-search-list-size", "16");
        }
        int count = data.length / dimension;
        long[] ids = new long[count];
        for (int row = 0; row < count; row++) {
            ids[row] = LABEL_BASE + row;
        }
        VectorIndexNativeValidationTest.ByteArrayPositionOutputStream output =
                new VectorIndexNativeValidationTest.ByteArrayPositionOutputStream();
        try (VectorIndexTraining training = VectorIndexTrainer.train(options, data, count);
                VectorIndexWriter writer = new VectorIndexWriter(training)) {
            writer.addVectors(ids, data, count);
            writer.writeIndex(output);
        }
        return output.toByteArray();
    }

    private static byte[] filter() {
        ByteBuffer bytes = ByteBuffer.allocate(36).order(ByteOrder.LITTLE_ENDIAN);
        bytes.putLong(1).putInt(2).putInt(12346).putInt(1);
        bytes.putShort((short) 0).putShort((short) 3).putInt(16);
        bytes.putShort((short) 0).putShort((short) 1).putShort((short) 16).putShort((short) 63);
        return bytes.array();
    }

    private static void assertShape(VectorRangeSearchResult result, int count) {
        check(result.queryCount() == count && result.lims().length == count + 1, "CSR shape");
        check(
                result.lims()[0] == 0 && result.lims()[count] == result.labels().length,
                "CSR limits");
        check(result.labels().length == result.distances().length, "parallel results");
        for (int queryIndex = 0; queryIndex < count; queryIndex++) {
            check(
                    result.rowsCommitted()[queryIndex] == result.labelsForQuery(queryIndex).length,
                    "committed counter");
            check(
                    result.rowsScanned()[queryIndex] >= result.rowsCommitted()[queryIndex],
                    "scanned counter");
            check(
                    result.earlyAbandoned()[queryIndex] <= result.rowsScanned()[queryIndex],
                    "abandon counter");
        }
    }

    private static Map<Long, Float> rows(VectorRangeSearchResult result, int queryIndex) {
        Map<Long, Float> rows = new HashMap<Long, Float>();
        long[] labels = result.labelsForQuery(queryIndex);
        float[] distances = result.distancesForQuery(queryIndex);
        for (int row = 0; row < labels.length; row++) {
            check(Float.isFinite(distances[row]), "finite raw distance");
            check(rows.put(labels[row], distances[row]) == null, "unique labels");
        }
        return rows;
    }

    private static void assertRows(
            VectorRangeSearchResult expected,
            int expectedQuery,
            VectorRangeSearchResult actual,
            int actualQuery) {
        assertShape(actual, actual.queryCount());
        check(
                rows(expected, expectedQuery).equals(rows(actual, actualQuery)),
                "single/batch row equality");
    }

    private static void assertSelected(
            VectorRangeSearchResult full,
            VectorRangeSearchResult actual,
            Float lower,
            Float upper,
            boolean filtered) {
        assertShape(actual, full.queryCount());
        for (int queryIndex = 0; queryIndex < full.queryCount(); queryIndex++) {
            Map<Long, Float> expected = rows(full, queryIndex);
            expected.entrySet()
                    .removeIf(
                            row ->
                                    (lower != null && row.getValue() < lower)
                                            || (upper != null && row.getValue() >= upper)
                                            || (filtered
                                                    && row.getKey() != LABEL_BASE
                                                    && row.getKey() != LABEL_BASE + 1
                                                    && row.getKey() != LABEL_BASE + 16
                                                    && row.getKey() != LABEL_BASE + 63));
            check(expected.equals(rows(actual, queryIndex)), "half-open raw membership and filter");
        }
    }

    private static void check(boolean condition, String message) {
        if (!condition) {
            throw new AssertionError(message);
        }
    }

    private static void expectMessage(
            Class<? extends Throwable> type, String message, Runnable action) {
        Throwable error = expect(type, action);
        check(
                error.getMessage() != null && error.getMessage().contains(message),
                "exception message: " + error.getMessage());
    }

    private static Throwable expect(Class<? extends Throwable> type, Runnable action) {
        try {
            action.run();
        } catch (Throwable error) {
            if (!type.isInstance(error)) {
                throw new AssertionError("expected " + type.getName(), error);
            }
            if (error.getMessage() != null
                    && error.getMessage().contains("Rust panic in JNI call")) {
                throw new AssertionError("validation must not panic", error);
            }
            return error;
        }
        throw new AssertionError("expected " + type.getName());
    }
}
