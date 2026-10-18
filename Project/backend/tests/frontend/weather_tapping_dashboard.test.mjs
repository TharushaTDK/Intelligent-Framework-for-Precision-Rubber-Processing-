// Mocked frontend tests for the Component 1 testing dashboard.
//
// These never touch the real backend: fetch is replaced with fakes so we can
// check payload building, validation and error handling on their own.
// Uses only Node's built-in test runner, so there's nothing to install.
//
// Run from the Project folder:
//   node --test backend/tests/frontend/weather_tapping_dashboard.test.mjs

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const htmlPath = fileURLToPath(
    new URL("../../app/static/weather_tapping_dashboard.html", import.meta.url)
);

// Grab the page's script and run it without a DOM. The page only exports its
// pure helpers when "module" exists, and skips all DOM code when "document"
// doesn't, so this gives us just the logic.
function loadDashboard() {
    const html = readFileSync(htmlPath, "utf-8");
    const match = html.match(/<script>([\s\S]*?)<\/script>/);
    assert.ok(match, "inline script not found");

    // Run it in this realm (not a vm context) so returned objects compare
    // cleanly with deepStrictEqual.
    const fakeModule = { exports: {} };
    new Function("module", match[1])(fakeModule);
    return fakeModule.exports;
}

const page = loadDashboard();

const sampleWeather = {
    observation_timestamp: "2026-10-18T14:00:00+05:30",
    temperature_c: 29.1,
    relative_humidity_pct: 78,
    wind_speed_kmh: 9.4,
    rainfall_last_6h_mm: 0.4,
    rainfall_last_3days_mm: 21.7,
    forecast_rain_next_6h_mm: 3.2,
};

const sampleMeasurements = {
    soil_moisture_pct: 41,
    soil_ph: 5.8,
    tree_count: 250,
    tapping_system: "S/2 D2",
    days_since_last_tap: 2,
    previous_collection_yield_litres: 38,
};

function fakeResponse(status, body, { json = true } = {}) {
    return {
        ok: status >= 200 && status < 300,
        status,
        json: async () => {
            if (!json) throw new SyntaxError("not json");
            return body;
        },
    };
}

// ---------------- estates ----------------

test("every estate uses a trained estate ID and the dataset's placeholder name", () => {
    assert.ok(page.ESTATES.length > 0);
    for (const e of page.ESTATES) {
        assert.match(e.estate_id, /^EST-(0\d\d|1[01]\d|120)$/);
        assert.equal(e.estate_name, "TBD_" + e.estate_id);
    }
});

test("every estate is labelled as demo and has usable coordinates", () => {
    for (const e of page.ESTATES) {
        assert.equal(e.coordinate_status, "demo_placeholder");
        assert.ok(e.latitude >= -90 && e.latitude <= 90);
        assert.ok(e.longitude >= -180 && e.longitude <= 180);
    }
});

// ---------------- weather reuse ----------------

test("weather cache key changes with estate or time", () => {
    const [a, b] = page.ESTATES;
    const t1 = "2026-10-18T14:00:00+05:30";
    const t2 = "2026-10-18T15:00:00+05:30";

    assert.equal(page.weatherCacheKey(a, t1), page.weatherCacheKey({ ...a }, t1));
    assert.notEqual(page.weatherCacheKey(a, t1), page.weatherCacheKey(b, t1));
    assert.notEqual(page.weatherCacheKey(a, t1), page.weatherCacheKey(a, t2));
});

// ---------------- price DRC choice ----------------

test("measured DRC wins over the model estimate", () => {
    assert.deepEqual(page.choosePriceDrc(33.1, 30.2), { value: 33.1, source: "measured" });
});

test("model estimate is used only when nothing was measured", () => {
    assert.deepEqual(page.choosePriceDrc(null, 30.2), { value: 30.2, source: "model_estimate" });
});

test("no DRC at all means no price request", () => {
    assert.equal(page.choosePriceDrc(null, null), null);
    assert.equal(page.choosePriceDrc(undefined, NaN), null);
});

// ---------------- timestamps ----------------

test("datetime-local value is sent with an explicit +05:30 offset", () => {
    assert.equal(page.toSriLankaTimestamp("2025-01-02T08:00"), "2025-01-02T08:00:00+05:30");
    assert.equal(page.toSriLankaTimestamp("2025-01-02T08:00:15"), "2025-01-02T08:00:15+05:30");
    assert.equal(page.toSriLankaTimestamp(""), null);
    assert.equal(page.toSriLankaTimestamp("02/01/2025 08:00"), null);
});

test("current Sri Lanka time is UTC + 5:30", () => {
    const utc = new Date("2026-10-18T20:00:00Z");
    assert.equal(page.currentSriLankaLocalValue(utc), "2026-10-19T01:30");
});

// ---------------- validation ----------------

test("blank measurements become null, not zero", () => {
    assert.deepEqual(page.parseMeasurement("soil_ph", ""), { value: null });
    assert.deepEqual(page.parseMeasurement("soil_ph", "   "), { value: null });
});

test("invalid and out-of-range measurements are rejected", () => {
    assert.ok(page.parseMeasurement("soil_ph", "abc").error);
    assert.ok(page.parseMeasurement("soil_ph", "15").error);
    assert.ok(page.parseMeasurement("soil_moisture_pct", "-1").error);
    assert.ok(page.parseMeasurement("latex_drc_pct", "101").error);
    assert.ok(page.parseMeasurement("tree_count", "-5").error);
    assert.deepEqual(page.parseMeasurement("soil_ph", "5.8"), { value: 5.8 });
});

// ---------------- payloads ----------------

test("DRC payload has weather, measurements and estate_id but no estate_name", () => {
    const { payload, missing } = page.buildPredictionPayload("drc", {
        observationTimestamp: sampleWeather.observation_timestamp,
        weather: sampleWeather,
        measurements: sampleMeasurements,
        estateId: "EST-001",
    });

    assert.equal(payload.estate_id, "EST-001");
    assert.equal(payload.temperature_c, 29.1);
    assert.equal(payload.observation_timestamp, "2026-10-18T14:00:00+05:30");
    assert.ok(!("estate_name" in payload));
    assert.deepEqual(missing, []);
    assert.deepEqual(
        Object.keys(payload).sort(),
        ["observation_timestamp", ...page.ENDPOINT_FIELDS.drc].sort()
    );
});

test("tapping payload never includes estate_id", () => {
    const { payload } = page.buildPredictionPayload("tapping", {
        observationTimestamp: sampleWeather.observation_timestamp,
        weather: sampleWeather,
        measurements: sampleMeasurements,
        estateId: "EST-001",
    });

    assert.ok(!("estate_id" in payload));
    assert.ok(!("previous_collection_yield_litres" in payload));
});

test("price payload carries the supplied DRC and only price fields", () => {
    const { payload } = page.buildPredictionPayload("price", {
        observationTimestamp: sampleWeather.observation_timestamp,
        weather: sampleWeather,
        measurements: sampleMeasurements,
        estateId: "EST-001",
        latexDrc: 32.4,
    });

    assert.equal(payload.latex_drc_pct, 32.4);
    assert.ok(!("forecast_rain_next_6h_mm" in payload));
    assert.ok(!("estate_id" in payload));
});

test("estate-only request sends weather, timestamp and estate_id, and lists the rest as missing", () => {
    const { payload, missing } = page.buildPredictionPayload("drc", {
        observationTimestamp: sampleWeather.observation_timestamp,
        weather: sampleWeather,
        measurements: {},
        estateId: "EST-001",
    });

    assert.deepEqual(Object.keys(payload).sort(), [
        "estate_id", "forecast_rain_next_6h_mm", "observation_timestamp", "rainfall_last_3days_mm",
        "rainfall_last_6h_mm", "relative_humidity_pct", "temperature_c", "wind_speed_kmh",
    ]);
    assert.deepEqual(missing.sort(), [
        "days_since_last_tap", "previous_collection_yield_litres", "soil_moisture_pct",
        "soil_ph", "tapping_system", "tree_count",
    ]);
});

test("missing weather and blank measurements are left out, never zero-filled", () => {
    const weather = { ...sampleWeather, wind_speed_kmh: null };
    delete weather.rainfall_last_3days_mm;

    const { payload, missing } = page.buildPredictionPayload("drc", {
        observationTimestamp: sampleWeather.observation_timestamp,
        weather,
        measurements: { ...sampleMeasurements, soil_ph: null, tapping_system: null },
        estateId: "EST-001",
    });

    for (const field of ["wind_speed_kmh", "rainfall_last_3days_mm", "soil_ph", "tapping_system"]) {
        assert.ok(!(field in payload), `${field} should be omitted`);
        assert.ok(missing.includes(field), `${field} should be listed as missing`);
    }
});

// ---------------- API error handling (mocked fetch) ----------------

test("network failure is reported as backend unreachable", async () => {
    const failingFetch = async () => { throw new TypeError("Failed to fetch"); };

    await assert.rejects(
        page.apiRequest("/health", {}, failingFetch),
        err => err.kind === "unreachable" && /Cannot reach the backend/.test(err.message)
    );
});

test("aborted request is reported as a timeout", async () => {
    const abortingFetch = async () => {
        const err = new Error("aborted");
        err.name = "AbortError";
        throw err;
    };

    await assert.rejects(
        page.apiRequest("/health", {}, abortingFetch),
        err => err.kind === "timeout"
    );
});

test("422 validation errors are turned into readable field messages", async () => {
    const body = {
        detail: [
            { loc: ["body", "observation_timestamp"], msg: "must include an explicit timezone offset" },
            { loc: ["body", "estate_name"], msg: "Extra inputs are not permitted" },
        ],
    };
    const fetch422 = async () => fakeResponse(422, body);

    await assert.rejects(page.apiRequest("/x", {}, fetch422), err => {
        assert.equal(err.kind, "http");
        assert.equal(err.status, 422);
        assert.match(err.message, /observation_timestamp: must include/);
        assert.match(err.message, /estate_name: Extra inputs/);
        return true;
    });
});

test("503 from the weather provider shows the backend's message", async () => {
    const fetch503 = async () => fakeResponse(503, { detail: "Weather data is currently unavailable." });

    await assert.rejects(
        page.apiRequest("/x", {}, fetch503),
        err => err.status === 503 && /Weather data is currently unavailable/.test(err.message)
    );
});

test("non-JSON error responses are handled", async () => {
    const fetchHtml = async () => fakeResponse(500, null, { json: false });

    await assert.rejects(
        page.apiRequest("/x", {}, fetchHtml),
        err => err.status === 500 && /non-JSON/.test(err.message)
    );
});

test("successful responses are returned untouched", async () => {
    const body = { predicted_latex_drc_pct: 31.23456 };
    const fetchOk = async () => fakeResponse(200, body);

    assert.deepEqual(await page.apiRequest("/x", {}, fetchOk), body);
});
