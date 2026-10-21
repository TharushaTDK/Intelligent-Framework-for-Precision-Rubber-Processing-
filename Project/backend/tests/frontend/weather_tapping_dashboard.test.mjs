// Mocked frontend tests for the Component 1 testing dashboard.
//
// These never touch the real backend or a real GPS: fetch and
// navigator.geolocation are replaced with fakes so we can check the location,
// payload, validation and error-handling logic on their own.
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

// Grab the page's inline script and run it without a DOM. The page only
// exports its pure helpers when "module" exists, and skips all DOM code when
// "document" doesn't, so this gives us just the logic.
function loadDashboard() {
    const html = readFileSync(htmlPath, "utf-8");
    // Matches the attribute-less inline <script>, not the Leaflet <script src>.
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

// A fake navigator.geolocation that records how it was called.
function fakeGeolocation({ position = null, error = null } = {}) {
    const calls = [];
    return {
        calls,
        getCurrentPosition(success, failure, options) {
            calls.push(options);
            if (error) failure(error); else success(position);
        },
    };
}

// ---------------- manual pin selection ----------------

test("map click becomes a location with rounded coordinates", () => {
    const loc = page.makeLocation(6.927079123456, 79.861244987654, "map");
    assert.deepEqual(loc, {
        latitude: 6.927079,
        longitude: 79.861245,
        source: "map",
        accuracy: null,
        errors: [],
    });
});

test("typed coordinates are accepted as strings", () => {
    const loc = page.makeLocation("7.2513", " 80.3464 ", "typed");
    assert.equal(loc.latitude, 7.2513);
    assert.equal(loc.longitude, 80.3464);
    assert.equal(loc.source, "typed");
});

test("invalid coordinates are rejected with a message per field", () => {
    assert.equal(page.makeLocation("", "80", "typed").errors.length, 1);
    assert.equal(page.makeLocation("91", "181", "typed").errors.length, 2);
    assert.equal(page.makeLocation("abc", "80", "typed").errors.length, 1);
    assert.equal(page.makeLocation(-90, 180, "map").errors.length, 0);
});

test("weather cache key changes when the pin or time changes", () => {
    const a = page.makeLocation(6.5854, 79.9607, "map");
    const moved = page.makeLocation(6.5855, 79.9607, "map");
    const t1 = "2026-10-18T14:00:00+05:30";
    const t2 = "2026-10-18T15:00:00+05:30";

    assert.equal(page.weatherCacheKey(a, t1), page.weatherCacheKey({ ...a, source: "gps" }, t1));
    assert.notEqual(page.weatherCacheKey(a, t1), page.weatherCacheKey(moved, t1));
    assert.notEqual(page.weatherCacheKey(a, t1), page.weatherCacheKey(a, t2));
});

// ---------------- GPS permission handling ----------------

test("granted GPS returns the coordinates and accuracy", async () => {
    const geo = fakeGeolocation({
        position: { coords: { latitude: 6.9271, longitude: 79.8612, accuracy: 18.4 } },
    });

    const pos = await page.requestCurrentPosition(geo);

    assert.deepEqual(pos, { latitude: 6.9271, longitude: 79.8612, accuracy: 18.4 });
    assert.equal(geo.calls.length, 1);
    assert.deepEqual(geo.calls[0], page.GEOLOCATION_OPTIONS);

    const loc = page.makeLocation(pos.latitude, pos.longitude, "gps", pos.accuracy);
    assert.equal(loc.source, "gps");
    assert.equal(loc.accuracy, 18);
});

test("denied permission tells the farmer to use the map", async () => {
    const geo = fakeGeolocation({ error: { code: 1, message: "User denied Geolocation" } });

    await assert.rejects(page.requestCurrentPosition(geo), err => {
        assert.match(err.message, /permission was denied/i);
        assert.match(err.message, /click the map/i);
        return true;
    });
});

test("unavailable position and timeout have their own messages", async () => {
    await assert.rejects(
        page.requestCurrentPosition(fakeGeolocation({ error: { code: 2 } })),
        /could not find its location/
    );
    await assert.rejects(
        page.requestCurrentPosition(fakeGeolocation({ error: { code: 3 } })),
        /took too long/
    );
});

test("a browser without geolocation is handled", async () => {
    await assert.rejects(page.requestCurrentPosition(undefined), /cannot share its location/);
    await assert.rejects(page.requestCurrentPosition({}), /cannot share its location/);
});

// ---------------- weather API integration (mocked fetch) ----------------

test("weather request hits the real route with exactly the schema fields", async () => {
    const loc = page.makeLocation(6.5854, 79.9607, "map");
    let seen = null;
    const fetchSpy = async (url, options) => {
        seen = { url, options };
        return fakeResponse(200, sampleWeather);
    };

    const body = await page.fetchWeatherSnapshot(loc, "2026-10-18T14:20:00+05:30", fetchSpy);

    assert.equal(seen.url, "/api/v1/weather-tapping/weather-snapshot");
    assert.equal(seen.options.method, "POST");
    assert.deepEqual(JSON.parse(seen.options.body), {
        latitude: 6.5854,
        longitude: 79.9607,
        observation_timestamp: "2026-10-18T14:20:00+05:30",
    });
    assert.deepEqual(body, sampleWeather);
});

test("weather provider outage surfaces the backend's message", async () => {
    const loc = page.makeLocation(6.5854, 79.9607, "map");
    const fetch503 = async () => fakeResponse(503, { detail: "Weather data is currently unavailable." });

    await assert.rejects(
        page.fetchWeatherSnapshot(loc, "2026-10-18T14:00:00+05:30", fetch503),
        err => err.status === 503 && /Weather data is currently unavailable/.test(err.message)
    );
});

test("weather request when the backend is down is reported as unreachable", async () => {
    const loc = page.makeLocation(6.5854, 79.9607, "map");
    const failingFetch = async () => { throw new TypeError("Failed to fetch"); };

    await assert.rejects(
        page.fetchWeatherSnapshot(loc, "2026-10-18T14:00:00+05:30", failingFetch),
        err => err.kind === "unreachable"
    );
});

// ---------------- timestamps ----------------

test("datetime-local value is sent with an explicit +05:30 offset", () => {
    assert.equal(page.toSriLankaTimestamp("2025-01-02T08:00"), "2025-01-02T08:00:00+05:30");
    assert.equal(page.toSriLankaTimestamp("2025-01-02T08:00:15"), "2025-01-02T08:00:15+05:30");
    assert.equal(page.toSriLankaTimestamp(""), null);
    assert.equal(page.toSriLankaTimestamp("02/01/2025 08:00"), null);
});

test("advisory times up to 7 days ahead are allowed", () => {
    const now = new Date("2026-10-19T15:00:00Z");   // 20:30 in Sri Lanka
    assert.equal(page.checkAdvisoryTime("2026-10-19T20:30:00+05:30", now), null);
    assert.equal(page.checkAdvisoryTime("2026-10-20T08:32:00+05:30", now), null);   // the time from the screenshot
    assert.equal(page.checkAdvisoryTime("2026-10-26T20:30:00+05:30", now), null);   // exactly 7 days
});

test("more than 7 days ahead or too far back is refused with a clear message", () => {
    const now = new Date("2026-10-19T15:00:00Z");
    assert.match(page.checkAdvisoryTime("2026-10-26T21:30:00+05:30", now), /up to 7 days ahead/);
    assert.match(page.checkAdvisoryTime("2026-07-01T08:00:00+05:30", now), /goes back about 89 days/);
    assert.match(page.checkAdvisoryTime(null, now), /Pick an advisory date/);
});

test("forecast flag comes from the backend's data_quality", () => {
    assert.equal(page.isForecastBased({ data_quality: { observation_in_future: true } }), true);
    assert.equal(page.isForecastBased({ data_quality: { observation_in_future: false } }), false);
    assert.equal(page.isForecastBased({ data_quality: {} }), false);
    assert.equal(page.isForecastBased(null), false);
});

test("current Sri Lanka time is UTC + 5:30", () => {
    assert.equal(page.currentSriLankaLocalValue(new Date("2026-10-18T20:00:00Z")), "2026-10-19T01:30");
});

// ---------------- payloads ----------------

test("DRC request sends only weather and timestamp, nothing invented", () => {
    const { payload, missing } = page.buildPredictionPayload("drc", {
        observationTimestamp: sampleWeather.observation_timestamp,
        weather: sampleWeather,
    });

    assert.deepEqual(Object.keys(payload).sort(), [
        "forecast_rain_next_6h_mm", "observation_timestamp", "rainfall_last_3days_mm",
        "rainfall_last_6h_mm", "relative_humidity_pct", "temperature_c", "wind_speed_kmh",
    ]);
    assert.deepEqual(missing.sort(), [
        "estate_id", "soil_moisture_pct", "soil_ph", "visible_latex_condition",
    ]);
    assert.ok(!("latitude" in payload) && !("longitude" in payload));
});

test("DRC request carries the latex condition the farmer picked", () => {
    const { payload, missing } = page.buildPredictionPayload("drc", {
        observationTimestamp: sampleWeather.observation_timestamp,
        weather: sampleWeather,
        latexCondition: "Watery",
    });

    assert.equal(payload.visible_latex_condition, "Watery");
    assert.ok(!missing.includes("visible_latex_condition"));
});

test("unknown or empty latex condition is left out, not sent", () => {
    for (const latexCondition of [null, "", "watery", "Very watery"]) {
        const { payload, missing } = page.buildPredictionPayload("drc", {
            observationTimestamp: sampleWeather.observation_timestamp,
            weather: sampleWeather,
            latexCondition,
        });
        assert.ok(!("visible_latex_condition" in payload), `${latexCondition} should not be sent`);
        assert.ok(missing.includes("visible_latex_condition"));
    }
});

test("latex condition only goes to the DRC model", () => {
    for (const kind of ["tapping", "price"]) {
        const { payload } = page.buildPredictionPayload(kind, {
            observationTimestamp: sampleWeather.observation_timestamp,
            weather: sampleWeather,
            latexDrc: 36.3,
            latexCondition: "Watery",
        });
        assert.ok(!("visible_latex_condition" in payload), `${kind} must not get the condition`);
    }
});

test("latex condition choices match the backend schema", () => {
    assert.deepEqual(page.LATEX_CONDITIONS, ["Normal", "Watery", "Early coagulation"]);
});

test("every sent field is one the schema accepts", () => {
    for (const kind of ["drc", "tapping", "price"]) {
        const { payload } = page.buildPredictionPayload(kind, {
            observationTimestamp: sampleWeather.observation_timestamp,
            weather: sampleWeather,
            latexDrc: 30.2,
        });
        const allowed = new Set(["observation_timestamp", ...page.ENDPOINT_FIELDS[kind]]);
        for (const key of Object.keys(payload)) {
            assert.ok(allowed.has(key), `${kind} payload has unexpected field ${key}`);
        }
    }
});

test("tapping payload never includes estate_id", () => {
    const { payload } = page.buildPredictionPayload("tapping", {
        observationTimestamp: sampleWeather.observation_timestamp,
        weather: sampleWeather,
    });

    assert.ok(!("estate_id" in payload));
});

test("price payload carries the DRC estimate and only price fields", () => {
    const { payload, missing } = page.buildPredictionPayload("price", {
        observationTimestamp: sampleWeather.observation_timestamp,
        weather: sampleWeather,
        latexDrc: 32.4,
    });

    assert.equal(payload.latex_drc_pct, 32.4);
    assert.ok(!("forecast_rain_next_6h_mm" in payload));
    assert.ok(!missing.includes("latex_drc_pct"));
});

test("price payload without a DRC lists it as missing, not zero", () => {
    const { payload, missing } = page.buildPredictionPayload("price", {
        observationTimestamp: sampleWeather.observation_timestamp,
        weather: sampleWeather,
        latexDrc: NaN,
    });

    assert.ok(!("latex_drc_pct" in payload));
    assert.ok(missing.includes("latex_drc_pct"));
});

test("missing weather values are left out, never zero-filled", () => {
    const weather = { ...sampleWeather, wind_speed_kmh: null };
    delete weather.rainfall_last_3days_mm;

    const { payload, missing } = page.buildPredictionPayload("drc", {
        observationTimestamp: sampleWeather.observation_timestamp,
        weather,
    });

    for (const field of ["wind_speed_kmh", "rainfall_last_3days_mm"]) {
        assert.ok(!(field in payload), `${field} should be omitted`);
        assert.ok(missing.includes(field), `${field} should be listed as missing`);
    }
});

test("not-provided list merges all requests without duplicates", () => {
    const names = page.describeNotProvided([
        ["estate_id", "soil_ph", "visible_latex_condition"],
        ["soil_ph", "visible_latex_condition"],
        ["soil_ph"],
    ]);
    assert.deepEqual(names, ["estate ID", "soil pH", "latex condition"]);
});


// ---------------- saving for Component 2 ----------------

const drcBody = {
    predicted_latex_drc_pct: 30.26572224824792,
    model_name: "drc_prediction",
    runtime_artifact_sha256: "b".repeat(64),
};
const tappingBody = {
    predicted_label: "High",
    probabilities: { high: 0.597, low: 0.137, moderate: 0.267 },
};
const priceBody = { predicted_price_lkr_per_kg: 413.0509 };

test("saved record carries location, time, weather and the exact DRC", () => {
    const loc = page.makeLocation(7.411142, 79.822483, "gps", 25);
    const record = page.buildAdvisoryRecord(loc, sampleWeather, drcBody, tappingBody, priceBody);

    assert.deepEqual(record, {
        latitude: 7.411142,
        longitude: 79.822483,
        location_source: "gps",
        observation_timestamp: sampleWeather.observation_timestamp,
        forecast_based: false,
        temperature_c: 29.1,
        relative_humidity_pct: 78,
        wind_speed_kmh: 9.4,
        rainfall_last_6h_mm: 0.4,
        rainfall_last_3days_mm: 21.7,
        forecast_rain_next_6h_mm: 3.2,
        predicted_latex_drc_pct: 30.26572224824792,
        drc_model_name: "drc_prediction",
        drc_runtime_artifact_sha256: "b".repeat(64),
        tapping_label: "High",
        tapping_probability_high: 0.597,
        tapping_probability_low: 0.137,
        tapping_probability_moderate: 0.267,
        predicted_price_lkr_per_kg: 413.0509,
    });
});

test("record without tapping or price leaves those fields out", () => {
    const loc = page.makeLocation(6.9271, 79.8612, "map");
    const record = page.buildAdvisoryRecord(loc, sampleWeather, drcBody);

    assert.equal(record.predicted_latex_drc_pct, drcBody.predicted_latex_drc_pct);
    for (const key of ["tapping_label", "tapping_probability_high", "predicted_price_lkr_per_kg"]) {
        assert.ok(!(key in record), `${key} should be omitted`);
    }
});

test("forecast-based weather is flagged in the saved record", () => {
    const loc = page.makeLocation(6.9271, 79.8612, "map");
    const future = { ...sampleWeather, data_quality: { observation_in_future: true } };

    assert.equal(page.buildAdvisoryRecord(loc, future, drcBody).forecast_based, true);
});

test("records are saved to the Component 1 advisory-records route", () => {
    assert.equal(page.ADVISORY_RECORDS_URL, "/api/v1/weather-tapping/advisory-records");
});

// ---------------- general API error handling ----------------

test("aborted request is reported as a timeout", async () => {
    const abortingFetch = async () => {
        const err = new Error("aborted");
        err.name = "AbortError";
        throw err;
    };

    await assert.rejects(page.apiRequest("/health", {}, abortingFetch), err => err.kind === "timeout");
});

test("422 validation errors are turned into readable field messages", async () => {
    const body = {
        detail: [
            { loc: ["body", "observation_timestamp"], msg: "must include an explicit timezone offset" },
            { loc: ["body", "latitude"], msg: "Input should be less than or equal to 90" },
        ],
    };
    const fetch422 = async () => fakeResponse(422, body);

    await assert.rejects(page.apiRequest("/x", {}, fetch422), err => {
        assert.equal(err.status, 422);
        assert.match(err.message, /observation_timestamp: must include/);
        assert.match(err.message, /latitude: Input should be/);
        return true;
    });
});

test("non-JSON error responses are handled", async () => {
    const fetchHtml = async () => fakeResponse(500, null, { json: false });

    await assert.rejects(
        page.apiRequest("/x", {}, fetchHtml),
        err => err.status === 500 && /non-JSON/.test(err.message)
    );
});
