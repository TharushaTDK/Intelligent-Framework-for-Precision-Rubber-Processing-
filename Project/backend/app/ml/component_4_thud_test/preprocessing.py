"""Component 4 audio preprocessing and acoustic feature construction.

The functions are moved unchanged from the previous audio, segmentation,
technical-check and feature modules. Raw recordings are never rewritten.
"""
import hashlib
import io
from pathlib import Path
import wave

import numpy as np


# AUDIO IO


def load_wav(path):
    path = Path(path)

    if path.stat().st_size > 10 * 1024 * 1024:
        raise ValueError("WAV exceeds the 10 MiB prototype limit.")

    content = path.read_bytes()

    with wave.open(io.BytesIO(content), "rb") as wav:
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        rate = wav.getframerate()
        frames = wav.getnframes()

        if wav.getcomptype() != "NONE":
            raise ValueError("Uncompressed PCM WAV required.")

        if channels != 1:
            raise ValueError(
                "Mono WAV required; stereo is not silently mixed."
            )

        if width not in (2, 3, 4):
            raise ValueError(
                "Supported PCM widths: 16, 24 and 32 bits."
            )

        if rate <= 0 or frames <= 0:
            raise ValueError("Recording contains no usable frames.")

        raw = wav.readframes(frames)

    if len(raw) != frames * channels * width:
        raise ValueError(
            "Incomplete WAV: fewer PCM bytes than declared."
        )

    if width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)

        values = (
            b[:, 0].astype(np.int32)
            | (b[:, 1].astype(np.int32) << 8)
            | (b[:, 2].astype(np.int32) << 16)
        )

        values = np.where(
            values & 0x800000,
            values - 0x1000000,
            values,
        )
    else:
        values = np.frombuffer(
            raw,
            dtype="<i2" if width == 2 else "<i4",
        )

    full_scale = 2 ** (width * 8 - 1)
    samples = values.astype(np.float64) / full_scale

    metadata = {
        "filename": path.name,
        "sha256": hashlib.sha256(content).hexdigest(),
        "sample_rate_hz": rate,
        "channels": channels,
        "bit_depth": width * 8,
        "frames": frames,
        "duration_seconds": frames / rate,
        "file_bytes": len(content),
        "pcm_bytes": len(raw),
        "digital_rail_samples": int(
            np.count_nonzero(
                (values == -full_scale)
                | (values == full_scale - 1)
            )
        ),
    }

    return samples, metadata


def inspect_wav(path):
    samples, report = load_wav(path)

    rms = float(np.sqrt(np.mean(samples * samples)))

    report.update({
        "processing_status": "AUDIO_INSPECTED",
        "quality_status": "NOT_EVALUATED",
        "peak_percent_full_scale": round(
            float(np.max(np.abs(samples))) * 100,
            5,
        ),
        "rms_dbfs": (
            round(float(20 * np.log10(rms)), 3)
            if rms > 0
            else None
        ),
        "dc_offset": float(np.mean(samples)),
        "near_full_scale_fraction": float(
            np.mean(np.abs(samples) >= 0.999)
        ),
        "silent": bool(np.all(samples == 0)),
    })

    return report


# SEGMENTATION

CONFIG = {
    "version": "segmentation-pilot-v1",
    "window_seconds": 0.005,
    "background_seconds": 0.2,
    "noise_multiplier": 6.0,
    "relative_threshold": 0.12,
    "merge_gap_seconds": 0.06,
    "pre_seconds": 0.05,
    "post_seconds": 0.75,
}


def segment_wav(path):
    samples, metadata = load_wav(path)
    rate = metadata["sample_rate_hz"]

    report = {
        "filename": metadata["filename"],
        "source_sha256": metadata["sha256"],
        "config": CONFIG.copy(),
        "quality_status": "NOT_EVALUATED",
        "flags": [],
    }

    if len(samples) < round(rate * 0.25):
        report["segmentation_status"] = "CAPTURE_TOO_SHORT"
        return None, report

    # DC removal is used for detection only.
    centered = samples - np.mean(samples)

    window = max(1, round(rate * CONFIG["window_seconds"]))
    blocks = len(centered) // window

    envelope = np.sqrt(
        np.mean(
            centered[:blocks * window].reshape(blocks, window) ** 2,
            axis=1,
        )
    )

    background_blocks = max(
        1,
        round(rate * CONFIG["background_seconds"] / window),
    )

    noise = float(
        np.percentile(envelope[:background_blocks], 90)
    )

    threshold = max(
        noise * CONFIG["noise_multiplier"],
        float(envelope.max()) * CONFIG["relative_threshold"],
        1e-7,
    )

    active = np.flatnonzero(envelope > threshold)

    report["background_rms_estimate"] = noise
    report["detection_threshold"] = threshold

    if active.size == 0:
        report.update(
            segmentation_status="NO_EVENT",
            candidate_count=0,
        )
        return None, report

    boundaries = np.flatnonzero(
        np.diff(active) * window / rate
        > CONFIG["merge_gap_seconds"]
    ) + 1

    groups = np.split(active, boundaries)

    energies = [
        float(np.sum(envelope[group] ** 2))
        for group in groups
    ]

    # Select the strongest candidate for initial review.
    selected = int(np.argmax(energies))
    onset = int(groups[selected][0]) * window

    requested_start = onset - round(
        rate * CONFIG["pre_seconds"]
    )
    requested_end = onset + round(
        rate * CONFIG["post_seconds"]
    )

    start = max(0, requested_start)
    end = min(len(samples), requested_end)

    if len(groups) > 1:
        report["flags"].append("MULTIPLE_EVENT_CANDIDATES")

    if requested_start < 0 or requested_end > len(samples):
        report["flags"].append("SEGMENT_TRUNCATED")

    if onset < round(rate * CONFIG["background_seconds"]):
        report["flags"].append("EVENT_IN_BACKGROUND_INTERVAL")

    report.update({
        "segmentation_status": "SEGMENT_EXTRACTED",
        "candidate_count": len(groups),
        "candidate_onsets_seconds": [
            int(group[0]) * window / rate
            for group in groups
        ],
        "selected_candidate_index": selected,
        "onset_seconds": onset / rate,
        "segment_start_seconds": start / rate,
        "segment_end_seconds": end / rate,
        "segment_duration_seconds": (end - start) / rate,
        "review_required": True,
    })

    # Return original sample values from the selected interval.
    return samples[start:end].copy(), report


# TECHNICAL CHECKS

RULE_VERSION = "technical-rules-pilot-v1"

NEAR_FULL_SCALE_LIMIT = 0.001


def check_recording(path):
    result = {
        "filename": Path(path).name,
        "rule_version": RULE_VERSION,
        "quality_status": "NOT_EVALUATED",
        "rejection_reasons": [],
        "warnings": [],
        "moisture_percent": None,
    }

    try:
        audio = inspect_wav(path)
        _, segment = segment_wav(path)
    except (OSError, EOFError, wave.Error, ValueError) as exc:
        result.update(
            technical_status="TECHNICAL_REJECTED",
            rejection_reasons=["FILE_OR_FORMAT_ERROR"],
            error=str(exc),
        )
        return result

    reasons = result["rejection_reasons"]
    warnings = result["warnings"]

    if audio["silent"]:
        reasons.append("SILENT_RECORDING")

    if segment["segmentation_status"] != "SEGMENT_EXTRACTED":
        reasons.append(segment["segmentation_status"])

    if "MULTIPLE_EVENT_CANDIDATES" in segment["flags"]:
        reasons.append("MULTIPLE_EVENT_CANDIDATES")

    if "SEGMENT_TRUNCATED" in segment["flags"]:
        reasons.append("INCOMPLETE_RESPONSE")

    if "EVENT_IN_BACKGROUND_INTERVAL" in segment["flags"]:
        reasons.append("BACKGROUND_INTERVAL_CONTAMINATED")

    if audio["near_full_scale_fraction"] > NEAR_FULL_SCALE_LIMIT:
        reasons.append("EXCESSIVE_NEAR_FULL_SCALE_SAMPLES")

    if audio["peak_percent_full_scale"] >= 99.0:
        warnings.append("HIGH_PEAK_REVIEW")

    result.update({
        "source_sha256": audio["sha256"],
        "technical_status": (
            "TECHNICAL_REJECTED"
            if reasons
            else "TECHNICAL_PASSED"
        ),
        "classifier_status": (
            "SKIPPED_TECHNICAL_REJECTION"
            if reasons
            else "QUALITY_MODEL_NOT_CONNECTED"
        ),
        "candidate_count": segment.get("candidate_count", 0),
        "onset_seconds": segment.get("onset_seconds"),
        "segment_duration_seconds": segment.get(
            "segment_duration_seconds"
        ),
        "peak_percent_full_scale": audio[
            "peak_percent_full_scale"
        ],
        "near_full_scale_fraction": audio[
            "near_full_scale_fraction"
        ],
    })

    return result


# FEATURES

FEATURE_VERSION = "quality-features-pilot-v1"

EPS = 1e-12


def rms(x):
    return float(np.sqrt(np.mean(x * x))) if len(x) else 0.0


def extract_features(path):
    full, meta = load_wav(path)
    segment, detected = segment_wav(path)
    checks = check_recording(path)

    if segment is None:
        return None, checks

    rate = meta["sample_rate_hz"]
    full = full - np.mean(full)
    x = segment - np.mean(segment)

    level = rms(x)
    noise = rms(full[:max(1, round(rate * 0.2))])

    start = round(detected["segment_start_seconds"] * rate)
    end = round(detected["segment_end_seconds"] * rate)
    outside = np.concatenate((full[:start], full[end:]))

    power = np.abs(
        np.fft.rfft(x * np.hanning(len(x)))
    ) ** 2

    frequencies = np.fft.rfftfreq(len(x), d=1.0 / rate)
    power[0] = 0.0

    total = float(power.sum())
    weights = power / max(total, EPS)

    centroid = float(np.sum(frequencies * weights))

    bandwidth = float(
        np.sqrt(
            np.sum((frequencies - centroid) ** 2 * weights)
        )
    )

    roll_index = min(
        int(np.searchsorted(np.cumsum(weights), 0.85)),
        len(power) - 1,
    )

    quarter = max(1, len(x) // 4)
    positive = power[1:]
    distribution = weights[weights > 0]

    features = {
        "segment_rms": level,
        "segment_peak": float(np.max(np.abs(x))),
        "crest_factor": (
            float(np.max(np.abs(x))) / max(level, EPS)
        ),
        "zero_crossing_rate": float(
            np.mean(np.diff(np.signbit(x)) != 0)
        ),
        "background_rms": noise,
        "outside_segment_rms": rms(outside),
        "segment_to_background_db": float(
            20 * np.log10((level + EPS) / (noise + EPS))
        ),
        "near_full_scale_fraction": checks[
            "near_full_scale_fraction"
        ],
        "event_candidate_count": float(
            detected["candidate_count"]
        ),
        "dominant_frequency_hz": float(
            frequencies[np.argmax(power)]
        ),
        "spectral_centroid_hz": centroid,
        "spectral_bandwidth_hz": bandwidth,
        "spectral_rolloff85_hz": float(
            frequencies[roll_index]
        ),
        "spectral_flatness": float(
            np.exp(np.mean(np.log(positive + EPS)))
            / (np.mean(positive) + EPS)
        ),
        "spectral_entropy": float(
            -np.sum(distribution * np.log2(distribution))
            / np.log2(len(power))
        ),
        "late_to_early_energy_db": float(
            10 * np.log10(
                (np.mean(x[-quarter:] ** 2) + EPS)
                / (np.mean(x[:quarter] ** 2) + EPS)
            )
        ),
    }

    if not all(np.isfinite(v) for v in features.values()):
        raise ValueError(
            f"Non-finite feature in {Path(path).name}"
        )

    return features, checks
