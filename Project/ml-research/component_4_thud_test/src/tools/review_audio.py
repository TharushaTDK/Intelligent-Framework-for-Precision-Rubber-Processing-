"""Create a waveform review page without changing raw audio."""

import argparse
from html import escape
import os
from pathlib import Path
from urllib.parse import quote

import numpy as np

import sys

PROJECT = Path(__file__).resolve().parents[4]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from backend.app.ml.component_4_thud_test.preprocessing import load_wav, segment_wav


def waveform_svg(samples, rate, offset=0.0, bounds=None, onsets=()):
    duration = len(samples) / rate

    if duration <= 0:
        return "<p>No samples to plot.</p>"

    def xpos(time):
        return 35 + 930 * (time - offset) / duration

    svg = [
        '<svg viewBox="0 0 1000 230" role="img" '
        'aria-label="Audio waveform">'
    ]

    if bounds is not None:
        left = max(35, xpos(bounds[0]))
        right = min(965, xpos(bounds[1]))
        svg.append(
            f'<rect x="{left}" y="20" '
            f'width="{max(0, right-left)}" height="160" '
            'fill="#dbeafe"/>'
        )

    svg.append(
        '<path d="M35 100 H965" stroke="#94a3b8"/>'
    )

    for time in onsets:
        if offset <= time <= offset + duration:
            x = xpos(time)
            svg.append(
                f'<path d="M{x} 20 V180" '
                'stroke="#b45309" stroke-dasharray="5 4"/>'
            )

    edges = np.linspace(
        0,
        len(samples),
        min(900, len(samples)) + 1,
        dtype=int,
    )

    points = []

    for start, end in zip(edges[:-1], edges[1:]):
        block = samples[start:end]
        x = 35 + 930 * ((start + end) / 2) / len(samples)

        points.extend([
            f"{x:.2f},{100-75*float(block.min()):.2f}",
            f"{x:.2f},{100-75*float(block.max()):.2f}",
        ])

    svg.append(
        '<polyline points="' + " ".join(points)
        + '" fill="none" stroke="#0f172a" stroke-width="0.8"/>'
    )

    for t in np.linspace(0, duration, 6):
        x = xpos(offset + t)
        svg.append(
            f'<text x="{x}" y="210" text-anchor="middle" '
            f'font-size="15">{offset+t:.3f}s</text>'
        )

    svg.append(
        '<text x="4" y="30" font-size="14">+1</text>'
        '<text x="4" y="178" font-size="14">-1</text></svg>'
    )

    return "".join(svg)


def build_review(raw_dir, output):
    raw_dir = Path(raw_dir).resolve()

    files = sorted(
        p for p in raw_dir.iterdir()
        if p.suffix.lower() == ".wav"
    )

    if not files:
        raise ValueError("No WAV files found.")

    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    sections = []

    for file in files:
        samples, metadata = load_wav(file)
        segment, result = segment_wav(file)
        rate = metadata["sample_rate_hz"]

        bounds = None if segment is None else (
            result["segment_start_seconds"],
            result["segment_end_seconds"],
        )

        events = result.get("candidate_onsets_seconds", [])

        audio_url = quote(
            os.path.relpath(file, output.parent).replace("\\", "/")
        )

        flags = ", ".join(result["flags"]) or "None"

        sections.append(
            f"<section><h2>{escape(file.name)}</h2>"
        )

        sections.append(
            f'<p>Status: {escape(result["segmentation_status"])}'
            f' | Candidates: {result.get("candidate_count", 0)}'
            f' | Flags: {escape(flags)}</p>'
        )

        sections.append(
            '<audio controls preload="none" '
            f'src="{escape(audio_url, quote=True)}"></audio>'
        )

        sections.append("<h3>Full recording</h3>")

        sections.append(
            waveform_svg(
                samples,
                rate,
                bounds=bounds,
                onsets=events,
            )
        )

        if segment is not None:
            sections.append(
                f"<h3>Selected response: {bounds[0]:.3f}s "
                f"to {bounds[1]:.3f}s</h3>"
            )
            sections.append(
                waveform_svg(
                    segment,
                    rate,
                    offset=bounds[0],
                    onsets=events,
                )
            )

        sections.append("</section>")

    html = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Acoustic Recording Review</title>
<style>
body {
    font-family: Arial, sans-serif;
    max-width: 1100px;
    margin: 30px auto;
    padding: 0 20px;
    color: #172033;
}
section {
    border-top: 1px solid #cbd5e1;
    margin-top: 32px;
    padding-top: 16px;
}
svg { width: 100%; height: auto; }
audio { width: 100%; }
p { line-height: 1.6; }
</style>
</head>
<body>
<h1>Acoustic Recording Review</h1>
<p>
Pilot segmentation only. Blue shading marks the selected interval.
Dashed lines mark event candidates.
Both plots use the same digital full-scale amplitude.
This page shows signal segmentation only; consult recordings.csv for researcher quality annotations.
Listen at a comfortable volume.
The strongest event is not automatically the intended sheet strike.
</p>
"""

    output.write_text(
        html + "".join(sections) + "</body></html>",
        encoding="utf-8",
    )

    print(f"Reviewed recordings: {len(files)}")
    print(f"Open in browser: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    build_review(args.raw_dir, args.output)