"""Event-selection behavior checked with controlled temporary audio fixtures."""
import wave

import numpy as np
import pytest

from backend.app.ml.component_4_thud_test.preprocessing import load_wav
from backend.app.ml.component_4_thud_test.preprocessing import segment_wav


RATE = 16000


def recording(tmp_path, events=(), duration=3.0, dc=0.0):
    samples = np.full(round(RATE * duration), dc, dtype=float)
    for onset, amplitude, length in events:
        start, count = round(onset * RATE), round(length * RATE)
        pulse = amplitude * np.where(np.arange(count) % 2 == 0, 1, -1)
        samples[start:start + count] += pulse[:len(samples[start:start + count])]
    path = tmp_path / 'events.wav'
    with wave.open(str(path), 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(RATE)
        wav.writeframes(np.rint(samples * 32768).astype('<i2').tobytes())
    return path


def test_one_event_extracts_complete_pre_and_post_window(tmp_path):
    path = recording(tmp_path, [(1.0, 0.5, 0.02)])
    original = path.read_bytes()
    segment, report = segment_wav(path)
    assert report['segmentation_status'] == 'SEGMENT_EXTRACTED'
    assert report['candidate_count'] == 1
    assert report['onset_seconds'] == pytest.approx(1.0, abs=0.005)
    assert report['segment_start_seconds'] == pytest.approx(0.95, abs=0.005)
    assert report['segment_end_seconds'] == pytest.approx(1.75, abs=0.005)
    assert report['segment_duration_seconds'] == pytest.approx(0.8)
    assert len(segment) == 12800
    assert report['flags'] == []
    assert report['quality_status'] == 'NOT_EVALUATED'
    assert path.read_bytes() == original


def test_silence_has_no_event_or_segment(tmp_path):
    segment, report = segment_wav(recording(tmp_path))
    assert segment is None
    assert report['segmentation_status'] == 'NO_EVENT'
    assert report['candidate_count'] == 0


def test_dc_only_recording_does_not_create_false_event(tmp_path):
    segment, report = segment_wav(recording(tmp_path, dc=0.2))
    assert segment is None
    assert report['segmentation_status'] == 'NO_EVENT'


def test_two_separate_events_are_retained_in_report(tmp_path):
    path = recording(tmp_path, [(0.8, 0.3, 0.02), (1.2, 0.6, 0.02)])
    segment, report = segment_wav(path)
    assert segment is not None
    assert report['candidate_count'] == 2
    assert report['candidate_onsets_seconds'] == pytest.approx([0.8, 1.2], abs=0.005)
    assert 'MULTIPLE_EVENT_CANDIDATES' in report['flags']
    assert report['selected_candidate_index'] == 1


def test_selection_uses_event_energy_not_just_highest_peak(tmp_path):
    path = recording(tmp_path, [(0.8, 0.3, 0.1), (1.3, 0.6, 0.01)])
    _, report = segment_wav(path)
    assert report['candidate_count'] == 2
    assert report['selected_candidate_index'] == 0
    assert report['onset_seconds'] == pytest.approx(0.8, abs=0.005)


def test_close_fragments_are_merged_as_one_candidate(tmp_path):
    path = recording(tmp_path, [(1.0, 0.4, 0.02), (1.05, 0.4, 0.02)])
    _, report = segment_wav(path)
    assert report['candidate_count'] == 1
    assert 'MULTIPLE_EVENT_CANDIDATES' not in report['flags']


def test_late_event_flags_missing_post_window(tmp_path):
    segment, report = segment_wav(recording(tmp_path, [(2.8, 0.4, 0.02)]))
    assert segment is not None
    assert 'SEGMENT_TRUNCATED' in report['flags']
    assert report['segment_end_seconds'] == 3.0
    assert report['segment_duration_seconds'] < 0.8


def test_early_event_is_flagged_for_background_contamination(tmp_path):
    _, report = segment_wav(recording(tmp_path, [(0.1, 0.5, 0.01)]))
    assert 'EVENT_IN_BACKGROUND_INTERVAL' in report['flags']


def test_too_short_recording_does_not_produce_segment(tmp_path):
    segment, report = segment_wav(recording(tmp_path, duration=0.1))
    assert segment is None
    assert report['segmentation_status'] == 'CAPTURE_TOO_SHORT'


def test_detection_dc_removal_does_not_rewrite_returned_samples(tmp_path):
    path = recording(tmp_path, [(1.0, 0.4, 0.02)], dc=0.05)
    samples, _ = load_wav(path)
    segment, report = segment_wav(path)
    start = round(report['segment_start_seconds'] * RATE)
    end = round(report['segment_end_seconds'] * RATE)
    np.testing.assert_array_equal(segment, samples[start:end])
    assert segment[0] == pytest.approx(0.05, abs=1 / 32768)
