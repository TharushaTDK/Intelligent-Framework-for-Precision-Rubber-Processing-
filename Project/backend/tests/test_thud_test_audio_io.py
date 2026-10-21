"""PCM format and validation checks using generated temporary WAV fixtures."""
import hashlib
import wave

import numpy as np
import pytest

from backend.app.ml.component_4_thud_test.preprocessing import inspect_wav, load_wav


def write_pcm(path, values, width=3, channels=1, rate=16000):
    values = np.asarray(values, dtype=np.int64)
    if width == 3:
        raw = np.column_stack((values & 255, (values >> 8) & 255,
                               (values >> 16) & 255)).astype(np.uint8).tobytes()
    else:
        raw = values.astype({1: 'u1', 2: '<i2', 4: '<i4'}[width]).tobytes()
    with wave.open(str(path), 'wb') as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(width)
        wav.setframerate(rate)
        wav.writeframes(raw)
    return path


@pytest.mark.parametrize('width', [2, 3, 4], ids=['16-bit', '24-bit', '32-bit'])
def test_signed_pcm_boundaries_and_fixed_full_scale(tmp_path, width):
    scale = 2 ** (width * 8 - 1)
    values = np.array([-scale, -1, 0, 1, scale - 1], dtype=np.int64)
    path = write_pcm(tmp_path / 'boundaries.wav', values, width)
    samples, metadata = load_wav(path)
    np.testing.assert_array_equal(samples, values.astype(np.float64) / scale)
    assert metadata['bit_depth'] == width * 8
    assert metadata['digital_rail_samples'] == 2
    assert metadata['frames'] == len(values)
    assert metadata['pcm_bytes'] == len(values) * width


def test_sub_full_scale_audio_is_not_peak_normalized(tmp_path):
    scale = 2 ** 23
    path = write_pcm(tmp_path / 'quiet.wav', [-scale // 4, 0, scale // 4])
    samples, _ = load_wav(path)
    np.testing.assert_array_equal(samples, [-0.25, 0, 0.25])


def test_inspection_keeps_original_bytes_and_reports_metadata(tmp_path):
    path = write_pcm(tmp_path / 'recording.wav', [16384, -16384] * 8000, width=2)
    original = path.read_bytes()
    result = inspect_wav(path)
    assert path.read_bytes() == original
    assert result['sha256'] == hashlib.sha256(original).hexdigest()
    assert result['sample_rate_hz'] == 16000
    assert result['channels'] == 1
    assert result['duration_seconds'] == 1.0
    assert result['peak_percent_full_scale'] == 50.0
    assert result['rms_dbfs'] == pytest.approx(-6.021, abs=0.001)
    assert result['dc_offset'] == 0.0
    assert result['quality_status'] == 'NOT_EVALUATED'


def test_zero_pcm_is_reported_as_silent(tmp_path):
    path = write_pcm(tmp_path / 'silence.wav', np.zeros(16000, dtype=np.int64))
    result = inspect_wav(path)
    assert result['silent'] is True
    assert result['peak_percent_full_scale'] == 0.0
    assert result['rms_dbfs'] is None


def test_stereo_is_rejected_without_silent_channel_mixing(tmp_path):
    path = write_pcm(tmp_path / 'stereo.wav', [100, -100] * 100, width=2, channels=2)
    with pytest.raises(ValueError, match='Mono'):
        load_wav(path)


def test_unsupported_8_bit_pcm_is_rejected(tmp_path):
    path = write_pcm(tmp_path / '8bit.wav', [128] * 100, width=1)
    with pytest.raises(ValueError, match='Supported PCM widths'):
        load_wav(path)


def test_truncated_pcm_is_not_accepted_as_complete(tmp_path):
    path = write_pcm(tmp_path / 'truncated.wav', [100, -100] * 100)
    path.write_bytes(path.read_bytes()[:-6])
    with pytest.raises(ValueError, match='Incomplete WAV'):
        load_wav(path)


def test_empty_wav_is_rejected(tmp_path):
    path = write_pcm(tmp_path / 'empty.wav', [])
    with pytest.raises(ValueError, match='no usable frames'):
        load_wav(path)


def test_non_wav_bytes_are_rejected(tmp_path):
    path = tmp_path / 'invalid.wav'
    path.write_bytes(b'This is not a RIFF WAV recording.')
    with pytest.raises((wave.Error, EOFError, ValueError)):
        load_wav(path)


def test_oversized_input_is_rejected_before_audio_decoding(tmp_path):
    path = tmp_path / 'oversized.wav'
    with path.open('wb') as file:
        file.truncate(10 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match='10 MiB'):
        load_wav(path)
