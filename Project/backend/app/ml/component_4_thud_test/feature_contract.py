"""Ordered input feature contract for Component 4 quality training."""

FEATURE_NAMES = [
    "segment_rms", "segment_peak", "crest_factor", "zero_crossing_rate",
    "background_rms", "outside_segment_rms", "segment_to_background_db",
    "near_full_scale_fraction", "event_candidate_count", "dominant_frequency_hz",
    "spectral_centroid_hz", "spectral_bandwidth_hz", "spectral_rolloff85_hz",
    "spectral_flatness", "spectral_entropy", "late_to_early_energy_db",
]
