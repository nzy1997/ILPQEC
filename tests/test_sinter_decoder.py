"""Tests for the sinter adapter."""

import numpy as np
import pytest

from ilpdecoder import get_available_solvers


def _require_sinter_and_stim():
    pytest.importorskip("sinter")
    return pytest.importorskip("stim")


def _require_highs():
    if "highs" not in get_available_solvers():
        pytest.skip("HiGHS solver not available")


def test_sinter_compiled_decoder_bit_packed_roundtrip():
    _require_highs()
    stim = _require_sinter_and_stim()
    from ilpdecoder.sinter_decoder import SinterIlpDecoder

    dem = stim.DetectorErrorModel("error(0.1) D0 L0")
    decoder = SinterIlpDecoder()
    compiled = decoder.compile_decoder_for_dem(dem=dem)

    shots = np.array([[1], [0], [1]], dtype=np.uint8)
    bit_packed = np.packbits(shots, axis=1, bitorder="little")
    predictions = compiled.decode_shots_bit_packed(
        bit_packed_detection_event_data=bit_packed
    )

    pred_bits = np.unpackbits(predictions, axis=1, bitorder="little")[:, 0:1]
    np.testing.assert_array_equal(pred_bits, shots)


def test_sinter_decode_via_files(tmp_path):
    _require_highs()
    stim = _require_sinter_and_stim()
    from ilpdecoder.sinter_decoder import SinterIlpDecoder

    dem = stim.DetectorErrorModel("error(0.1) D0 L0")
    dem_path = tmp_path / "model.dem"
    dets_path = tmp_path / "dets.b8"
    obs_path = tmp_path / "obs.b8"
    dem.to_file(dem_path)

    shots = np.array([[0], [1], [1], [0]], dtype=np.uint8)
    stim.write_shot_data_file(
        data=shots, path=dets_path, format="b8", num_detectors=1
    )

    decoder = SinterIlpDecoder()
    decoder.decode_via_files(
        num_shots=shots.shape[0],
        num_dets=1,
        num_obs=1,
        dem_path=dem_path,
        dets_b8_in_path=dets_path,
        obs_predictions_b8_out_path=obs_path,
        tmp_dir=tmp_path,
    )

    predictions = stim.read_shot_data_file(
        path=obs_path, format="b8", num_observables=1
    )
    np.testing.assert_array_equal(predictions, shots)
