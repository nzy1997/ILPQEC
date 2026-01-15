"""Sinter adapter for ILPDecoder (HiGHS direct backend only)."""

from __future__ import annotations

import pathlib
from typing import Any, Dict, Optional

import numpy as np
import sinter
import stim

from ilpdecoder.decoder import Decoder


class _IlpCompiledDecoder(sinter.CompiledDecoder):
    def __init__(self, decoder: Decoder, num_dets: int, num_obs: int):
        self._decoder = decoder
        self._num_dets = num_dets
        self._num_obs = num_obs

    def decode_shots_bit_packed(
        self,
        *,
        bit_packed_detection_event_data: np.ndarray,
    ) -> np.ndarray:
        num_shots = bit_packed_detection_event_data.shape[0]
        if self._num_dets == 0:
            num_obs_bytes = (self._num_obs + 7) // 8
            return np.zeros((num_shots, num_obs_bytes), dtype=np.uint8)

        shots = np.unpackbits(
            bit_packed_detection_event_data, axis=1, bitorder="little"
        )
        shots = shots[:, : self._num_dets].astype(np.uint8, copy=False)
        predictions = self._decoder.decode_batch(shots)
        predictions = np.asarray(predictions, dtype=np.uint8)
        return np.packbits(predictions, axis=1, bitorder="little")


class SinterIlpDecoder(sinter.Decoder):
    """Use ILPDecoder (direct HiGHS backend) to predict observables from detections."""

    def __init__(
        self,
        *,
        time_limit: Optional[float] = None,
        gap: Optional[float] = None,
        threads: Optional[int] = None,
        verbose: bool = False,
        options: Optional[Dict[str, Any]] = None,
        merge_parallel: bool = True,
        flatten_dem: bool = True,
    ):
        self._time_limit = time_limit
        self._gap = gap
        self._threads = threads
        self._verbose = verbose
        self._options = dict(options) if options else {}
        self._merge_parallel = merge_parallel
        self._flatten_dem = flatten_dem

    def _build_decoder(self, dem: stim.DetectorErrorModel) -> Decoder:
        return Decoder.from_stim_dem(
            dem,
            solver="highs",
            direct=True,
            time_limit=self._time_limit,
            gap=self._gap,
            threads=self._threads,
            verbose=self._verbose,
            merge_parallel_edges=self._merge_parallel,
            flatten_dem=self._flatten_dem,
            **self._options,
        )

    def compile_decoder_for_dem(self, *, dem: stim.DetectorErrorModel) -> sinter.CompiledDecoder:
        decoder = self._build_decoder(dem)
        return _IlpCompiledDecoder(decoder, dem.num_detectors, dem.num_observables)

    def decode_via_files(
        self,
        *,
        num_shots: int,
        num_dets: int,
        num_obs: int,
        dem_path: pathlib.Path,
        dets_b8_in_path: pathlib.Path,
        obs_predictions_b8_out_path: pathlib.Path,
        tmp_dir: pathlib.Path,
    ) -> None:
        if num_dets == 0:
            with open(obs_predictions_b8_out_path, "wb") as handle:
                handle.write(b"\0" * (num_obs * num_shots))
            return

        dem = stim.DetectorErrorModel.from_file(dem_path)
        decoder = self._build_decoder(dem)

        shots = stim.read_shot_data_file(
            path=dets_b8_in_path, format="b8", num_detectors=num_dets
        )
        predictions = decoder.decode_batch(shots)
        predictions = np.asarray(predictions, dtype=np.uint8)
        stim.write_shot_data_file(
            data=predictions,
            path=obs_predictions_b8_out_path,
            format="b8",
            num_observables=num_obs,
        )
