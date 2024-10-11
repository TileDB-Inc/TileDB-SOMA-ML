# Copyright (c) 2021-2024 The Chan Zuckerberg Initiative Foundation
# Copyright (c) 2021-2024 TileDB, Inc.
#
# Licensed under the MIT License.

from pathlib import Path
from typing import Optional, Sequence

import pytest
from tiledbsoma import Collection, Experiment, Measurement

from tests.utils import XValueGen, add_dataframe, add_sparse_array


@pytest.fixture
def X_layer_names() -> list[str]:
    return ["raw"]


@pytest.fixture
def obsp_layer_names() -> Optional[list[str]]:
    return None


@pytest.fixture
def varp_layer_names() -> Optional[list[str]]:
    return None


@pytest.fixture(scope="function")
def soma_experiment(
    tmp_path: Path,
    obs_range: int | range,
    var_range: int | range,
    X_value_gen: XValueGen,
    obsp_layer_names: Sequence[str],
    varp_layer_names: Sequence[str],
) -> Experiment:
    with Experiment.create((tmp_path / "exp").as_posix()) as exp:
        if isinstance(obs_range, int):
            obs_range = range(obs_range)
        if isinstance(var_range, int):
            var_range = range(var_range)

        add_dataframe(exp, "obs", obs_range)
        ms = exp.add_new_collection("ms")
        rna = ms.add_new_collection("RNA", Measurement)
        add_dataframe(rna, "var", var_range)
        rna_x = rna.add_new_collection("X", Collection)
        add_sparse_array(rna_x, "raw", obs_range, var_range, X_value_gen)

        if obsp_layer_names:
            obsp = rna.add_new_collection("obsp")
            for obsp_layer_name in obsp_layer_names:
                add_sparse_array(
                    obsp, obsp_layer_name, obs_range, var_range, X_value_gen
                )

        if varp_layer_names:
            varp = rna.add_new_collection("varp")
            for varp_layer_name in varp_layer_names:
                add_sparse_array(
                    varp, varp_layer_name, obs_range, var_range, X_value_gen
                )

    return Experiment.open((tmp_path / "exp").as_posix())
