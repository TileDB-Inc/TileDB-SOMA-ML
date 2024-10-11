# Copyright (c) 2021-2024 The Chan Zuckerberg Initiative Foundation
# Copyright (c) 2021-2024 TileDB, Inc.
#
# Licensed under the MIT License.

from __future__ import annotations

from functools import partial
from typing import Tuple
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal
from utz import parametrize

from tests.case import Case
from tests.utils import assert_array_equal
from tiledbsoma_ml import (
    ExperimentAxisQueryIterableDataset,
    ExperimentAxisQueryIterDataPipe,
    experiment_dataloader,
)
from tiledbsoma_ml.pytorch import (
    NDArrayNumber,
    XObsDatum,
)

# Only test the DataPipe and Dataset classes in this file
# (they each wrap ``ExperimentAxisQueryIterable``)
pipeclasses = partial(
    parametrize,
    PipeClass=(
        ExperimentAxisQueryIterDataPipe,
        ExperimentAxisQueryIterableDataset,
    ),
)
pipeclasses_eagerlazy = partial(pipeclasses, use_eager_fetch=[True, False])


@pipeclasses(
    Case(
        obs_range=6,
        obs_column_names=["soma_joinid", "label"],
        io_batch_size=3,  # two chunks, one per worker
        num_workers=2,
    )
)
def test_multiprocessing__returns_full_result(batches: list[XObsDatum]):
    """Test that ``ExperimentAxisQueryIter*Data{set,Pipe}`` provides all data, as collected from
    multiple processes managed by a PyTorch DataLoader with multiple workers."""
    soma_joinids = np.concatenate([t[1]["soma_joinid"].to_numpy() for t in batches])
    assert sorted(soma_joinids) == list(range(6))


@pipeclasses_eagerlazy(Case(obs_range=3, obs_column_names=["label"], shuffle=False))
def test_experiment_dataloader__non_batched(batches):
    for X, obs in batches:
        assert X.shape == (3,)
        assert obs.shape == (1, 1)

    X, obs = batches[0]
    assert_array_equal(X, np.array([0, 1, 0], dtype=np.float32))
    assert_frame_equal(obs, pd.DataFrame({"label": ["0"]}))


@pipeclasses_eagerlazy(Case(obs_range=6, batch_size=3, shuffle=False))
def test_experiment_dataloader__batched(batches):
    X, obs = batches[0]
    assert_array_equal(X, np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32))
    assert_frame_equal(obs, pd.DataFrame({"soma_joinid": [0, 1, 2]}))


@pipeclasses_eagerlazy(Case(obs_range=10))
def test_experiment_dataloader__batched_length(dataloader):
    with dataloader() as dl:
        assert len(dl) == len(list(dl))


@pipeclasses(
    Case(obs_range=10, batch_size=batch_size, expected_nbatches=expected_nbatches)
    for batch_size, expected_nbatches in [(1, 10), (3, 4), (10, 1)]
)
def test_experiment_dataloader__collate_fn(
    dataloader, batch_size: int, expected_nbatches: int
):
    def collate_fn(
        batch: Tuple[NDArrayNumber, pd.DataFrame]
    ) -> Tuple[NDArrayNumber, pd.DataFrame]:
        assert isinstance(batch, tuple)
        assert len(batch) == 2
        X, obs = batch
        assert isinstance(X, np.ndarray) and isinstance(obs, pd.DataFrame)
        if batch_size > 1:
            assert X.shape[0] == obs.shape[0]
            assert X.shape[0] <= batch_size
        else:
            assert X.ndim == 1
        assert obs.shape[1] <= batch_size
        return batch

    with dataloader(collate_fn=collate_fn) as dl:
        batches = list(dl)

    assert len(batches) == expected_nbatches


@parametrize(
    Case(obs_range=10, var_range=1, obs_column_names=["label"]),
)
def test__pytorch_splitting(datapipe):
    with datapipe as dp:
        # ``random_split`` not available for ``IterableDataset``, yet...
        dp_train, dp_test = dp.random_split(
            weights={"train": 0.7, "test": 0.3}, seed=1234
        )
        dl = experiment_dataloader(dp_train)
        train_batches = list(dl)
        assert len(train_batches) == 7

        dl = experiment_dataloader(dp_test)
        test_batches = list(dl)
        assert len(test_batches) == 3


def test_experiment_dataloader__unsupported_params__fails():
    with patch(
        "tiledbsoma_ml.pytorch.ExperimentAxisQueryIterDataPipe"
    ) as dummy_exp_data_pipe:
        with pytest.raises(ValueError):
            experiment_dataloader(dummy_exp_data_pipe, shuffle=True)
        with pytest.raises(ValueError):
            experiment_dataloader(dummy_exp_data_pipe, batch_size=3)
        with pytest.raises(ValueError):
            experiment_dataloader(dummy_exp_data_pipe, batch_sampler=[])
        with pytest.raises(ValueError):
            experiment_dataloader(dummy_exp_data_pipe, sampler=[])
