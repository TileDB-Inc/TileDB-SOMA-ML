# Copyright (c) 2021-2024 The Chan Zuckerberg Initiative Foundation
# Copyright (c) 2021-2024 TileDB, Inc.
#
# Licensed under the MIT License.

from functools import partial
from typing import Tuple, Type
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal
from tiledbsoma import Experiment

from tests.utils import assert_array_equal, eager_lazy, pytorch_x_value_gen
from tiledbsoma_ml import (
    ExperimentAxisQueryIterableDataset,
    ExperimentAxisQueryIterDataPipe,
    experiment_dataloader,
)
from tiledbsoma_ml.pytorch import NDArrayNumber

IterableWrapperType = (
    Type[ExperimentAxisQueryIterDataPipe] | Type[ExperimentAxisQueryIterableDataset]
)
iterable_wrappers = pytest.mark.parametrize(
    "PipeClass", (ExperimentAxisQueryIterDataPipe, ExperimentAxisQueryIterableDataset)
)


@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen", [(6, 3, pytorch_x_value_gen)]
)
@iterable_wrappers
def test_multiprocessing__returns_full_result(
    PipeClass: IterableWrapperType,
    soma_experiment: Experiment,
):
    """Tests the ExperimentAxisQueryIterDataPipe provides all data, as collected from multiple processes that are managed by a
    PyTorch DataLoader with multiple workers configured."""
    with soma_experiment.axis_query(measurement_name="RNA") as query:
        dp = PipeClass(
            query,
            X_name="raw",
            obs_column_names=["soma_joinid", "label"],
            io_batch_size=3,  # two chunks, one per worker
        )
        # Note we're testing the ExperimentAxisQueryIterDataPipe via a DataLoader, since this is what sets up the multiprocessing
        dl = experiment_dataloader(dp, num_workers=2)

        batches = list(dl)

        soma_joinids = np.concatenate(
            [obs["soma_joinid"].to_numpy() for _, obs in batches]
        )
        assert sorted(soma_joinids) == list(range(6))


@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen", [(3, 3, pytorch_x_value_gen)]
)
@eager_lazy
@iterable_wrappers
def test_experiment_dataloader__non_batched(
    PipeClass: IterableWrapperType,
    soma_experiment: Experiment,
    use_eager_fetch: bool,
):
    with soma_experiment.axis_query(measurement_name="RNA") as query:
        dp = PipeClass(
            query,
            X_name="raw",
            obs_column_names=["label"],
            use_eager_fetch=use_eager_fetch,
        )
        dl = experiment_dataloader(dp)
        batches = list(dl)
        assert all(X.shape == (3,) for X, _ in batches)
        assert all(obs.shape == (1, 1) for _, obs in batches)

        X, obs = batches[0]
        assert_array_equal(X, np.array([0, 1, 0], dtype=np.float32))
        assert_frame_equal(obs, pd.DataFrame({"label": ["0"]}))


@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen",
    [(6, 3, pytorch_x_value_gen)],
)
@eager_lazy
@iterable_wrappers
def test_experiment_dataloader__batched(
    PipeClass: IterableWrapperType,
    soma_experiment: Experiment,
    use_eager_fetch: bool,
):
    with soma_experiment.axis_query(measurement_name="RNA") as query:
        dp = PipeClass(
            query,
            X_name="raw",
            batch_size=3,
            use_eager_fetch=use_eager_fetch,
        )
        dl = experiment_dataloader(dp)
        batches = list(dl)

        X, obs = batches[0]
        assert_array_equal(
            X, np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
        )
        assert_frame_equal(obs, pd.DataFrame({"soma_joinid": [0, 1, 2]}))


@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen",
    [(10, 3, pytorch_x_value_gen)],
)
@eager_lazy
@iterable_wrappers
def test_experiment_dataloader__batched_length(
    PipeClass: IterableWrapperType,
    soma_experiment: Experiment,
    use_eager_fetch: bool,
):
    with soma_experiment.axis_query(measurement_name="RNA") as query:
        dp = PipeClass(
            query,
            X_name="raw",
            obs_column_names=["label"],
            batch_size=3,
            use_eager_fetch=use_eager_fetch,
        )
        dl = experiment_dataloader(dp)
        assert len(dl) == len(list(dl))


@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen,batch_size",
    [(10, 3, pytorch_x_value_gen, batch_size) for batch_size in (1, 3, 10)],
)
@iterable_wrappers
def test_experiment_dataloader__collate_fn(
    PipeClass: IterableWrapperType,
    soma_experiment: Experiment,
    batch_size: int,
):
    def collate_fn(
        batch_size: int, batch: Tuple[NDArrayNumber, pd.DataFrame]
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

    with soma_experiment.axis_query(measurement_name="RNA") as query:
        dp = PipeClass(
            query,
            X_name="raw",
            obs_column_names=["label"],
            batch_size=batch_size,
        )
        dl = experiment_dataloader(dp, collate_fn=partial(collate_fn, batch_size))
        assert len(list(dl)) > 0


@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen", [(10, 1, pytorch_x_value_gen)]
)
def test__pytorch_splitting(soma_experiment: Experiment):
    with soma_experiment.axis_query(measurement_name="RNA") as query:
        dp = ExperimentAxisQueryIterDataPipe(
            query,
            X_name="raw",
            obs_column_names=["label"],
        )
        # function not available for IterableDataset, yet....
        dp_train, dp_test = dp.random_split(
            weights={"train": 0.7, "test": 0.3}, seed=1234
        )
        dl = experiment_dataloader(dp_train)

        batches = list(dl)
        assert len(batches) == 7


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
