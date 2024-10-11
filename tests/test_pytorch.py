# Copyright (c) 2021-2024 The Chan Zuckerberg Initiative Foundation
# Copyright (c) 2021-2024 TileDB, Inc.
#
# Licensed under the MIT License.

from __future__ import annotations

from typing import Type
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal
from scipy import sparse
from somacore import AxisQuery
from tiledbsoma import Experiment
from torch.utils.data._utils.worker import WorkerInfo

from tests.utils import (
    assert_array_equal,
    eager_lazy,
    init_world,
    pytorch_seq_x_value_gen,
    pytorch_x_value_gen,
)
from tiledbsoma_ml.pytorch import (
    ExperimentAxisQueryIterable,
    ExperimentAxisQueryIterableDataset,
    ExperimentAxisQueryIterDataPipe,
)

# These control which classes are tested (for most, but not all tests).
# Centralized to allow easy add/delete of specific test parameters.
PipeClassType = (
    Type[ExperimentAxisQueryIterable]
    | Type[ExperimentAxisQueryIterDataPipe]
    | Type[ExperimentAxisQueryIterableDataset]
)
PipeClasses = (
    ExperimentAxisQueryIterable,
    ExperimentAxisQueryIterDataPipe,
    ExperimentAxisQueryIterableDataset,
)

pipeclasses = pytest.mark.parametrize("PipeClass", PipeClasses)
sparse_dense = pytest.mark.parametrize("return_sparse_X", [True, False])


@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen",
    [(6, 3, pytorch_x_value_gen)],
)
@eager_lazy
@sparse_dense
@pipeclasses
def test_non_batched(
    PipeClass: PipeClassType,
    soma_experiment: Experiment,
    use_eager_fetch: bool,
    return_sparse_X: bool,
):
    """Check batches of size 1 (the default)"""
    with soma_experiment.axis_query(measurement_name="RNA") as query:
        exp_data_pipe = PipeClass(
            query,
            X_name="raw",
            obs_column_names=["label"],
            shuffle=False,
            use_eager_fetch=use_eager_fetch,
            return_sparse_X=return_sparse_X,
        )
        assert exp_data_pipe.shape == (6, 3)
        batch_iter = iter(exp_data_pipe)
        for idx, (X, obs) in enumerate(batch_iter):
            expected_X = [0, 1, 0] if idx % 2 == 0 else [1, 0, 1]
            if return_sparse_X:
                assert isinstance(X, sparse.csr_matrix)
                # Sparse slices are always 2D
                assert X.shape == (1, 3)
                assert X.todense().tolist() == [expected_X]
            else:
                assert isinstance(X, np.ndarray)
                if PipeClass is ExperimentAxisQueryIterable:
                    assert X.shape == (1, 3)
                    assert X.tolist() == [expected_X]
                else:
                    # ExperimentAxisQueryIterData{Pipe,set} "squeeze" dense single-row batches
                    assert X.shape == (3,)
                    assert X.tolist() == expected_X

            assert_frame_equal(obs, pd.DataFrame({"label": [str(idx)]}))


@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen",
    [(6, 3, pytorch_x_value_gen)],
)
@eager_lazy
@sparse_dense
@pipeclasses
def test_uneven_soma_and_result_batches(
    PipeClass: PipeClassType,
    soma_experiment: Experiment,
    use_eager_fetch: bool,
    return_sparse_X: bool,
):
    """Check that batches are correctly created when they require fetching multiple chunks."""
    with soma_experiment.axis_query(measurement_name="RNA") as query:
        exp_data_pipe = PipeClass(
            query,
            X_name="raw",
            obs_column_names=["label"],
            batch_size=3,
            shuffle=False,
            io_batch_size=2,
            use_eager_fetch=use_eager_fetch,
            return_sparse_X=return_sparse_X,
        )
        assert exp_data_pipe.shape == (2, 3)
        batch_iter = iter(exp_data_pipe)

        X, obs = next(batch_iter)
        assert X.shape == (3, 3)
        if return_sparse_X:
            assert isinstance(X, sparse.csr_matrix)
            X = X.todense()
        else:
            assert isinstance(X, np.ndarray)
        assert X.tolist() == [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
        assert_frame_equal(obs, pd.DataFrame({"label": ["0", "1", "2"]}))

        X, obs = next(batch_iter)
        assert X.shape == (3, 3)
        if return_sparse_X:
            assert isinstance(X, sparse.csr_matrix)
            X = X.todense()
        else:
            assert isinstance(X, np.ndarray)
        assert X.tolist() == [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
        assert_frame_equal(obs, pd.DataFrame({"label": ["3", "4", "5"]}))


@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen",
    [(6, 3, pytorch_x_value_gen)],
)
@eager_lazy
@sparse_dense
@pipeclasses
def test_batching__all_batches_full_size(
    PipeClass: PipeClassType,
    soma_experiment: Experiment,
    use_eager_fetch: bool,
    return_sparse_X: bool,
):
    with soma_experiment.axis_query(measurement_name="RNA") as query:
        exp_data_pipe = PipeClass(
            query,
            X_name="raw",
            obs_column_names=["label"],
            batch_size=3,
            shuffle=False,
            use_eager_fetch=use_eager_fetch,
            return_sparse_X=return_sparse_X,
        )
        batch_iter = iter(exp_data_pipe)
        assert exp_data_pipe.shape == (2, 3)

        X, obs = next(batch_iter)
        if return_sparse_X:
            assert isinstance(X, sparse.csr_matrix)
            X = X.todense()
        assert X.tolist() == [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
        assert_frame_equal(obs, pd.DataFrame({"label": ["0", "1", "2"]}))

        X, obs = next(batch_iter)
        if return_sparse_X:
            assert isinstance(X, sparse.csr_matrix)
            X = X.todense()
        assert X.tolist() == [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
        assert_frame_equal(obs, pd.DataFrame({"label": ["3", "4", "5"]}))

        with pytest.raises(StopIteration):
            next(batch_iter)


@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen",
    [(range(100_000_000, 100_000_003), 3, pytorch_x_value_gen)],
)
@eager_lazy
@pipeclasses
def test_soma_joinids(
    PipeClass: PipeClassType,
    soma_experiment: Experiment,
    use_eager_fetch: bool,
):
    with soma_experiment.axis_query(measurement_name="RNA") as query:
        exp_data_pipe = PipeClass(
            query,
            X_name="raw",
            obs_column_names=["soma_joinid", "label"],
            batch_size=3,
            shuffle=False,
            use_eager_fetch=use_eager_fetch,
        )
        assert exp_data_pipe.shape == (1, 3)

        soma_joinids = np.concatenate(
            [batch[1]["soma_joinid"].to_numpy() for batch in exp_data_pipe]
        )
        assert_array_equal(soma_joinids, np.arange(100_000_000, 100_000_003))


@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen",
    [(5, 3, pytorch_x_value_gen)],
)
@eager_lazy
@sparse_dense
@pipeclasses
def test_batching__partial_final_batch_size(
    PipeClass: PipeClassType,
    soma_experiment: Experiment,
    use_eager_fetch: bool,
    return_sparse_X: bool,
):
    with soma_experiment.axis_query(measurement_name="RNA") as query:
        exp_data_pipe = PipeClass(
            query,
            X_name="raw",
            obs_column_names=["label"],
            batch_size=3,
            shuffle=False,
            use_eager_fetch=use_eager_fetch,
            return_sparse_X=return_sparse_X,
        )
        assert exp_data_pipe.shape == (2, 3)
        batch_iter = iter(exp_data_pipe)

        next(batch_iter)
        X, obs = next(batch_iter)
        if return_sparse_X:
            assert isinstance(X, sparse.csr_matrix)
            X = X.todense()
        assert X.tolist() == [[1, 0, 1], [0, 1, 0]]
        assert_frame_equal(obs, pd.DataFrame({"label": ["3", "4"]}))

        with pytest.raises(StopIteration):
            next(batch_iter)


@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen",
    [(3, 3, pytorch_x_value_gen)],
)
@eager_lazy
@pipeclasses
def test_batching__exactly_one_batch(
    PipeClass: PipeClassType,
    soma_experiment: Experiment,
    use_eager_fetch: bool,
):
    with soma_experiment.axis_query(measurement_name="RNA") as query:
        exp_data_pipe = PipeClass(
            query,
            X_name="raw",
            obs_column_names=["label"],
            batch_size=3,
            shuffle=False,
            use_eager_fetch=use_eager_fetch,
        )
        assert exp_data_pipe.shape == (1, 3)
        batch_iter = iter(exp_data_pipe)
        X, obs = next(batch_iter)
        assert X.tolist() == [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
        assert_frame_equal(obs, pd.DataFrame({"label": ["0", "1", "2"]}))

        with pytest.raises(StopIteration):
            next(batch_iter)


@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen",
    [(6, 3, pytorch_x_value_gen)],
)
@eager_lazy
@pipeclasses
def test_batching__empty_query_result(
    PipeClass: PipeClassType,
    soma_experiment: Experiment,
    use_eager_fetch: bool,
):
    with soma_experiment.axis_query(
        measurement_name="RNA", obs_query=AxisQuery(coords=([],))
    ) as query:
        exp_data_pipe = PipeClass(
            query,
            X_name="raw",
            obs_column_names=["label"],
            batch_size=3,
            use_eager_fetch=use_eager_fetch,
        )
        assert exp_data_pipe.shape == (0, 3)
        batch_iter = iter(exp_data_pipe)

        with pytest.raises(StopIteration):
            next(batch_iter)


@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen",
    [(10, 1, pytorch_x_value_gen)],
)
@eager_lazy
@pipeclasses
def test_batching__partial_soma_batches_are_concatenated(
    PipeClass: PipeClassType, soma_experiment: Experiment, use_eager_fetch: bool
):
    with soma_experiment.axis_query(measurement_name="RNA") as query:
        exp_data_pipe = PipeClass(
            query,
            X_name="raw",
            obs_column_names=["label"],
            batch_size=3,
            # set SOMA batch read size such that PyTorch batches will span the tail and head of two SOMA batches
            io_batch_size=4,
            use_eager_fetch=use_eager_fetch,
        )

        batches = list(exp_data_pipe)

        assert [len(X) for X, _ in batches] == [3, 3, 3, 1]


@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen",
    [(6, 3, pytorch_x_value_gen), (7, 3, pytorch_x_value_gen)],
)
@pytest.mark.parametrize(
    "world_size,rank",
    [(3, 0), (3, 1), (3, 2), (2, 0), (2, 1)],
)
@pipeclasses
def test_distributed__returns_data_partition_for_rank(
    PipeClass: PipeClassType,
    soma_experiment: Experiment,
    obs_range: int,
    world_size: int,
    rank: int,
):
    """Tests pytorch._partition_obs_joinids() behavior in a simulated PyTorch distributed processing mode,
    using mocks to avoid having to do real PyTorch distributed setup."""

    with (
        init_world(world_size, rank),
        soma_experiment.axis_query(measurement_name="RNA") as query,
    ):
        dp = PipeClass(
            query,
            X_name="raw",
            obs_column_names=["soma_joinid"],
            io_batch_size=2,
            shuffle=False,
        )
        batches = list(dp)
        soma_joinids = np.concatenate(
            [batch[1]["soma_joinid"].to_numpy() for batch in batches]
        )

        expected_joinids = np.array_split(np.arange(obs_range), world_size)[rank][
            0 : obs_range // world_size
        ].tolist()
        assert sorted(soma_joinids) == expected_joinids


# fmt: off
@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen,world_size,num_workers,splits",
    [
        (12, 3, pytorch_x_value_gen, 3, 2, [[0, 2, 4], [4,  6,  8], [ 8, 10, 12]]),
        (13, 3, pytorch_x_value_gen, 3, 2, [[0, 2, 4], [5,  7,  9], [ 9, 11, 13]]),
        (15, 3, pytorch_x_value_gen, 3, 2, [[0, 4, 5], [5,  9, 10], [10, 14, 15]]),
        (16, 3, pytorch_x_value_gen, 3, 2, [[0, 4, 5], [6, 10, 11], [11, 15, 16]]),
        (18, 3, pytorch_x_value_gen, 3, 2, [[0, 4, 6], [6, 10, 12], [12, 16, 18]]),
        (19, 3, pytorch_x_value_gen, 3, 2, [[0, 4, 6], [7, 11, 13], [13, 17, 19]]),
        (20, 3, pytorch_x_value_gen, 3, 2, [[0, 4, 6], [7, 11, 13], [14, 18, 20]]),
        (21, 3, pytorch_x_value_gen, 3, 2, [[0, 4, 7], [7, 11, 14], [14, 18, 21]]),
        (25, 3, pytorch_x_value_gen, 3, 2, [[0, 4, 8], [9, 13, 17], [17, 21, 25]]),
        (27, 3, pytorch_x_value_gen, 3, 2, [[0, 6, 9], [9, 15, 18], [18, 24, 27]]),
    ],
)
# fmt: on
def test_distributed_and_multiprocessing__returns_data_partition_for_rank(
    soma_experiment: Experiment,
    world_size: int,
    num_workers: int,
    splits: list[list[int]],
):
    """Tests pytorch._partition_obs_joinids() behavior in a simulated PyTorch distributed processing mode and
    DataLoader multiprocessing mode, using mocks to avoid having to do distributed pytorch
    setup or real DataLoader multiprocessing."""

    for rank in range(world_size):
        proc_splits = splits[rank]
        for worker_id in range(num_workers):
            expected_joinids = list(
                range(proc_splits[worker_id], proc_splits[worker_id + 1])
            )
            with (
                init_world(world_size, rank),
                patch("torch.utils.data.get_worker_info") as mock_get_worker_info,
            ):
                mock_get_worker_info.return_value = WorkerInfo(
                    id=worker_id, num_workers=num_workers, seed=1234
                )

                with soma_experiment.axis_query(measurement_name="RNA") as query:
                    dp = ExperimentAxisQueryIterable(
                        query,
                        X_name="raw",
                        obs_column_names=["soma_joinid"],
                        io_batch_size=2,
                        shuffle=False,
                    )

                    batches = list(dp)

                    soma_joinids = np.concatenate(
                        [batch[1]["soma_joinid"].to_numpy() for batch in batches]
                    ).tolist()

                    assert soma_joinids == expected_joinids


@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen", [(16, 1, pytorch_seq_x_value_gen)]
)
@pipeclasses
def test__shuffle(PipeClass: PipeClassType, soma_experiment: Experiment) -> None:
    with soma_experiment.axis_query(measurement_name="RNA") as query:
        dp = PipeClass(
            query,
            X_name="raw",
        )

        batches = list(dp)
        if PipeClass is ExperimentAxisQueryIterable:
            assert all(np.squeeze(X, axis=0).shape == (1,) for X, _ in batches)
        else:
            assert all(X.shape == (1,) for X, _ in batches)
        soma_joinids = [obs["soma_joinid"].iloc[0] for _, obs in batches]
        X_values = [X[0].item() for X, _ in batches]

        # same elements
        assert set(soma_joinids) == set(range(16))
        # not ordered! (...with a `1/16!` probability of being ordered)
        assert soma_joinids != list(range(16))
        # randomizes X in same order as obs
        # note: X values were explicitly set to match obs_joinids to allow for this simple assertion
        assert X_values == soma_joinids


@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen", [(6, 3, pytorch_x_value_gen)]
)
def test_experiment_axis_query_iterable_error_checks(
    soma_experiment: Experiment,
) -> None:
    with soma_experiment.axis_query(measurement_name="RNA") as query:
        dp = ExperimentAxisQueryIterable(
            query,
            X_name="raw",
        )
        with pytest.raises(NotImplementedError):
            dp[0]

        with pytest.raises(ValueError):
            ExperimentAxisQueryIterable(
                query,
                obs_column_names=(),
                X_name="raw",
            )
