# Copyright (c) 2021-2024 The Chan Zuckerberg Initiative Foundation
# Copyright (c) 2021-2024 TileDB, Inc.
#
# Licensed under the MIT License.

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from tempfile import TemporaryDirectory
from typing import Sequence, Type
from unittest.mock import patch

from somacore import ExperimentAxisQuery
from tiledbsoma import AxisQuery, Collection, Experiment, Measurement
from torch.utils.data import DataLoader

from tests.utils import (
    XValueGen,
    add_dataframe,
    add_sparse_array,
    pytorch_x_value_gen,
)
from tiledbsoma_ml._utils import Yield
from tiledbsoma_ml.dataloader import experiment_dataloader
from tiledbsoma_ml.pytorch import (
    ExperimentAxisQueryIterable,
    ExperimentAxisQueryIterableDataset,
    ExperimentAxisQueryIterDataPipe,
    XObsDatum,
)

PipeClassType = (
    Type[ExperimentAxisQueryIterable]
    | Type[ExperimentAxisQueryIterDataPipe]
    | Type[ExperimentAxisQueryIterableDataset]
)


@dataclass
class Case:
    obs_range: int | range
    var_range: int | range = 3
    PipeClass: PipeClassType = ExperimentAxisQueryIterDataPipe
    shuffle: bool = True
    X_value_gen: XValueGen = pytorch_x_value_gen
    obsp_layer_names: Sequence[str] | None = None
    varp_layer_names: Sequence[str] | None = None
    measurement_name: str = "RNA"
    X_name: str = "raw"
    obs_column_names: Sequence[str] = ("soma_joinid",)
    obs_query: AxisQuery | None = None
    var_query: AxisQuery | None = None
    batch_size: int = 1
    io_batch_size: int = 2**16
    return_sparse_X: bool = False
    use_eager_fetch: bool = True
    world_size: int | None = None
    rank: int | None = None
    num_workers: int | None = None
    worker_id: int | None = None
    expected_splits: list[list[int]] | None = None
    expected_nbatches: int | None = None

    @property
    @contextmanager
    def experiment(self) -> Yield[Experiment]:
        with TemporaryDirectory() as tmpdir:
            with Experiment.create(tmpdir) as exp:
                obs_range = self.obs_range
                var_range = self.var_range
                if isinstance(obs_range, int):
                    obs_range = range(obs_range)
                if isinstance(var_range, int):
                    var_range = range(var_range)

                add_dataframe(exp, "obs", obs_range)
                ms = exp.add_new_collection("ms")
                rna = ms.add_new_collection("RNA", Measurement)
                add_dataframe(rna, "var", var_range)
                rna_x = rna.add_new_collection("X", Collection)
                add_sparse_array(rna_x, "raw", obs_range, var_range, self.X_value_gen)

                if self.obsp_layer_names:
                    obsp = rna.add_new_collection("obsp")
                    for obsp_layer_name in self.obsp_layer_names:
                        add_sparse_array(
                            obsp,
                            obsp_layer_name,
                            obs_range,
                            var_range,
                            self.X_value_gen,
                        )

                if self.varp_layer_names:
                    varp = rna.add_new_collection("varp")
                    for varp_layer_name in self.varp_layer_names:
                        add_sparse_array(
                            varp,
                            varp_layer_name,
                            obs_range,
                            var_range,
                            self.X_value_gen,
                        )
            # Must let the ``Experiment.create`` close, before re-opening / yielding
            yield Experiment.open(tmpdir)

    @property
    @contextmanager
    def query_ctx(self) -> Yield[ExperimentAxisQuery]:
        with self.experiment as exp:
            yield exp.axis_query(
                measurement_name=self.measurement_name,
                obs_query=self.obs_query,
                var_query=self.var_query,
            )

    @property
    @contextmanager
    def datapipe(self) -> Yield[PipeClass]:
        with self.query_ctx as query:
            dp = self.PipeClass(
                query,
                **{
                    k: getattr(self, k)
                    for k in [
                        "X_name",
                        "obs_column_names",
                        "shuffle",
                        "batch_size",
                        "io_batch_size",
                        "use_eager_fetch",
                        "return_sparse_X",
                    ]
                },
            )
            yield dp

    @contextmanager
    def dataloader(
        self, num_workers: int | None = None, **dataloader_kwargs
    ) -> Yield[DataLoader]:
        if num_workers is None:
            num_workers = self.num_workers
        if num_workers is not None:
            dataloader_kwargs["num_workers"] = num_workers
        with self.datapipe as dp:
            dl = experiment_dataloader(dp, **dataloader_kwargs)
            yield dl

    @property
    def batches(self) -> list[XObsDatum]:
        if self.PipeClass is ExperimentAxisQueryIterable:
            with self.datapipe as dp:
                return list(dp)
        else:
            with self.dataloader() as dl:
                return list(dl)

    @property
    @contextmanager
    def init_world(self) -> Yield[None]:
        assert self.rank is not None
        assert self.world_size is not None
        with (
            patch("torch.distributed.is_initialized") as mock_dist_is_initialized,
            patch("torch.distributed.get_rank") as mock_dist_get_rank,
            patch("torch.distributed.get_world_size") as mock_dist_get_world_size,
        ):
            mock_dist_is_initialized.return_value = True
            mock_dist_get_rank.return_value = self.rank
            mock_dist_get_world_size.return_value = self.world_size
            yield

    # Nicer string reprs for test IDs
    _id_fmts = {
        "obs_range": lambda n: f"obs{n}",
        "var_range": lambda n: f"var{n}",
        "PipeClass": lambda cls: cls.__name__.replace("ExperimentAxisQuery", ""),
        "shuffle": lambda b: "shuffle" if b else None,
        "use_eager_fetch": lambda b: "eager" if b else "lazy",
        "return_sparse_X": lambda b: "sparse" if b else "dense",
        "batch_size": lambda n: f"batch{n}",
        "world_size": lambda n: f"world{n}",
        "rank": lambda n: f"rank{n}",
        "expected_splits": lambda s: None,
        "expected_nbatches": lambda n: None,
        "worker_id": lambda n: f"worker{n}",
    }
