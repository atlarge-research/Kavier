from __future__ import annotations

from typing import Optional

import pytest
from coastline.sdk.models.context import SystemContext
from coastline.sdk.models.recommendation import Prediction, Recommendation
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.policies.base import BaseStrategy
from coastline.sdk.predictors.base import BasePredictor

from kavier.experimental.conservative_performance_div4 import (
    BATCH_DIVISOR,
    POLICY_KEY,
    POLICY_NAME,
    ConservativePerformanceDiv4Strategy,
    divide_effective_batch,
)


class _PerformanceStrategy(BaseStrategy):
    def __init__(self, recommendations: list[Recommendation]) -> None:
        self.recommendations = recommendations

    def recommend(self, workload: WorkloadSpec, context: SystemContext) -> list[Recommendation]:
        return self.recommendations

    def get_name(self) -> str:
        return "performance"


class _FeasibilityChecker:
    def __init__(self, rejected_batches: set[int] | None = None) -> None:
        self.checked_batches: list[int] = []
        self.rejected_batches = rejected_batches or set()

    def is_feasible(self, workload: WorkloadSpec) -> tuple[bool, dict]:
        self.checked_batches.append(workload.batch_size)
        feasible = workload.batch_size not in self.rejected_batches
        return feasible, {"checked_effective_batch": workload.batch_size}


class _Predictor(BasePredictor):
    def __init__(self, *, power: float = 300.0) -> None:
        self.seen_batches: list[int] = []
        self.power = power

    def predict(self, workload: WorkloadSpec, context: SystemContext) -> Optional[Prediction]:
        self.seen_batches.append(workload.batch_size)
        return Prediction(
            gpus_per_node=workload.gpus_per_node or 1,
            number_of_nodes=workload.number_of_nodes or 1,
            total_gpus=workload.total_gpus,
            predicted_throughput=float(workload.batch_size * workload.total_gpus),
            predicted_runtime_seconds=10.0,
            predicted_power=self.power,
        )

    def get_name(self) -> str:
        return "fake"


class KavierPredictor(_Predictor):
    """Name intentionally matches Coastline's physics predictor for adapter coverage."""


class KavierPowerPredictor(_Predictor):
    """Name intentionally matches Coastline's power adapter for adapter coverage."""


def _workload() -> WorkloadSpec:
    return WorkloadSpec(
        llm_model="mistral-7b-v0.1",
        fine_tuning_method="lora",
        gpu_model="NVIDIA-A100-SXM4-80GB",
        tokens_per_sample=1024,
        batch_size=64,
        gpus_per_node=4,
        number_of_nodes=1,
    )


def _context() -> SystemContext:
    return SystemContext.for_gpus(
        ["NVIDIA-A100-SXM4-80GB"],
        max_gpus=8,
        gpus_per_node=8,
    )


def _recommendation(*, total_gpus: int, effective_batch: int) -> Recommendation:
    return Recommendation(
        gpus_per_node=total_gpus,
        number_of_nodes=1,
        total_gpus=total_gpus,
        strategy="multi_objective_performance",
        predicted_throughput=10_000.0,
        metadata={
            "batch_size": effective_batch,
            "combined_score": 0.9,
            "feasibility": {"original": True},
        },
    )


@pytest.mark.parametrize(
    ("effective", "gpus", "expected"),
    [
        (64, 4, (16, 4, 16)),
        (32, 8, (8, 1, 4)),
        # A micro-batch is integral and cannot be reduced below one.
        (8, 8, (8, 1, 1)),
    ],
)
def test_divide_effective_batch_uses_per_device_semantics(
    effective: int,
    gpus: int,
    expected: tuple[int, int, int],
) -> None:
    assert divide_effective_batch(effective, gpus) == expected


@pytest.mark.parametrize(("effective", "gpus"), [(7, 4), (4, 8)])
def test_divide_effective_batch_rejects_a_non_distributable_global_batch(effective: int, gpus: int) -> None:
    with pytest.raises(ValueError, match="cannot be split evenly"):
        divide_effective_batch(effective, gpus)


def test_policy_keeps_performance_gpu_layout_and_rechecks_quarter_batch() -> None:
    throughput = _Predictor(power=275.0)
    power = _Predictor(power=325.0)
    feasibility = _FeasibilityChecker()
    strategy = ConservativePerformanceDiv4Strategy(
        performance_strategy=_PerformanceStrategy([_recommendation(total_gpus=4, effective_batch=64)]),
        throughput_predictor=throughput,
        power_predictor=power,
        feasibility_checker=feasibility,
        top_k=1,
    )

    recommendation = strategy.recommend(_workload(), _context())[0]
    metadata = recommendation.metadata

    assert strategy.get_name() == POLICY_NAME
    assert recommendation.strategy == POLICY_KEY
    assert recommendation.total_gpus == 4
    assert recommendation.gpus_per_node == 4
    assert metadata["performance_effective_batch_size"] == 64
    assert metadata["performance_per_device_batch_size"] == 16
    assert metadata["per_device_batch_size"] == 4
    assert metadata["effective_batch_size"] == 16
    assert metadata["batch_size"] == 16
    assert metadata["batch_divisor"] == BATCH_DIVISOR
    assert feasibility.checked_batches == [16]
    # Generic/data-driven predictors consume the effective batch.
    assert throughput.seen_batches == [16]
    assert power.seen_batches == [16]
    assert recommendation.predicted_throughput == 64.0
    assert metadata["predicted_power_watts"] == 325.0


def test_kavier_predictors_receive_micro_batch_while_metadata_stays_effective() -> None:
    throughput = KavierPredictor(power=275.0)
    power = KavierPowerPredictor(power=325.0)
    strategy = ConservativePerformanceDiv4Strategy(
        performance_strategy=_PerformanceStrategy([_recommendation(total_gpus=4, effective_batch=64)]),
        throughput_predictor=throughput,
        power_predictor=power,
        feasibility_checker=_FeasibilityChecker(),
        top_k=1,
    )

    recommendation = strategy.recommend(_workload(), _context())[0]

    # The final global batch is 16, but Kavier's engine must see micro-batch 4.
    assert recommendation.metadata["effective_batch_size"] == 16
    assert recommendation.metadata["per_device_batch_size"] == 4
    assert throughput.seen_batches == [4]
    assert power.seen_batches == [4]


def test_policy_falls_through_when_a_performance_result_cannot_be_evenly_split() -> None:
    strategy = ConservativePerformanceDiv4Strategy(
        performance_strategy=_PerformanceStrategy(
            [
                _recommendation(total_gpus=8, effective_batch=4),
                _recommendation(total_gpus=4, effective_batch=32),
            ]
        ),
        throughput_predictor=_Predictor(),
        power_predictor=_Predictor(),
        feasibility_checker=_FeasibilityChecker(),
        top_k=1,
    )

    recommendation = strategy.recommend(_workload(), _context())[0]

    assert recommendation.total_gpus == 4
    assert recommendation.metadata["effective_batch_size"] == 8
    assert recommendation.metadata["per_device_batch_size"] == 2


def test_policy_falls_through_when_reduced_configuration_fails_second_feasibility_check() -> None:
    feasibility = _FeasibilityChecker(rejected_batches={16})
    strategy = ConservativePerformanceDiv4Strategy(
        performance_strategy=_PerformanceStrategy(
            [
                _recommendation(total_gpus=4, effective_batch=64),
                _recommendation(total_gpus=2, effective_batch=32),
            ]
        ),
        throughput_predictor=_Predictor(),
        power_predictor=_Predictor(),
        feasibility_checker=feasibility,
        top_k=1,
    )

    recommendation = strategy.recommend(_workload(), _context())[0]

    assert feasibility.checked_batches == [16, 8]
    assert recommendation.total_gpus == 2
    assert recommendation.metadata["effective_batch_size"] == 8


def test_policy_marks_the_integer_floor_when_original_micro_batch_is_below_four() -> None:
    strategy = ConservativePerformanceDiv4Strategy(
        performance_strategy=_PerformanceStrategy([_recommendation(total_gpus=8, effective_batch=8)]),
        throughput_predictor=_Predictor(),
        power_predictor=_Predictor(),
        feasibility_checker=_FeasibilityChecker(),
        top_k=1,
    )

    recommendation = strategy.recommend(_workload(), _context())[0]

    assert recommendation.metadata["per_device_batch_size"] == 1
    assert recommendation.metadata["effective_batch_size"] == 8
    assert recommendation.metadata["batch_division_clamped"] is True
