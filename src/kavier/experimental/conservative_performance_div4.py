"""The experimental ``Conservative Performance Div 4`` Coastline policy.

Coastline recommendations expose an effective data-parallel batch size, while
Kavier's training simulator accepts a per-device micro-batch.  The relationship is::

    effective_batch_size = per_device_batch_size * total_gpus

This policy wraps Coastline's performance strategy.  It retains that strategy's
GPU ranking, quarters the recommended *per-device* batch, reconstructs the matching
effective batch, rechecks feasibility, and recomputes throughput and power at the
reduced load.

Coastline is currently a development dependency, so this module is explicitly
experimental and is not re-exported from :mod:`kavier`.
"""

from __future__ import annotations

import copy
import logging
import math
from typing import Any, Optional

from coastline.sdk.models.context import SystemContext
from coastline.sdk.models.recommendation import Prediction, Recommendation
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.pipeline.feasibility import FeasibilityChecker, create_feasibility_checker
from coastline.sdk.pipeline.workflow import GridWorkflowPipeline
from coastline.sdk.policies import PolicyFactory
from coastline.sdk.policies.base import BaseStrategy
from coastline.sdk.policies.multi_objective import MultiObjectiveStrategy
from coastline.sdk.predictors.base import BasePredictor

logger = logging.getLogger(__name__)

POLICY_KEY = "conservative_performance_div4"
POLICY_NAME = "Conservative Performance Div 4"
BATCH_DIVISOR = 4


def divide_effective_batch(
    effective_batch_size: int,
    total_gpus: int,
    *,
    divisor: int = BATCH_DIVISOR,
) -> tuple[int, int, int]:
    """Return ``(conservative effective, conservative/device, performance/device)``.

    The effective batch must split evenly across the selected devices. Integer
    micro-batches cannot go below one, so a performance per-device batch smaller
    than ``divisor`` is clamped to one.
    """
    if effective_batch_size < 1:
        raise ValueError(f"effective_batch_size must be >= 1, got {effective_batch_size}")
    if total_gpus < 1:
        raise ValueError(f"total_gpus must be >= 1, got {total_gpus}")
    if divisor < 1:
        raise ValueError(f"divisor must be >= 1, got {divisor}")

    performance_per_device, remainder = divmod(effective_batch_size, total_gpus)
    if remainder or performance_per_device < 1:
        raise ValueError(f"effective batch {effective_batch_size} cannot be split evenly across {total_gpus} GPUs")

    conservative_per_device = max(1, performance_per_device // divisor)
    conservative_effective = conservative_per_device * total_gpus
    return conservative_effective, conservative_per_device, performance_per_device


def _predictor_expects_per_device_batch(predictor: BasePredictor) -> bool:
    """Whether ``predictor`` ultimately calls Kavier's per-device training API."""
    predictor_type = type(predictor)
    module = predictor_type.__module__
    return predictor_type.__name__ in {"KavierPredictor", "KavierPowerPredictor"} or module.startswith(
        (
            "coastline.sdk.predictors.performance.physics",
            "coastline.sdk.predictors.energy.kavier",
        )
    )


def _prediction_workload(
    workload: WorkloadSpec,
    predictor: BasePredictor,
    per_device_batch_size: int,
) -> WorkloadSpec:
    """Present the correct batch meaning to physics versus data-driven predictors."""
    if not _predictor_expects_per_device_batch(predictor):
        return workload
    return workload.model_copy(update={"batch_size": per_device_batch_size})


class ConservativePerformanceDiv4Strategy(BaseStrategy):
    """Performance GPU ranking with a quarter-sized per-device final batch."""

    def __init__(
        self,
        *,
        performance_strategy: BaseStrategy,
        throughput_predictor: BasePredictor,
        power_predictor: BasePredictor,
        feasibility_checker: FeasibilityChecker,
        top_k: int = 5,
        divisor: int = BATCH_DIVISOR,
    ) -> None:
        self._performance_strategy = performance_strategy
        self._throughput_predictor = throughput_predictor
        self._power_predictor = power_predictor
        self._feasibility_checker = feasibility_checker
        self._top_k = max(1, int(top_k))
        self._divisor = divisor

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ConservativePerformanceDiv4Strategy":
        """Build the policy from the same config accepted by Coastline policies."""
        config = copy.deepcopy(config)
        predictor_config = config.get("predictors", {})
        grid_config = config.setdefault("grid", {})
        requested_top_k = max(1, int(grid_config.get("top_k", 5)))

        throughput = PolicyFactory.throughput_predictor(predictor_config)
        power = PolicyFactory.power_predictor(predictor_config)
        feasibility = create_feasibility_checker(predictor_config)

        # Ask the base policy for its complete ranking. If a top result cannot be
        # divided evenly or fails the conservative re-check, the next performance
        # result can still be returned.
        batch_count = len(grid_config.get("batch_sizes", ()))
        gpu_count = len(grid_config.get("total_gpus", ()))
        grid_config["top_k"] = max(requested_top_k, batch_count * gpu_count, 1)
        performance_pipeline = GridWorkflowPipeline.from_config(
            config=config,
            selection_policy="performance",
            strategy_name="multi_objective_performance",
            throughput_predictor=throughput,
            power_predictor=power,
            feasibility_checker=feasibility,
            preset="performance",
        )
        performance_strategy = MultiObjectiveStrategy(
            throughput_predictor=throughput,
            power_predictor=power,
            preset="performance",
            config=config,
            pipeline=performance_pipeline,
        )
        return cls(
            performance_strategy=performance_strategy,
            throughput_predictor=throughput,
            power_predictor=power,
            feasibility_checker=feasibility,
            top_k=requested_top_k,
        )

    def get_name(self) -> str:
        return POLICY_NAME

    def recommend(self, workload: WorkloadSpec, context: SystemContext) -> list[Recommendation]:
        performance_recommendations = self._performance_strategy.recommend(workload, context)
        conservative: list[Recommendation] = []

        for performance_rec in performance_recommendations:
            try:
                recommendation = self._make_conservative(performance_rec, workload, context)
            except ValueError as exc:
                logger.debug("Skipping an indivisible performance recommendation: %s", exc)
                continue
            if recommendation is None:
                continue
            conservative.append(recommendation)
            if len(conservative) >= self._top_k:
                break

        if not conservative:
            raise RuntimeError(
                f"{POLICY_NAME}: no performance-ranked configuration remained feasible after batch division"
            )
        return conservative

    def _make_conservative(
        self,
        performance_rec: Recommendation,
        workload: WorkloadSpec,
        context: SystemContext,
    ) -> Optional[Recommendation]:
        performance_meta = performance_rec.metadata or {}
        performance_effective = int(performance_meta.get("batch_size", 0))
        effective_batch, per_device_batch, performance_per_device = divide_effective_batch(
            performance_effective,
            performance_rec.total_gpus,
            divisor=self._divisor,
        )

        final_workload = WorkloadSpec(
            llm_model=workload.llm_model,
            fine_tuning_method=workload.fine_tuning_method,
            gpu_model=workload.gpu_model,
            tokens_per_sample=workload.tokens_per_sample,
            batch_size=effective_batch,
            gpus_per_node=performance_rec.gpus_per_node,
            number_of_nodes=performance_rec.number_of_nodes,
            torch_dtype=workload.torch_dtype,
            enable_roce=workload.enable_roce,
            feasibility_model=workload.feasibility_model,
        )

        feasible, final_feasibility = self._feasibility_checker.is_feasible(final_workload)
        if not feasible:
            logger.debug(
                "Reduced config rejected: effective batch=%d, total_gpus=%d",
                effective_batch,
                performance_rec.total_gpus,
            )
            return None

        throughput_prediction = self._throughput_predictor.predict(
            _prediction_workload(final_workload, self._throughput_predictor, per_device_batch),
            context,
        )
        if not self._valid_throughput(throughput_prediction):
            return None

        power = self._power_for(final_workload, per_device_batch, throughput_prediction, context)
        if power is None:
            return None

        throughput = float(throughput_prediction.predicted_throughput)
        metadata = {
            "policy": POLICY_KEY,
            "policy_name": POLICY_NAME,
            "base_policy": "performance",
            "batch_divisor": self._divisor,
            # Coastline compatibility: batch_size remains effective/global.
            "batch_size": effective_batch,
            "effective_batch_size": effective_batch,
            "per_device_batch_size": per_device_batch,
            "performance_effective_batch_size": performance_effective,
            "performance_per_device_batch_size": performance_per_device,
            "batch_division_clamped": performance_per_device < self._divisor,
            "predicted_power_watts": power,
            "tokens_per_watt": throughput / power,
            "feasibility": final_feasibility,
            "performance_feasibility": performance_meta.get("feasibility", {}),
            "performance_combined_score": performance_meta.get("combined_score"),
            "workflow": "performance_rank_then_per_device_batch_div4_then_recheck",
        }
        return Recommendation(
            gpus_per_node=performance_rec.gpus_per_node,
            number_of_nodes=performance_rec.number_of_nodes,
            total_gpus=performance_rec.total_gpus,
            strategy=POLICY_KEY,
            predicted_throughput=throughput,
            predicted_runtime_seconds=throughput_prediction.predicted_runtime_seconds,
            metadata=metadata,
        )

    @staticmethod
    def _valid_throughput(prediction: Optional[Prediction]) -> bool:
        if prediction is None or prediction.predicted_throughput is None:
            return False
        throughput = float(prediction.predicted_throughput)
        return math.isfinite(throughput) and throughput > 0

    def _power_for(
        self,
        workload: WorkloadSpec,
        per_device_batch_size: int,
        throughput_prediction: Prediction,
        context: SystemContext,
    ) -> Optional[float]:
        # Kavier throughput already carries power, so preserve Coastline's one-call
        # optimization when the power adapter wraps the same engine.
        if getattr(self._power_predictor, "WRAPS_THROUGHPUT_ENGINE", False):
            candidate = throughput_prediction.predicted_power
            if candidate is not None and math.isfinite(float(candidate)) and candidate > 0:
                return float(candidate)

        power_prediction = self._power_predictor.predict(
            _prediction_workload(workload, self._power_predictor, per_device_batch_size),
            context,
        )
        if power_prediction is None or power_prediction.predicted_power is None:
            return None
        power = float(power_prediction.predicted_power)
        return power if math.isfinite(power) and power > 0 else None
