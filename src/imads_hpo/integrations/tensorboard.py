"""TensorBoard dashboard sink."""

from __future__ import annotations

from collections.abc import Mapping


class TensorBoardSink:
    """Log HPO trials to TensorBoard."""

    def __init__(self, log_dir: str) -> None:
        from tensorboardX import SummaryWriter

        self._writer = SummaryWriter(log_dir)
        self._trial_step: dict[str, int] = {}

    def on_trial_start(self, trial_id: str, params: dict) -> None:
        self._trial_step[trial_id] = 0
        for k, v in params.items():
            if isinstance(v, (int, float)):
                self._writer.add_scalar(f"{trial_id}/params/{k}", v, 0)

    def on_trial_metric(self, trial_id: str, epoch: int, metrics: Mapping[str, float]) -> None:
        for k, v in metrics.items():
            self._writer.add_scalar(f"{trial_id}/{k}", v, epoch)

    def on_trial_end(self, trial_id: str, result: float, constraints: list[float]) -> None:
        step = self._trial_step.get(trial_id, 0)
        self._writer.add_scalar(f"{trial_id}/final_value", result, step)
        for i, c in enumerate(constraints):
            self._writer.add_scalar(f"{trial_id}/constraint_{i}", c, step)

    def on_study_end(self, best_params: dict, best_value: float) -> None:
        self._writer.add_scalar("study/best_value", best_value, 0)

    def flush(self) -> None:
        self._writer.flush()
