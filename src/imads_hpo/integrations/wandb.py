"""Weights & Biases dashboard sink."""

from __future__ import annotations

from collections.abc import Mapping


class WandbSink:
    """Log HPO trials to Weights & Biases."""

    def __init__(self, project: str, **wandb_kwargs: object) -> None:
        import wandb

        self._wandb = wandb
        self._project = project
        self._wandb_kwargs = wandb_kwargs
        self._run = None

    def on_trial_start(self, trial_id: str, params: dict) -> None:
        self._run = self._wandb.init(
            project=self._project,
            name=trial_id,
            config=params,
            reinit=True,
            **self._wandb_kwargs,
        )

    def on_trial_metric(self, trial_id: str, epoch: int, metrics: Mapping[str, float]) -> None:
        if self._run is not None:
            self._wandb.log({"epoch": epoch, **metrics})

    def on_trial_end(self, trial_id: str, result: float, constraints: list[float]) -> None:
        if self._run is not None:
            self._wandb.log({"final_value": result, "constraints": constraints})
            self._run.finish()
            self._run = None

    def on_study_end(self, best_params: dict, best_value: float) -> None:
        summary_run = self._wandb.init(
            project=self._project,
            name="study_summary",
            reinit=True,
            **self._wandb_kwargs,
        )
        self._wandb.log({"best_value": best_value, **{f"best_{k}": v for k, v in best_params.items()}})
        summary_run.finish()

    def flush(self) -> None:
        pass
