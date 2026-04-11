"""MLflow dashboard sink."""

from __future__ import annotations

from collections.abc import Mapping


class MLflowSink:
    """Log HPO trials to MLflow."""

    def __init__(self, experiment_name: str, tracking_uri: str | None = None) -> None:
        import mlflow

        self._mlflow = mlflow
        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    def on_trial_start(self, trial_id: str, params: dict) -> None:
        self._mlflow.start_run(run_name=trial_id, nested=True)
        self._mlflow.log_params(params)

    def on_trial_metric(self, trial_id: str, epoch: int, metrics: Mapping[str, float]) -> None:
        self._mlflow.log_metrics(dict(metrics), step=epoch)

    def on_trial_end(self, trial_id: str, result: float, constraints: list[float]) -> None:
        self._mlflow.log_metric("final_value", result)
        for i, c in enumerate(constraints):
            self._mlflow.log_metric(f"constraint_{i}", c)
        self._mlflow.end_run()

    def on_study_end(self, best_params: dict, best_value: float) -> None:
        self._mlflow.log_metric("best_value", best_value)

    def flush(self) -> None:
        pass
