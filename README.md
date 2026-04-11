# imads-hpo

PyTorch hyperparameter optimization via [IMADS](https://github.com/Honey-Be/imads) (Integrated Mesh Adaptive Direct Search).

## Features

- **Multi-fidelity**: cheap evaluations first (few epochs), expensive truth last (full training)
- **Constraints**: GPU memory, latency, model size — as first-class citizens
- **Conditional constraints**: `ConditionalConstraint`, `DynamicBoundConstraint`, `CompositeConstraint`
- **Deterministic**: identical seeds + dataset = identical results
- **Functional core**: `curry` + `Run` monad for explicit state management
- **Run monad pipeline**: `build_trial_program()` composes determinism, checkpoint, and execution stages
- **Categorical variables**: 64-bit integer encoding
- **Checkpoint prefix reuse**: resume from earlier fidelity levels
- **Flexible fidelity schedules**: `EpochFidelity`, `DataFractionFidelity`, `ExplicitFidelity` via the `FidelitySchedule` protocol
- **Early stopping**: `PatientStopper`, `MedianStopper`, `ThresholdStopper` rules
- **Dashboard integration**: `WandbSink`, `MLflowSink`, `TensorBoardSink` via the `DashboardSink` protocol
- **Multi-objective HPO**: optimize multiple objectives simultaneously with Pareto front extraction

## Quick Start

```python
import imads_hpo as hpo

space = hpo.Space({
    "lr":         hpo.LogReal(1e-5, 1e-1),
    "batch_size": hpo.Integer(16, 256, step=16),
    "optimizer":  hpo.Categorical(["adam", "sgd", "adamw"]),
    "dropout":    hpo.Real(0.0, 0.5),
})

@hpo.objective(space, num_constraints=1)
def train(params, fidelity):
    model = build_model(params)
    opt = build_optimizer(model, params)
    for epoch in range(fidelity.epochs):
        train_one_epoch(model, opt, data_fraction=fidelity.data_fraction)
    val_loss = evaluate(model)
    gpu_gb = torch.cuda.max_memory_allocated() / 1e9
    return val_loss, [gpu_gb - 8.0]

result = hpo.minimize(
    train,
    preset="balanced",
    workers=4,
    fidelity=hpo.EpochFidelity(min_epochs=5, max_epochs=100, num_seeds=3),
    seed=42,
)
print(result.best_params, result.best_value)
```

### Multi-Objective Example

```python
@hpo.objective(space, num_objectives=2, num_constraints=1)
def train_multi(params, fidelity):
    model = build_model(params)
    opt = build_optimizer(model, params)
    for epoch in range(fidelity.epochs):
        train_one_epoch(model, opt)
    val_loss = evaluate(model)
    latency = measure_inference_time(model)
    gpu_gb = torch.cuda.max_memory_allocated() / 1e9
    return [val_loss, latency], [gpu_gb - 8.0]

result = hpo.minimize(
    train_multi,
    preset="balanced",
    workers=4,
    fidelity=hpo.EpochFidelity(min_epochs=5, max_epochs=100, num_seeds=3),
    seed=42,
)
for trial in result.pareto_front:
    print(trial.params, trial.value)
```

## Installation

```bash
uv add imads-hpo

# With optional dashboard integrations:
uv add imads-hpo[wandb]
uv add imads-hpo[mlflow]
uv add imads-hpo[tensorboard]
uv add imads-hpo[all]
```

## Documentation

- [English](docs/en/guide.md)
- [한국어](docs/ko/guide.md)
- [日本語](docs/ja/guide.md)
- [Deutsch](docs/de/guide.md)

## License

Licensed under any of:

- [MIT](LICENSE-MIT.txt)
- [Apache 2.0](LICENSE-APACHE.txt)
- [LGPL 2.1+](LICENSE-LGPL.txt)

at your option.
