# imads-hpo: PyTorch Hyperparameter Optimization Guide

## Overview

`imads-hpo` is a hyperparameter optimization package for PyTorch, powered by the IMADS (Integrated Mesh Adaptive Direct Search) engine. It provides derivative-free, constrained, multi-fidelity optimization with full deterministic reproducibility.

### Key Features

- **Multi-fidelity evaluation**: cheap evaluations first (few epochs), expensive truth last (full training)
- **Constraint support**: GPU memory, latency, model size — as first-class citizens
- **Conditional constraints**: predicates, dynamic bounds, and composite constraints
- **Deterministic reproducibility**: identical seeds + dataset = identical results, regardless of hardware
- **Functional core**: currying and Run monad for explicit state management
- **Run monad pipeline**: composable trial execution via `build_trial_program()`
- **Categorical variables**: 64-bit integer encoding (unlike NOMAD 4 / PyNomadBBO which lack categorical support)
- **Checkpoint prefix reuse**: resume training from earlier fidelity levels instead of restarting
- **Flexible fidelity schedules**: epoch-based, data-fraction-based, or explicit schedules via the `FidelitySchedule` protocol
- **Early stopping**: patient, median, and threshold stopping rules
- **Dashboard integration**: Weights & Biases, MLflow, and TensorBoard via the `DashboardSink` protocol
- **Multi-objective HPO**: optimize multiple objectives with Pareto front extraction

## Installation

```bash
uv add imads-hpo
# or
pip install imads-hpo

# With optional dashboard integrations:
uv add imads-hpo[wandb]
uv add imads-hpo[mlflow]
uv add imads-hpo[tensorboard]
uv add imads-hpo[all]
```

Dependencies: `imads` (built from git), `torch>=2.0`, `safetensors`. Optional: `wandb`, `mlflow`, `tensorboardX`.

## Quick Start

```python
import imads_hpo as hpo

# 1. Define search space
space = hpo.Space({
    "lr":           hpo.LogReal(1e-5, 1e-1),
    "weight_decay": hpo.LogReal(1e-6, 1e-2),
    "batch_size":   hpo.Integer(16, 256, step=16),
    "optimizer":    hpo.Categorical(["adam", "sgd", "adamw"]),
    "dropout":      hpo.Real(0.0, 0.5),
})

# 2. Define objective
@hpo.objective(space, num_constraints=1)
def train(params, fidelity):
    model = build_model(params)
    opt = build_optimizer(model, params)
    for epoch in range(fidelity.epochs):
        train_one_epoch(model, opt, data_fraction=fidelity.data_fraction)
    val_loss = evaluate(model)
    gpu_gb = torch.cuda.max_memory_allocated() / 1e9
    return val_loss, [gpu_gb - 8.0]  # constraint: GPU <= 8 GB

# 3. Run optimization
result = hpo.minimize(
    train,
    preset="balanced",
    max_evals=200,
    workers=4,
    fidelity=hpo.EpochFidelity(min_epochs=5, max_epochs=100, num_seeds=3),
    seed=42,
)
print(result.best_params)
print(result.best_value)
```

## Search Space

| Type | Description | Mesh encoding |
|------|-------------|:------------:|
| `Real(low, high)` | Continuous in [low, high] | float → int64 via `base_step` |
| `LogReal(low, high)` | Log-scale continuous (both > 0) | log-transform → int64 |
| `Integer(low, high, step)` | Integer with optional step | direct int64 |
| `Categorical(choices)` | Fixed set of choices | index as int64 (0, 1, 2, ...) |

All dimensions are encoded as 64-bit integers for IMADS mesh compatibility.

> **Note:** The search space dimensionality is automatically propagated to the IMADS engine. The `HpoEvaluator` exposes a `search_dim()` method (via the `SpaceEncoder.search_dim` property), so the engine discovers the number of dimensions from the evaluator. There is no need to manually set `EngineConfig.search_dim`.

```python
space = hpo.Space({
    "lr":        hpo.LogReal(1e-5, 1e-1),    # log-scale
    "layers":    hpo.Integer(2, 8),            # integer
    "act":       hpo.Categorical(["relu", "gelu", "silu"]),  # categorical
    "dropout":   hpo.Real(0.0, 0.5),          # linear
})
```

## Multi-Fidelity: Progressive Evaluation

IMADS uses a 2-axis fidelity ladder `(τ, S)`:

| IMADS | HPO meaning |
|-------|-------------|
| `τ` (tau) | Inversely proportional to epoch count. Large τ = few epochs (cheap). |
| `S` (smc) | Number of independent training seeds for noise averaging. |
| TRUTH | Final level: full `max_epochs` with all seeds. |

### EpochFidelity

The default epoch-based fidelity schedule:

```python
fidelity = hpo.EpochFidelity(
    min_epochs=5,     # cheapest: 5 epochs (τ = 20)
    max_epochs=100,   # TRUTH: 100 epochs (τ = 1)
    num_seeds=3,      # S levels: [1, 3]
)
```

### FidelitySchedule Protocol

Any fidelity schedule implements the `FidelitySchedule` protocol (runtime-checkable):

```python
class FidelitySchedule(Protocol):
    tau_levels: list[int]
    smc_levels: list[int]
    def resolve_fidelity(self, tau: int, smc: int) -> Fidelity: ...
```

The `Fidelity` dataclass includes a `data_fraction: float = 1.0` field, enabling data-subsampling-based fidelity.

### DataFractionFidelity

Fidelity based on data subsampling rather than epochs:

```python
fidelity = hpo.DataFractionFidelity(
    fractions=[0.1, 0.25, 0.5, 1.0],  # progressive data fractions
    num_seeds=3,
)
```

### ExplicitFidelity

User-specified epoch steps for full control:

```python
fidelity = hpo.ExplicitFidelity(
    epoch_steps=[10, 25, 50, 100],  # explicit epoch counts per level
    num_seeds=3,
)
```

The `minimize()` function accepts any `FidelitySchedule`, not just `EpochFidelity`:

```python
result = hpo.minimize(
    train,
    preset="balanced",
    fidelity=hpo.DataFractionFidelity(fractions=[0.1, 0.5, 1.0], num_seeds=2),
    seed=42,
)
```

### Checkpoint Prefix Reuse

When evaluating at a tighter fidelity (more epochs), `imads-hpo` automatically loads the checkpoint from the looser level and resumes training:

```
Trial x, seed k:
  τ=20 → train 5 epochs   → save checkpoint
  τ=10 → load checkpoint   → train 5 more (total 10) → save
  τ=1  → load checkpoint   → train 90 more (total 100) → save (TRUTH)
```

This avoids redundant computation: each fidelity level builds on the previous one.

## Deterministic Reproducibility

`imads-hpo` follows the patterns from [framesmoothie](https://github.com/Honey-Be/framesmoothie):

### SeedPath — Hierarchical Seed Derivation

```python
from imads_hpo import SeedPath

root = SeedPath(master_seed=42)
trial_seed = root.child("trial", 7).child("sample", 0).seed()
gen = root.child("dataloader").torch_generator()
```

Every RNG stream is derived deterministically from the master seed via BLAKE2b hashing.

### DeterminismConfig — PyTorch Determinism

```python
from imads_hpo import DeterminismConfig, configure_determinism

config = DeterminismConfig(
    master_seed=42,
    deterministic_algorithms=True,
    cudnn_benchmark=False,
    allow_tf32=False,
)
configure_determinism(config)
```

Sets `torch.use_deterministic_algorithms(True)`, disables cuDNN benchmark, TF32, and configures CUBLAS workspace.

### RngRegistry — Explicit RNG Management

```python
from imads_hpo import RngRegistry

registry = RngRegistry()
train_gen = registry.register_torch("train", SeedPath(42).child("train").torch_generator())
snapshot = registry.snapshot()   # serialize all RNG states
# ... later ...
registry.restore(snapshot)       # restore exact RNG states
```

## Functional Programming Primitives

### curry — Automatic Partial Application

```python
from imads_hpo import curry

@curry
def make_optimizer(lr, weight_decay, params):
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

adam_factory = make_optimizer(0.001)(1e-5)  # partially applied
optimizer = adam_factory(model.parameters())  # fully applied
```

### Run Monad — Deterministic State Transitions

```python
from imads_hpo import Run, ask, get_state, put_state, tell, sequence

# A Run[Env, State, A] wraps: (env, state) -> (new_state, value, log)

program = (
    ask()                                    # read environment
    .bind(lambda env: Run.pure(env["lr"]))   # extract lr
    .bind(lambda lr: tell({"lr": lr}))       # log it
    .then(get_state())                       # read state
)

state, value, log = program.execute({"lr": 0.01}, initial_state)
```

### Run Monad Pipeline (Trial Execution)

`WrappedObjective.evaluate()` uses a Run monad pipeline instead of a direct function call. The pipeline is composed by `build_trial_program()`:

```
_configure_determinism() → _load_or_init_checkpoint() → _execute_objective() → _save_checkpoint()
```

```python
from imads_hpo import build_trial_program

program = build_trial_program(fn, checkpoint_enabled=True)
```

#### RunLogSink

The `RunLogSink` protocol consumes `RunLog` events emitted during pipeline execution. `WrappedObjective` has a `log_sink` field that accepts any `RunLogSink`:

```python
from imads_hpo import NullSink, ListSink

# NullSink (default) — discards all events
obj = hpo.objective(space)(train)
obj.log_sink = NullSink()

# ListSink — accumulates events (useful for testing)
sink = ListSink()
obj.log_sink = sink
# ... run optimization ...
print(sink.events)
```

## Constraints

```python
from imads_hpo import GpuMemoryConstraint, LatencyConstraint

# In the objective function:
@hpo.objective(space, num_constraints=2)
def train(params, fidelity):
    ...
    gpu_gb = torch.cuda.max_memory_allocated() / 1e9
    latency_ms = measure_inference_time(model)
    return val_loss, [gpu_gb - 8.0, latency_ms - 50.0]
    #                 c0 <= 0: GPU <= 8GB
    #                 c1 <= 0: latency <= 50ms
```

Constraints are evaluated alongside the objective. IMADS uses its conservative early-infeasible screening to skip obviously infeasible candidates cheaply.

### Conditional Constraints

#### ConditionalConstraint

A constraint that is active only when a predicate on the hyperparameters returns `True`. When inactive, the constraint returns `inactive_value` (default `-1.0`, i.e. feasible):

```python
from imads_hpo import ConditionalConstraint

# Only enforce GPU constraint when batch_size > 128
gpu_constraint = ConditionalConstraint(
    predicate=lambda params: params["batch_size"] > 128,
    inner=lambda result: result.gpu_gb - 8.0,
    inactive_value=-1.0,
)
```

#### DynamicBoundConstraint

A constraint whose bound depends on the hyperparameters themselves:

```python
from imads_hpo import DynamicBoundConstraint

# Latency bound varies with model size
latency_constraint = DynamicBoundConstraint(
    bound_fn=lambda params: 50.0 if params["layers"] <= 4 else 100.0,
)
```

#### CompositeConstraint

Combines multiple constraints by taking the maximum (most violated):

```python
from imads_hpo import CompositeConstraint

combined = CompositeConstraint(gpu_constraint, latency_constraint)
```

## Early Stopping

The `EarlyStoppingRule` protocol defines rules that can terminate a trial early based on its training trajectory:

```python
class EarlyStoppingRule(Protocol):
    def should_stop(self, epoch: int, metric: float, history: list[float]) -> bool: ...
    def reset(self) -> None: ...
```

### PatientStopper

Stop if no improvement for `patience` consecutive epochs:

```python
from imads_hpo import PatientStopper

stopper = PatientStopper(patience=10, min_delta=1e-4)
```

### MedianStopper

Stop if the current trial's metric is below the median of completed trials:

```python
from imads_hpo import MedianStopper

stopper = MedianStopper(completed_curves=completed, min_epochs=5)
```

### ThresholdStopper

Stop if the metric exceeds a fixed threshold:

```python
from imads_hpo import ThresholdStopper

stopper = ThresholdStopper(threshold=2.0, min_epochs=3)
```

## Dashboard Integration

The `DashboardSink` protocol routes trial metrics to external dashboards:

```python
class DashboardSink(Protocol):
    def on_trial_start(self, trial_id: int, params: dict) -> None: ...
    def on_trial_metric(self, trial_id: int, epoch: int, metric: float) -> None: ...
    def on_trial_end(self, trial_id: int, value: float) -> None: ...
    def on_study_end(self) -> None: ...
    def flush(self) -> None: ...
```

`RunLogAdapter` bridges `RunLog` events to a `DashboardSink`.

### Backend Implementations

Available in the `integrations/` subpackage:

```python
from imads_hpo.integrations import WandbSink, MLflowSink, TensorBoardSink

# Weights & Biases
sink = WandbSink(project="my-hpo-study")

# MLflow
sink = MLflowSink(experiment_name="hpo-run")

# TensorBoard
sink = TensorBoardSink(log_dir="runs/hpo")
```

Each backend requires its optional dependency. Install with extras:

```bash
uv add imads-hpo[wandb]
uv add imads-hpo[mlflow]
uv add imads-hpo[tensorboard]
uv add imads-hpo[all]         # all integrations
```

## Multi-Objective HPO

Optimize multiple objectives simultaneously by specifying `num_objectives`:

```python
@hpo.objective(space, num_objectives=2, num_constraints=1)
def train(params, fidelity):
    model = build_model(params)
    opt = build_optimizer(model, params)
    for epoch in range(fidelity.epochs):
        train_one_epoch(model, opt)
    val_loss = evaluate(model)
    latency = measure_inference_time(model)
    gpu_gb = torch.cuda.max_memory_allocated() / 1e9
    return [val_loss, latency], [gpu_gb - 8.0]
```

The `WrappedObjective` has a `num_objectives` field reflecting the number of objectives. When `num_objectives > 1`, the `TrialRecord.value` type is `list[float]` instead of `float`.

The result includes a Pareto front:

```python
result = hpo.minimize(
    train,
    preset="balanced",
    workers=4,
    fidelity=hpo.EpochFidelity(min_epochs=5, max_epochs=100, num_seeds=3),
    seed=42,
)

# Pareto-optimal trials
for trial in result.pareto_front:
    print(trial.params, trial.value)  # value is [loss, latency]
```

## Presets

| Preset | Use case |
|--------|----------|
| `"balanced"` | Recommended default. Moderate throughput, good quality. |
| `"conservative"` | Minimize false-infeasible risk. Slower but safer. |
| `"throughput"` | Fast sweeps. More candidates, earlier pruning. |

## API Reference

### Top-level

| Function / Class | Description |
|-----------------|-------------|
| `Space({...})` | Define search space |
| `@objective(space, num_constraints, num_objectives)` | Wrap training function |
| `minimize(obj, preset, ...)` | Run optimization |
| `EpochFidelity(min, max, seeds)` | Epoch-based multi-fidelity |
| `DataFractionFidelity(fractions, seeds)` | Data subsampling fidelity |
| `ExplicitFidelity(epoch_steps, seeds)` | User-specified epoch steps |
| `FidelitySchedule` | Protocol for custom fidelity schedules |
| `Result` | Optimization result with `.best_params`, `.best_value`, `.pareto_front` |

### Constraints

| Function / Class | Description |
|-----------------|-------------|
| `ConditionalConstraint(predicate, inner, inactive_value)` | Active only when predicate is True |
| `DynamicBoundConstraint(bound_fn)` | Bound depends on hyperparameters |
| `CompositeConstraint(*constraints)` | Max of multiple constraints |

### Early Stopping

| Function / Class | Description |
|-----------------|-------------|
| `EarlyStoppingRule` | Protocol for stopping rules |
| `PatientStopper(patience, min_delta)` | Stop on no improvement |
| `MedianStopper(completed_curves, min_epochs)` | Stop if below median |
| `ThresholdStopper(threshold, min_epochs)` | Stop if above threshold |

### Dashboard Integration

| Function / Class | Description |
|-----------------|-------------|
| `DashboardSink` | Protocol for dashboard backends |
| `RunLogAdapter` | Routes RunLog events to DashboardSink |
| `WandbSink` | Weights & Biases integration |
| `MLflowSink` | MLflow integration |
| `TensorBoardSink` | TensorBoard integration |

### Reproducibility

| Function / Class | Description |
|-----------------|-------------|
| `DeterminismConfig(seed, ...)` | PyTorch determinism settings |
| `configure_determinism(config)` | Apply settings globally |
| `SeedPath(seed).child(...)` | Hierarchical seed derivation |
| `RngRegistry` | Named RNG stream management |
| `derive_seed(master, ns, ...)` | Curried seed derivation |

### Functional

| Function / Class | Description |
|-----------------|-------------|
| `curry(fn)` | Auto partial application |
| `Run[E, S, A]` | Reader/Writer/State monad |
| `ask()`, `get_state()`, `put_state(s)` | Monad primitives |
| `tell(event)` | Emit log event |
| `sequence(programs)` | Run programs in order |
| `build_trial_program(fn, checkpoint_enabled)` | Compose trial execution pipeline |
| `RunLogSink` | Protocol for consuming RunLog events |
| `NullSink` | Discards all RunLog events (default) |
| `ListSink` | Accumulates RunLog events (testing) |
