# imads-hpo: PyTorch Hyperparameter Optimization Guide

## Overview

`imads-hpo` is a hyperparameter optimization package for PyTorch, powered by the IMADS (Integrated Mesh Adaptive Direct Search) engine. It provides derivative-free, constrained, multi-fidelity optimization with full deterministic reproducibility.

### Key Features

- **Multi-fidelity evaluation**: cheap evaluations first (few epochs), expensive truth last (full training)
- **Constraint support**: GPU memory, latency, model size — as first-class citizens
- **Deterministic reproducibility**: identical seeds + dataset = identical results, regardless of hardware
- **Functional core**: currying and Run monad for explicit state management
- **Categorical variables**: 64-bit integer encoding (unlike NOMAD 4 / PyNomadBBO which lack categorical support)
- **Checkpoint prefix reuse**: resume training from earlier fidelity levels instead of restarting

## Installation

```bash
uv add imads-hpo
# or
pip install imads-hpo
```

Dependencies: `imads` (built from git), `torch>=2.0`, `safetensors`.

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
        train_one_epoch(model, opt)
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

## Multi-Fidelity: Epoch-Based Progressive Evaluation

IMADS uses a 2-axis fidelity ladder `(τ, S)`:

| IMADS | HPO meaning |
|-------|-------------|
| `τ` (tau) | Inversely proportional to epoch count. Large τ = few epochs (cheap). |
| `S` (smc) | Number of independent training seeds for noise averaging. |
| TRUTH | Final level: full `max_epochs` with all seeds. |

```python
fidelity = hpo.EpochFidelity(
    min_epochs=5,     # cheapest: 5 epochs (τ = 20)
    max_epochs=100,   # TRUTH: 100 epochs (τ = 1)
    num_seeds=3,      # S levels: [1, 3]
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
| `@objective(space, num_constraints)` | Wrap training function |
| `minimize(obj, preset, ...)` | Run optimization |
| `EpochFidelity(min, max, seeds)` | Configure multi-fidelity |
| `Result` | Optimization result with `.best_params`, `.best_value` |

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
