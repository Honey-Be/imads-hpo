# imads-hpo: PyTorch-Leitfaden zur Hyperparameter-Optimierung

## Überblick

`imads-hpo` ist ein Paket zur Hyperparameter-Optimierung für PyTorch, angetrieben durch die IMADS-Engine (Integrated Mesh Adaptive Direct Search). Es bietet ableitungsfreie, beschränkte, Multi-Fidelity-Optimierung mit vollständiger deterministischer Reproduzierbarkeit.

### Hauptmerkmale

- **Multi-Fidelity-Auswertung**: günstige Auswertungen zuerst (wenige Epochen), teure TRUTH-Auswertung zuletzt (vollständiges Training)
- **Unterstützung von Beschränkungen**: GPU-Speicher, Latenz, Modellgröße — als erstklassige Bürger
- **Deterministische Reproduzierbarkeit**: identische Seeds + Datensatz = identische Ergebnisse, unabhängig von der Hardware
- **Funktionaler Kern**: curry und Run monad für explizites Zustandsmanagement
- **Kategorische Variablen**: 64-Bit-Ganzzahlkodierung (im Gegensatz zu NOMAD 4 / PyNomadBBO, die keine kategorische Unterstützung bieten)
- **Checkpoint-Präfix-Wiederverwendung**: Training von früheren Fidelity-Stufen fortsetzen, statt neu zu starten

## Installation

```bash
uv add imads-hpo
# or
pip install imads-hpo
```

Abhängigkeiten: `imads` (aus Git gebaut), `torch>=2.0`, `safetensors`.

## Schnellstart

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

## Suchraum

| Typ | Beschreibung | Mesh-Kodierung |
|------|-------------|:------------:|
| `Real(low, high)` | Kontinuierlich in [low, high] | float → int64 über `base_step` |
| `LogReal(low, high)` | Logarithmisch-skaliert kontinuierlich (beide > 0) | Log-Transformation → int64 |
| `Integer(low, high, step)` | Ganzzahl mit optionalem Schritt | direkt int64 |
| `Categorical(choices)` | Feste Menge von Auswahlmöglichkeiten | Index als int64 (0, 1, 2, ...) |

Alle Dimensionen werden als 64-Bit-Ganzzahlen für die IMADS-Mesh-Kompatibilität kodiert.

> **Hinweis:** Die Dimensionalität des Suchraums wird automatisch an die IMADS-Engine weitergegeben. Der `HpoEvaluator` stellt eine `search_dim()`-Methode bereit (über die `SpaceEncoder.search_dim`-Eigenschaft), sodass die Engine die Anzahl der Dimensionen automatisch vom Evaluator erkennt. Es ist nicht nötig, `EngineConfig.search_dim` manuell zu setzen.

```python
space = hpo.Space({
    "lr":        hpo.LogReal(1e-5, 1e-1),    # log-scale
    "layers":    hpo.Integer(2, 8),            # integer
    "act":       hpo.Categorical(["relu", "gelu", "silu"]),  # categorical
    "dropout":   hpo.Real(0.0, 0.5),          # linear
})
```

## Multi-Fidelity: Epochenbasierte progressive Auswertung

IMADS verwendet eine 2-Achsen-Fidelity-Leiter `(τ, S)`:

| IMADS | HPO-Bedeutung |
|-------|-------------|
| `τ` (tau) | Umgekehrt proportional zur Epochenanzahl. Großes τ = wenige Epochen (günstig). |
| `S` (smc) | Anzahl unabhängiger Trainings-Seeds zur Rauschglättung. |
| TRUTH | Letzte Stufe: volles `max_epochs` mit allen Seeds. |

```python
fidelity = hpo.EpochFidelity(
    min_epochs=5,     # cheapest: 5 epochs (τ = 20)
    max_epochs=100,   # TRUTH: 100 epochs (τ = 1)
    num_seeds=3,      # S levels: [1, 3]
)
```

### Checkpoint-Präfix-Wiederverwendung

Bei der Auswertung auf einer engeren Fidelity-Stufe (mehr Epochen) lädt `imads-hpo` automatisch den Checkpoint der lockereren Stufe und setzt das Training fort:

```
Trial x, seed k:
  τ=20 → train 5 epochs   → save checkpoint
  τ=10 → load checkpoint   → train 5 more (total 10) → save
  τ=1  → load checkpoint   → train 90 more (total 100) → save (TRUTH)
```

Dies vermeidet redundante Berechnungen: Jede Fidelity-Stufe baut auf der vorherigen auf.

## Deterministische Reproduzierbarkeit

`imads-hpo` folgt den Mustern von [framesmoothie](https://github.com/Honey-Be/framesmoothie):

### SeedPath — Hierarchische Seed-Ableitung

```python
from imads_hpo import SeedPath

root = SeedPath(master_seed=42)
trial_seed = root.child("trial", 7).child("sample", 0).seed()
gen = root.child("dataloader").torch_generator()
```

Jeder RNG-Strom wird deterministisch aus dem Master-Seed über BLAKE2b-Hashing abgeleitet.

### DeterminismConfig — PyTorch-Determinismus

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

Setzt `torch.use_deterministic_algorithms(True)`, deaktiviert cuDNN-Benchmark, TF32 und konfiguriert den CUBLAS-Arbeitsbereich.

### RngRegistry — Explizite RNG-Verwaltung

```python
from imads_hpo import RngRegistry

registry = RngRegistry()
train_gen = registry.register_torch("train", SeedPath(42).child("train").torch_generator())
snapshot = registry.snapshot()   # serialize all RNG states
# ... later ...
registry.restore(snapshot)       # restore exact RNG states
```

## Funktionale Programmierprimitiven

### curry — Automatische partielle Anwendung

```python
from imads_hpo import curry

@curry
def make_optimizer(lr, weight_decay, params):
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

adam_factory = make_optimizer(0.001)(1e-5)  # partially applied
optimizer = adam_factory(model.parameters())  # fully applied
```

### Run Monad — Deterministische Zustandsübergänge

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

## Beschränkungen

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

Beschränkungen werden zusammen mit der Zielfunktion ausgewertet. IMADS verwendet sein konservatives Early-Infeasible-Screening, um offensichtlich unzulässige Kandidaten günstig auszusortieren.

## Voreinstellungen

| Voreinstellung | Anwendungsfall |
|--------|----------|
| `"balanced"` | Empfohlene Standardeinstellung. Moderater Durchsatz, gute Qualität. |
| `"conservative"` | Minimiert das Risiko falsch-unzulässiger Bewertungen. Langsamer, aber sicherer. |
| `"throughput"` | Schnelle Durchläufe. Mehr Kandidaten, früheres Pruning. |

## API-Referenz

### Oberste Ebene

| Funktion / Klasse | Beschreibung |
|-----------------|-------------|
| `Space({...})` | Suchraum definieren |
| `@objective(space, num_constraints)` | Trainingsfunktion umschließen |
| `minimize(obj, preset, ...)` | Optimierung ausführen |
| `EpochFidelity(min, max, seeds)` | Multi-Fidelity konfigurieren |
| `Result` | Optimierungsergebnis mit `.best_params`, `.best_value` |

### Reproduzierbarkeit

| Funktion / Klasse | Beschreibung |
|-----------------|-------------|
| `DeterminismConfig(seed, ...)` | PyTorch-Determinismus-Einstellungen |
| `configure_determinism(config)` | Einstellungen global anwenden |
| `SeedPath(seed).child(...)` | Hierarchische Seed-Ableitung |
| `RngRegistry` | Verwaltung benannter RNG-Ströme |
| `derive_seed(master, ns, ...)` | Curried Seed-Ableitung |

### Funktional

| Funktion / Klasse | Beschreibung |
|-----------------|-------------|
| `curry(fn)` | Automatische partielle Anwendung |
| `Run[E, S, A]` | Reader/Writer/State Monad |
| `ask()`, `get_state()`, `put_state(s)` | Monad-Primitiven |
| `tell(event)` | Log-Ereignis ausgeben |
| `sequence(programs)` | Programme der Reihe nach ausführen |
