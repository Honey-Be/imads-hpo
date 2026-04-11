# imads-hpo: PyTorch-Leitfaden zur Hyperparameter-Optimierung

## Überblick

`imads-hpo` ist ein Paket zur Hyperparameter-Optimierung für PyTorch, angetrieben durch die IMADS-Engine (Integrated Mesh Adaptive Direct Search). Es bietet ableitungsfreie, beschränkte, Multi-Fidelity-Optimierung mit vollständiger deterministischer Reproduzierbarkeit.

### Hauptmerkmale

- **Multi-Fidelity-Auswertung**: günstige Auswertungen zuerst (wenige Epochen), teure TRUTH-Auswertung zuletzt (vollständiges Training)
- **Unterstützung von Beschränkungen**: GPU-Speicher, Latenz, Modellgröße — als erstklassige Bürger
- **Bedingte Beschränkungen**: Prädikate, dynamische Grenzen und zusammengesetzte Beschränkungen
- **Deterministische Reproduzierbarkeit**: identische Seeds + Datensatz = identische Ergebnisse, unabhängig von der Hardware
- **Funktionaler Kern**: curry und Run monad für explizites Zustandsmanagement
- **Run-Monad-Pipeline**: zusammensetzbare Trial-Ausführung über `build_trial_program()`
- **Kategorische Variablen**: 64-Bit-Ganzzahlkodierung (im Gegensatz zu NOMAD 4 / PyNomadBBO, die keine kategorische Unterstützung bieten)
- **Checkpoint-Präfix-Wiederverwendung**: Training von früheren Fidelity-Stufen fortsetzen, statt neu zu starten
- **Flexible Fidelity-Zeitpläne**: epochenbasiert, datenfraktionsbasiert oder explizite Zeitpläne über das `FidelitySchedule`-Protokoll
- **Frühzeitiges Stoppen**: Patient-, Median- und Schwellenwert-Stoppregeln
- **Dashboard-Integration**: Weights & Biases, MLflow und TensorBoard über das `DashboardSink`-Protokoll
- **Mehrziel-HPO**: Optimierung mehrerer Ziele mit Pareto-Front-Extraktion

## Installation

```bash
uv add imads-hpo
# or
pip install imads-hpo

# Mit optionalen Dashboard-Integrationen:
uv add imads-hpo[wandb]
uv add imads-hpo[mlflow]
uv add imads-hpo[tensorboard]
uv add imads-hpo[all]
```

Abhängigkeiten: `imads` (aus Git gebaut), `torch>=2.0`, `safetensors`. Optional: `wandb`, `mlflow`, `tensorboardX`.

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

## Multi-Fidelity: Progressive Auswertung

IMADS verwendet eine 2-Achsen-Fidelity-Leiter `(τ, S)`:

| IMADS | HPO-Bedeutung |
|-------|-------------|
| `τ` (tau) | Umgekehrt proportional zur Epochenanzahl. Großes τ = wenige Epochen (günstig). |
| `S` (smc) | Anzahl unabhängiger Trainings-Seeds zur Rauschglättung. |
| TRUTH | Letzte Stufe: volles `max_epochs` mit allen Seeds. |

### EpochFidelity

Der standardmäßige epochenbasierte Fidelity-Zeitplan:

```python
fidelity = hpo.EpochFidelity(
    min_epochs=5,     # cheapest: 5 epochs (τ = 20)
    max_epochs=100,   # TRUTH: 100 epochs (τ = 1)
    num_seeds=3,      # S levels: [1, 3]
)
```

### FidelitySchedule-Protokoll

Jeder Fidelity-Zeitplan implementiert das `FidelitySchedule`-Protokoll (zur Laufzeit überprüfbar):

```python
class FidelitySchedule(Protocol):
    tau_levels: list[int]
    smc_levels: list[int]
    def resolve_fidelity(self, tau: int, smc: int) -> Fidelity: ...
```

Die `Fidelity`-Datenklasse enthält ein `data_fraction: float = 1.0`-Feld, das datensubsampling-basierte Fidelity ermöglicht.

### DataFractionFidelity

Fidelity basierend auf Datensubsampling statt Epochen:

```python
fidelity = hpo.DataFractionFidelity(
    fractions=[0.1, 0.25, 0.5, 1.0],  # progressive data fractions
    num_seeds=3,
)
```

### ExplicitFidelity

Benutzerdefinierte Epochenschritte für volle Kontrolle:

```python
fidelity = hpo.ExplicitFidelity(
    epoch_steps=[10, 25, 50, 100],  # explicit epoch counts per level
    num_seeds=3,
)
```

Die `minimize()`-Funktion akzeptiert jeden `FidelitySchedule`, nicht nur `EpochFidelity`:

```python
result = hpo.minimize(
    train,
    preset="balanced",
    fidelity=hpo.DataFractionFidelity(fractions=[0.1, 0.5, 1.0], num_seeds=2),
    seed=42,
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

### Run-Monad-Pipeline (Trial-Ausführung)

`WrappedObjective.evaluate()` verwendet eine Run-Monad-Pipeline anstelle eines direkten Funktionsaufrufs. Die Pipeline wird durch `build_trial_program()` zusammengesetzt:

```
_configure_determinism() → _load_or_init_checkpoint() → _execute_objective() → _save_checkpoint()
```

```python
from imads_hpo import build_trial_program

program = build_trial_program(fn, checkpoint_enabled=True)
```

#### RunLogSink

Das `RunLogSink`-Protokoll konsumiert `RunLog`-Ereignisse, die während der Pipeline-Ausführung emittiert werden. `WrappedObjective` hat ein `log_sink`-Feld, das jeden `RunLogSink` akzeptiert:

```python
from imads_hpo import NullSink, ListSink

# NullSink (Standard) — verwirft alle Ereignisse
obj = hpo.objective(space)(train)
obj.log_sink = NullSink()

# ListSink — sammelt Ereignisse (nützlich für Tests)
sink = ListSink()
obj.log_sink = sink
# ... run optimization ...
print(sink.events)
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

### Bedingte Beschränkungen

#### ConditionalConstraint

Eine Beschränkung, die nur aktiv ist, wenn ein Prädikat auf den Hyperparametern `True` zurückgibt. Wenn inaktiv, gibt die Beschränkung `inactive_value` zurück (Standard `-1.0`, d.h. zulässig):

```python
from imads_hpo import ConditionalConstraint

# GPU-Beschränkung nur anwenden, wenn batch_size > 128
gpu_constraint = ConditionalConstraint(
    predicate=lambda params: params["batch_size"] > 128,
    inner=lambda result: result.gpu_gb - 8.0,
    inactive_value=-1.0,
)
```

#### DynamicBoundConstraint

Eine Beschränkung, deren Grenze von den Hyperparametern selbst abhängt:

```python
from imads_hpo import DynamicBoundConstraint

# Latenzgrenze variiert mit der Modellgröße
latency_constraint = DynamicBoundConstraint(
    bound_fn=lambda params: 50.0 if params["layers"] <= 4 else 100.0,
)
```

#### CompositeConstraint

Kombiniert mehrere Beschränkungen durch Übernahme des Maximums (am stärksten verletzt):

```python
from imads_hpo import CompositeConstraint

combined = CompositeConstraint(gpu_constraint, latency_constraint)
```

## Frühzeitiges Stoppen

Das `EarlyStoppingRule`-Protokoll definiert Regeln, die einen Trial basierend auf seiner Trainingstrajektorie frühzeitig beenden können:

```python
class EarlyStoppingRule(Protocol):
    def should_stop(self, epoch: int, metric: float, history: list[float]) -> bool: ...
    def reset(self) -> None: ...
```

### PatientStopper

Stoppt, wenn für `patience` aufeinanderfolgende Epochen keine Verbesserung eintritt:

```python
from imads_hpo import PatientStopper

stopper = PatientStopper(patience=10, min_delta=1e-4)
```

### MedianStopper

Stoppt, wenn die Metrik des aktuellen Trials unter dem Median der abgeschlossenen Trials liegt:

```python
from imads_hpo import MedianStopper

stopper = MedianStopper(completed_curves=completed, min_epochs=5)
```

### ThresholdStopper

Stoppt, wenn die Metrik einen festen Schwellenwert überschreitet:

```python
from imads_hpo import ThresholdStopper

stopper = ThresholdStopper(threshold=2.0, min_epochs=3)
```

## Dashboard-Integration

Das `DashboardSink`-Protokoll leitet Trial-Metriken an externe Dashboards weiter:

```python
class DashboardSink(Protocol):
    def on_trial_start(self, trial_id: int, params: dict) -> None: ...
    def on_trial_metric(self, trial_id: int, epoch: int, metric: float) -> None: ...
    def on_trial_end(self, trial_id: int, value: float) -> None: ...
    def on_study_end(self) -> None: ...
    def flush(self) -> None: ...
```

`RunLogAdapter` verbindet `RunLog`-Ereignisse mit einem `DashboardSink`.

### Backend-Implementierungen

Verfügbar im `integrations/`-Unterpaket:

```python
from imads_hpo.integrations import WandbSink, MLflowSink, TensorBoardSink

# Weights & Biases
sink = WandbSink(project="my-hpo-study")

# MLflow
sink = MLflowSink(experiment_name="hpo-run")

# TensorBoard
sink = TensorBoardSink(log_dir="runs/hpo")
```

Jedes Backend benötigt seine optionale Abhängigkeit. Installieren Sie mit Extras:

```bash
uv add imads-hpo[wandb]
uv add imads-hpo[mlflow]
uv add imads-hpo[tensorboard]
uv add imads-hpo[all]         # alle Integrationen
```

## Mehrziel-HPO

Optimieren Sie mehrere Ziele gleichzeitig, indem Sie `num_objectives` angeben:

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

`WrappedObjective` hat ein `num_objectives`-Feld, das die Anzahl der Ziele widerspiegelt. Wenn `num_objectives > 1`, ist der Typ von `TrialRecord.value` `list[float]` statt `float`.

Das Ergebnis enthält eine Pareto-Front:

```python
result = hpo.minimize(
    train,
    preset="balanced",
    workers=4,
    fidelity=hpo.EpochFidelity(min_epochs=5, max_epochs=100, num_seeds=3),
    seed=42,
)

# Pareto-optimale Trials
for trial in result.pareto_front:
    print(trial.params, trial.value)  # value is [loss, latency]
```

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
| `@objective(space, num_constraints, num_objectives)` | Trainingsfunktion umschließen |
| `minimize(obj, preset, ...)` | Optimierung ausführen |
| `EpochFidelity(min, max, seeds)` | Epochenbasierte Multi-Fidelity konfigurieren |
| `DataFractionFidelity(fractions, seeds)` | Datensubsampling-Fidelity konfigurieren |
| `ExplicitFidelity(epoch_steps, seeds)` | Benutzerdefinierte Epochenschritte konfigurieren |
| `FidelitySchedule` | Protokoll für benutzerdefinierte Fidelity-Zeitpläne |
| `Result` | Optimierungsergebnis mit `.best_params`, `.best_value`, `.pareto_front` |

### Beschränkungen

| Funktion / Klasse | Beschreibung |
|-----------------|-------------|
| `ConditionalConstraint(predicate, inner, inactive_value)` | Nur aktiv, wenn Prädikat True ergibt |
| `DynamicBoundConstraint(bound_fn)` | Grenze hängt von Hyperparametern ab |
| `CompositeConstraint(*constraints)` | Maximum mehrerer Beschränkungen |

### Frühzeitiges Stoppen

| Funktion / Klasse | Beschreibung |
|-----------------|-------------|
| `EarlyStoppingRule` | Protokoll für Stoppregeln |
| `PatientStopper(patience, min_delta)` | Stoppt bei fehlender Verbesserung |
| `MedianStopper(completed_curves, min_epochs)` | Stoppt bei Unterschreitung des Medians |
| `ThresholdStopper(threshold, min_epochs)` | Stoppt bei Überschreitung des Schwellenwerts |

### Dashboard-Integration

| Funktion / Klasse | Beschreibung |
|-----------------|-------------|
| `DashboardSink` | Protokoll für Dashboard-Backends |
| `RunLogAdapter` | Leitet RunLog-Ereignisse an DashboardSink weiter |
| `WandbSink` | Weights & Biases-Integration |
| `MLflowSink` | MLflow-Integration |
| `TensorBoardSink` | TensorBoard-Integration |

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
| `build_trial_program(fn, checkpoint_enabled)` | Trial-Ausführungspipeline zusammensetzen |
| `RunLogSink` | Protokoll zum Konsumieren von RunLog-Ereignissen |
| `NullSink` | Verwirft alle RunLog-Ereignisse (Standard) |
| `ListSink` | Sammelt RunLog-Ereignisse (zum Testen) |
