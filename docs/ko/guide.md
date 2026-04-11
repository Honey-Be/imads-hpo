# imads-hpo: PyTorch 하이퍼파라미터 최적화 가이드

## 개요

`imads-hpo`는 IMADS (Integrated Mesh Adaptive Direct Search) 엔진을 기반으로 한 PyTorch 하이퍼파라미터 최적화 패키지입니다. 도함수를 사용하지 않는(derivative-free) 제약 조건 기반 다중 충실도 최적화를 완전한 결정론적 재현성과 함께 제공합니다.

### 주요 기능

- **다중 충실도 평가**: 저비용 평가를 먼저 수행하고(적은 에포크), 고비용 TRUTH 평가를 마지막에 수행합니다(전체 학습)
- **제약 조건 지원**: GPU 메모리, 지연 시간, 모델 크기 등을 일급 객체로 지원합니다
- **조건부 제약 조건**: 술어, 동적 바운드, 복합 제약 조건을 지원합니다
- **결정론적 재현성**: 동일한 시드와 데이터셋이면 하드웨어에 관계없이 동일한 결과를 보장합니다
- **함수형 핵심 설계**: curry와 Run monad를 통한 명시적 상태 관리를 지원합니다
- **Run monad 파이프라인**: `build_trial_program()`을 통한 조합 가능한 시행 실행을 지원합니다
- **범주형 변수**: 64비트 정수 인코딩을 사용합니다 (범주형 변수를 지원하지 않는 NOMAD 4 / PyNomadBBO와 차별화됩니다)
- **Checkpoint prefix 재사용**: 처음부터 다시 학습하지 않고 이전 충실도 수준의 checkpoint에서 학습을 재개합니다
- **유연한 충실도 스케줄**: 에포크 기반, 데이터 비율 기반, 또는 `FidelitySchedule` 프로토콜을 통한 명시적 스케줄을 지원합니다
- **조기 종료**: patience, median, threshold 기반 종료 규칙을 지원합니다
- **대시보드 연동**: `DashboardSink` 프로토콜을 통해 Weights & Biases, MLflow, TensorBoard를 지원합니다
- **다중 목적 HPO**: 파레토 프론트 추출을 통한 다중 목적 최적화를 지원합니다

## 설치

```bash
uv add imads-hpo
# or
pip install imads-hpo

# 선택적 대시보드 연동 설치:
uv add imads-hpo[wandb]
uv add imads-hpo[mlflow]
uv add imads-hpo[tensorboard]
uv add imads-hpo[all]
```

의존성: `imads` (git에서 빌드), `torch>=2.0`, `safetensors`. 선택 사항: `wandb`, `mlflow`, `tensorboardX`.

## 빠른 시작

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

## 탐색 공간

| 타입 | 설명 | Mesh 인코딩 |
|------|------|:------------:|
| `Real(low, high)` | [low, high] 범위의 연속값 | float를 `base_step`을 통해 int64로 변환 |
| `LogReal(low, high)` | 로그 스케일 연속값 (둘 다 양수) | 로그 변환 후 int64로 변환 |
| `Integer(low, high, step)` | 선택적 step이 있는 정수 | int64 직접 사용 |
| `Categorical(choices)` | 고정된 선택지 집합 | int64 인덱스 (0, 1, 2, ...) |

모든 차원은 IMADS mesh 호환성을 위해 64비트 정수로 인코딩됩니다.

> **참고:** 탐색 공간의 차원 수는 IMADS 엔진에 자동으로 전달됩니다. `HpoEvaluator`가 `search_dim()` 메서드를 제공하며(`SpaceEncoder.search_dim` 속성을 통해), 엔진이 evaluator로부터 차원 수를 자동으로 파악합니다. `EngineConfig.search_dim`을 수동으로 설정할 필요가 없습니다.

```python
space = hpo.Space({
    "lr":        hpo.LogReal(1e-5, 1e-1),    # log-scale
    "layers":    hpo.Integer(2, 8),            # integer
    "act":       hpo.Categorical(["relu", "gelu", "silu"]),  # categorical
    "dropout":   hpo.Real(0.0, 0.5),          # linear
})
```

## 다중 충실도: 점진적 평가

IMADS는 2축 충실도 사다리 `(τ, S)`를 사용합니다:

| IMADS | HPO에서의 의미 |
|-------|----------------|
| `τ` (tau) | 에포크 수에 반비례합니다. τ가 크면 적은 에포크(저비용)를 의미합니다. |
| `S` (smc) | 노이즈 평균화를 위한 독립적인 학습 시드 수입니다. |
| TRUTH | 최종 수준: 모든 시드로 전체 `max_epochs`를 수행합니다. |

### EpochFidelity

기본 에포크 기반 충실도 스케줄입니다:

```python
fidelity = hpo.EpochFidelity(
    min_epochs=5,     # cheapest: 5 epochs (τ = 20)
    max_epochs=100,   # TRUTH: 100 epochs (τ = 1)
    num_seeds=3,      # S levels: [1, 3]
)
```

### FidelitySchedule 프로토콜

모든 충실도 스케줄은 `FidelitySchedule` 프로토콜을 구현합니다 (런타임 검사 가능):

```python
class FidelitySchedule(Protocol):
    tau_levels: list[int]
    smc_levels: list[int]
    def resolve_fidelity(self, tau: int, smc: int) -> Fidelity: ...
```

`Fidelity` 데이터클래스에는 `data_fraction: float = 1.0` 필드가 포함되어 있어 데이터 서브샘플링 기반 충실도를 지원합니다.

### DataFractionFidelity

에포크 대신 데이터 서브샘플링을 기반으로 한 충실도입니다:

```python
fidelity = hpo.DataFractionFidelity(
    fractions=[0.1, 0.25, 0.5, 1.0],  # progressive data fractions
    num_seeds=3,
)
```

### ExplicitFidelity

완전한 제어를 위한 사용자 지정 에포크 단계입니다:

```python
fidelity = hpo.ExplicitFidelity(
    epoch_steps=[10, 25, 50, 100],  # explicit epoch counts per level
    num_seeds=3,
)
```

`minimize()` 함수는 `EpochFidelity`뿐만 아니라 모든 `FidelitySchedule`을 받아들입니다:

```python
result = hpo.minimize(
    train,
    preset="balanced",
    fidelity=hpo.DataFractionFidelity(fractions=[0.1, 0.5, 1.0], num_seeds=2),
    seed=42,
)
```

### Checkpoint Prefix 재사용

더 엄격한 충실도(더 많은 에포크)로 평가할 때, `imads-hpo`는 이전의 느슨한 수준에서 저장된 checkpoint를 자동으로 불러와 학습을 재개합니다:

```
Trial x, seed k:
  τ=20 → train 5 epochs   → save checkpoint
  τ=10 → load checkpoint   → train 5 more (total 10) → save
  τ=1  → load checkpoint   → train 90 more (total 100) → save (TRUTH)
```

이를 통해 중복 연산을 방지합니다: 각 충실도 수준은 이전 수준을 기반으로 합니다.

## 결정론적 재현성

`imads-hpo`는 [framesmoothie](https://github.com/Honey-Be/framesmoothie)의 패턴을 따릅니다:

### SeedPath — 계층적 시드 파생

```python
from imads_hpo import SeedPath

root = SeedPath(master_seed=42)
trial_seed = root.child("trial", 7).child("sample", 0).seed()
gen = root.child("dataloader").torch_generator()
```

모든 RNG 스트림은 BLAKE2b 해싱을 통해 마스터 시드로부터 결정론적으로 파생됩니다.

### DeterminismConfig — PyTorch 결정론 설정

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

`torch.use_deterministic_algorithms(True)`를 설정하고, cuDNN benchmark와 TF32를 비활성화하며, CUBLAS 워크스페이스를 구성합니다.

### RngRegistry — 명시적 RNG 관리

```python
from imads_hpo import RngRegistry

registry = RngRegistry()
train_gen = registry.register_torch("train", SeedPath(42).child("train").torch_generator())
snapshot = registry.snapshot()   # serialize all RNG states
# ... later ...
registry.restore(snapshot)       # restore exact RNG states
```

## 함수형 프로그래밍 프리미티브

### curry — 자동 부분 적용

```python
from imads_hpo import curry

@curry
def make_optimizer(lr, weight_decay, params):
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

adam_factory = make_optimizer(0.001)(1e-5)  # partially applied
optimizer = adam_factory(model.parameters())  # fully applied
```

### Run Monad — 결정론적 상태 전이

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

### Run Monad 파이프라인 (시행 실행)

`WrappedObjective.evaluate()`는 직접 함수 호출 대신 Run monad 파이프라인을 사용합니다. 파이프라인은 `build_trial_program()`으로 구성됩니다:

```
_configure_determinism() → _load_or_init_checkpoint() → _execute_objective() → _save_checkpoint()
```

```python
from imads_hpo import build_trial_program

program = build_trial_program(fn, checkpoint_enabled=True)
```

#### RunLogSink

`RunLogSink` 프로토콜은 파이프라인 실행 중 발생하는 `RunLog` 이벤트를 소비합니다. `WrappedObjective`에는 모든 `RunLogSink`를 받아들이는 `log_sink` 필드가 있습니다:

```python
from imads_hpo import NullSink, ListSink

# NullSink (기본값) — 모든 이벤트를 폐기합니다
obj = hpo.objective(space)(train)
obj.log_sink = NullSink()

# ListSink — 이벤트를 축적합니다 (테스트에 유용합니다)
sink = ListSink()
obj.log_sink = sink
# ... run optimization ...
print(sink.events)
```

## 제약 조건

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

제약 조건은 목적 함수와 함께 평가됩니다. IMADS는 보수적 조기 비실현가능성 스크리닝을 사용하여 명백히 비실현가능한 후보를 저비용으로 건너뜁니다.

### 조건부 제약 조건

#### ConditionalConstraint

하이퍼파라미터에 대한 술어가 `True`를 반환할 때만 활성화되는 제약 조건입니다. 비활성일 때는 `inactive_value` (기본값 `-1.0`, 즉 실현가능)를 반환합니다:

```python
from imads_hpo import ConditionalConstraint

# batch_size > 128일 때만 GPU 제약 조건을 적용합니다
gpu_constraint = ConditionalConstraint(
    predicate=lambda params: params["batch_size"] > 128,
    inner=lambda result: result.gpu_gb - 8.0,
    inactive_value=-1.0,
)
```

#### DynamicBoundConstraint

바운드가 하이퍼파라미터 자체에 의존하는 제약 조건입니다:

```python
from imads_hpo import DynamicBoundConstraint

# 지연 시간 바운드가 모델 크기에 따라 달라집니다
latency_constraint = DynamicBoundConstraint(
    bound_fn=lambda params: 50.0 if params["layers"] <= 4 else 100.0,
)
```

#### CompositeConstraint

최대값(가장 많이 위반된 값)을 취하여 여러 제약 조건을 결합합니다:

```python
from imads_hpo import CompositeConstraint

combined = CompositeConstraint(gpu_constraint, latency_constraint)
```

## 조기 종료

`EarlyStoppingRule` 프로토콜은 학습 궤적을 기반으로 시행을 조기에 종료할 수 있는 규칙을 정의합니다:

```python
class EarlyStoppingRule(Protocol):
    def should_stop(self, epoch: int, metric: float, history: list[float]) -> bool: ...
    def reset(self) -> None: ...
```

### PatientStopper

`patience`회 연속으로 개선이 없으면 종료합니다:

```python
from imads_hpo import PatientStopper

stopper = PatientStopper(patience=10, min_delta=1e-4)
```

### MedianStopper

현재 시행의 지표가 완료된 시행들의 중앙값보다 낮으면 종료합니다:

```python
from imads_hpo import MedianStopper

stopper = MedianStopper(completed_curves=completed, min_epochs=5)
```

### ThresholdStopper

지표가 고정된 임계값을 초과하면 종료합니다:

```python
from imads_hpo import ThresholdStopper

stopper = ThresholdStopper(threshold=2.0, min_epochs=3)
```

## 대시보드 연동

`DashboardSink` 프로토콜은 시행 지표를 외부 대시보드로 전달합니다:

```python
class DashboardSink(Protocol):
    def on_trial_start(self, trial_id: int, params: dict) -> None: ...
    def on_trial_metric(self, trial_id: int, epoch: int, metric: float) -> None: ...
    def on_trial_end(self, trial_id: int, value: float) -> None: ...
    def on_study_end(self) -> None: ...
    def flush(self) -> None: ...
```

`RunLogAdapter`는 `RunLog` 이벤트를 `DashboardSink`로 연결합니다.

### 백엔드 구현

`integrations/` 하위 패키지에서 사용할 수 있습니다:

```python
from imads_hpo.integrations import WandbSink, MLflowSink, TensorBoardSink

# Weights & Biases
sink = WandbSink(project="my-hpo-study")

# MLflow
sink = MLflowSink(experiment_name="hpo-run")

# TensorBoard
sink = TensorBoardSink(log_dir="runs/hpo")
```

각 백엔드는 선택적 의존성이 필요합니다. extras와 함께 설치하세요:

```bash
uv add imads-hpo[wandb]
uv add imads-hpo[mlflow]
uv add imads-hpo[tensorboard]
uv add imads-hpo[all]         # 모든 연동
```

## 다중 목적 HPO

`num_objectives`를 지정하여 여러 목적을 동시에 최적화합니다:

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

`WrappedObjective`에는 목적 수를 나타내는 `num_objectives` 필드가 있습니다. `num_objectives > 1`일 때, `TrialRecord.value`의 타입은 `float` 대신 `list[float]`입니다.

결과에는 파레토 프론트가 포함됩니다:

```python
result = hpo.minimize(
    train,
    preset="balanced",
    workers=4,
    fidelity=hpo.EpochFidelity(min_epochs=5, max_epochs=100, num_seeds=3),
    seed=42,
)

# 파레토 최적 시행
for trial in result.pareto_front:
    print(trial.params, trial.value)  # value is [loss, latency]
```

## 프리셋

| 프리셋 | 사용 사례 |
|--------|-----------|
| `"balanced"` | 권장 기본값입니다. 적절한 처리량과 좋은 품질을 제공합니다. |
| `"conservative"` | 잘못된 비실현가능 판정 위험을 최소화합니다. 느리지만 안전합니다. |
| `"throughput"` | 빠른 탐색에 적합합니다. 더 많은 후보를 평가하고 조기 가지치기를 수행합니다. |

## API 레퍼런스

### 최상위 레벨

| 함수 / 클래스 | 설명 |
|---------------|------|
| `Space({...})` | 탐색 공간을 정의합니다 |
| `@objective(space, num_constraints, num_objectives)` | 학습 함수를 래핑합니다 |
| `minimize(obj, preset, ...)` | 최적화를 실행합니다 |
| `EpochFidelity(min, max, seeds)` | 에포크 기반 다중 충실도를 설정합니다 |
| `DataFractionFidelity(fractions, seeds)` | 데이터 서브샘플링 충실도를 설정합니다 |
| `ExplicitFidelity(epoch_steps, seeds)` | 사용자 지정 에포크 단계를 설정합니다 |
| `FidelitySchedule` | 사용자 정의 충실도 스케줄을 위한 프로토콜입니다 |
| `Result` | `.best_params`, `.best_value`, `.pareto_front`를 포함하는 최적화 결과입니다 |

### 제약 조건

| 함수 / 클래스 | 설명 |
|---------------|------|
| `ConditionalConstraint(predicate, inner, inactive_value)` | 술어가 True일 때만 활성화됩니다 |
| `DynamicBoundConstraint(bound_fn)` | 바운드가 하이퍼파라미터에 의존합니다 |
| `CompositeConstraint(*constraints)` | 여러 제약 조건의 최대값을 취합니다 |

### 조기 종료

| 함수 / 클래스 | 설명 |
|---------------|------|
| `EarlyStoppingRule` | 종료 규칙을 위한 프로토콜입니다 |
| `PatientStopper(patience, min_delta)` | 개선이 없으면 종료합니다 |
| `MedianStopper(completed_curves, min_epochs)` | 중앙값 미만이면 종료합니다 |
| `ThresholdStopper(threshold, min_epochs)` | 임계값 초과 시 종료합니다 |

### 대시보드 연동

| 함수 / 클래스 | 설명 |
|---------------|------|
| `DashboardSink` | 대시보드 백엔드를 위한 프로토콜입니다 |
| `RunLogAdapter` | RunLog 이벤트를 DashboardSink로 전달합니다 |
| `WandbSink` | Weights & Biases 연동입니다 |
| `MLflowSink` | MLflow 연동입니다 |
| `TensorBoardSink` | TensorBoard 연동입니다 |

### 재현성

| 함수 / 클래스 | 설명 |
|---------------|------|
| `DeterminismConfig(seed, ...)` | PyTorch 결정론 설정입니다 |
| `configure_determinism(config)` | 설정을 전역적으로 적용합니다 |
| `SeedPath(seed).child(...)` | 계층적 시드 파생입니다 |
| `RngRegistry` | 이름 기반 RNG 스트림 관리입니다 |
| `derive_seed(master, ns, ...)` | curry된 시드 파생 함수입니다 |

### 함수형

| 함수 / 클래스 | 설명 |
|---------------|------|
| `curry(fn)` | 자동 부분 적용입니다 |
| `Run[E, S, A]` | Reader/Writer/State monad입니다 |
| `ask()`, `get_state()`, `put_state(s)` | Monad 프리미티브입니다 |
| `tell(event)` | 로그 이벤트를 발생시킵니다 |
| `sequence(programs)` | 프로그램을 순서대로 실행합니다 |
| `build_trial_program(fn, checkpoint_enabled)` | 시행 실행 파이프라인을 구성합니다 |
| `RunLogSink` | RunLog 이벤트를 소비하는 프로토콜입니다 |
| `NullSink` | 모든 RunLog 이벤트를 폐기합니다 (기본값) |
| `ListSink` | RunLog 이벤트를 축적합니다 (테스트용) |
