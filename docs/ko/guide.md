# imads-hpo: PyTorch 하이퍼파라미터 최적화 가이드

## 개요

`imads-hpo`는 IMADS (Integrated Mesh Adaptive Direct Search) 엔진을 기반으로 한 PyTorch 하이퍼파라미터 최적화 패키지입니다. 도함수를 사용하지 않는(derivative-free) 제약 조건 기반 다중 충실도 최적화를 완전한 결정론적 재현성과 함께 제공합니다.

### 주요 기능

- **다중 충실도 평가**: 저비용 평가를 먼저 수행하고(적은 에포크), 고비용 TRUTH 평가를 마지막에 수행합니다(전체 학습)
- **제약 조건 지원**: GPU 메모리, 지연 시간, 모델 크기 등을 일급 객체로 지원합니다
- **결정론적 재현성**: 동일한 시드와 데이터셋이면 하드웨어에 관계없이 동일한 결과를 보장합니다
- **함수형 핵심 설계**: curry와 Run monad를 통한 명시적 상태 관리를 지원합니다
- **범주형 변수**: 64비트 정수 인코딩을 사용합니다 (범주형 변수를 지원하지 않는 NOMAD 4 / PyNomadBBO와 차별화됩니다)
- **Checkpoint prefix 재사용**: 처음부터 다시 학습하지 않고 이전 충실도 수준의 checkpoint에서 학습을 재개합니다

## 설치

```bash
uv add imads-hpo
# or
pip install imads-hpo
```

의존성: `imads` (git에서 빌드), `torch>=2.0`, `safetensors`.

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

## 탐색 공간

| 타입 | 설명 | Mesh 인코딩 |
|------|------|:------------:|
| `Real(low, high)` | [low, high] 범위의 연속값 | float를 `base_step`을 통해 int64로 변환 |
| `LogReal(low, high)` | 로그 스케일 연속값 (둘 다 양수) | 로그 변환 후 int64로 변환 |
| `Integer(low, high, step)` | 선택적 step이 있는 정수 | int64 직접 사용 |
| `Categorical(choices)` | 고정된 선택지 집합 | int64 인덱스 (0, 1, 2, ...) |

모든 차원은 IMADS mesh 호환성을 위해 64비트 정수로 인코딩됩니다.

```python
space = hpo.Space({
    "lr":        hpo.LogReal(1e-5, 1e-1),    # log-scale
    "layers":    hpo.Integer(2, 8),            # integer
    "act":       hpo.Categorical(["relu", "gelu", "silu"]),  # categorical
    "dropout":   hpo.Real(0.0, 0.5),          # linear
})
```

## 다중 충실도: 에포크 기반 점진적 평가

IMADS는 2축 충실도 사다리 `(τ, S)`를 사용합니다:

| IMADS | HPO에서의 의미 |
|-------|----------------|
| `τ` (tau) | 에포크 수에 반비례합니다. τ가 크면 적은 에포크(저비용)를 의미합니다. |
| `S` (smc) | 노이즈 평균화를 위한 독립적인 학습 시드 수입니다. |
| TRUTH | 최종 수준: 모든 시드로 전체 `max_epochs`를 수행합니다. |

```python
fidelity = hpo.EpochFidelity(
    min_epochs=5,     # cheapest: 5 epochs (τ = 20)
    max_epochs=100,   # TRUTH: 100 epochs (τ = 1)
    num_seeds=3,      # S levels: [1, 3]
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
| `@objective(space, num_constraints)` | 학습 함수를 래핑합니다 |
| `minimize(obj, preset, ...)` | 최적화를 실행합니다 |
| `EpochFidelity(min, max, seeds)` | 다중 충실도를 설정합니다 |
| `Result` | `.best_params`, `.best_value`를 포함하는 최적화 결과입니다 |

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
