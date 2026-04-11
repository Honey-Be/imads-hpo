# imads-hpo: PyTorch ハイパーパラメータ最適化ガイド

## 概要

`imads-hpo` は、IMADS (Integrated Mesh Adaptive Direct Search) エンジンを搭載した PyTorch 向けハイパーパラメータ最適化パッケージです。導関数不要の制約付き多忠実度最適化を、完全な決定論的再現性のもとで提供します。

### 主な機能

- **多忠実度評価**: 安価な評価（少数エポック）を先に行い、最後に高価な TRUTH（フル学習）を実行します
- **制約サポート**: GPU メモリ、レイテンシ、モデルサイズなどをファーストクラスとしてサポートします
- **条件付き制約**: 述語、動的バウンド、複合制約をサポートします
- **決定論的再現性**: 同一のシードとデータセットであれば、ハードウェアに関係なく同一の結果が得られます
- **関数型コア**: curry と Run monad による明示的な状態管理を行います
- **Run monad パイプライン**: `build_trial_program()` による合成可能なトライアル実行をサポートします
- **カテゴリカル変数**: 64 ビット整数エンコーディングに対応しています（NOMAD 4 / PyNomadBBO にはないカテゴリカルサポート）
- **Checkpoint prefix reuse**: 最初からやり直すのではなく、前の忠実度レベルの checkpoint から学習を再開します
- **柔軟な忠実度スケジュール**: エポックベース、データ比率ベース、または `FidelitySchedule` プロトコルによる明示的スケジュールをサポートします
- **早期停止**: patience、median、threshold ベースの停止ルールをサポートします
- **ダッシュボード統合**: `DashboardSink` プロトコルを通じて Weights & Biases、MLflow、TensorBoard をサポートします
- **多目的 HPO**: パレートフロント抽出による多目的最適化をサポートします

## インストール

```bash
uv add imads-hpo
# or
pip install imads-hpo

# オプションのダッシュボード統合をインストール:
uv add imads-hpo[wandb]
uv add imads-hpo[mlflow]
uv add imads-hpo[tensorboard]
uv add imads-hpo[all]
```

依存パッケージ: `imads`（git からビルド）、`torch>=2.0`、`safetensors`。オプション: `wandb`、`mlflow`、`tensorboardX`。

## クイックスタート

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

## 探索空間

| 型 | 説明 | メッシュエンコーディング |
|------|-------------|:------------:|
| `Real(low, high)` | [low, high] の連続値 | float → int64（`base_step` 経由） |
| `LogReal(low, high)` | 対数スケールの連続値（両方 > 0） | 対数変換 → int64 |
| `Integer(low, high, step)` | オプションの step 付き整数 | 直接 int64 |
| `Categorical(choices)` | 固定された選択肢の集合 | インデックスとして int64（0, 1, 2, ...） |

すべての次元は IMADS メッシュとの互換性のために 64 ビット整数としてエンコーディングされます。

> **注意:** 探索空間の次元数は IMADS エンジンに自動的に伝播されます。`HpoEvaluator` が `search_dim()` メソッドを提供し（`SpaceEncoder.search_dim` プロパティ経由）、エンジンは evaluator から次元数を自動的に検出します。`EngineConfig.search_dim` を手動で設定する必要はありません。

```python
space = hpo.Space({
    "lr":        hpo.LogReal(1e-5, 1e-1),    # log-scale
    "layers":    hpo.Integer(2, 8),            # integer
    "act":       hpo.Categorical(["relu", "gelu", "silu"]),  # categorical
    "dropout":   hpo.Real(0.0, 0.5),          # linear
})
```

## 多忠実度: 段階的評価

IMADS は 2 軸の忠実度ラダー `(τ, S)` を使用します:

| IMADS | HPO における意味 |
|-------|-------------|
| `τ` (tau) | エポック数に反比例します。τ が大きい = エポック数が少ない（安価）。 |
| `S` (smc) | ノイズ平均化のための独立した学習シードの数です。 |
| TRUTH | 最終レベル: すべてのシードで `max_epochs` をフル実行します。 |

### EpochFidelity

デフォルトのエポックベース忠実度スケジュールです:

```python
fidelity = hpo.EpochFidelity(
    min_epochs=5,     # cheapest: 5 epochs (τ = 20)
    max_epochs=100,   # TRUTH: 100 epochs (τ = 1)
    num_seeds=3,      # S levels: [1, 3]
)
```

### FidelitySchedule プロトコル

すべての忠実度スケジュールは `FidelitySchedule` プロトコルを実装します（ランタイムチェック可能）:

```python
class FidelitySchedule(Protocol):
    tau_levels: list[int]
    smc_levels: list[int]
    def resolve_fidelity(self, tau: int, smc: int) -> Fidelity: ...
```

`Fidelity` データクラスには `data_fraction: float = 1.0` フィールドが含まれており、データサブサンプリングベースの忠実度を可能にします。

### DataFractionFidelity

エポックではなくデータサブサンプリングに基づく忠実度です:

```python
fidelity = hpo.DataFractionFidelity(
    fractions=[0.1, 0.25, 0.5, 1.0],  # progressive data fractions
    num_seeds=3,
)
```

### ExplicitFidelity

完全な制御のためのユーザー指定エポックステップです:

```python
fidelity = hpo.ExplicitFidelity(
    epoch_steps=[10, 25, 50, 100],  # explicit epoch counts per level
    num_seeds=3,
)
```

`minimize()` 関数は `EpochFidelity` だけでなく、あらゆる `FidelitySchedule` を受け付けます:

```python
result = hpo.minimize(
    train,
    preset="balanced",
    fidelity=hpo.DataFractionFidelity(fractions=[0.1, 0.5, 1.0], num_seeds=2),
    seed=42,
)
```

### Checkpoint Prefix Reuse

より厳しい忠実度（より多くのエポック）で評価する際、`imads-hpo` は自動的に前の緩い忠実度レベルの checkpoint を読み込み、学習を再開します:

```
Trial x, seed k:
  τ=20 → train 5 epochs   → save checkpoint
  τ=10 → load checkpoint   → train 5 more (total 10) → save
  τ=1  → load checkpoint   → train 90 more (total 100) → save (TRUTH)
```

これにより冗長な計算を回避します。各忠実度レベルは前のレベルの上に構築されます。

## 決定論的再現性

`imads-hpo` は [framesmoothie](https://github.com/Honey-Be/framesmoothie) のパターンに従います:

### SeedPath — 階層的シード導出

```python
from imads_hpo import SeedPath

root = SeedPath(master_seed=42)
trial_seed = root.child("trial", 7).child("sample", 0).seed()
gen = root.child("dataloader").torch_generator()
```

すべての RNG ストリームは、BLAKE2b ハッシュを介してマスターシードから決定論的に導出されます。

### DeterminismConfig — PyTorch の決定論的設定

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

`torch.use_deterministic_algorithms(True)` を設定し、cuDNN benchmark と TF32 を無効化し、CUBLAS ワークスペースを構成します。

### RngRegistry — 明示的な RNG 管理

```python
from imads_hpo import RngRegistry

registry = RngRegistry()
train_gen = registry.register_torch("train", SeedPath(42).child("train").torch_generator())
snapshot = registry.snapshot()   # serialize all RNG states
# ... later ...
registry.restore(snapshot)       # restore exact RNG states
```

## 関数型プログラミングプリミティブ

### curry — 自動部分適用

```python
from imads_hpo import curry

@curry
def make_optimizer(lr, weight_decay, params):
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

adam_factory = make_optimizer(0.001)(1e-5)  # partially applied
optimizer = adam_factory(model.parameters())  # fully applied
```

### Run Monad — 決定論的状態遷移

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

### Run Monad パイプライン（トライアル実行）

`WrappedObjective.evaluate()` は直接の関数呼び出しではなく、Run monad パイプラインを使用します。パイプラインは `build_trial_program()` によって構成されます:

```
_configure_determinism() → _load_or_init_checkpoint() → _execute_objective() → _save_checkpoint()
```

```python
from imads_hpo import build_trial_program

program = build_trial_program(fn, checkpoint_enabled=True)
```

#### RunLogSink

`RunLogSink` プロトコルは、パイプライン実行中に発生する `RunLog` イベントを消費します。`WrappedObjective` には任意の `RunLogSink` を受け付ける `log_sink` フィールドがあります:

```python
from imads_hpo import NullSink, ListSink

# NullSink（デフォルト）— すべてのイベントを破棄します
obj = hpo.objective(space)(train)
obj.log_sink = NullSink()

# ListSink — イベントを蓄積します（テストに便利です）
sink = ListSink()
obj.log_sink = sink
# ... run optimization ...
print(sink.events)
```

## 制約

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

制約は目的関数と同時に評価されます。IMADS は保守的な早期非実行可能性スクリーニングを使用して、明らかに非実行可能な候補を安価にスキップします。

### 条件付き制約

#### ConditionalConstraint

ハイパーパラメータに対する述語が `True` を返す場合にのみ有効になる制約です。無効な場合は `inactive_value`（デフォルト `-1.0`、すなわち実行可能）を返します:

```python
from imads_hpo import ConditionalConstraint

# batch_size > 128 の場合のみ GPU 制約を適用します
gpu_constraint = ConditionalConstraint(
    predicate=lambda params: params["batch_size"] > 128,
    inner=lambda result: result.gpu_gb - 8.0,
    inactive_value=-1.0,
)
```

#### DynamicBoundConstraint

バウンドがハイパーパラメータ自体に依存する制約です:

```python
from imads_hpo import DynamicBoundConstraint

# レイテンシバウンドがモデルサイズに応じて変化します
latency_constraint = DynamicBoundConstraint(
    bound_fn=lambda params: 50.0 if params["layers"] <= 4 else 100.0,
)
```

#### CompositeConstraint

最大値（最も違反している値）を取ることで複数の制約を結合します:

```python
from imads_hpo import CompositeConstraint

combined = CompositeConstraint(gpu_constraint, latency_constraint)
```

## 早期停止

`EarlyStoppingRule` プロトコルは、学習軌跡に基づいてトライアルを早期に終了できるルールを定義します:

```python
class EarlyStoppingRule(Protocol):
    def should_stop(self, epoch: int, metric: float, history: list[float]) -> bool: ...
    def reset(self) -> None: ...
```

### PatientStopper

`patience` 回連続で改善がない場合に停止します:

```python
from imads_hpo import PatientStopper

stopper = PatientStopper(patience=10, min_delta=1e-4)
```

### MedianStopper

現在のトライアルの指標が完了済みトライアルの中央値を下回る場合に停止します:

```python
from imads_hpo import MedianStopper

stopper = MedianStopper(completed_curves=completed, min_epochs=5)
```

### ThresholdStopper

指標が固定のしきい値を超えた場合に停止します:

```python
from imads_hpo import ThresholdStopper

stopper = ThresholdStopper(threshold=2.0, min_epochs=3)
```

## ダッシュボード統合

`DashboardSink` プロトコルは、トライアル指標を外部ダッシュボードにルーティングします:

```python
class DashboardSink(Protocol):
    def on_trial_start(self, trial_id: int, params: dict) -> None: ...
    def on_trial_metric(self, trial_id: int, epoch: int, metric: float) -> None: ...
    def on_trial_end(self, trial_id: int, value: float) -> None: ...
    def on_study_end(self) -> None: ...
    def flush(self) -> None: ...
```

`RunLogAdapter` は `RunLog` イベントを `DashboardSink` にブリッジします。

### バックエンド実装

`integrations/` サブパッケージで利用できます:

```python
from imads_hpo.integrations import WandbSink, MLflowSink, TensorBoardSink

# Weights & Biases
sink = WandbSink(project="my-hpo-study")

# MLflow
sink = MLflowSink(experiment_name="hpo-run")

# TensorBoard
sink = TensorBoardSink(log_dir="runs/hpo")
```

各バックエンドにはオプションの依存パッケージが必要です。extras と一緒にインストールしてください:

```bash
uv add imads-hpo[wandb]
uv add imads-hpo[mlflow]
uv add imads-hpo[tensorboard]
uv add imads-hpo[all]         # すべての統合
```

## 多目的 HPO

`num_objectives` を指定して複数の目的を同時に最適化します:

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

`WrappedObjective` には目的の数を反映する `num_objectives` フィールドがあります。`num_objectives > 1` の場合、`TrialRecord.value` の型は `float` ではなく `list[float]` になります。

結果にはパレートフロントが含まれます:

```python
result = hpo.minimize(
    train,
    preset="balanced",
    workers=4,
    fidelity=hpo.EpochFidelity(min_epochs=5, max_epochs=100, num_seeds=3),
    seed=42,
)

# パレート最適なトライアル
for trial in result.pareto_front:
    print(trial.params, trial.value)  # value is [loss, latency]
```

## プリセット

| プリセット | ユースケース |
|--------|----------|
| `"balanced"` | 推奨デフォルトです。適度なスループットと良好な品質を提供します。 |
| `"conservative"` | 偽非実行可能リスクを最小化します。低速ですがより安全です。 |
| `"throughput"` | 高速スイープ向けです。より多くの候補を生成し、早期に枝刈りを行います。 |

## API リファレンス

### トップレベル

| 関数 / クラス | 説明 |
|-----------------|-------------|
| `Space({...})` | 探索空間を定義します |
| `@objective(space, num_constraints, num_objectives)` | 学習関数をラップします |
| `minimize(obj, preset, ...)` | 最適化を実行します |
| `EpochFidelity(min, max, seeds)` | エポックベースの多忠実度を構成します |
| `DataFractionFidelity(fractions, seeds)` | データサブサンプリング忠実度を構成します |
| `ExplicitFidelity(epoch_steps, seeds)` | ユーザー指定エポックステップを構成します |
| `FidelitySchedule` | カスタム忠実度スケジュールのためのプロトコルです |
| `Result` | `.best_params`、`.best_value`、`.pareto_front` を持つ最適化結果です |

### 制約

| 関数 / クラス | 説明 |
|-----------------|-------------|
| `ConditionalConstraint(predicate, inner, inactive_value)` | 述語が True の場合のみ有効です |
| `DynamicBoundConstraint(bound_fn)` | バウンドがハイパーパラメータに依存します |
| `CompositeConstraint(*constraints)` | 複数の制約の最大値を取ります |

### 早期停止

| 関数 / クラス | 説明 |
|-----------------|-------------|
| `EarlyStoppingRule` | 停止ルールのためのプロトコルです |
| `PatientStopper(patience, min_delta)` | 改善がない場合に停止します |
| `MedianStopper(completed_curves, min_epochs)` | 中央値を下回る場合に停止します |
| `ThresholdStopper(threshold, min_epochs)` | しきい値を超えた場合に停止します |

### ダッシュボード統合

| 関数 / クラス | 説明 |
|-----------------|-------------|
| `DashboardSink` | ダッシュボードバックエンドのためのプロトコルです |
| `RunLogAdapter` | RunLog イベントを DashboardSink にルーティングします |
| `WandbSink` | Weights & Biases 統合です |
| `MLflowSink` | MLflow 統合です |
| `TensorBoardSink` | TensorBoard 統合です |

### 再現性

| 関数 / クラス | 説明 |
|-----------------|-------------|
| `DeterminismConfig(seed, ...)` | PyTorch の決定論的設定です |
| `configure_determinism(config)` | 設定をグローバルに適用します |
| `SeedPath(seed).child(...)` | 階層的シード導出です |
| `RngRegistry` | 名前付き RNG ストリーム管理です |
| `derive_seed(master, ns, ...)` | curry されたシード導出です |

### 関数型

| 関数 / クラス | 説明 |
|-----------------|-------------|
| `curry(fn)` | 自動部分適用です |
| `Run[E, S, A]` | Reader/Writer/State monad です |
| `ask()`、`get_state()`、`put_state(s)` | Monad プリミティブです |
| `tell(event)` | ログイベントを出力します |
| `sequence(programs)` | プログラムを順番に実行します |
| `build_trial_program(fn, checkpoint_enabled)` | トライアル実行パイプラインを構成します |
| `RunLogSink` | RunLog イベントを消費するプロトコルです |
| `NullSink` | すべての RunLog イベントを破棄します（デフォルト） |
| `ListSink` | RunLog イベントを蓄積します（テスト用） |
