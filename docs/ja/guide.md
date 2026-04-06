# imads-hpo: PyTorch ハイパーパラメータ最適化ガイド

## 概要

`imads-hpo` は、IMADS (Integrated Mesh Adaptive Direct Search) エンジンを搭載した PyTorch 向けハイパーパラメータ最適化パッケージです。導関数不要の制約付き多忠実度最適化を、完全な決定論的再現性のもとで提供します。

### 主な機能

- **多忠実度評価**: 安価な評価（少数エポック）を先に行い、最後に高価な TRUTH（フル学習）を実行します
- **制約サポート**: GPU メモリ、レイテンシ、モデルサイズなどをファーストクラスとしてサポートします
- **決定論的再現性**: 同一のシードとデータセットであれば、ハードウェアに関係なく同一の結果が得られます
- **関数型コア**: curry と Run monad による明示的な状態管理を行います
- **カテゴリカル変数**: 64 ビット整数エンコーディングに対応しています（NOMAD 4 / PyNomadBBO にはないカテゴリカルサポート）
- **Checkpoint prefix reuse**: 最初からやり直すのではなく、前の忠実度レベルの checkpoint から学習を再開します

## インストール

```bash
uv add imads-hpo
# or
pip install imads-hpo
```

依存パッケージ: `imads`（git からビルド）、`torch>=2.0`、`safetensors`。

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

## 多忠実度: エポックベースの段階的評価

IMADS は 2 軸の忠実度ラダー `(τ, S)` を使用します:

| IMADS | HPO における意味 |
|-------|-------------|
| `τ` (tau) | エポック数に反比例します。τ が大きい = エポック数が少ない（安価）。 |
| `S` (smc) | ノイズ平均化のための独立した学習シードの数です。 |
| TRUTH | 最終レベル: すべてのシードで `max_epochs` をフル実行します。 |

```python
fidelity = hpo.EpochFidelity(
    min_epochs=5,     # cheapest: 5 epochs (τ = 20)
    max_epochs=100,   # TRUTH: 100 epochs (τ = 1)
    num_seeds=3,      # S levels: [1, 3]
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
| `@objective(space, num_constraints)` | 学習関数をラップします |
| `minimize(obj, preset, ...)` | 最適化を実行します |
| `EpochFidelity(min, max, seeds)` | 多忠実度を構成します |
| `Result` | `.best_params`、`.best_value` を持つ最適化結果です |

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
