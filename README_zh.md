# NetGen — 批量神经网络模型生成器 & 管理器

根据参数量范围生成 PyTorch 模型训练文件夹，并提供管理、对比、基准测试功能。

适用：基准测试、NAS 原型、教学演示、训练管线测试。

## 快速开始

```bash
pip install -e .
```

## 命令一览

```
netgen generate --range <min>-<max> --count <N> [选项]
netgen list     [--dir <path>]
netgen info     <id> [--dir <path>]
netgen compare  [--dir <path>] [--sort params|accuracy|loss] [--top N]
netgen benchmark [--dir <path>] [--epochs N] [--lr X]
netgen clean    [--dir <path>] [--untrained] [--dry-run|--force]
netgen export   [--dir <path>] [--format md|csv|json]
netgen archs    [--tree|--list]
```

### `generate` — 生成模型

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--range` | *必填* | 参数量范围，如 `10000-20000` |
| `--count` | 20 | 生成数量 |
| `--output` `-o` | `./generated_models` | 输出目录 |
| `--arch` | 全部 | 架构筛选，如 `mlp,cnn,lstm` |
| `--preset` `-p` | — | 快捷筛选：`cv`、`nlp`、`gen`、`light` |
| `--dataset` | `syn` | 数据集名称 |
| `--seed` | 42 | 随机种子 |
| `--jobs` `-j` | 1 | 并行进程数 |
| `--device` | cuda,mps | 训练设备优先级，如 `cuda,mps`。`cpu` 永远是最后兜底（可省略：`cuda,cpu` == `cuda`） |

```bash
netgen generate --range 10000-20000 --count 20
netgen generate --device cuda,mps --range 10000-20000 --count 20
netgen generate --device cpu --range 10000-20000 --count 20   # 只用 CPU
netgen generate --preset cv --range 5000-50000 --count 10
netgen generate --preset nlp --arch lstm --range 10000-50000 --count 5
```

### `benchmark` — 一键对比训练

```bash
netgen benchmark --epochs 10
netgen benchmark --epochs 20 --lr 0.01 --batch-size 128
```

自动训练目录内所有未训练模型，输出排行榜并保存 `benchmark_report.md`。

### `list` / `info` / `compare` / `clean` / `export` — 模型管理

```bash
netgen list                          # 模型状态一览
netgen info 001                      # 单个模型详情
netgen compare --sort accuracy       # 横向对比
netgen clean --untrained --dry-run   # 预览删除
netgen clean --keep-best 3 --force   # 保留最优 3 个
netgen export --format md            # 导出对比报告
```

### `archs` — 架构浏览

```bash
netgen archs          # 家族树（9 个家族，31 种架构）
netgen archs --list   # 平铺列表 + 描述
```

## 文件架构（3 层）

| 层级 | 参数范围 | 文件数 | 亮点 |
|------|---------|:-----:|------|
| **Quick** | < 5 万 | 8 | 基础训练、training_log.md、data_explore.py、checkpoint & best_model |
| **Standard** | 5 万~5000 万 | 12 | + lr 调度、早停、sweep、真实可视化 |
| **Production** | > 5000 万 | 17+ | + DDP、AMP、模型子包、benchmark、profile、ONNX |

### Quick 层

```
001-mlp-200x150x80/
├── config.py         # 全部超参数（LR、epoch、optimizer、scheduler...）
├── data.py           # 数据集加载器
├── data_explore.py   # 一键数据探查
├── model.py          # PyTorch nn.Module
├── train.py          # 训练脚本（支持全部 CLI 参数）
├── eval.py           # 评估
├── predict.py        # 推理演示
├── visualize.py      # 读取 training_log.md 绘制真实曲线
├── checkpoints/      # 定时检查点
├── best_model.pth    # 自动保存最佳模型
├── model.pth         # 最终模型
├── requirements.txt
└── README.md         # 任务说明
```

## 运行模型

```bash
cd 001-mlp-200x150x80

# 1. 数据探查
python data_explore.py

# 2. 训练
python train.py --epochs 50 --lr 0.001 --batch-size 128

# 3. 从 checkpoint 恢复
python train.py --resume checkpoints/ckpt_0010.pth --epochs 100

# 4. 评估
python eval.py

# 5. 可视化
python visualize.py
```

### 训练参数

```
python train.py [选项]

--lr           学习率（默认 0.001）
--epochs       训练轮数（默认 30）
--batch-size   批量大小（默认 64）
--save-every   checkpoint 间隔（默认 10）
--optimizer    adam | sgd | adamw
--weight-decay L2 正则化
--momentum     SGD 动量
--scheduler    none | cosine | plateau | step
--patience     早停耐心值
--grad-clip    梯度裁剪阈值
--seed         随机种子
--resume       从 checkpoint 恢复
```

架构专属参数：VAE 有 `BETA`，GAN 有 `G_LR`/`D_LR`，对比学习有 `TEMPERATURE`，孪生网络有 `MARGIN`，分类有 `LABEL_SMOOTHING`。

## 数据集兼容

| 任务类型 | 数据集 | 自动架构筛选 |
|---------|--------|-----------|
| 分类 | iris, wine, breast_cancer, moons, circles, blobs, mnist, cifar10, text | linear, mlp, deep, wide, resblock, highway, moe, cnn, multitask |
| 回归 | line | linear, mlp, deep, wide, resblock |
| 合成 | syn | 全部 31 种 |

## 架构总览（31 种）

### 通用 — 8 种
`mlp` `deep` `wide` `resblock` `highway` `moe` `transformer` `sae`

### 经典（上限 500万~1000万）— 12 种
`unary`(a/b/c) `linear` `cnn` `lstm` `gru` `bilstm` `ae` `vae` `gan` `multitask` `contrastive` `siamese`

### 中型（≥ 10万）— 6 种
`rescnn` `sepcnn` `densecnn` `attnlstm` `selfattn` `gcn`

### 大型（≥ 1000万）— 5 种
`vit` `unet` `mixer` `gpt` `t5`

## 依赖

- Python ≥ 3.8
- PyTorch ≥ 2.0
- numpy
- matplotlib
- scikit-learn（可选）
- torchvision（可选）
