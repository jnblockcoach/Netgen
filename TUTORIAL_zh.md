# NetGen 教学指南

从零开始掌握 NetGen 全部功能。

---

## 目录

1. [安装](#1-安装)
2. [第一个命令](#2-第一个命令)
3. [理解生成的文件](#3-理解生成的文件)
4. [数据在哪里](#4-数据在哪里)
5. [训练与评估](#5-训练与评估)
6. [恢复训练](#6-恢复训练)
7. [控制参数量范围](#7-控制参数量范围)
8. [筛选架构](#8-筛选架构)
9. [使用真实数据集](#9-使用真实数据集)
10. [三层文件架构](#10-三层文件架构)
11. [架构全景（31 种）](#11-架构全景31-种)
12. [模型管理](#12-模型管理)
13. [一键对比基准测试](#13-一键对比基准测试)
14. [Python API](#14-python-api)
15. [常见问题](#15-常见问题)

---

## 1. 安装

```bash
cd netgen
pip install -e .
```

验证：

```bash
netgen --help
```

---

## 2. 第一个命令

```bash
netgen generate --range 500-2000 --count 5 --output ./my_models
```

输出：

```
NetGen - Generating 5 models in range 500-2,000

  [  1] 001-linear-29x27:          810 params  [reg]
  [  2] 002-siamese-16x18:          914 params  [sim]
  [  3] 003-mlp-42x6x3:            279 params  [cls]
  [  4] 004-resblock-11x22x2:       592 params  [cls]
  [  5] 005-cnn-66f:              1,330 params  [img]

Generated 5 models in ./my_models
```

---

## 3. 理解生成的文件

以 Quick 层为例：

```
001-mlp-200x150x80/
├── config.py         # 所有超参数（LR、epoch、optimizer、scheduler…）
├── data.py           # 数据集加载器
├── data_explore.py   # 一键数据探查
├── model.py          # 模型定义
├── train.py          # 训练（argparse 支持全部参数）
├── eval.py           # 评估
├── predict.py        # 推理演示
├── visualize.py      # 读取训练日志画真实曲线
├── checkpoints/      # 定时检查点
├── best_model.pth    # 自动保存最佳模型
├── model.pth         # 训练结束模型
├── requirements.txt
└── README.md         # 任务说明
```

所有文件无需额外配置，直接运行。

---

## 4. 数据在哪里

### 查看数据源

打开 `config.py`：

```python
DATASET = 'syn'        # synthetic Gaussian random data
INPUT_DIM = 25
OUTPUT_DIM = 28
```

### 一键数据探查

```bash
python data_explore.py
```

输出：

```
==================================================
  DATA EXPLORER
==================================================
Dataset:    syn
Samples:    2000
Input dim:  25
Output dim: 28

Sample[0] (x) shape: torch.Size([25])
Sample[1] (y/target): 3

Batch x stats — mean: 0.025, std: 0.991
Done.
```

---

## 5. 训练与评估

```bash
cd 001-mlp-200x150x80

# 默认参数
python train.py

# 自定义超参数
python train.py --epochs 50 --lr 0.01 --batch-size 32 --optimizer adamw

# 带调度器 + 早停
python train.py --scheduler cosine --patience 15 --grad-clip 2.0

# 评估
python eval.py

# 可视化
python visualize.py
```

### 训练显示

```
Model: 2512 parameters | Epochs: 5 | Batch: 64

  Epoch   1/5 [####################] 100.0%
  Epoch   1/5 | loss=1.6075  acc=0.3335

  Epoch   2/5 [####################] 100.0%
  Epoch   2/5 | loss=1.4621  acc=0.3920
  ...
```

### 训练日志

`training_log.md` 自动生成：

```markdown
# Training Log

**Model**: 2512 parameters
**Dataset**: syn
**Epochs**: 5

| Epoch | Loss | Accuracy |
|-------|------|----------|
|     0 | 1.61 | 0.3335   |
|     1 | 1.46 | 0.3920   |
```

分类显示 Loss+Accuracy，回归只显示 Loss，VAE 显示 Recon Loss，GAN 显示 D Loss+G Loss。

---

## 6. 恢复训练

```bash
# 从 checkpoint 继续
python train.py --resume checkpoints/ckpt_0010.pth --epochs 50
```

输出：

```
Resumed from checkpoints/ckpt_0010.pth (epoch 11)
  Epoch  12/50 [####################] 100.0%  Epoch  12/50 | loss=0.8755  acc=0.6085
```

恢复内容：模型权重、优化器状态、从断点 +1 epoch 继续。

---

## 7. 控制参数量范围

```bash
netgen generate --range 100-500 --count 5           # 极小
netgen generate --range 10000-50000 --count 5        # 中等
netgen generate --range 1000000-5000000 --count 5    # 大型
netgen generate --range 5000000000-10000000000 --count 3  # 超大
```

---

## 8. 筛选架构

### `--arch` 精确筛选

```bash
netgen generate --range 1000-10000 --count 5 --arch cnn,lstm
netgen generate --range 50000-500000 --count 3 --arch transformer
```

### `--preset` 快捷筛选

```bash
netgen generate --preset cv --range 5000-50000 --count 5    # 图像
netgen generate --preset nlp --range 1000-10000 --count 3   # 序列
netgen generate --preset gen --range 500-5000 --count 5     # 生成
netgen generate --preset light --range 100-500 --count 5    # 轻量
```

`--preset` 和 `--arch` 可叠加——取二者交集。

### 浏览所有架构

```bash
netgen archs          # 家族树
netgen archs --list   # 平铺列表
```

---

## 9. 使用真实数据集

```bash
netgen generate --range 500-2000 --count 5 --dataset iris
netgen generate --range 50000-200000 --count 3 --dataset mnist
netgen generate --range 500-2000 --count 3 --dataset line
```

NetGen 自动：

- 筛选兼容架构（iris → 分类架构，line → 回归架构）
- 重写模型输入/输出维度匹配数据集
- 切换损失函数

---

## 10. 三层文件架构

| 层级 | 参数 | 文件 | 特点 |
|------|------|:--:|------|
| Quick | < 5 万 | 8 | 基础训练 + checkpoint + best_model |
| Standard | 5 万~5000 万 | 12 | + 早停、sweep、真实可视化 |
| Production | > 5000 万 | 17+ | + DDP、AMP、子包、benchmark |

所有三层都包含 `checkpoints/` + `best_model.pth` + `model.pth`。

---

## 11. 架构全景（31 种）

查看完整家族树：

```bash
netgen archs
```

- **通用**：mlp, deep, wide, resblock, highway, moe, transformer, sae
- **经典**：unary, linear, cnn, lstm, gru, bilstm, ae, vae, gan, multitask, contrastive, siamese
- **中型**（≥ 10 万）：rescnn, sepcnn, densecnn, attnlstm, selfattn, gcn
- **大型**（≥ 1000 万）：vit, unet, mixer, gpt, t5

---

## 12. 模型管理

```bash
# 列出所有模型及状态
netgen list

# 单个模型详情
netgen info 001

# 按准确率排名
netgen compare --sort accuracy --top 5

# 清理未训练模型（预览）
netgen clean --untrained --dry-run

# 真删
netgen clean --untrained --force

# 保留最好的 3 个
netgen clean --keep-best 3 --force

# 导出对比报告
netgen export --format md
```

---

## 13. 一键对比基准测试

```bash
# 训练所有未训练模型
netgen benchmark --epochs 10

# 自定义参数
netgen benchmark --epochs 20 --lr 0.01 --batch-size 128
```

输出：

```
============================================================
  BENCHMARK: 5 models × 10 epochs
============================================================

  [1/5] 001-mlp-200x150     OK (5.1s)  accuracy=0.8912
  [2/5] 002-lstm-32h-2l     OK (6.3s)  accuracy=0.9431
  ...

======================================================================
  BENCHMARK RESULTS
======================================================================

Rank  Model                  Params  Metric            Time
   1  002-lstm-32h-2l        15,618  accuracy=0.9431   6.3s
   2  001-mlp-200x150        37,330  accuracy=0.8912   5.1s
   ...

Saved benchmark_report.md
```

---

## 14. Python API

```python
from netgen import find_candidates, gen_folder, list_architectures

# 搜索候选架构
candidates = find_candidates(
    lo=10000, hi=50000, count=10, seed=42,
    arch_filter=['mlp', 'cnn', 'rescnn']
)

# 生成文件夹
for i, (desc, code, params, inp, outp, mtype) in enumerate(candidates):
    gen_folder('./output', i+1, desc, code, 'M{}',
               params, inp, outp, mtype, dataset='iris')
```

---

## 15. 常见问题

**Q: 模型太少？** 放宽 `--range`，或加大 `--count`。默认最多尝试 `count × 20` 次。

**Q: 训练中断如何恢复？** `python train.py --resume checkpoints/ckpt_XXXX.pth --epochs 50`

**Q: 复现某次生成？** 加 `--seed 42`。

**Q: `M{}` 是什么？** 类名占位符，生成时替换为 M1、M2 等。

**Q: checkpoint 在哪？** 每个模型文件夹内的 `checkpoints/` 目录。

**Q: 如何查看最佳模型？** `best_model.pth` 是训练过程中 loss 最低时自动保存的。
