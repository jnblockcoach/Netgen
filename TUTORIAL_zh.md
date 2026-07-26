# NetGen 教学指南

从零开始掌握 NetGen 全部功能。

---

## 目录

1. [安装](#1-安装)
2. [第一个命令](#2-第一个命令)
3. [理解生成的文件](#3-理解生成的文件)
4. [数据在哪里](#4-数据在哪里)
5. [训练与评估](#5-训练与评估)
6. [控制参数量范围](#6-控制参数量范围)
7. [筛选架构](#7-筛选架构)
8. [使用真实数据集](#8-使用真实数据集)
9. [三层文件架构](#9-三层文件架构)
10. [架构全景（31 种）](#10-架构全景31-种)
11. [Python API](#11-python-api)
12. [常见问题](#12-常见问题)

---

## 1. 安装

```bash
cd netgen
pip install -e .
```

验证：

```bash
python -m netgen --help
```

---

## 2. 第一个命令

```bash
python -m netgen --range 500-2000 --count 5 --output ./my_models
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

每个文件夹是一个完整的独立项目。以 Quick 层为例：

```
001-mlp-200x150x80/
├── config.py         # 所有超参数（LR、epoch、维度、数据集）
├── data.py           # 数据加载器
├── data_explore.py   # 一键数据探查
├── model.py          # PyTorch 模型定义
├── train.py          # 训练脚本（支持命令行参数）
├── eval.py           # 评估脚本
├── predict.py        # 推理演示
├── visualize.py      # 绘制训练曲线
├── requirements.txt
└── README.md         # 任务说明 + 数据来源
```

所有文件即可运行，无需额外配置。

---

## 4. 数据在哪里

### 数据源

打开 `config.py`，第一行就告诉你：

```python
DATASET = 'syn'        # synthetic Gaussian random data
```

### 用 data_explore.py 查看

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
Batch x range: [-3.241, 3.926]
Done.
```

### 数据文件对应关系

| 文件 | 作用 |
|------|------|
| `config.py` → `DATASET` | 指定用哪个数据源 |
| `config.py` → `INPUT_DIM` / `OUTPUT_DIM` | 定义特征数和类别/输出维度 |
| `data.py` | 真正的 PyTorch Dataset，负责加载和预处理 |

---

## 5. 训练与评估

```bash
cd 001-mlp-200x150x80

# 训练
python train.py --epochs 30 --lr 0.001 --batch-size 128

# 评估
python eval.py

# 推理演示
python predict.py
```

### training_log.md

训练结束后**自动生成** Markdown 格式的训练日志，无需额外脚本：

```
# Training Log

**Model**: 2512 parameters
**Dataset**: syn
**Epochs**: 30
**Batch Size**: 64
**Learning Rate**: 0.001

| Epoch | Loss | Accuracy |
|-------|------|----------|
|     0 | 1.99 | 0.1960   |
|     1 | 1.59 | 0.3045   |
|   ... |  ... |    ...   |
|    29 | 0.42 | 0.8910   |
```

每个 epoch 都记录，分类任务显示 Loss+Accuracy，回归任务显示 Loss。

---

## 6. 控制参数量范围

```bash
# 极小模型（100~500 参数）
python -m netgen --range 100-500 --count 5

# 中型模型（1万~5万）
python -m netgen --range 10000-50000 --count 5

# 大型模型（100万~500万）
python -m netgen --range 1000000-5000000 --count 5

# 超大模型（50亿~100亿）
python -m netgen --range 5000000000-10000000000 --count 3
```

NetGen 自动在参数范围内搜索合适的维度组合。

---

## 7. 筛选架构

```bash
# 只要 CNN 和 LSTM
python -m netgen --range 1000-10000 --count 5 --arch cnn,lstm

# 只要 Transformer
python -m netgen --range 50000-500000 --count 3 --arch transformer

# 查看所有可用架构
python -m netgen --help
```

---

## 8. 使用真实数据集

```bash
# Iris 鸢尾花
python -m netgen --range 500-2000 --count 5 --dataset iris

# MNIST 手写数字
python -m netgen --range 50000-200000 --count 3 --dataset mnist

# CIFAR-10
python -m netgen --range 100000-500000 --count 3 --dataset cifar10

# 线性回归
python -m netgen --range 500-2000 --count 3 --dataset line
```

使用真实数据集时，NetGen 自动：
- 筛选兼容架构（分类数据集不会生成 GAN/AE）
- 重写模型维度匹配数据集的特征数
- 切换损失函数（分类→CrossEntropy，回归→MSE）

---

## 9. 三层文件架构

NetGen 根据参数量自动选择文件复杂度：

### Quick（< 5 万参数）

8 个文件，极简训练循环，适合快速实验：

```
├── train.py    # ~35 行，固定 lr
├── eval.py     # 计算指标
└── ...
```

### Standard（5 万 ~ 5000 万参数）

12 个文件，适合正经实验：

| 新增 | 作用 |
|------|------|
| lr scheduler | ReduceLROnPlateau 自动降学习率 |
| 早停 | patience=10，最佳模型存 best_model.pth |
| checkpoint | 每 SAVE_EVERY epoch 存到 checkpoints/ |
| 梯度裁剪 | clip_grad_norm_ |
| sweep.py | 网格搜索 lr × batch_size |
| visualize.py | 读取 training_log.md 画真实曲线 |

### Production（> 5000 万参数）

17+ 个文件，生产级工程：

| 新增 | 作用 |
|------|------|
| DDP | 多 GPU 分布式训练 |
| AMP | 混合精度（fp16） |
| 梯度累积 | grad_accum_steps |
| 验证集 | 自动切 80/20 |
| model/ | 子包（__init__.py + layers.py） |
| configs/ | YAML 预设（default.yaml / large.yaml） |
| benchmark.py | 推理延迟 + 吞吐量 |
| profile.py | FLOPs + 显存分析 |
| export.py | ONNX / TorchScript 导出 |

---

## 10. 架构全景（31 种）

### 通用（任意参数量都可用）— 8 种

`mlp` `deep` `wide` `resblock` `highway` `moe` `transformer` `sae`

### 经典（上限 500 万~1000 万）— 12 种

`unary`（3变体：a/b/c） `linear` `cnn` `lstm` `gru` `bilstm` `ae` `vae` `gan` `multitask` `contrastive` `siamese`

### 中型（≥ 10 万参数才出现）— 6 种

| 架构 | 领域 | 特点 |
|------|------|------|
| `rescnn` | 图像 | 多阶段残差 CNN（ResNet 风格） |
| `sepcnn` | 图像 | 深度可分离卷积（MobileNet 风格） |
| `densecnn` | 图像 | 密集连接（DenseNet 风格） |
| `attnlstm` | 序列 | LSTM + 多头自注意力池化 |
| `selfattn` | 通用 | 纯自注意力编码器 |
| `gcn` | 图 | 2 层图卷积网络 |

### 大型（≥ 1000 万参数才出现）— 5 种

| 架构 | 领域 | 特点 |
|------|------|------|
| `vit` | 图像 | Vision Transformer |
| `unet` | 图像 | 编码-解码 + 跳跃连接 |
| `mixer` | 图像 | MLP-Mixer |
| `gpt` | 语言 | 小型 GPT 解码器 |
| `t5` | 序列 | Encoder-Decoder Transformer |

---

## 11. Python API

```python
from netgen import find_candidates, gen_folder, list_architectures

# 查看可用架构
print(list_architectures())
# 31 种

# 搜索候选
candidates = find_candidates(
    lo=10000, hi=50000, count=10, seed=42,
    arch_filter=['mlp', 'cnn', 'rescnn', 'selfattn']
)

# 每个 candidate 是 (描述, 代码, 参数量, 输入维度, 输出维度, 模型类型)
for desc, code, params, inp, outp, mtype in candidates:
    print(f"{desc}: {params:,} params")

# 生成文件夹
gen_folder(
    base_dir='./output',
    index=1,
    description=desc,
    code=code,
    class_name_template='M{}',
    params=params,
    input_dim=inp,
    output_dim=outp,
    model_type=mtype,
    dataset='iris'
)
```

---

## 12. 常见问题

### Q: 模型太多/太少？

调节 `--count`，或放宽 `--range`。

### Q: 找不到合适架构？

范围太窄或 `--arch` 筛选太苛刻。去掉 `--arch` 或扩大范围。

### Q: 训练报 CUDA out of memory？

生成的大模型有 DDP/AMP 支持但需要相应 GPU。小模型直接 CPU 跑。

### Q: 复现结果？

```bash
python -m netgen --range 1000-5000 --count 5 --seed 42
```

### Q: `M{}` 是什么？

类名占位符，生成时替换为 M1、M2 等唯一类名。

### Q: 如何添加自定义架构？

1. `architectures.py` 加 `make_*` 函数
2. `search.py` 加 `_sample_*` 函数并注册到 SAMPLERS
3. 如需新模板，在对应 tier 文件加训练代码

### Q: 三层文件架构如何选择？

自动的——参数量决定层级。也可以手动查看 `generator.py` → `_get_tier()` 调整阈值。
