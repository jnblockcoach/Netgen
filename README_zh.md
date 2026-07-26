# NetGen — 批量神经网络模型生成器

根据指定参数量范围，批量生成可直接运行的 PyTorch 模型训练文件夹。

适用：基准测试、NAS 原型、教学演示、训练管线测试。

## 快速开始

```bash
pip install -e .
```

## 命令格式

```
python -m netgen --range <最小值>-<最大值> --count <数量> [--output <目录>] [--arch <筛选>] [--dataset <名称>] [--seed <种子>]
```

| 参数 | 必填 | 默认值 | 说明 |
|------|:--:|--------|------|
| `--range` | ✓ | — | 参数量范围，如 `10000-20000` |
| `--count` | | 20 | 生成模型数量 |
| `--output` `-o` | | `./generated_models` | 输出目录 |
| `--arch` | | 全部 | 架构筛选，逗号分隔，如 `mlp,cnn,lstm` |
| `--dataset` | | `syn` | 数据集：`syn`、`iris`、`wine`、`breast_cancer`、`moons`、`circles`、`blobs`、`mnist`、`cifar10`、`text`、`line` |
| `--seed` | | 42 | 随机种子 |

### 示例

```bash
# 生成 20 个 1万~2万 参数的模型
python -m netgen --range 10000-20000 --count 20

# 只生成 MLP 和 CNN
python -m netgen --range 5000-50000 --count 10 --arch mlp,cnn

# 使用真实数据集
python -m netgen --range 1000-10000 --count 5 --dataset iris

# 超大模型
python -m netgen --range 5000000000-10000000000 --count 3
```

## 文件架构（3 层）

生成的文件夹根据参数量自动分层：

| 层级 | 参数范围 | 文件数 | 能力 |
|------|---------|:-----:|------|
| **Quick** | < 5 万 | 8 | 基础训练 + training_log.md + data_explore.py |
| **Standard** | 5 万~5000 万 | 12 | + lr 调度、早停、checkpoint、sweep、真实可视化 |
| **Production** | > 5000 万 | 17+ | + DDP、AMP、模型子包、benchmark、profile、ONNX |

### Quick 层

```
001-mlp-200x150x80/
├── config.py         # 超参数（学习率、epoch、维度、数据集）
├── data.py           # 数据加载器
├── data_explore.py   # 数据探查 — 统计信息与形状
├── model.py          # PyTorch nn.Module
├── train.py          # 训练（argparse: --lr --epochs --batch-size）
├── eval.py           # 评估
├── predict.py        # 推理演示
├── visualize.py      # 训练曲线
├── requirements.txt
└── README.md         # 任务说明
```

### Standard 层（新增）

```
├── sweep.py          # 超参数网格搜索
├── visualize.py      # 读取 training_log.md 绘制真实曲线
├── predict.py        # 批量推理（加载 best_model.pth）
├── checkpoints/      # 定期检查点
└── best_model.pth    # 早停最佳模型
```

### Production 层（新增）

```
├── model/            # 模块化子包
│   ├── __init__.py
│   └── layers.py
├── configs/          # YAML 预设
│   ├── default.yaml
│   └── large.yaml
├── scripts/
│   ├── benchmark.py  # 推理延迟与吞吐量
│   ├── profile.py    # FLOPs / 显存分析
│   └── export.py     # ONNX / TorchScript 导出
├── logs/             # CSV 详细日志
└── checkpoints/
```

## 运行模型

```bash
cd 001-mlp-200x150x80

# 1. 先看数据
python data_explore.py

# 2. 训练（自动生成 training_log.md）
python train.py --epochs 50 --lr 0.001 --batch-size 128

# 3. 评估
python eval.py

# 4. 可视化（Standard+）— 读取 training_log.md
python visualize.py

# 5. 超参搜索（Standard+）
python sweep.py
```

## 数据集兼容性

`--dataset` 自动筛选兼容架构并重写模型维度：

| 任务类型 | 数据集 | 自动选择的架构 |
|---------|--------|-------------|
| 分类 | iris, wine, breast_cancer, moons, circles, blobs, mnist, cifar10, text | linear, mlp, deep, wide, resblock, highway, moe, cnn, multitask |
| 回归 | line | linear, mlp, deep, wide, resblock |
| 合成 | syn | 全部 31 种架构 |

## 架构总览（31 种）

### 通用架构（任意参数量）

| 键 | 架构 | 参数量级 |
|----|------|---------|
| `mlp` | 3 层全连接 | d1·d2 + d2·d3 |
| `deep` | 深度 MLP | N × d² |
| `wide` | 宽网络 | d_in × d_hidden |
| `resblock` | 残差块 | N × 2·d·h |
| `highway` | 高速网络 | N × 2·d² |
| `moe` | 混合专家 | E × 2·d·h |
| `transformer` | Transformer 编码器 | L × d_model² |
| `sae` | 堆叠自编码器 | N × 2·h² |

### 经典架构（上限 500 万~1000 万）

| 键 | 架构 | 上限 |
|----|------|:----:|
| `unary` | 1 参数变体（a/b/c） | 1 |
| `linear` | 单层线性 | 500 万 |
| `cnn` | 卷积网络 | 500 万 |
| `lstm` | LSTM | 500 万 |
| `gru` | GRU | 500 万 |
| `bilstm` | 双向 LSTM | 500 万 |
| `ae` | 自编码器 | 500 万 |
| `vae` | 变分自编码器 | 500 万 |
| `gan` | 生成对抗网络 | 1000 万 |
| `multitask` | 多任务 | 1000 万 |
| `contrastive` | 对比学习 | 1000 万 |
| `siamese` | 孪生网络 | 1000 万 |

### 中型架构（≥ 10 万参数）

| 键 | 架构 | 描述 |
|----|------|------|
| `rescnn` | 残差 CNN | 多阶段跳跃连接 |
| `sepcnn` | 分离卷积 | 深度可分离（MobileNet 风格） |
| `densecnn` | 密集 CNN | 密集连接（DenseNet 风格） |
| `attnlstm` | 注意力 LSTM | LSTM + 多头自注意力池化 |
| `selfattn` | 自注意力 | 纯注意力，无 RNN/CNN |
| `gcn` | 图卷积网络 | 2 层 GCN 节点分类 |

### 大型架构（≥ 1000 万参数）

| 键 | 架构 | 描述 |
|----|------|------|
| `vit` | Vision Transformer | Patch 嵌入 + Transformer 编码器 |
| `unet` | U-Net | 编码-解码 + 跳跃连接 |
| `mixer` | MLP-Mixer | Token-mixing + Channel-mixing MLP |
| `gpt` | GPT 解码器 | Causal Transformer 语言建模 |
| `t5` | T5 编解码器 | 完整编码器-解码器 Transformer |

## 数据集

| 名称 | 描述 | 特征数 | 类别数 |
|------|------|--------|--------|
| `syn` | 合成高斯数据 | 可配置 | 可配置 |
| `iris` | 鸢尾花 | 4 | 3 |
| `wine` | 葡萄酒 | 13 | 3 |
| `breast_cancer` | 乳腺癌 | 30 | 2 |
| `moons` | 双半月形 | 2 | 2 |
| `circles` | 同心圆 | 2 | 2 |
| `blobs` | 高斯团块 | 6 | 5 |
| `mnist` | 手写数字 | 1×28×28 | 10 |
| `cifar10` | 自然图像 | 3×32×32 | 10 |
| `text` | 字符序列 | 20 | 10 |
| `line` | 线性回归 | 1 | 1 |

## Python API

```python
from netgen import find_candidates, gen_folder, list_architectures

# 搜索架构
candidates = find_candidates(lo=10000, hi=20000, count=10, seed=42,
                             arch_filter=['mlp', 'cnn', 'lstm'])

# 生成文件夹
for desc, code, params, inp, outp, mtype in candidates:
    gen_folder('./output', 1, desc, code, 'M{}', params, inp, outp, mtype,
               dataset='iris')
```

## 依赖

- Python ≥ 3.8
- PyTorch ≥ 2.0
- numpy
- matplotlib（visualize.py 需要）
- scikit-learn（真实数据集，可选）
- torchvision（MNIST / CIFAR-10，可选）
