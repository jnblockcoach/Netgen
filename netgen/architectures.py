"""Model architecture generators.

Each make_* function returns a tuple of:
    (code: str, params: int, input_dim: int, output_dim: int, model_type: str)

code is a class template string containing `{}` for the class name placeholder.
"""
from typing import Tuple
import torch
import torch.nn as nn


def count_params(module: nn.Module) -> int:
    """Count total trainable parameters in a PyTorch module."""
    return sum(p.numel() for p in module.parameters())


# ── Linear ──

def make_linear(in_features: int, out_features: int) -> Tuple[str, int, int, int, str]:
    model = nn.Linear(in_features, out_features)
    params = count_params(model)
    code = (
        f"class M{{}}(nn.Module):\n"
        f"    def __init__(self):\n"
        f"        super().__init__()\n"
        f"        self.linear = nn.Linear({in_features}, {out_features})\n"
        f"    def forward(self, x):\n"
        f"        return self.linear(x)\n"
    )
    return code, params, in_features, out_features, 'mse'


# ── MLP ──

def make_mlp(dims: list) -> Tuple[str, int, int, int, str]:
    layers = []
    for i in range(len(dims) - 2):
        layers.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU()])
    layers.append(nn.Linear(dims[-2], dims[-1]))
    model = nn.Sequential(*layers)
    params = count_params(model)

    code = f"class M{{}}(nn.Module):\n    def __init__(self):\n        super().__init__()\n        layers = []\n"
    for i in range(len(dims) - 2):
        code += f"        layers.append(nn.Linear({dims[i]}, {dims[i + 1]}))\n        layers.append(nn.ReLU())\n"
    code += f"        layers.append(nn.Linear({dims[-2]}, {dims[-1]}))\n        self.net = nn.Sequential(*layers)\n"
    code += "    def forward(self, x):\n        return self.net(x)\n"
    return code, params, dims[0], dims[-1], 'ce'


# ── Unary (1-parameter) variants ──

def make_unary_a() -> Tuple[str, int, int, int, str]:
    model = nn.Linear(1, 1, bias=False)
    params = count_params(model)
    code = (
        "class M{}(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.linear = nn.Linear(1, 1, bias=False)\n"
        "    def forward(self, x):\n"
        "        return self.linear(x)\n"
    )
    return code, params, 1, 1, 'mse'


def make_unary_b() -> Tuple[str, int, int, int, str]:
    code = (
        "class M{}(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.weight = nn.Parameter(torch.randn(1))\n"
        "    def forward(self, x):\n"
        "        return x * self.weight\n"
    )
    return code, 1, 1, 1, 'mse'


def make_unary_c() -> Tuple[str, int, int, int, str]:
    code = (
        "class M{}(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.bias = nn.Parameter(torch.tensor(0.0))\n"
        "    def forward(self, x):\n"
        "        return x + self.bias\n"
    )
    return code, 1, 1, 1, 'mse'


# ── RNN variants ──

def make_lstm(input_size: int, hidden_size: int, num_layers: int,
              output_size: int) -> Tuple[str, int, int, int, str]:
    lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    fc = nn.Linear(hidden_size, output_size)
    params = count_params(lstm) + count_params(fc)
    code = (
        f"class M{{}}(nn.Module):\n"
        f"    def __init__(self):\n"
        f"        super().__init__()\n"
        f"        self.lstm = nn.LSTM({input_size}, {hidden_size}, {num_layers}, batch_first=True)\n"
        f"        self.fc = nn.Linear({hidden_size}, {output_size})\n"
        f"    def forward(self, x):\n"
        f"        out, _ = self.lstm(x)\n"
        f"        return self.fc(out[:, -1, :])\n"
    )
    return code, params, input_size, output_size, 'rnn'


def make_gru(input_size: int, hidden_size: int, num_layers: int,
             output_size: int) -> Tuple[str, int, int, int, str]:
    gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
    fc = nn.Linear(hidden_size, output_size)
    params = count_params(gru) + count_params(fc)
    code = (
        f"class M{{}}(nn.Module):\n"
        f"    def __init__(self):\n"
        f"        super().__init__()\n"
        f"        self.gru = nn.GRU({input_size}, {hidden_size}, {num_layers}, batch_first=True)\n"
        f"        self.fc = nn.Linear({hidden_size}, {output_size})\n"
        f"    def forward(self, x):\n"
        f"        out, _ = self.gru(x)\n"
        f"        return self.fc(out[:, -1, :])\n"
    )
    return code, params, input_size, output_size, 'rnn'


def make_bilstm(input_size: int, hidden_size: int, num_layers: int,
                output_size: int) -> Tuple[str, int, int, int, str]:
    lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
    fc = nn.Linear(2 * hidden_size, output_size)
    params = count_params(lstm) + count_params(fc)
    code = (
        f"class M{{}}(nn.Module):\n"
        f"    def __init__(self):\n"
        f"        super().__init__()\n"
        f"        self.lstm = nn.LSTM({input_size}, {hidden_size}, {num_layers}, "
        f"batch_first=True, bidirectional=True)\n"
        f"        self.fc = nn.Linear({2 * hidden_size}, {output_size})\n"
        f"    def forward(self, x):\n"
        f"        out, _ = self.lstm(x)\n"
        f"        return self.fc(out[:, -1, :])\n"
    )
    return code, params, input_size, output_size, 'rnn'


# ── CNN ──

def make_cnn(in_channels: int, filters: list, fc_sizes: list,
             num_classes: int) -> Tuple[str, int, int, int, str]:
    layers = []
    prev = in_channels
    for out_ch, pool_size in filters:
        layers.extend([nn.Conv2d(prev, out_ch, 3, padding=1), nn.ReLU()])
        if pool_size:
            layers.append(nn.MaxPool2d(pool_size))
        prev = out_ch
    layers.append(nn.AdaptiveAvgPool2d(1))
    conv_model = nn.Sequential(*layers)
    params = count_params(conv_model)

    if fc_sizes:
        fc_layers = []
        cur = prev
        for f in fc_sizes:
            fc_layers.extend([nn.Linear(cur, f), nn.ReLU()])
            cur = f
        fc_layers.append(nn.Linear(cur, num_classes))
        params += count_params(nn.Sequential(*fc_layers))
    else:
        params += prev * num_classes + num_classes

    # Build code string
    code = f"class M{{}}(nn.Module):\n    def __init__(self):\n        super().__init__()\n        conv = []\n"
    prev = in_channels
    for out_ch, pool_size in filters:
        code += f"        conv.append(nn.Conv2d({prev}, {out_ch}, 3, padding=1))\n        conv.append(nn.ReLU())\n"
        if pool_size:
            code += f"        conv.append(nn.MaxPool2d({pool_size}))\n"
        prev = out_ch
    code += "        conv.append(nn.AdaptiveAvgPool2d(1))\n        self.conv = nn.Sequential(*conv)\n"

    if fc_sizes:
        fc_parts = []
        cur = prev
        for f in fc_sizes:
            fc_parts.append(f"nn.Linear({cur}, {f})")
            fc_parts.append("nn.ReLU()")
            cur = f
        fc_parts.append(f"nn.Linear({cur}, {num_classes})")
        code += f"        self.fc = nn.Sequential({', '.join(fc_parts)})\n"
    else:
        code += f"        self.fc = nn.Linear({prev}, {num_classes})\n"

    code += "    def forward(self, x):\n        return self.fc(self.conv(x).view(x.size(0), -1))\n"
    return code, params, in_channels, num_classes, 'cnn'


# ── Autoencoder ──

def make_ae(input_dim: int, hidden_dim: int) -> Tuple[str, int, int, int, str]:
    params = 2 * (input_dim * hidden_dim + hidden_dim)
    code = (
        f"class M{{}}(nn.Module):\n"
        f"    def __init__(self):\n"
        f"        super().__init__()\n"
        f"        self.encoder = nn.Sequential(nn.Linear({input_dim}, {hidden_dim}), nn.ReLU())\n"
        f"        self.decoder = nn.Sequential(nn.Linear({hidden_dim}, {input_dim}))\n"
        f"    def forward(self, x):\n"
        f"        latent = self.encoder(x)\n"
        f"        return self.decoder(latent), latent\n"
    )
    return code, params, input_dim, input_dim, 'ae'


# ── Variational Autoencoder ──

def make_vae(input_dim: int, hidden_dim: int, latent_dim: int) -> Tuple[str, int, int, int, str]:
    params = (
        (input_dim * hidden_dim + hidden_dim)
        + 2 * (hidden_dim * latent_dim + latent_dim)
        + (latent_dim * hidden_dim + hidden_dim)
        + (hidden_dim * input_dim + input_dim)
    )
    code = (
        f"class M{{}}(nn.Module):\n"
        f"    def __init__(self):\n"
        f"        super().__init__()\n"
        f"        self.enc_h = nn.Linear({input_dim}, {hidden_dim})\n"
        f"        self.mu = nn.Linear({hidden_dim}, {latent_dim})\n"
        f"        self.logvar = nn.Linear({hidden_dim}, {latent_dim})\n"
        f"        self.dec_h = nn.Linear({latent_dim}, {hidden_dim})\n"
        f"        self.recon = nn.Linear({hidden_dim}, {input_dim})\n"
        f"    def encode(self, x):\n"
        f"        h = torch.relu(self.enc_h(x))\n"
        f"        return self.mu(h), self.logvar(h)\n"
        f"    def reparameterize(self, mu, logvar):\n"
        f"        return mu + torch.randn_like(logvar) * torch.exp(0.5 * logvar)\n"
        f"    def decode(self, z):\n"
        f"        return self.recon(torch.relu(self.dec_h(z)))\n"
        f"    def forward(self, x):\n"
        f"        mu, lv = self.encode(x)\n"
        f"        return self.decode(self.reparameterize(mu, lv)), mu, lv\n"
    )
    return code, params, input_dim, input_dim, 'vae'


# ── Deep MLP ──

def make_deep_mlp(dim: int, num_layers: int, out_dim: int = 10) -> Tuple[str, int, int, int, str]:
    params = dim * dim + dim + (num_layers - 2) * (dim * dim + dim) + dim * out_dim + out_dim
    code = f"class M{{}}(nn.Module):\n    def __init__(self):\n        super().__init__()\n        layers = []\n"
    code += f"        layers.append(nn.Linear({dim}, {dim}))\n        layers.append(nn.ReLU())\n"
    for _ in range(num_layers - 2):
        code += f"        layers.append(nn.Linear({dim}, {dim}))\n        layers.append(nn.ReLU())\n"
    code += f"        layers.append(nn.Linear({dim}, {out_dim}))\n"
    code += "        self.net = nn.Sequential(*layers)\n    def forward(self, x):\n        return self.net(x)\n"
    return code, params, dim, out_dim, 'ce'


# ── Stacked AE ──

def make_stacked_ae(dim: int, num_layers: int,
                    bottleneck_ratio: int = 4) -> Tuple[str, int, int, int, str]:
    hidden = max(1, dim // bottleneck_ratio)
    params = dim * hidden + hidden + (num_layers - 1) * (hidden * hidden + hidden) * 2 + hidden * dim + dim
    code = f"class M{{}}(nn.Module):\n    def __init__(self):\n        super().__init__()\n"
    code += "        self.encoder = nn.Sequential(\n"
    code += f"            nn.Linear({dim}, {hidden}), nn.ReLU(),\n"
    for _ in range(num_layers - 1):
        code += f"            nn.Linear({hidden}, {hidden}), nn.ReLU(),\n"
    code += "        )\n        self.decoder = nn.Sequential(\n"
    for _ in range(num_layers - 1):
        code += f"            nn.Linear({hidden}, {hidden}), nn.ReLU(),\n"
    code += f"            nn.Linear({hidden}, {dim})\n        )\n"
    code += "    def forward(self, x):\n        latent = self.encoder(x)\n        return self.decoder(latent), latent\n"
    return code, params, dim, dim, 'ae'


# ── Transformer ──

def make_transformer(d_model: int, nhead: int, num_layers: int,
                     dim_feedforward: int = None) -> Tuple[str, int, int, int, str]:
    if dim_feedforward is None:
        dim_feedforward = d_model * 4
    block_params = (
        (4 * d_model * d_model + 4 * d_model)
        + (d_model * dim_feedforward * 2 + dim_feedforward + d_model)
        + 4 * d_model
    )
    params = block_params * num_layers + d_model * 10 + 10
    code = f"class M{{}}(nn.Module):\n    def __init__(self):\n        super().__init__()\n"
    code += f"        self.blocks = nn.ModuleList()\n"
    for _ in range(num_layers):
        code += (
            f"        self.blocks.append(nn.TransformerEncoderLayer("
            f"d_model={d_model}, nhead={nhead}, dim_feedforward={dim_feedforward}, batch_first=True))\n"
        )
    code += f"        self.fc = nn.Linear({d_model}, 10)\n"
    code += (
        "    def forward(self, x):\n"
        "        if x.dim() == 2: x = x.unsqueeze(1)\n"
        "        for block in self.blocks:\n            x = block(x)\n"
        "        return self.fc(x.mean(dim=1))\n"
    )
    return code, params, d_model, 10, 'ce'


# ── Wide Net ──

def make_wide_net(in_dim: int, hidden_dim: int, out_dim: int = 10) -> Tuple[str, int, int, int, str]:
    params = in_dim * hidden_dim + hidden_dim + hidden_dim * out_dim + out_dim
    code = (
        f"class M{{}}(nn.Module):\n"
        f"    def __init__(self):\n"
        f"        super().__init__()\n"
        f"        self.net = nn.Sequential("
        f"nn.Linear({in_dim}, {hidden_dim}), nn.ReLU(), nn.Linear({hidden_dim}, {out_dim}))\n"
        f"    def forward(self, x):\n"
        f"        return self.net(x)\n"
    )
    return code, params, in_dim, out_dim, 'ce'


# ── ResBlock ──

def make_resblock(in_dim: int, hidden_dim: int, num_blocks: int,
                  num_classes: int = 10) -> Tuple[str, int, int, int, str]:
    params = 0
    code = f"class M{{}}(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.blocks = nn.ModuleList()\n"
    for _ in range(num_blocks):
        params += in_dim * hidden_dim + hidden_dim + hidden_dim * in_dim + in_dim
        code += (
            f"        self.blocks.append(nn.Sequential("
            f"nn.Linear({in_dim}, {hidden_dim}), nn.ReLU(), nn.Linear({hidden_dim}, {in_dim})))\n"
        )
    params += in_dim * num_classes + num_classes
    code += f"        self.fc = nn.Linear({in_dim}, {num_classes})\n"
    code += (
        "    def forward(self, x):\n"
        "        for block in self.blocks:\n            x = torch.relu(x + block(x))\n"
        "        return self.fc(x)\n"
    )
    return code, params, in_dim, num_classes, 'ce'


# ── Highway Network ──

def make_highway(dim: int, num_layers: int, num_classes: int = 10) -> Tuple[str, int, int, int, str]:
    params = 0
    code = (
        f"class M{{}}(nn.Module):\n"
        f"    def __init__(self):\n"
        f"        super().__init__()\n"
        f"        self.transforms = nn.ModuleList()\n"
        f"        self.gates = nn.ModuleList()\n"
    )
    for _ in range(num_layers):
        params += dim * dim + dim
        code += f"        self.transforms.append(nn.Linear({dim}, {dim}))\n"
        params += dim * dim + dim
        code += f"        self.gates.append(nn.Linear({dim}, {dim}))\n"
    params += dim * num_classes + num_classes
    code += f"        self.fc = nn.Linear({dim}, {num_classes})\n"
    code += (
        "    def forward(self, x):\n"
        "        for t, g in zip(self.transforms, self.gates):\n"
        "            gate = torch.sigmoid(g(x))\n"
        "            x = gate * torch.relu(t(x)) + (1 - gate) * x\n"
        "        return self.fc(x)\n"
    )
    return code, params, dim, num_classes, 'ce'


# ── Mixture of Experts ──

def make_moe(in_dim: int, hidden_dim: int, num_experts: int,
             num_classes: int = 10) -> Tuple[str, int, int, int, str]:
    params = in_dim * num_experts + num_experts
    code = (
        f"class M{{}}(nn.Module):\n"
        f"    def __init__(self):\n"
        f"        super().__init__()\n"
        f"        self.router = nn.Linear({in_dim}, {num_experts})\n"
        f"        self.experts = nn.ModuleList()\n"
    )
    for _ in range(num_experts):
        params += in_dim * hidden_dim + hidden_dim + hidden_dim * in_dim + in_dim
        code += (
            f"        self.experts.append(nn.Sequential("
            f"nn.Linear({in_dim}, {hidden_dim}), nn.ReLU(), nn.Linear({hidden_dim}, {in_dim})))\n"
        )
    params += in_dim * num_classes + num_classes
    code += f"        self.fc = nn.Linear({in_dim}, {num_classes})\n"
    code += (
        "    def forward(self, x):\n"
        "        weights = torch.softmax(self.router(x), dim=-1)\n"
        "        out = sum(weights[:, e:e+1] * self.experts[e](x) for e in range(len(self.experts)))\n"
        "        return self.fc(out)\n"
    )
    return code, params, in_dim, num_classes, 'ce'


# ── Multi-Task ──

def make_multitask(in_dim: int, hidden_dim: int, num_tasks: int = 2,
                   out_per_task: int = 10) -> Tuple[str, int, int, int, str]:
    actual_tasks = 2
    params = in_dim * hidden_dim + hidden_dim
    code = (
        f"class M{{}}(nn.Module):\n"
        f"    def __init__(self):\n"
        f"        super().__init__()\n"
        f"        self.shared = nn.Sequential(nn.Linear({in_dim}, {hidden_dim}), nn.ReLU())\n"
        f"        self.heads = nn.ModuleList()\n"
    )
    for _ in range(actual_tasks):
        params += hidden_dim * out_per_task + out_per_task
        code += f"        self.heads.append(nn.Linear({hidden_dim}, {out_per_task}))\n"
    code += (
        "    def forward(self, x):\n"
        "        h = self.shared(x)\n"
        "        return self.heads[0](h), self.heads[1](h)\n"
    )
    return code, params, in_dim, out_per_task, 'mt'


# ── GAN ──

def make_gan(z_dim: int, g_hidden: int, d_hidden: int,
             data_dim: int = 2) -> Tuple[str, int, int, int, str]:
    g_params = z_dim * g_hidden + g_hidden + g_hidden * data_dim + data_dim
    d_params = data_dim * d_hidden + d_hidden + d_hidden * 1 + 1
    params = g_params + d_params
    code = (
        f"class M{{}}(nn.Module):\n"
        f"    def __init__(self):\n"
        f"        super().__init__()\n"
        f"        self.generator = nn.Sequential(nn.Linear({z_dim}, {g_hidden}), nn.ReLU(), "
        f"nn.Linear({g_hidden}, {data_dim}))\n"
        f"        self.discriminator = nn.Sequential(nn.Linear({data_dim}, {d_hidden}), nn.ReLU(), "
        f"nn.Linear({d_hidden}, 1))\n"
        f"    def forward_gen(self, z):\n        return self.generator(z)\n"
        f"    def forward_disc(self, x):\n        return torch.sigmoid(self.discriminator(x))\n"
        f"    def forward(self, z):\n        return self.forward_gen(z)\n"
    )
    return code, params, z_dim, data_dim, 'gan'


# ── Contrastive ──

def make_contrastive(in_dim: int, hidden_dim: int,
                     out_dim: int = 64) -> Tuple[str, int, int, int, str]:
    params = in_dim * hidden_dim + hidden_dim + hidden_dim * out_dim + out_dim
    code = (
        f"class M{{}}(nn.Module):\n"
        f"    def __init__(self):\n"
        f"        super().__init__()\n"
        f"        self.net = nn.Sequential("
        f"nn.Linear({in_dim}, {hidden_dim}), nn.ReLU(), nn.Linear({hidden_dim}, {out_dim}))\n"
        f"    def forward(self, x):\n        return self.net(x)\n"
    )
    return code, params, in_dim, out_dim, 'contrastive'


# ── Siamese ──

def make_siamese(in_dim: int, hidden_dim: int,
                 embed_dim: int = 32) -> Tuple[str, int, int, int, str]:
    params = in_dim * hidden_dim + hidden_dim + hidden_dim * embed_dim + embed_dim
    code = (
        f"class M{{}}(nn.Module):\n"
        f"    def __init__(self):\n"
        f"        super().__init__()\n"
        f"        self.encoder = nn.Sequential("
        f"nn.Linear({in_dim}, {hidden_dim}), nn.ReLU(), nn.Linear({hidden_dim}, {embed_dim}))\n"
        f"    def forward(self, x):\n        return self.encoder(x)\n"
    )
    return code, params, in_dim, embed_dim, 'siamese'


# ═══════════════════════════════════════════════════
#  Medium-tier architectures (≥ 100K params)
# ═══════════════════════════════════════════════════

# ── ResCNN (ResNet-style multi-stage CNN) ──

def make_rescnn(in_channels: int, stages: list, num_classes: int = 10):
    """Multi-stage residual CNN. stages = [(out_ch, num_blocks), ...].
    Each block: Conv→BN→ReLU→Conv→BN, then +skip."""
    params = 0
    code = f"class M{{}}(nn.Module):\n    def __init__(self):\n        super().__init__()\n"
    code += f"        self.stem = nn.Sequential(nn.Conv2d({in_channels}, {stages[0][0]}, 3, padding=1), nn.BatchNorm2d({stages[0][0]}), nn.ReLU())\n"
    params += in_channels * stages[0][0] * 9 + stages[0][0] * 2  # Conv weight+bias + BN

    prev_ch = stages[0][0]
    all_stages = []
    for i, (out_ch, num_blocks) in enumerate(stages):
        stage_code = f"        self.stage{i} = nn.ModuleList()\n"
        for b in range(num_blocks):
            ch_in = prev_ch if b == 0 else out_ch
            code += f"        self.stage{i}.append(nn.Sequential(\n"
            code += f"            nn.Conv2d({ch_in}, {out_ch}, 3, padding=1),\n"
            code += f"            nn.BatchNorm2d({out_ch}), nn.ReLU(),\n"
            code += f"            nn.Conv2d({out_ch}, {out_ch}, 3, padding=1),\n"
            code += f"            nn.BatchNorm2d({out_ch})\n"
            code += f"        ))\n"
            params += ch_in * out_ch * 9 + out_ch + out_ch * out_ch * 9 + out_ch + out_ch * 4  # 2 Conv + 2 BN
            if b == 0 and ch_in != out_ch:
                code += f"        self.stage{i}_skip = nn.Conv2d({ch_in}, {out_ch}, 1)\n"
                params += ch_in * out_ch
            prev_ch = out_ch
        all_stages.append(i)

    code += "        self.pool = nn.AdaptiveAvgPool2d(1)\n"
    code += f"        self.fc = nn.Linear({prev_ch}, {num_classes})\n"
    params += prev_ch * num_classes + num_classes

    code += "    def forward(self, x):\n        x = self.stem(x)\n"
    for i in range(len(stages)):
        code += f"        for blk in self.stage{i}:\n            residual = x\n            x = blk(x)\n"
        code += f"            if residual.shape != x.shape:\n                x = x + self.stage{i}_skip(residual)\n"
        code += f"            else:\n                x = x + residual\n"
        code += f"            x = torch.relu(x)\n"
    code += "        x = self.pool(x).flatten(1)\n        return self.fc(x)\n"

    return code, params, in_channels, num_classes, 'cnn'


# ── SepCNN (Depthwise Separable CNN, MobileNet style) ──

def make_sepcnn(in_channels: int, channels: list, num_classes: int = 10):
    """Depthwise separable CNN. channels = [ch1, ch2, ...] per stage."""
    params = 0
    code = f"class M{{}}(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.convs = nn.ModuleList()\n"
    prev = in_channels
    for i, ch in enumerate(channels):
        # Depthwise
        code += f"        self.convs.append(nn.Conv2d({prev}, {prev}, 3, padding=1, groups={prev}))\n"
        params += prev * 9 + prev
        # Pointwise
        code += f"        self.convs.append(nn.Conv2d({prev}, {ch}, 1))\n"
        params += prev * ch + ch
        code += f"        self.convs.append(nn.BatchNorm2d({ch}))\n"
        code += f"        self.convs.append(nn.ReLU())\n"
        params += ch * 2
        prev = ch
    code += "        self.pool = nn.AdaptiveAvgPool2d(1)\n"
    code += f"        self.fc = nn.Linear({prev}, {num_classes})\n"
    params += prev * num_classes + num_classes
    code += "    def forward(self, x):\n        for layer in self.convs:\n            x = layer(x)\n        x = self.pool(x).flatten(1)\n        return self.fc(x)\n"
    return code, params, in_channels, num_classes, 'cnn'


# ── DenseCNN (DenseNet-style) ──

def make_densecnn(in_channels: int, growth_rate: int, num_layers: int, num_classes: int = 10):
    """Dense CNN: each layer receives all previous feature maps."""
    params = 0
    code = f"class M{{}}(nn.Module):\n    def __init__(self):\n        super().__init__()\n"
    code += f"        self.stem = nn.Conv2d({in_channels}, {growth_rate*2}, 3, padding=1)\n"
    params += in_channels * growth_rate * 2 * 9 + growth_rate * 2
    prev_total = growth_rate * 2
    code += "        self.layers = nn.ModuleList()\n"
    for i in range(num_layers):
        code += f"        self.layers.append(nn.Sequential(\n"
        code += f"            nn.BatchNorm2d({prev_total}), nn.ReLU(),\n"
        code += f"            nn.Conv2d({prev_total}, {growth_rate}, 3, padding=1)\n"
        code += f"        ))\n"
        params += prev_total * 2 + growth_rate + prev_total * growth_rate * 9 + growth_rate
        prev_total += growth_rate
    code += "        self.pool = nn.AdaptiveAvgPool2d(1)\n"
    code += f"        self.fc = nn.Linear({prev_total}, {num_classes})\n"
    params += prev_total * num_classes + num_classes
    code += "    def forward(self, x):\n        x = self.stem(x)\n"
    code += "        for layer in self.layers:\n            out = layer(x)\n            x = torch.cat([x, out], dim=1)\n"
    code += "        x = self.pool(x).flatten(1)\n        return self.fc(x)\n"
    return code, params, in_channels, num_classes, 'cnn'


# ── AttnLSTM (LSTM + multi-head self-attention pooling) ──

def make_attnlstm(input_size: int, hidden_size: int, num_layers: int,
                  num_heads: int, num_classes: int = 10):
    """LSTM with multi-head self-attention pooling over hidden states."""
    lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    lstm_p = sum(p.numel() for p in lstm.parameters())
    # Multi-head attention: Q, K, V projections
    attn_p = 3 * hidden_size * hidden_size + 3 * hidden_size
    fc_p = hidden_size * num_classes + num_classes
    params = lstm_p + attn_p + fc_p

    code = (f"class M{{}}(nn.Module):\n"
            f"    def __init__(self):\n        super().__init__()\n"
            f"        self.lstm = nn.LSTM({input_size}, {hidden_size}, {num_layers}, batch_first=True)\n"
            f"        self.attn = nn.MultiheadAttention({hidden_size}, {num_heads}, batch_first=True)\n"
            f"        self.fc = nn.Linear({hidden_size}, {num_classes})\n"
            f"    def forward(self, x):\n"
            f"        out, _ = self.lstm(x)\n"
            f"        attn_out, _ = self.attn(out, out, out)\n"
            f"        pooled = attn_out.mean(dim=1)\n"
            f"        return self.fc(pooled)\n")
    return code, params, input_size, num_classes, 'rnn'


# ── SelfAttn (pure self-attention, no FFN — lightweight) ──

def make_selfattn(d_model: int, num_layers: int, num_heads: int, num_classes: int = 10):
    """Stacked self-attention layers (no feed-forward, minimal version)."""
    attn_per_layer = 3 * d_model * d_model + 3 * d_model  # Q,K,V projections
    params = attn_per_layer * num_layers + d_model * num_classes + num_classes
    code = (f"class M{{}}(nn.Module):\n"
            f"    def __init__(self):\n        super().__init__()\n"
            f"        self.layers = nn.ModuleList()\n")
    for _ in range(num_layers):
        code += f"        self.layers.append(nn.MultiheadAttention({d_model}, {num_heads}, batch_first=True))\n"
    code += (f"        self.fc = nn.Linear({d_model}, {num_classes})\n"
             f"    def forward(self, x):\n"
             f"        if x.dim() == 2: x = x.unsqueeze(1)\n"
             f"        for layer in self.layers:\n"
             f"            x, _ = layer(x, x, x)\n"
             f"        return self.fc(x.mean(dim=1))\n")
    return code, params, d_model, num_classes, 'ce'


# ── GCN (Graph Convolutional Network) ──

def make_gcn(in_features: int, hidden_dim: int, num_classes: int = 10):
    """2-layer GCN: GCN(in→hidden) → ReLU → GCN(hidden→out)."""
    params = in_features * hidden_dim + hidden_dim + hidden_dim * num_classes + num_classes
    code = (f"class M{{}}(nn.Module):\n"
            f"    def __init__(self):\n        super().__init__()\n"
            f"        self.conv1 = nn.Linear({in_features}, {hidden_dim})\n"
            f"        self.conv2 = nn.Linear({hidden_dim}, {num_classes})\n"
            f"    def forward(self, x, adj):\n"
            f"        x = torch.relu(adj @ self.conv1(x))\n"
            f"        return adj @ self.conv2(x)\n")
    return code, params, in_features, num_classes, 'gcn'


# ═══════════════════════════════════════════════════
#  Large-tier architectures (≥ 10M params)
# ═══════════════════════════════════════════════════

# ── ViT (Vision Transformer) ──

def make_vit(patch_size: int, d_model: int, num_layers: int, num_heads: int,
             image_size: int = 32, num_classes: int = 10):
    """Vision Transformer: patch embedding + positional encoding + Transformer encoder."""
    num_patches = (image_size // patch_size) ** 2
    patch_dim = patch_size * patch_size * 3  # assume RGB
    patch_emb_p = patch_dim * d_model + d_model
    pos_emb_p = num_patches * d_model
    # Single TransformerEncoderLayer: self-attn + FFN
    layer_p = (4 * d_model * d_model + 4 * d_model) + (d_model * d_model * 8 + d_model * 4 + d_model)
    total_layer_p = layer_p * num_layers
    cls_head_p = d_model * num_classes + num_classes
    params = patch_emb_p + pos_emb_p + total_layer_p + cls_head_p

    code = (f"class M{{}}(nn.Module):\n"
            f"    def __init__(self):\n        super().__init__()\n"
            f"        self.patch_size = {patch_size}\n"
            f"        self.num_patches = {num_patches}\n"
            f"        self.patch_emb = nn.Linear({patch_dim}, {d_model})\n"
            f"        self.pos_emb = nn.Parameter(torch.randn(1, {num_patches}, {d_model}))\n"
            f"        self.encoder = nn.TransformerEncoder(\n"
            f"            nn.TransformerEncoderLayer(d_model={d_model}, nhead={num_heads}, batch_first=True),\n"
            f"            num_layers={num_layers}\n"
            f"        )\n"
            f"        self.cls_token = nn.Parameter(torch.randn(1, 1, {d_model}))\n"
            f"        self.fc = nn.Linear({d_model}, {num_classes})\n"
            f"    def forward(self, x):\n"
            f"        B, C, H, W = x.shape\n"
            f"        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)\n"
            f"        x = x.permute(0, 2, 3, 1, 4, 5).reshape(B, -1, C*self.patch_size*self.patch_size)\n"
            f"        x = self.patch_emb(x) + self.pos_emb\n"
            f"        cls = self.cls_token.expand(B, -1, -1)\n"
            f"        x = torch.cat([cls, x], dim=1)\n"
            f"        x = self.encoder(x)\n"
            f"        return self.fc(x[:, 0, :])\n")
    return code, params, 3, num_classes, 'cnn'


# ── UNet (encoder-decoder with skip connections) ──

def make_unet(in_channels: int, base_ch: int, num_stages: int, num_classes: int = 10):
    """U-Net: encoder-decoder with skip connections."""
    params = 0
    code = (f"class M{{}}(nn.Module):\n    def __init__(self):\n        super().__init__()\n"
            f"        self.enc_blocks = nn.ModuleList()\n"
            f"        self.pools = nn.ModuleList()\n"
            f"        self.dec_blocks = nn.ModuleList()\n"
            f"        self.ups = nn.ModuleList()\n")

    enc_chs = []
    prev_ch = in_channels
    ch = base_ch
    for i in range(num_stages):
        enc_chs.append(ch)
        code += f"        self.enc_blocks.append(nn.Sequential(\n"
        code += f"            nn.Conv2d({prev_ch}, {ch}, 3, padding=1), nn.ReLU(),\n"
        code += f"            nn.Conv2d({ch}, {ch}, 3, padding=1), nn.ReLU()\n"
        code += f"        ))\n"
        params += prev_ch * ch * 9 + ch + ch * ch * 9 + ch
        code += f"        self.pools.append(nn.MaxPool2d(2))\n"
        prev_ch = ch
        ch *= 2

    # Bottleneck
    code += f"        self.bottleneck = nn.Sequential(\n"
    code += f"            nn.Conv2d({prev_ch}, {ch}, 3, padding=1), nn.ReLU(),\n"
    code += f"            nn.Conv2d({ch}, {ch}, 3, padding=1), nn.ReLU()\n"
    code += f"        ))\n"
    params += prev_ch * ch * 9 + ch + ch * ch * 9 + ch
    prev_ch = ch

    # Decoder
    for i in range(num_stages - 1, -1, -1):
        skip_ch = enc_chs[i]
        ch = skip_ch
        code += f"        self.ups.append(nn.ConvTranspose2d({prev_ch}, {ch}, 2, stride=2))\n"
        params += prev_ch * ch * 4 + ch
        code += f"        self.dec_blocks.append(nn.Sequential(\n"
        code += f"            nn.Conv2d({ch + skip_ch}, {ch}, 3, padding=1), nn.ReLU(),\n"
        code += f"            nn.Conv2d({ch}, {ch}, 3, padding=1), nn.ReLU()\n"
        code += f"        ))\n"
        params += (ch + skip_ch) * ch * 9 + ch + ch * ch * 9 + ch
        prev_ch = ch

    code += f"        self.out_conv = nn.Conv2d({prev_ch}, {num_classes}, 1)\n"
    params += prev_ch * num_classes + num_classes

    code += ("    def forward(self, x):\n"
             "        skips = []\n"
             "        for i, (enc, pool) in enumerate(zip(self.enc_blocks, self.pools)):\n"
             "            x = enc(x)\n"
             "            skips.append(x)\n"
             "            x = pool(x)\n"
             "        x = self.bottleneck(x)\n"
             "        for i, (up, dec) in enumerate(zip(self.ups, self.dec_blocks)):\n"
             "            x = up(x)\n"
             "            skip = skips[-1 - i]\n"
             "            x = torch.cat([x, skip], dim=1)\n"
             "            x = dec(x)\n"
             "        return self.out_conv(x)\n")
    return code, params, in_channels, num_classes, 'cnn'


# ── Mixer (MLP-Mixer) ──

def make_mixer(patch_size: int, d_model: int, num_layers: int,
               image_size: int = 32, num_classes: int = 10):
    """MLP-Mixer: patch embedding → alternating token-mix and channel-mix MLPs."""
    num_patches = (image_size // patch_size) ** 2
    patch_dim = patch_size * patch_size * 3
    emb_p = patch_dim * d_model + d_model
    # Per layer: token-mix MLP (2 linear across patches) + channel-mix MLP (2 linear across dims)
    token_mix_p = num_patches * num_patches * 2 + num_patches * 2
    channel_mix_p = d_model * d_model * 2 + d_model * 2
    layer_p = token_mix_p + channel_mix_p
    total_p = layer_p * num_layers
    head_p = d_model * num_classes + num_classes
    params = emb_p + total_p + head_p

    code = (f"class M{{}}(nn.Module):\n"
            f"    def __init__(self):\n        super().__init__()\n"
            f"        self.patch_emb = nn.Linear({patch_dim}, {d_model})\n"
            f"        self.mixers = nn.ModuleList()\n")
    for _ in range(num_layers):
        code += f"        self.mixers.append(nn.Sequential(\n"
        code += f"            nn.LayerNorm({d_model}),\n"
        code += f"            nn.Linear({num_patches}, {num_patches}), nn.GELU(),\n"
        code += f"            nn.Linear({num_patches}, {num_patches}),\n"
        code += f"            nn.LayerNorm({d_model}),\n"
        code += f"            nn.Linear({d_model}, {d_model}), nn.GELU(),\n"
        code += f"            nn.Linear({d_model}, {d_model})\n"
        code += f"        ))\n"
    code += f"        self.fc = nn.Linear({d_model}, {num_classes})\n"
    code += (f"    def forward(self, x):\n"
             f"        B, C, H, W = x.shape\n"
             f"        x = x.unfold(2, {patch_size}, {patch_size}).unfold(3, {patch_size}, {patch_size})\n"
             f"        x = x.permute(0, 2, 3, 1, 4, 5).reshape(B, -1, C*{patch_size}*{patch_size})\n"
             f"        x = self.patch_emb(x)\n"
             f"        for mixer in self.mixers:\n"
             f"            x = x + mixer[:4](x.permute(0, 2, 1)).permute(0, 2, 1)\n"
             f"            x = x + mixer[4:](x)\n"
             f"        return self.fc(x.mean(dim=1))\n")
    return code, params, 3, num_classes, 'cnn'


# ── GPT (small decoder-only transformer) ──

def make_gpt(vocab_size: int, d_model: int, num_layers: int, num_heads: int,
             block_size: int = 128):
    """Small GPT-style decoder: token + position embedding → Transformer decoder → LM head."""
    tok_emb_p = vocab_size * d_model
    pos_emb_p = block_size * d_model
    # Decoder layer: masked MHA + FFN
    layer_p = (4 * d_model * d_model + 4 * d_model) + (d_model * d_model * 8 + d_model * 4 + d_model)
    layers_p = layer_p * num_layers
    lm_head_p = d_model * vocab_size + vocab_size
    params = tok_emb_p + pos_emb_p + layers_p + lm_head_p

    code = (f"class M{{}}(nn.Module):\n"
            f"    def __init__(self):\n        super().__init__()\n"
            f"        self.tok_emb = nn.Embedding({vocab_size}, {d_model})\n"
            f"        self.pos_emb = nn.Embedding({block_size}, {d_model})\n"
            f"        self.decoder = nn.TransformerDecoder(\n"
            f"            nn.TransformerDecoderLayer(d_model={d_model}, nhead={num_heads}, batch_first=True),\n"
            f"            num_layers={num_layers}\n"
            f"        )\n"
            f"        self.lm_head = nn.Linear({d_model}, {vocab_size})\n"
            f"    def forward(self, x):\n"
            f"        B, T = x.shape\n"
            f"        pos = torch.arange(T, device=x.device).unsqueeze(0)\n"
            f"        x = self.tok_emb(x) + self.pos_emb(pos)\n"
            f"        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)\n"
            f"        x = self.decoder(x, x, tgt_mask=mask)\n"
            f"        return self.lm_head(x)\n")
    return code, params, vocab_size, vocab_size, 'ce'


# ── T5 (Encoder-Decoder Transformer) ──

def make_t5(vocab_size: int, d_model: int, num_layers: int, num_heads: int,
            num_classes: int = 10):
    """Encoder-decoder transformer (T5-style)."""
    tok_emb_p = vocab_size * d_model * 2  # shared embedding
    # Encoder layer + Decoder layer
    enc_layer_p = (4 * d_model * d_model + 4 * d_model) + (d_model * d_model * 8 + d_model * 4 + d_model)
    dec_layer_p = enc_layer_p + (4 * d_model * d_model + 4 * d_model)  # + cross-attention
    params = tok_emb_p + enc_layer_p * num_layers + dec_layer_p * num_layers + d_model * num_classes + num_classes

    code = (f"class M{{}}(nn.Module):\n"
            f"    def __init__(self):\n        super().__init__()\n"
            f"        self.tok_emb = nn.Embedding({vocab_size}, {d_model})\n"
            f"        self.encoder = nn.TransformerEncoder(\n"
            f"            nn.TransformerEncoderLayer(d_model={d_model}, nhead={num_heads}, batch_first=True),\n"
            f"            num_layers={num_layers}\n"
            f"        )\n"
            f"        self.decoder = nn.TransformerDecoder(\n"
            f"            nn.TransformerDecoderLayer(d_model={d_model}, nhead={num_heads}, batch_first=True),\n"
            f"            num_layers={num_layers}\n"
            f"        )\n"
            f"        self.fc = nn.Linear({d_model}, {num_classes})\n"
            f"    def forward(self, x):\n"
            f"        mem = self.encoder(self.tok_emb(x))\n"
            f"        out = self.decoder(self.tok_emb(x), mem)\n"
            f"        return self.fc(out.mean(dim=1))\n")
    return code, params, d_model, num_classes, 'ce'
