"""Production-tier templates (> 50M params).

Adds: DDP multi-GPU, mixed precision (AMP), gradient accumulation, validation split,
      model/ sub-package, configs/ YAML, scripts/ (benchmark, profile, export),
      TensorBoard logging.
"""
import os as _os


def get_templates(model_type):
    """Returns (data_code, train_code, eval_code) for production tier."""
    from . import quick
    data, _, eval_ = quick.get_templates(model_type)
    train = _build_train(model_type)
    return data, train, eval_


def _build_train(model_type: str) -> str:
    """Build production-tier train.py with DDP, AMP, grad accumulation."""
    header = (
        '"""Production training script — DDP, AMP, gradient accumulation."""\n'
        'import torch,torch.nn as nn,os,argparse\n'
        'from torch.utils.data import DataLoader,random_split\n'
        'from config import *\n'
        'from model import {cn}\n'
        'from data import SynData\n\n'
        '# ── CLI overrides ──\n'
        'parser=argparse.ArgumentParser()\n'
        'parser.add_argument("--lr",type=float)\n'
        'parser.add_argument("--epochs",type=int)\n'
        'parser.add_argument("--batch-size",type=int,dest="batch_size")\n'
        'parser.add_argument("--local_rank",type=int,default=-1)\n'
        'parser.add_argument("--grad-accum",type=int,default=1,dest="grad_accum")\n'
        'parser.add_argument("--fp16",action="store_true",default=False)\n'
        'parser.add_argument("--device",type=str,default=None)\n'
        'opts=parser.parse_args()\n'
        'if opts.lr is not None: LR=opts.lr\n'
        'if opts.epochs is not None: EPOCHS=opts.epochs\n'
        'if opts.batch_size is not None: BATCH_SIZE=opts.batch_size\n'
        'GRAD_ACCUM=opts.grad_accum;USE_AMP=opts.fp16\n'
        'LOCAL_RANK=opts.local_rank\n'
        'if opts.device is not None:\n'
        '    DEVICE_PRIORITY=[d.strip().lower() for d in opts.device.split(",") if d.strip()]\n'
        '    DEVICE=resolve_device()\n\n'
        '# ── DDP setup ──\n'
        'if LOCAL_RANK!=-1:\n'
        '    torch.cuda.set_device(LOCAL_RANK)\n'
        '    torch.distributed.init_process_group(backend="nccl")\n'
        '    DEVICE=torch.device(f"cuda:{LOCAL_RANK}")\n'
        '    IS_MAIN=(LOCAL_RANK==0)\n'
        'else:\n'
        '    # DEVICE already resolved from config DEVICE_PRIORITY (cuda/mps/cpu priority list)\n'
        '    DEVICE=DEVICE\n'
        '    IS_MAIN=True\n\n'
    )

    body = (
        '# ── Data ──\n'
        'ds=SynData()\n'
        'n_val=int(len(ds)*0.2)\n'
        'n_train=len(ds)-n_val\n'
        'train_ds,val_ds=random_split(ds,[n_train,n_val],generator=torch.Generator().manual_seed(42))\n'
        'train_lo=DataLoader(train_ds,BATCH_SIZE,shuffle=True,num_workers=4,pin_memory=True)\n'
        'val_lo=DataLoader(val_ds,BATCH_SIZE*2,shuffle=False,num_workers=2,pin_memory=True)\n\n'
        '# ── Model ──\n'
        'm={cn}().to(DEVICE)\n'
        'if LOCAL_RANK!=-1:\n'
        '    m=nn.parallel.DistributedDataParallel(m,device_ids=[LOCAL_RANK])\n'
        'total_params=sum(p.numel() for p in m.parameters())\n'
        'if IS_MAIN: print(f"Model: {total_params} parameters, Device: {DEVICE}")\n\n'
        '# ── Optimizer & scaler ──\n'
        'o=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=0.01)\n'
        'scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(o,T_0=10,T_mult=2)\n'
        'scaler=torch.cuda.amp.GradScaler() if USE_AMP else None\n'
        'os.makedirs("checkpoints",exist_ok=True)\n'
        'os.makedirs("logs",exist_ok=True)\n'
        'best_loss=float("inf");patience_counter=0;PATIENCE=20\n'
        'history=[];global_step=0\n'
    )

    if model_type in ('ce', 'cnn', 'rnn'):
        train_loop = (
            'criterion=nn.CrossEntropyLoss()\n'
            'for e in range(EPOCHS):\n'
            '    m.train()\n'
            '    train_loss=0;train_correct=0;train_n=0\n'
            '    for i,(x,y) in enumerate(train_lo):\n'
            '        x,y=x.to(DEVICE),y.to(DEVICE)\n'
            '        if USE_AMP:\n'
            '            with torch.cuda.amp.autocast():\n'
            '                out=m(x);loss=criterion(out,y)\n'
            '            loss=loss/GRAD_ACCUM\n'
            '            scaler.scale(loss).backward()\n'
            '        else:\n'
            '            out=m(x);loss=criterion(out,y)/GRAD_ACCUM\n'
            '            loss.backward()\n'
            '        if (i+1)%GRAD_ACCUM==0:\n'
            '            if USE_AMP:\n'
            '                scaler.unscale_(o);nn.utils.clip_grad_norm_(m.parameters(),1.0)\n'
            '                scaler.step(o);scaler.update()\n'
            '            else:\n'
            '                nn.utils.clip_grad_norm_(m.parameters(),1.0);o.step()\n'
            '            o.zero_grad()\n'
            '        train_loss+=loss.item()*GRAD_ACCUM*x.size(0);train_correct+=(out.argmax(1)==y).sum().item()\n'
            '        train_n+=x.size(0);global_step+=1\n'
            '    train_loss/=train_n;train_acc=train_correct/train_n\n'
            '    # Validation\n'
            '    m.eval()\n'
            '    val_loss=0;val_correct=0;val_n=0\n'
            '    with torch.no_grad():\n'
            '        for x,y in val_lo:\n'
            '            x,y=x.to(DEVICE),y.to(DEVICE)\n'
            '            out=m(x);val_loss+=criterion(out,y).item()*x.size(0)\n'
            '            val_correct+=(out.argmax(1)==y).sum().item();val_n+=x.size(0)\n'
            '    val_loss/=val_n;val_acc=val_correct/val_n\n'
            '    scheduler.step()\n'
            '    history.append((e,train_loss,train_acc,val_loss,val_acc))\n'
            '    if IS_MAIN:\n'
            '        print(f"Epoch {e:3d}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")\n'
            '    if val_loss<best_loss:\n'
            '        best_loss=val_loss;patience_counter=0\n'
            '        if IS_MAIN: torch.save(m.state_dict(),"best_model.pth")\n'
            '    else:\n'
            '        patience_counter+=1\n'
            '    if IS_MAIN and e%SAVE_EVERY==0:\n'
            '        torch.save({"epoch":e,"model":m.state_dict(),"opt":o.state_dict()},f"checkpoints/ckpt_{e:04d}.pth")\n'
            '    if patience_counter>=PATIENCE:\n'
            '        if IS_MAIN: print(f"Early stopping at epoch {e}");break\n'
            'if IS_MAIN: torch.save(m.state_dict(),"model.pth")\n'
            + _FOOTER_PROD_CLS
        )
    elif model_type == 'mse':
        train_loop = (
            '_losses={"mse":nn.MSELoss,"l1":nn.L1Loss,"smooth_l1":nn.SmoothL1Loss}\n'
            'criterion=_losses.get(LOSS_TYPE,nn.MSELoss)()\n'
            'for e in range(EPOCHS):\n'
            '    m.train()\n'
            '    train_loss=0;train_n=0\n'
            '    for i,(x,y) in enumerate(train_lo):\n'
            '        x,y=x.to(DEVICE),y.to(DEVICE)\n'
            '        if USE_AMP:\n'
            '            with torch.cuda.amp.autocast():\n'
            '                loss=criterion(m(x),y)\n'
            '            loss=loss/GRAD_ACCUM;scaler.scale(loss).backward()\n'
            '        else:\n'
            '            loss=criterion(m(x),y)/GRAD_ACCUM;loss.backward()\n'
            '        if (i+1)%GRAD_ACCUM==0:\n'
            '            if USE_AMP: scaler.unscale_(o);nn.utils.clip_grad_norm_(m.parameters(),1.0);scaler.step(o);scaler.update()\n'
            '            else: nn.utils.clip_grad_norm_(m.parameters(),1.0);o.step()\n'
            '            o.zero_grad()\n'
            '        train_loss+=loss.item()*GRAD_ACCUM*x.size(0);train_n+=x.size(0);global_step+=1\n'
            '    train_loss/=train_n\n'
            '    m.eval()\n'
            '    val_loss=0;val_n=0\n'
            '    with torch.no_grad():\n'
            '        for x,y in val_lo:\n'
            '            x,y=x.to(DEVICE),y.to(DEVICE)\n'
            '            val_loss+=criterion(m(x),y).item()*x.size(0);val_n+=x.size(0)\n'
            '    val_loss/=val_n\n'
            '    scheduler.step()\n'
            '    history.append((e,train_loss,val_loss))\n'
            '    if IS_MAIN: print(f"Epoch {e:3d}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")\n'
            '    if val_loss<best_loss:\n        best_loss=val_loss;patience_counter=0\n        if IS_MAIN: torch.save(m.state_dict(),"best_model.pth")\n'
            '    else:\n        patience_counter+=1\n'
            '    if IS_MAIN and e%SAVE_EVERY==0:\n        torch.save({"epoch":e,"model":m.state_dict()},f"checkpoints/ckpt_{e:04d}.pth")\n'
            '    if patience_counter>=PATIENCE:\n        if IS_MAIN: print(f"Early stopping at epoch {e}");break\n'
            'if IS_MAIN: torch.save(m.state_dict(),"model.pth")\n'
            + _FOOTER_PROD_REG
        )
    else:
        # Fallback to Standard-tier train for gan/contrastive/siamese/ae/vae/mt/gcn
        from . import standard
        return standard._build_train(model_type)

    return header + body + train_loop


_FOOTER_PROD_CLS = """
# ── Save training log ──
if IS_MAIN:
    md = '# Training Log (Production)\\n\\n'
    md += f'**Model**: {total_params} parameters  \\n'
    md += f'**Dataset**: {DATASET}  \\n'
    md += f'**Epochs**: {len(history)}  \\n'
    md += f'**Batch Size**: {BATCH_SIZE} × {GRAD_ACCUM} grad-accum  \\n'
    md += f'**Learning Rate**: {LR}  \\n'
    md += f'**Best Val Loss**: {best_loss:.4f}  \\n\\n'
    md += '| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |\\n'
    md += '|-------|------------|-----------|----------|--------|\\n'
    for e, tl, ta, vl, va in history:
        md += f'| {e:5d} | {tl:.4f} | {ta:.4f} | {vl:.4f} | {va:.4f} |\\n'
    with open('training_log.md', 'w') as f:
        f.write(md)
    print('Saved training_log.md')
    print(f'Best model saved as best_model.pth (val_loss={best_loss:.4f})')
"""

_FOOTER_PROD_REG = """
if IS_MAIN:
    md = '# Training Log (Production)\\n\\n'
    md += f'**Model**: {total_params} parameters  \\n'
    md += f'**Dataset**: {DATASET}  \\n'
    md += f'**Epochs**: {len(history)}  \\n'
    md += f'**Batch Size**: {BATCH_SIZE} × {GRAD_ACCUM} grad-accum  \\n'
    md += f'**Learning Rate**: {LR}  \\n\\n'
    md += '| Epoch | Train Loss | Val Loss |\\n'
    md += '|-------|------------|----------|\\n'
    for e, tl, vl in history:
        md += f'| {e:5d} | {tl:.4f} | {vl:.4f} |\\n'
    with open('training_log.md', 'w') as f:
        f.write(md)
    print('Saved training_log.md')
"""


# ── Extra files: model sub-package, configs, scripts ──

def get_extra_files(model_type: str, class_name: str) -> dict:
    """Return {relative_path: content} for production-tier extras."""
    return {
        'model/__init__.py': _gen_model_init(class_name),
        'model/layers.py': '# Sub-layers for large models (customize as needed)\n',
        'configs/default.yaml': _gen_yaml('default', 'base'),
        'configs/large.yaml': _gen_yaml('large', '2x'),
        'scripts/benchmark.py': _gen_benchmark(class_name),
        'scripts/profile.py': _gen_profile(class_name),
        'scripts/export.py': _gen_export(class_name),
    }


def _gen_model_init(class_name: str) -> str:
    return (
        f'"""Modular model — import sub-components from layers.py."""\n'
        f'import torch.nn as nn\n'
        f'from .layers import *\n\n'
        f'class {class_name}(nn.Module):\n'
        f'    def __init__(self):\n'
        f'        super().__init__()\n'
        f'        # TODO: define model using layer components\n'
        f'        self.net = nn.Sequential(\n'
        f'            nn.Linear(1, 1)  # placeholder\n'
        f'        )\n'
        f'    def forward(self, x):\n'
        f'        return self.net(x)\n'
    )


def _gen_yaml(name: str, description: str) -> str:
    return (
        f'# {name} config\n'
        f'description: "{description}"\n'
        f'LR: 0.001\n'
        f'EPOCHS: 100\n'
        f'BATCH_SIZE: 64\n'
        f'GRAD_ACCUM: 1\n'
    )


def _gen_benchmark(class_name: str) -> str:
    return (
        '"""Inference latency & throughput benchmark."""\n'
        'import torch,time\n'
        f'from model import {class_name}\n'
        'from config import INPUT_DIM\n\n'
        f'm = {class_name}().eval()\n'
        'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n'
        'm = m.to(device)\n\n'
        '# Warmup\n'
        'for _ in range(10):\n'
        '    _ = m(torch.randn(1, INPUT_DIM).to(device))\n'
        'if device.type == "cuda": torch.cuda.synchronize()\n\n'
        '# Latency (batch=1)\n'
        'N = 100\n'
        'x = torch.randn(1, INPUT_DIM).to(device)\n'
        't0 = time.perf_counter()\n'
        'for _ in range(N):\n'
        '    _ = m(x)\n'
        'if device.type == "cuda": torch.cuda.synchronize()\n'
        'latency = (time.perf_counter() - t0) / N * 1000\n'
        'print(f"Latency (batch=1): {latency:.2f} ms")\n\n'
        '# Throughput (batch=64)\n'
        'x = torch.randn(64, INPUT_DIM).to(device)\n'
        't0 = time.perf_counter()\n'
        'for _ in range(N):\n'
        '    _ = m(x)\n'
        'if device.type == "cuda": torch.cuda.synchronize()\n'
        'throughput = 64 * N / (time.perf_counter() - t0)\n'
        'print(f"Throughput (batch=64): {throughput:.0f} samples/sec")\n'
    )


def _gen_profile(class_name: str) -> str:
    return (
        '"""FLOPs & memory profiling (requires torch>=2.0)."""\n'
        'import torch\n'
        f'from model import {class_name}\n'
        'from config import INPUT_DIM\n\n'
        f'm = {class_name}()\n'
        'x = torch.randn(1, INPUT_DIM)\n\n'
        'print(f"Parameters: {sum(p.numel() for p in m.parameters()):,}")\n'
        'try:\n'
        '    from torch.profiler import profile, ProfilerActivity\n'
        '    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:\n'
        '        _ = m(x)\n'
        '    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))\n'
        'except ImportError:\n'
        '    print("torch.profiler not available (need PyTorch >= 1.8)")\n'
    )


def _gen_export(class_name: str) -> str:
    return (
        '"""Export model to ONNX / TorchScript."""\n'
        'import torch\n'
        f'from model import {class_name}\n'
        'from config import INPUT_DIM\n\n'
        f'm = {class_name}()\n'
        f"m.load_state_dict(torch.load('best_model.pth', weights_only=True))\n"
        'm.eval()\n'
        'x = torch.randn(1, INPUT_DIM)\n\n'
        '# ONNX export\n'
        'try:\n'
        '    torch.onnx.export(m, x, "model.onnx",\n'
        '                      input_names=["input"], output_names=["output"],\n'
        '                      dynamic_axes={"input": {0: "batch"})\n'
        '    print("Exported model.onnx")\n'
        'except Exception as e:\n'
        '    print(f"ONNX export failed: {e}")\n\n'
        '# TorchScript export\n'
        'try:\n'
        '    scripted = torch.jit.script(m)\n'
        '    scripted.save("model_scripted.pt")\n'
        '    print("Exported model_scripted.pt")\n'
        'except Exception as e:\n'
        '    print(f"TorchScript export failed: {e}")\n'
    )
