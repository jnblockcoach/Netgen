"""Standard-tier templates (50K ~ 50M params).

Adds: lr scheduler, early stopping, checkpoint save/resume, gradient clipping,
      sweep.py (grid search), visualize.py (reads training_log.md),
      predict.py (batch inference).
"""

import os as _os


def get_templates(model_type):
    """Returns (data_code, train_code, eval_code) for the standard tier."""
    # Data and eval templates are identical to quick tier — reuse them
    from . import quick
    data, _, eval_ = quick.get_templates(model_type)

    # Build upgraded train code based on model type
    train = _build_train(model_type)

    return data, train, eval_


def _build_train(model_type: str) -> str:
    """Build standard-tier train.py with lr scheduler, early stopping, checkpoints."""
    prefix = "import torch,torch.nn as nn,os\nfrom config import *\nfrom model import {cn}\nfrom data import SynData\n"
    common_mid = (
        "ds=SynData();lo=torch.utils.data.DataLoader(ds,BATCH_SIZE,shuffle=True)\n"
        "m={cn}();total_params=sum(p.numel() for p in m.parameters())\n"
        "print(f'Model: {total_params} parameters')\n"
        "o=torch.optim.Adam(m.parameters(),lr=LR)\n"
        "scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(o,mode='min',patience=5,factor=0.5)\n"
        "os.makedirs('checkpoints',exist_ok=True)\n"
        "best_loss=float('inf');patience_counter=0;PATIENCE=10\n"
        "history=[]\n"
    )

    if model_type in ('ce', 'cnn'):
        specific = (
            "criterion = {'ce': lambda: nn.CrossEntropyLoss(),'ce_smooth': lambda: nn.CrossEntropyLoss(label_smoothing=0.1)}.get(LOSS_TYPE, nn.CrossEntropyLoss)()\n"
            "for e in range(EPOCHS):\n"
            "    for x,y in lo:\n        l=criterion(m(x),y)\n        o.zero_grad();l.backward()\n"
            "        nn.utils.clip_grad_norm_(m.parameters(),1.0)\n        o.step()\n"
            "    with torch.no_grad():\n"
            "        loss_val=sum(criterion(m(x),y).item()*x.size(0) for x,y in lo)/len(ds)\n"
            "        correct=sum((m(x).argmax(1)==y).sum().item() for x,y in lo)\n"
            "        acc=correct/len(ds)\n"
            "    scheduler.step(loss_val)\n"
            "    history.append((e,loss_val,acc))\n"
            "    print(f'Epoch {e:3d}: loss={loss_val:.4f}, acc={acc:.4f}')\n"
            "    if loss_val<best_loss:\n        best_loss=loss_val;patience_counter=0\n"
            "        torch.save(m.state_dict(),'best_model.pth')\n"
            "    else:\n        patience_counter+=1\n"
            "    if e%SAVE_EVERY==0:\n        torch.save({'epoch':e,'model':m.state_dict(),'opt':o.state_dict()},f'checkpoints/ckpt_{e:04d}.pth')\n"
            "    if patience_counter>=PATIENCE:\n        print(f'Early stopping at epoch {e}');break\n"
            "torch.save(m.state_dict(),'model.pth')\n"
            + _FOOTER_STD_CLS
        )
    elif model_type == 'mse':
        specific = (
            "_losses = {'mse': nn.MSELoss, 'l1': nn.L1Loss, 'smooth_l1': nn.SmoothL1Loss}\n"
            "criterion = _losses.get(LOSS_TYPE, nn.MSELoss)()\n"
            "for e in range(EPOCHS):\n"
            "    for x,y in lo:\n        l=criterion(m(x),y)\n        o.zero_grad();l.backward()\n"
            "        nn.utils.clip_grad_norm_(m.parameters(),1.0)\n        o.step()\n"
            "    with torch.no_grad():\n"
            "        loss_val=sum(criterion(m(x),y).item()*x.size(0) for x,y in lo)/len(ds)\n"
            "    scheduler.step(loss_val)\n"
            "    history.append((e,loss_val))\n"
            "    print(f'Epoch {e:3d}: loss={loss_val:.4f}')\n"
            "    if loss_val<best_loss:\n        best_loss=loss_val;patience_counter=0\n"
            "        torch.save(m.state_dict(),'best_model.pth')\n"
            "    else:\n        patience_counter+=1\n"
            "    if e%SAVE_EVERY==0:\n        torch.save({'epoch':e,'model':m.state_dict(),'opt':o.state_dict()},f'checkpoints/ckpt_{e:04d}.pth')\n"
            "    if patience_counter>=PATIENCE:\n        print(f'Early stopping at epoch {e}');break\n"
            "torch.save(m.state_dict(),'model.pth')\n"
            + _FOOTER_STD_REG
        )
    elif model_type == 'rnn':
        specific = (
            "criterion=nn.CrossEntropyLoss()\n"
            "for e in range(EPOCHS):\n"
            "    for x,y in lo:\n        l=criterion(m(x),y)\n        o.zero_grad();l.backward()\n"
            "        nn.utils.clip_grad_norm_(m.parameters(),1.0)\n        o.step()\n"
            "    with torch.no_grad():\n"
            "        loss_val=sum(criterion(m(x),y).item()*x.size(0) for x,y in lo)/len(ds)\n"
            "        correct=sum((m(x).argmax(1)==y).sum().item() for x,y in lo)\n"
            "        acc=correct/len(ds)\n"
            "    scheduler.step(loss_val)\n"
            "    history.append((e,loss_val,acc))\n"
            "    print(f'Epoch {e:3d}: loss={loss_val:.4f}, acc={acc:.4f}')\n"
            "    if loss_val<best_loss:\n        best_loss=loss_val;patience_counter=0\n        torch.save(m.state_dict(),'best_model.pth')\n"
            "    else:\n        patience_counter+=1\n"
            "    if e%SAVE_EVERY==0:\n        torch.save({'epoch':e,'model':m.state_dict(),'opt':o.state_dict()},f'checkpoints/ckpt_{e:04d}.pth')\n"
            "    if patience_counter>=PATIENCE:\n        print(f'Early stopping at epoch {e}');break\n"
            "torch.save(m.state_dict(),'model.pth')\n"
            + _FOOTER_STD_CLS
        )
    elif model_type in ('ae', 'vae'):
        if model_type == 'ae':
            specific = (
                "for e in range(EPOCHS):\n"
                "    for x in lo:\n        r,_=m(x);l=nn.MSELoss()(r,x)\n        o.zero_grad();l.backward()\n"
                "        nn.utils.clip_grad_norm_(m.parameters(),1.0)\n        o.step()\n"
                "    with torch.no_grad():\n"
                "        recon_loss=sum(nn.MSELoss()(m(x)[0],x).item()*x.size(0) for x in lo)/len(ds)\n"
                "    scheduler.step(recon_loss)\n"
                "    history.append((e,recon_loss))\n"
                "    print(f'Epoch {e:3d}: recon_loss={recon_loss:.4f}')\n"
                "    if recon_loss<best_loss:\n        best_loss=recon_loss;patience_counter=0\n        torch.save(m.state_dict(),'best_model.pth')\n"
                "    else:\n        patience_counter+=1\n"
                "    if e%SAVE_EVERY==0:\n        torch.save({'epoch':e,'model':m.state_dict(),'opt':o.state_dict()},f'checkpoints/ckpt_{e:04d}.pth')\n"
                "    if patience_counter>=PATIENCE:\n        print(f'Early stopping at epoch {e}');break\n"
                "torch.save(m.state_dict(),'model.pth')\n"
                + _FOOTER_STD_AE
            )
        else:
            specific = (
                "for e in range(EPOCHS):\n"
                "    for x in lo:\n"
                "        r,mu,lv=m(x);rc=nn.functional.mse_loss(r,x)\n"
                "        kl=-0.5*(1+lv-mu.pow(2)-lv.exp()).sum(dim=1).mean()\n"
                "        l=rc+0.01*kl\n        o.zero_grad();l.backward()\n"
                "        nn.utils.clip_grad_norm_(m.parameters(),1.0)\n        o.step()\n"
                "    with torch.no_grad():\n"
                "        recon_loss=sum(nn.functional.mse_loss(m(x)[0],x).item()*x.size(0) for x in lo)/len(ds)\n"
                "    scheduler.step(recon_loss)\n"
                "    history.append((e,recon_loss))\n"
                "    print(f'Epoch {e:3d}: recon_loss={recon_loss:.4f}')\n"
                "    if recon_loss<best_loss:\n        best_loss=recon_loss;patience_counter=0\n        torch.save(m.state_dict(),'best_model.pth')\n"
                "    else:\n        patience_counter+=1\n"
                "    if e%SAVE_EVERY==0:\n        torch.save({'epoch':e,'model':m.state_dict(),'opt':o.state_dict()},f'checkpoints/ckpt_{e:04d}.pth')\n"
                "    if patience_counter>=PATIENCE:\n        print(f'Early stopping at epoch {e}');break\n"
                "torch.save(m.state_dict(),'model.pth')\n"
                + _FOOTER_STD_AE
            )
    elif model_type == 'mt':
        specific = (
            "for e in range(EPOCHS):\n"
            "    for x,y1,y2 in lo:\n"
            "        o1,o2=m(x)\n"
            "        l=nn.CrossEntropyLoss()(o1,y1)+nn.CrossEntropyLoss()(o2,y2)\n"
            "        o.zero_grad();l.backward()\n"
            "        nn.utils.clip_grad_norm_(m.parameters(),1.0)\n        o.step()\n"
            "    with torch.no_grad():\n"
            "        loss_val=0\n        a1s=a2s=0\n        n=0\n"
            "        for x,y1,y2 in lo:\n"
            "            o1,o2=m(x)\n"
            "            loss_val+=nn.CrossEntropyLoss()(o1,y1).item()*x.size(0)+nn.CrossEntropyLoss()(o2,y2).item()*x.size(0)\n"
            "            a1s+=(o1.argmax(1)==y1).sum().item()\n"
            "            a2s+=(o2.argmax(1)==y2).sum().item()\n"
            "            n+=x.size(0)\n"
            "        loss_val/=n;a1=a1s/n;a2=a2s/n\n"
            "    scheduler.step(loss_val)\n"
            "    history.append((e,loss_val,a1,a2))\n"
            "    print(f'Epoch {e:3d}: loss={loss_val:.4f}, acc1={a1:.4f}, acc2={a2:.4f}')\n"
            "    if loss_val<best_loss:\n        best_loss=loss_val;patience_counter=0\n        torch.save(m.state_dict(),'best_model.pth')\n"
            "    else:\n        patience_counter+=1\n"
            "    if e%SAVE_EVERY==0:\n        torch.save({'epoch':e,'model':m.state_dict(),'opt':o.state_dict()},f'checkpoints/ckpt_{e:04d}.pth')\n"
            "    if patience_counter>=PATIENCE:\n        print(f'Early stopping at epoch {e}');break\n"
            "torch.save(m.state_dict(),'model.pth')\n"
            + _FOOTER_STD_MT
        )
    elif model_type == 'gan':
        specific = (
            "og=torch.optim.Adam(m.generator.parameters(),lr=LR)\n"
            "od=torch.optim.Adam(m.discriminator.parameters(),lr=LR)\n"
            "os.makedirs('checkpoints',exist_ok=True)\n"
            "history=[]\n"
            "for e in range(EPOCHS):\n"
            "    dls=gls=0;n=0\n"
            "    for real in lo:\n"
            "        z=torch.randn(real.size(0),INPUT_DIM)\n"
            "        fake=m.forward_gen(z)\n"
            "        d_real=m.forward_disc(real)\n"
            "        d_fake=m.forward_disc(fake.detach())\n"
            "        ld=nn.BCELoss()(d_real,torch.ones_like(d_real))+nn.BCELoss()(d_fake,torch.zeros_like(d_fake))\n"
            "        od.zero_grad();ld.backward()\n"
            "        nn.utils.clip_grad_norm_(m.discriminator.parameters(),1.0)\n        od.step()\n"
            "        z=torch.randn(real.size(0),INPUT_DIM)\n"
            "        fake=m.forward_gen(z)\n"
            "        d_fake=m.forward_disc(fake)\n"
            "        lg=nn.BCELoss()(d_fake,torch.ones_like(d_fake))\n"
            "        og.zero_grad();lg.backward()\n"
            "        nn.utils.clip_grad_norm_(m.generator.parameters(),1.0)\n        og.step()\n"
            "        dls+=ld.item()*real.size(0);gls+=lg.item()*real.size(0);n+=real.size(0)\n"
            "    dl=dls/n;gl=gls/n\n"
            "    history.append((e,dl,gl))\n"
            "    print(f'Epoch {e:3d}: d_loss={dl:.4f}, g_loss={gl:.4f}')\n"
            "    if e%SAVE_EVERY==0:\n"
            "        torch.save({'epoch':e,'model':m.state_dict()},f'checkpoints/ckpt_{e:04d}.pth')\n"
            "torch.save(m.state_dict(),'model.pth')\n"
            + _FOOTER_STD_GAN
        )
        # GAN has custom init
        prefix = "import torch,torch.nn as nn,os\nfrom config import *\nfrom model import {cn}\nfrom data import SynData\n"
        common_mid = (
            "ds=SynData();lo=torch.utils.data.DataLoader(ds,BATCH_SIZE,shuffle=True)\n"
            "m={cn}();total_params=sum(p.numel() for p in m.parameters())\n"
            "print(f'Model: {total_params} parameters')\n"
        )
    elif model_type == 'contrastive':
        specific = (
            "for e in range(EPOCHS):\n"
            "    epoch_loss=0;n=0\n"
            "    for x1,x2 in lo:\n"
            "        h1=m(x1);h2=m(x2)\n"
            "        h=torch.cat([h1,h2],dim=0)\n"
            "        h=nn.functional.normalize(h,dim=1)\n"
            "        s=h@h.T/0.5\n"
            "        mask=torch.eye(len(h),dtype=torch.bool)\n"
            "        pos=torch.cat([torch.arange(len(h1)),torch.arange(len(h1))]).to(h.device)\n"
            "        s_pos=s[mask].reshape(len(h),-1)\n"
            "        s_neg=s[~mask].reshape(len(h),-1)\n"
            "        l=-torch.log(s_pos.exp()/s_neg.exp().sum(dim=1,keepdim=True)).mean()\n"
            "        o.zero_grad();l.backward()\n"
            "        nn.utils.clip_grad_norm_(m.parameters(),1.0)\n        o.step()\n"
            "        epoch_loss+=l.item()*x1.size(0)\n        n+=x1.size(0)\n"
            "    loss_val=epoch_loss/n\n"
            "    scheduler.step(loss_val)\n"
            "    history.append((e,loss_val))\n"
            "    print(f'Epoch {e:3d}: loss={loss_val:.4f}')\n"
            "    if loss_val<best_loss:\n        best_loss=loss_val;patience_counter=0\n        torch.save(m.state_dict(),'best_model.pth')\n"
            "    else:\n        patience_counter+=1\n"
            "    if e%SAVE_EVERY==0:\n        torch.save({'epoch':e,'model':m.state_dict(),'opt':o.state_dict()},f'checkpoints/ckpt_{e:04d}.pth')\n"
            "    if patience_counter>=PATIENCE:\n        print(f'Early stopping at epoch {e}');break\n"
            "torch.save(m.state_dict(),'model.pth')\n"
            + _FOOTER_STD_REG
        )
    elif model_type == 'siamese':
        specific = (
            "for e in range(EPOCHS):\n"
            "    epoch_loss=0;n=0\n"
            "    for x1,x2,y in lo:\n"
            "        e1=nn.functional.normalize(m(x1),dim=1)\n"
            "        e2=nn.functional.normalize(m(x2),dim=1)\n"
            "        d=torch.norm(e1-e2,dim=1)\n"
            "        l=torch.mean(y*d**2+(1-y)*torch.clamp(1.0-d,min=0)**2)\n"
            "        o.zero_grad();l.backward()\n"
            "        nn.utils.clip_grad_norm_(m.parameters(),1.0)\n        o.step()\n"
            "        epoch_loss+=l.item()*x1.size(0)\n        n+=x1.size(0)\n"
            "    loss_val=epoch_loss/n\n"
            "    scheduler.step(loss_val)\n"
            "    history.append((e,loss_val))\n"
            "    print(f'Epoch {e:3d}: loss={loss_val:.4f}')\n"
            "    if loss_val<best_loss:\n        best_loss=loss_val;patience_counter=0\n        torch.save(m.state_dict(),'best_model.pth')\n"
            "    else:\n        patience_counter+=1\n"
            "    if e%SAVE_EVERY==0:\n        torch.save({'epoch':e,'model':m.state_dict(),'opt':o.state_dict()},f'checkpoints/ckpt_{e:04d}.pth')\n"
            "    if patience_counter>=PATIENCE:\n        print(f'Early stopping at epoch {e}');break\n"
            "torch.save(m.state_dict(),'model.pth')\n"
            + _FOOTER_STD_REG
        )
    elif model_type == 'gcn':
        prefix = "import torch,torch.nn as nn,os\nfrom config import *\nfrom model import {cn}\nfrom data import SynData,build_adj\n"
        common_mid = (
            "ds=SynData()\n"
            "X=torch.stack([ds[i][0] for i in range(len(ds))])\n"
            "y=torch.tensor([ds[i][1] for i in range(len(ds))])\n"
            "adj=build_adj(len(ds),ds.edges)\n"
            "m={cn}();total_params=sum(p.numel() for p in m.parameters())\n"
            "print(f'Model: {total_params} parameters')\n"
            "o=torch.optim.Adam(m.parameters(),lr=LR)\n"
            "os.makedirs('checkpoints',exist_ok=True)\n"
            "best_loss=float('inf');patience_counter=0;PATIENCE=10\n"
            "history=[]\n"
        )
        specific = (
            "criterion=nn.CrossEntropyLoss()\n"
            "for e in range(EPOCHS):\n"
            "    out=m(X,adj)\n"
            "    l=criterion(out,y)\n"
            "    o.zero_grad();l.backward()\n"
            "    nn.utils.clip_grad_norm_(m.parameters(),1.0)\n    o.step()\n"
            "    with torch.no_grad():\n"
            "        pred=out.argmax(1)\n"
            "        acc=(pred==y).float().mean().item()\n"
            "        loss_val=l.item()\n"
            "    history.append((e,loss_val,acc))\n"
            "    print(f'Epoch {e:3d}: loss={loss_val:.4f}, acc={acc:.4f}')\n"
            "    if loss_val<best_loss:\n        best_loss=loss_val;patience_counter=0\n        torch.save(m.state_dict(),'best_model.pth')\n"
            "    else:\n        patience_counter+=1\n"
            "    if e%SAVE_EVERY==0:\n        torch.save({'epoch':e,'model':m.state_dict()},f'checkpoints/ckpt_{e:04d}.pth')\n"
            "    if patience_counter>=PATIENCE:\n        print(f'Early stopping at epoch {e}');break\n"
            "torch.save(m.state_dict(),'model.pth')\n"
            + _FOOTER_STD_CLS
        )
    else:
        raise ValueError(f'Unknown model_type: {model_type}')

    return prefix + common_mid + specific


# ── Standard-tier log footers ──

_FOOTER_STD_CLS = """
# ── Save training log ──
md = '# Training Log\\n\\n'
md += f'**Model**: {total_params} parameters  \\n'
md += f'**Dataset**: {DATASET}  \\n'
md += f'**Epochs**: {len(history)} (early stopped)' if patience_counter>=PATIENCE else f'**Epochs**: {EPOCHS}  \\n'
md += f'**Batch Size**: {BATCH_SIZE}  \\n'
md += f'**Learning Rate**: {LR}  \\n'
md += f'**Best Loss**: {best_loss:.4f}  \\n\\n'
md += '| Epoch | Loss | Accuracy | Val Loss | Val Acc |\\n'
md += '|-------|------|----------|----------|--------|\\n'
for e, loss, acc in history:
    _vl = val_history[e - start_epoch][0] if e - start_epoch < len(val_history) else float('nan')
    _va = val_history[e - start_epoch][1] if e - start_epoch < len(val_history) else float('nan')
    md += f'| {e:5d} | {loss:.4f} | {acc:.4f} | {_vl:.4f} | {_va:.4f} |\\n'
with open('training_log.md', 'w') as f:
    f.write(md)
print('Saved training_log.md')
print(f'Best model saved as best_model.pth (loss={best_loss:.4f})')
"""

_FOOTER_STD_REG = """
md = '# Training Log\\n\\n'
md += f'**Model**: {total_params} parameters  \\n'
md += f'**Dataset**: {DATASET}  \\n'
md += f'**Epochs**: {len(history)}  \\n'
md += f'**Batch Size**: {BATCH_SIZE}  \\n'
md += f'**Learning Rate**: {LR}  \\n'
md += f'**Best Loss**: {best_loss:.4f}  \\n\\n'
md += '| Epoch | Loss | Val Loss |\\n'
md += '|-------|------|----------|\\n'
for e, loss in history:
    _vl = val_history[e - start_epoch][0] if e - start_epoch < len(val_history) else float('nan')
    md += f'| {e:5d} | {loss:.4f} | {_vl:.4f} |\\n'
with open('training_log.md', 'w') as f:
    f.write(md)
print('Saved training_log.md')
print(f'Best model saved as best_model.pth (loss={best_loss:.4f})')
"""

_FOOTER_STD_AE = """
md = '# Training Log\\n\\n'
md += f'**Model**: {total_params} parameters  \\n'
md += f'**Dataset**: {DATASET}  \\n'
md += f'**Epochs**: {len(history)}  \\n'
md += f'**Batch Size**: {BATCH_SIZE}  \\n'
md += f'**Learning Rate**: {LR}  \\n'
md += f'**Best Recon Loss**: {best_loss:.4f}  \\n\\n'
md += '| Epoch | Recon Loss | Val Loss |\\n'
md += '|-------|------------|----------|\\n'
for e, loss in history:
    _vl = val_history[e - start_epoch][0] if e - start_epoch < len(val_history) else float('nan')
    md += f'| {e:5d} | {loss:.4f} | {_vl:.4f} |\\n'
with open('training_log.md', 'w') as f:
    f.write(md)
print('Saved training_log.md')
"""

_FOOTER_STD_MT = """
md = '# Training Log\\n\\n'
md += f'**Model**: {total_params} parameters  \\n'
md += f'**Dataset**: {DATASET}  \\n'
md += f'**Epochs**: {len(history)}  \\n'
md += f'**Batch Size**: {BATCH_SIZE}  \\n'
md += f'**Learning Rate**: {LR}  \\n\\n'
md += '| Epoch | Loss | Acc1 | Acc2 | Val Loss | Val Acc |\\n'
md += '|-------|------|------|------|----------|--------|\\n'
for e, loss, a1, a2 in history:
    _vl = val_history[e - start_epoch][0] if e - start_epoch < len(val_history) else float('nan')
    _va = val_history[e - start_epoch][1] if e - start_epoch < len(val_history) else float('nan')
    md += f'| {e:5d} | {loss:.4f} | {a1:.4f} | {a2:.4f} | {_vl:.4f} | {_va:.4f} |\\n'
with open('training_log.md', 'w') as f:
    f.write(md)
print('Saved training_log.md')
"""

_FOOTER_STD_GAN = """
md = '# Training Log\\n\\n'
md += f'**Model**: {total_params} parameters  \\n'
md += f'**Dataset**: {DATASET}  \\n'
md += f'**Epochs**: {EPOCHS}  \\n'
md += f'**Batch Size**: {BATCH_SIZE}  \\n'
md += f'**Learning Rate**: {LR}  \\n\\n'
md += '| Epoch | D Loss | G Loss |\\n'
md += '|-------|--------|--------|\\n'
for e, dl, gl in history:
    md += f'| {e:5d} | {dl:.4f} | {gl:.4f} |\\n'
with open('training_log.md', 'w') as f:
    f.write(md)
print('Saved training_log.md')
"""


# ── Extra files: sweep.py, visualize.py, predict.py ──

def get_extra_files(model_type: str, class_name: str) -> dict:
    """Return {filename: content} for standard-tier extra files."""
    return {
        'sweep.py': _gen_sweep(class_name),
        'visualize.py': _gen_visualize_std(),
        'predict.py': _gen_predict_std(class_name, model_type),
    }


def _gen_sweep(class_name: str) -> str:
    return (
        '"""Hyperparameter grid search — tries multiple lr × batch_size combos."""\n'
        'import torch,torch.nn as nn,itertools\n'
        'from config import *\n'
        f'from model import {class_name}\n'
        'from data import SynData\n\n'
        'lrs = [0.1, 0.01, 0.001, 0.0001]\n'
        'batch_sizes = [16, 32, 64, 128]\n'
        'epochs = 10  # short runs for sweep\n\n'
        'ds = SynData()\n'
        'results = []\n\n'
        'for lr, bs in itertools.product(lrs, batch_sizes):\n'
        '    lo = torch.utils.data.DataLoader(ds, bs, shuffle=True)\n'
        f'    m = {class_name}().to(DEVICE)\n'
        '    o = torch.optim.Adam(m.parameters(), lr=lr)\n'
        '    criterion = nn.CrossEntropyLoss()\n'
        '    for e in range(epochs):\n'
        '        for x, y in lo:\n'
        '            x, y = x.to(DEVICE), y.to(DEVICE)\n'
        '            l = criterion(m(x), y)\n'
        '            o.zero_grad(); l.backward(); o.step()\n'
        '    with torch.no_grad():\n'
        '        losses = []\n'
        '        for x, y in lo:\n'
        '            x, y = x.to(DEVICE), y.to(DEVICE)\n'
        '            losses.append(criterion(m(x), y).item() * x.size(0))\n'
        '        loss = sum(losses) / len(ds)\n'
        '    results.append((lr, bs, loss))\n'
        '    print(f"lr={lr:.4f}, bs={bs:3d} => loss={loss:.4f}")\n\n'
        'best = min(results, key=lambda r: r[2])\n'
        'print(f"\\nBest: lr={best[0]:.4f}, bs={best[1]}, loss={best[2]:.4f}")\n'
        'with open("sweep_results.txt", "w") as f:\n'
        '    f.write("lr,batch_size,loss\\n")\n'
        '    for lr, bs, loss in results:\n'
        '        f.write(f"{lr},{bs},{loss}\\n")\n'
        'print("Saved sweep_results.txt")\n'
    )


def _gen_visualize_std() -> str:
    return (
        '"""Plot training curves from training_log.md."""\n'
        'import re\n'
        'import matplotlib.pyplot as plt\n\n'
        '# Parse training_log.md\n'
        'try:\n'
        '    with open("training_log.md") as f:\n'
        '        text = f.read()\n'
        'except FileNotFoundError:\n'
        '    print("No training_log.md found. Train first with: python train.py")\n'
        '    exit(1)\n\n'
        '# Extract table rows: | epoch | value1 | [value2] |\n'
        'pattern = r"\\|\\s*(\\d+)\\s*\\|\\s*([\\d.]+)\\s*\\|\\s*([\\d.]+)?\\s*\\|"\n'
        'matches = re.findall(pattern, text)\n'
        'if not matches:\n'
        '    print("No metrics found in training_log.md")\n'
        '    exit(1)\n\n'
        'epochs = [int(m[0]) for m in matches]\n'
        'losses = [float(m[1]) for m in matches]\n'
        'has_acc = any(m[2] for m in matches)\n'
        'accs = [float(m[2]) for m in matches] if has_acc else None\n\n'
        'fig, ax1 = plt.subplots(figsize=(10, 5))\n'
        'ax1.plot(epochs, losses, "b-o", label="Loss")\n'
        'ax1.set_xlabel("Epoch")\n'
        'ax1.set_ylabel("Loss", color="b")\n'
        'ax1.tick_params(axis="y", labelcolor="b")\n\n'
        'if accs:\n'
        '    ax2 = ax1.twinx()\n'
        '    ax2.plot(epochs, accs, "r-s", label="Accuracy")\n'
        '    ax2.set_ylabel("Accuracy", color="r")\n'
        '    ax2.tick_params(axis="y", labelcolor="r")\n\n'
        'plt.title("Training Progress")\n'
        'fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.88))\n'
        'plt.grid(True, alpha=0.3)\n'
        'plt.tight_layout()\n'
        'plt.savefig("training_curve.png", dpi=150)\n'
        'print("Saved training_curve.png")\n'
    )


def _gen_predict_std(class_name: str, model_type: str = 'ce') -> str:
    if model_type == 'gcn':
        batch_setup = ('x = torch.randn(10, INPUT_DIM, device=DEVICE)\n'
                       'adj = torch.eye(10, device=DEVICE)\n')
        call = 'm(x, adj)'
    else:
        batch_setup = 'x = torch.randn(10, INPUT_DIM, device=DEVICE)\n'
        call = 'm(x)'
    return (
        '"""Batch inference demo."""\n'
        'import torch\n'
        'import numpy as np\n'
        f'from model import {class_name}\n'
        'from config import INPUT_DIM, OUTPUT_DIM\n\n'
        f'm = {class_name}().to(DEVICE)\n'
        "m.load_state_dict(torch.load('best_model.pth', weights_only=True))\n"
        'm.eval()\n\n'
        '# Batch inference on 10 random samples\n'
        + batch_setup +
        'with torch.no_grad():\n'
        f'    out = {call}\n'
        '    if isinstance(out, tuple):\n'
        '        out = out[0]\n'
        '    if out.shape[1] == 1:\n'
        '        print("Predictions:")\n'
        '        for i in range(min(5, len(x))):\n'
        '            print(f"  Sample {i}: {out[i].item():.4f}")\n'
        '    else:\n'
        '        preds = out.argmax(dim=1)\n'
        '        print("Predictions:")\n'
        '        for i in range(min(10, len(x))):\n'
        '            print(f"  Sample {i}: class={preds[i].item()}")\n'
        'print("Done.")\n'
    )
