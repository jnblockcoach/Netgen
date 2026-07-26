"""Quick-tier templates (< 50K params). Minimal, no-frills training scripts.

Each model type returns (data_code, train_code, eval_code).
Train code: basic loop + per-epoch logging + auto-save training_log.md.
"""

# ── Shared log footers ──

_FOOTER_CLS = """
# ── Save training log ──
md = '# Training Log\\n\\n'
md += f'**Model**: {total_params} parameters  \\n'
md += f'**Dataset**: {DATASET}  \\n'
md += f'**Epochs**: {EPOCHS}  \\n'
md += f'**Batch Size**: {BATCH_SIZE}  \\n'
md += f'**Learning Rate**: {LR}  \\n\\n'
md += '| Epoch | Loss | Accuracy |\\n'
md += '|-------|------|----------|\\n'
for e, loss, acc in history:
    md += f'| {e:5d} | {loss:.4f} | {acc:.4f} |\\n'
with open('training_log.md', 'w') as f:
    f.write(md)
print('Saved training_log.md')\nprint(f'Best model saved as best_model.pth (loss={best_loss:.4f})')
print(f'Best model saved as best_model.pth (loss={best_loss:.4f})')
"""

_FOOTER_REG = """
md = '# Training Log\\n\\n'
md += f'**Model**: {total_params} parameters  \\n'
md += f'**Dataset**: {DATASET}  \\n'
md += f'**Epochs**: {EPOCHS}  \\n'
md += f'**Batch Size**: {BATCH_SIZE}  \\n'
md += f'**Learning Rate**: {LR}  \\n\\n'
md += '| Epoch | Loss |\\n'
md += '|-------|------|\\n'
for e, loss in history:
    md += f'| {e:5d} | {loss:.4f} |\\n'
with open('training_log.md', 'w') as f:
    f.write(md)
print('Saved training_log.md')\nprint(f'Best model saved as best_model.pth (loss={best_loss:.4f})')
print(f'Best model saved as best_model.pth (loss={best_loss:.4f})')
"""

_FOOTER_AE = """
md = '# Training Log\\n\\n'
md += f'**Model**: {total_params} parameters  \\n'
md += f'**Dataset**: {DATASET}  \\n'
md += f'**Epochs**: {EPOCHS}  \\n'
md += f'**Batch Size**: {BATCH_SIZE}  \\n'
md += f'**Learning Rate**: {LR}  \\n\\n'
md += '| Epoch | Recon Loss |\\n'
md += '|-------|------------|\\n'
for e, loss in history:
    md += f'| {e:5d} | {loss:.4f} |\\n'
with open('training_log.md', 'w') as f:
    f.write(md)
print('Saved training_log.md')\nprint(f'Best model saved as best_model.pth (loss={best_loss:.4f})')
print(f'Best model saved as best_model.pth (loss={best_loss:.4f})')
"""

_FOOTER_GAN = """
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
print('Saved training_log.md')\nprint(f'Best model saved as best_model.pth (loss={best_loss:.4f})')
print(f'Best model saved as best_model.pth (loss={best_loss:.4f})')
"""

_FOOTER_MT = """
md = '# Training Log\\n\\n'
md += f'**Model**: {total_params} parameters  \\n'
md += f'**Dataset**: {DATASET}  \\n'
md += f'**Epochs**: {EPOCHS}  \\n'
md += f'**Batch Size**: {BATCH_SIZE}  \\n'
md += f'**Learning Rate**: {LR}  \\n\\n'
md += '| Epoch | Loss | Acc1 | Acc2 |\\n'
md += '|-------|------|------|------|\\n'
for e, loss, a1, a2 in history:
    md += f'| {e:5d} | {loss:.4f} | {a1:.4f} | {a2:.4f} |\\n'
with open('training_log.md', 'w') as f:
    f.write(md)
print('Saved training_log.md')\nprint(f'Best model saved as best_model.pth (loss={best_loss:.4f})')
print(f'Best model saved as best_model.pth (loss={best_loss:.4f})')
"""


def get_templates(model_type):
    if model_type == 'ce':
        data = ("from config import *\nimport torch,numpy as np\nclass SynData(torch.utils.data.Dataset):\n"
                "    def __init__(self,n=2000):\n        np.random.seed(42)\n"
                "        self.X=np.random.randn(n,INPUT_DIM).astype(np.float32)\n"
                "        c=self.X.shape[1]\n"
                "        if c>=6:\n            self.y=np.fmin(((self.X[:,0]*self.X[:,1]>0).astype(int)+(self.X[:,2]+self.X[:,3]>0).astype(int)*2+(self.X[:,4]**2-self.X[:,5]>0).astype(int)),OUTPUT_DIM-1)\n"
                "        elif c>=4:\n            self.y=np.fmin(((self.X[:,0]*self.X[:,1]>0).astype(int)+(self.X[:,2]+self.X[:,3]>0).astype(int)),OUTPUT_DIM-1)\n"
                "        else:\n            self.y=np.fmin((self.X[:,0]*self.X[:,1]>0).astype(int),OUTPUT_DIM-1)\n"
                "    def __len__(self):return len(self.X)\n    def __getitem__(self,i):return torch.from_numpy(self.X[i]),self.y[i]\n")
        train = ("import torch,torch.nn as nn,os\nfrom config import *\nfrom model import {cn}\nfrom data import SynData\n"
                 "ds=SynData();lo=torch.utils.data.DataLoader(ds,BATCH_SIZE,shuffle=True)\n"
                 "m={cn}();total_params=sum(p.numel() for p in m.parameters())\n"
                 "print(f'Model: {total_params} parameters')\n"
                 "os.makedirs('checkpoints',exist_ok=True)\n"
                 "best_loss=float('inf')\n"
                 "o=torch.optim.Adam(m.parameters(),lr=LR)\n"
                 "criterion = {'ce': lambda: nn.CrossEntropyLoss(),'ce_smooth': lambda: nn.CrossEntropyLoss(label_smoothing=0.1)}.get(LOSS_TYPE, nn.CrossEntropyLoss)()\n"
                 "history=[]\n"
                 "for e in range(EPOCHS):\n"
                 "    for x,y in lo:\n        l=criterion(m(x),y)\n        o.zero_grad();l.backward();o.step()\n"
                 "    with torch.no_grad():\n"
                 "        loss_val=sum(criterion(m(x),y).item()*x.size(0) for x,y in lo)/len(ds)\n"
                 "        correct=sum((m(x).argmax(1)==y).sum().item() for x,y in lo)\n"
                 "        acc=correct/len(ds)\n"
                 "    history.append((e,loss_val,acc))\n"
                 "    print(f'Epoch {e:3d}: loss={loss_val:.4f}, acc={acc:.4f}')\n"
                 "    if loss_val<best_loss:\n        best_loss=loss_val\n        torch.save(m.state_dict(),'best_model.pth')\n"
                 "    if e%SAVE_EVERY==0:\n        torch.save({'epoch':e,'model':m.state_dict(),'opt':o.state_dict()},f'checkpoints/ckpt_{e:04d}.pth')\n"
                 "torch.save(m.state_dict(),'model.pth')\n"
                 + _FOOTER_CLS)
        eval_ = ("import torch,numpy as np\nfrom config import *\nfrom model import {cn}\n"
                 "m={cn}();m.load_state_dict(torch.load('model.pth',weights_only=True));m.eval()\n"
                 "np.random.seed(0);X=np.random.randn(500,INPUT_DIM).astype(np.float32)\n"
                 "c=X.shape[1]\n"
                 "if c>=6:\n    y=np.fmin(((X[:,0]*X[:,1]>0).astype(int)+(X[:,2]+X[:,3]>0).astype(int)*2+(X[:,4]**2-X[:,5]>0).astype(int)),OUTPUT_DIM-1)\n"
                 "elif c>=4:\n    y=np.fmin(((X[:,0]*X[:,1]>0).astype(int)+(X[:,2]+X[:,3]>0).astype(int)),OUTPUT_DIM-1)\n"
                 "else:\n    y=np.fmin((X[:,0]*X[:,1]>0).astype(int),OUTPUT_DIM-1)\n"
                 "with torch.no_grad():\n    a=(m(torch.from_numpy(X)).argmax(1).numpy()==y).mean()\n    print('Test Accuracy:'+str(a))\n")
        return data, train, eval_

    elif model_type == 'mse':
        data = ("from config import *\nimport torch,numpy as np\nclass SynData(torch.utils.data.Dataset):\n"
                "    def __init__(self,n=2000):\n        np.random.seed(42)\n"
                "        self.X=np.random.randn(n,INPUT_DIM).astype(np.float32)\n"
                "        true_w=np.random.randn(INPUT_DIM,OUTPUT_DIM).astype(np.float32)\n"
                "        self.y=(self.X@true_w+np.random.randn(n,OUTPUT_DIM).astype(np.float32)*0.5)\n"
                "    def __len__(self):return len(self.X)\n    def __getitem__(self,i):return torch.from_numpy(self.X[i]),torch.from_numpy(self.y[i])\n")
        train = ("import torch,torch.nn as nn,os\nfrom config import *\nfrom model import {cn}\nfrom data import SynData\n"
                 "ds=SynData();lo=torch.utils.data.DataLoader(ds,BATCH_SIZE,shuffle=True)\n"
                 "m={cn}();total_params=sum(p.numel() for p in m.parameters())\n"
                 "print(f'Model: {total_params} parameters')\n"
                 "os.makedirs('checkpoints',exist_ok=True)\n"
                 "best_loss=float('inf')\n"
                 "o=torch.optim.Adam(m.parameters(),lr=LR)\n"
                 "_losses = {'mse': nn.MSELoss, 'l1': nn.L1Loss, 'smooth_l1': nn.SmoothL1Loss}\n"
                 "criterion = _losses.get(LOSS_TYPE, nn.MSELoss)()\n"
                 "history=[]\n"
                 "for e in range(EPOCHS):\n"
                 "    for x,y in lo:\n        l=criterion(m(x),y)\n        o.zero_grad();l.backward();o.step()\n"
                 "    with torch.no_grad():\n"
                 "        loss_val=sum(criterion(m(x),y).item()*x.size(0) for x,y in lo)/len(ds)\n"
                 "    history.append((e,loss_val))\n"
                 "    print(f'Epoch {e:3d}: loss={loss_val:.4f}')\n"
                 "    if loss_val<best_loss:\n        best_loss=loss_val\n        torch.save(m.state_dict(),'best_model.pth')\n"
                 "    if e%SAVE_EVERY==0:\n        torch.save({'epoch':e,'model':m.state_dict(),'opt':o.state_dict()},f'checkpoints/ckpt_{e:04d}.pth')\n"
                 "torch.save(m.state_dict(),'model.pth')\n"
                 + _FOOTER_REG)
        eval_ = ("import torch,numpy as np\nfrom config import *\nfrom model import {cn}\n"
                 "m={cn}();m.load_state_dict(torch.load('model.pth',weights_only=True));m.eval()\n"
                 "np.random.seed(0);X=np.random.randn(200,INPUT_DIM).astype(np.float32)\n"
                 "with torch.no_grad():\n    pred=m(torch.from_numpy(X)).numpy()\n    print('Pred range:'+str(pred.min())+'..'+str(pred.max()))\n")
        return data, train, eval_

    elif model_type == 'cnn':
        data = ("from config import *\nimport torch,numpy as np\nclass SynData(torch.utils.data.Dataset):\n"
                "    def __init__(self,n=2000):\n        np.random.seed(42)\n"
                "        self.X,self.y=[],[]\n        for _ in range(n):\n"
                "            img=np.zeros((8,8),dtype=np.float32)\n"
                "            k=np.random.randint(0,OUTPUT_DIM)\n"
                "            if k==0: img[3:5,:]=1.0\n            elif k==1: img[:,3:5]=1.0\n"
                "            elif k==2: img[2:6,2:6]=1.0\n            elif k==3: img[np.arange(8),np.arange(8)]=1.0\n"
                "            elif k==4: img[3,:]=1.0;img[:,4]=1.0\n"
                "            else: img[::2,::2]=0.5\n"
                "            img+=np.random.randn(8,8).astype(np.float32)*0.1\n"
                "            if INPUT_DIM==3:\n                self.X.append(np.stack([img,img,img]))\n"
                "            else:\n                self.X.append(img.reshape(1,8,8))\n"
                "            self.y.append(k%OUTPUT_DIM)\n"
                "        self.X=np.array(self.X,dtype=np.float32)\n        self.y=np.array(self.y,dtype=np.int64)\n"
                "    def __len__(self):return len(self.X)\n    def __getitem__(self,i):return torch.from_numpy(self.X[i]),self.y[i]\n")
        train = ("import torch,torch.nn as nn,os\nfrom config import *\nfrom model import {cn}\nfrom data import SynData\n"
                 "ds=SynData();lo=torch.utils.data.DataLoader(ds,BATCH_SIZE,shuffle=True)\n"
                 "m={cn}();total_params=sum(p.numel() for p in m.parameters())\n"
                 "print(f'Model: {total_params} parameters')\n"
                 "os.makedirs('checkpoints',exist_ok=True)\n"
                 "best_loss=float('inf')\n"
                 "o=torch.optim.Adam(m.parameters(),lr=LR)\n"
                 "criterion={'ce':lambda:nn.CrossEntropyLoss(),'ce_smooth':lambda:nn.CrossEntropyLoss(label_smoothing=0.1)}.get(LOSS_TYPE,nn.CrossEntropyLoss)()\n"
                 "history=[]\n"
                 "for e in range(EPOCHS):\n"
                 "    for x,y in lo:\n        l=criterion(m(x),y)\n        o.zero_grad();l.backward();o.step()\n"
                 "    with torch.no_grad():\n"
                 "        loss_val=sum(criterion(m(x),y).item()*x.size(0) for x,y in lo)/len(ds)\n"
                 "        correct=sum((m(x).argmax(1)==y).sum().item() for x,y in lo)\n"
                 "        acc=correct/len(ds)\n"
                 "    history.append((e,loss_val,acc))\n"
                 "    print(f'Epoch {e:3d}: loss={loss_val:.4f}, acc={acc:.4f}')\n"
                 "    if loss_val<best_loss:\n        best_loss=loss_val\n        torch.save(m.state_dict(),'best_model.pth')\n"
                 "    if e%SAVE_EVERY==0:\n        torch.save({'epoch':e,'model':m.state_dict(),'opt':o.state_dict()},f'checkpoints/ckpt_{e:04d}.pth')\n"
                 "torch.save(m.state_dict(),'model.pth')\n"
                 + _FOOTER_CLS)
        eval_ = ("import torch,numpy as np\nfrom config import *\nfrom model import {cn}\n"
                 "m={cn}();m.load_state_dict(torch.load('model.pth',weights_only=True));m.eval()\n"
                 "img=np.zeros((8,8),dtype=np.float32);img[3:5,:]=1.0\n"
                 "x=torch.from_numpy(img.reshape(1,INPUT_DIM,8,8).astype(np.float32))\n"
                 "if INPUT_DIM==3: x=x.repeat(1,3,1,1)\n"
                 "with torch.no_grad():\n    p=m(x).argmax(1).item()\n    print('Pred class:'+str(p))\n")
        return data, train, eval_

    elif model_type == 'rnn':
        data = ("from config import *\nimport torch,numpy as np\nclass SynData(torch.utils.data.Dataset):\n"
                "    def __init__(self,n=2000,sl=15):\n        np.random.seed(42)\n"
                "        self.X,self.y=[],[]\n        for _ in range(n):\n"
                "            s=np.random.uniform(0,4*np.pi)\n            t=np.linspace(s,s+sl,sl+1)\n"
                "            w=np.sin(t).astype(np.float32)\n"
                "            self.X.append(w[:-1].reshape(-1,1))\n"
                "            self.y.append(int((w[-1]+1)*(OUTPUT_DIM-1)/2))\n"
                "        self.X=np.array(self.X,dtype=np.float32)\n        self.y=np.array(self.y,dtype=np.int64)\n"
                "    def __len__(self):return len(self.X)\n    def __getitem__(self,i):return torch.from_numpy(self.X[i]),self.y[i]\n")
        train = ("import torch,torch.nn as nn,os\nfrom config import *\nfrom model import {cn}\nfrom data import SynData\n"
                 "ds=SynData();lo=torch.utils.data.DataLoader(ds,BATCH_SIZE,shuffle=True)\n"
                 "m={cn}();total_params=sum(p.numel() for p in m.parameters())\n"
                 "print(f'Model: {total_params} parameters')\n"
                 "os.makedirs('checkpoints',exist_ok=True)\n"
                 "best_loss=float('inf')\n"
                 "o=torch.optim.Adam(m.parameters(),lr=LR)\n"
                 "criterion=nn.CrossEntropyLoss()\n"
                 "history=[]\n"
                 "for e in range(EPOCHS):\n"
                 "    for x,y in lo:\n        l=criterion(m(x),y)\n        o.zero_grad();l.backward();o.step()\n"
                 "    with torch.no_grad():\n"
                 "        loss_val=sum(criterion(m(x),y).item()*x.size(0) for x,y in lo)/len(ds)\n"
                 "        correct=sum((m(x).argmax(1)==y).sum().item() for x,y in lo)\n"
                 "        acc=correct/len(ds)\n"
                 "    history.append((e,loss_val,acc))\n"
                 "    print(f'Epoch {e:3d}: loss={loss_val:.4f}, acc={acc:.4f}')\n"
                 "    if loss_val<best_loss:\n        best_loss=loss_val\n        torch.save(m.state_dict(),'best_model.pth')\n"
                 "    if e%SAVE_EVERY==0:\n        torch.save({'epoch':e,'model':m.state_dict(),'opt':o.state_dict()},f'checkpoints/ckpt_{e:04d}.pth')\n"
                 "torch.save(m.state_dict(),'model.pth')\n"
                 + _FOOTER_CLS)
        eval_ = ("import torch,numpy as np\nfrom config import *\nfrom model import {cn}\n"
                 "m={cn}();m.load_state_dict(torch.load('model.pth',weights_only=True));m.eval()\n"
                 "t=np.sin(np.linspace(0,6*np.pi,15).astype(np.float32)).reshape(1,-1,1)\n"
                 "with torch.no_grad(): print('Pred class:'+str(m(torch.from_numpy(t)).argmax(1).item()))\n")
        return data, train, eval_

    elif model_type in ('ae', 'vae'):
        data = ("from config import *\nimport torch,numpy as np\nclass SynData(torch.utils.data.Dataset):\n"
                "    def __init__(self,n=2000):\n        np.random.seed(42)\n"
                "        self.X=np.random.randn(n,INPUT_DIM).astype(np.float32)\n"
                "    def __len__(self):return len(self.X)\n    def __getitem__(self,i):return torch.from_numpy(self.X[i])\n")
        if model_type == 'ae':
            train = ("import torch,torch.nn as nn,os\nfrom config import *\nfrom model import {cn}\nfrom data import SynData\n"
                     "ds=SynData();lo=torch.utils.data.DataLoader(ds,BATCH_SIZE,shuffle=True)\n"
                     "m={cn}();total_params=sum(p.numel() for p in m.parameters())\n"
                     "print(f'Model: {total_params} parameters')\n"
                     "os.makedirs('checkpoints',exist_ok=True)\n"
                     "best_loss=float('inf')\n"
                     "o=torch.optim.Adam(m.parameters(),lr=LR)\n"
                     "history=[]\n"
                     "for e in range(EPOCHS):\n"
                     "    for x in lo:\n        r,_=m(x);l=nn.MSELoss()(r,x)\n        o.zero_grad();l.backward();o.step()\n"
                     "    with torch.no_grad():\n"
                     "        recon_loss=sum(nn.MSELoss()(m(x)[0],x).item()*x.size(0) for x in lo)/len(ds)\n"
                     "    history.append((e,recon_loss))\n"
                     "    print(f'Epoch {e:3d}: recon_loss={recon_loss:.4f}')\n"
                     "    if recon_loss<best_loss:\n        best_loss=recon_loss\n        torch.save(m.state_dict(),'best_model.pth')\n"
                     "    if e%SAVE_EVERY==0:\n        torch.save({'epoch':e,'model':m.state_dict(),'opt':o.state_dict()},f'checkpoints/ckpt_{e:04d}.pth')\n"
                     "torch.save(m.state_dict(),'model.pth')\n"
                     + _FOOTER_AE)
        else:
            train = ("import torch,torch.nn as nn,os\nfrom config import *\nfrom model import {cn}\nfrom data import SynData\n"
                     "ds=SynData();lo=torch.utils.data.DataLoader(ds,BATCH_SIZE,shuffle=True)\n"
                     "m={cn}();total_params=sum(p.numel() for p in m.parameters())\n"
                     "print(f'Model: {total_params} parameters')\n"
                     "os.makedirs('checkpoints',exist_ok=True)\n"
                     "best_loss=float('inf')\n"
                     "o=torch.optim.Adam(m.parameters(),lr=LR)\n"
                     "history=[]\n"
                     "for e in range(EPOCHS):\n"
                     "    for x in lo:\n"
                     "        r,mu,lv=m(x);rc=nn.functional.mse_loss(r,x)\n"
                     "        kl=-0.5*(1+lv-mu.pow(2)-lv.exp()).sum(dim=1).mean()\n"
                     "        l=rc+0.01*kl\n        o.zero_grad();l.backward();o.step()\n"
                     "    with torch.no_grad():\n"
                     "        recon_loss=sum(nn.functional.mse_loss(m(x)[0],x).item()*x.size(0) for x in lo)/len(ds)\n"
                     "    history.append((e,recon_loss))\n"
                     "    print(f'Epoch {e:3d}: recon_loss={recon_loss:.4f}')\n"
                     "    if recon_loss<best_loss:\n        best_loss=recon_loss\n        torch.save(m.state_dict(),'best_model.pth')\n"
                     "    if e%SAVE_EVERY==0:\n        torch.save({'epoch':e,'model':m.state_dict(),'opt':o.state_dict()},f'checkpoints/ckpt_{e:04d}.pth')\n"
                     "torch.save(m.state_dict(),'model.pth')\n"
                     + _FOOTER_AE)
        if model_type == 'ae':
            eval_ = ("import torch,numpy as np\nfrom config import *\nfrom model import {cn}\n"
                     "m={cn}();m.load_state_dict(torch.load('model.pth',weights_only=True));m.eval()\n"
                     "X=np.random.randn(50,INPUT_DIM).astype(np.float32)\n"
                     "with torch.no_grad():\n    r,_=m(torch.from_numpy(X))\n    mse=((r.numpy()-X)**2).mean()\n    print('Recon MSE:'+str(mse))\n")
        else:
            eval_ = ("import torch,numpy as np\nfrom config import *\nfrom model import {cn}\n"
                     "m={cn}();m.load_state_dict(torch.load('model.pth',weights_only=True));m.eval()\n"
                     "X=np.random.randn(50,INPUT_DIM).astype(np.float32)\n"
                     "with torch.no_grad():\n    r,*_=m(torch.from_numpy(X))\n    mse=((r.numpy()-X)**2).mean()\n    print('Recon MSE:'+str(mse))\n")
        return data, train, eval_

    elif model_type == 'mt':
        data = ("from config import *\nimport torch,numpy as np\nclass SynData(torch.utils.data.Dataset):\n"
                "    def __init__(self,n=2000):\n        np.random.seed(42)\n"
                "        self.X=np.random.randn(n,INPUT_DIM).astype(np.float32)\n"
                "        d=INPUT_DIM;X=self.X\n"
                "        if d>=8:\n"
                "            self.y1=np.fmin(((X[:,0]*X[:,1]>0).astype(int)+(X[:,2]+X[:,3]>0).astype(int)),2)\n"
                "            self.y2=np.fmin(((X[:,4]**2-X[:,5]>0).astype(int)+(X[:,6]*X[:,7]>0).astype(int)),2)\n"
                "        elif d>=4:\n"
                "            self.y1=np.fmin(((X[:,0]*X[:,1]>0).astype(int)+(X[:,2]+X[:,3]>0).astype(int)),2)\n"
                "            self.y2=np.fmin(((X[:,0]**2-X[:,1]>0).astype(int)),2)\n"
                "        else:\n"
                "            self.y1=np.fmin((X[:,0]*X[:,1]>0).astype(int),2)\n"
                "            self.y2=np.fmin((X[:,0]+X[:,1]>0).astype(int),2)\n"
                "    def __len__(self):return len(self.X)\n"
                "    def __getitem__(self,i):return torch.from_numpy(self.X[i]),self.y1[i],self.y2[i]\n")
        train = ("import torch,torch.nn as nn,os\nfrom config import *\nfrom model import {cn}\nfrom data import SynData\n"
                 "ds=SynData();lo=torch.utils.data.DataLoader(ds,BATCH_SIZE,shuffle=True)\n"
                 "m={cn}();total_params=sum(p.numel() for p in m.parameters())\n"
                 "print(f'Model: {total_params} parameters')\n"
                 "os.makedirs('checkpoints',exist_ok=True)\n"
                 "best_loss=float('inf')\n"
                 "o=torch.optim.Adam(m.parameters(),lr=LR)\n"
                 "history=[]\n"
                 "for e in range(EPOCHS):\n"
                 "    for x,y1,y2 in lo:\n"
                 "        o1,o2=m(x)\n"
                 "        l=nn.CrossEntropyLoss()(o1,y1)+nn.CrossEntropyLoss()(o2,y2)\n"
                 "        o.zero_grad();l.backward();o.step()\n"
                 "    with torch.no_grad():\n"
                 "        loss_val=0\n        a1s=a2s=0\n        n=0\n"
                 "        for x,y1,y2 in lo:\n"
                 "            o1,o2=m(x)\n"
                 "            loss_val+=nn.CrossEntropyLoss()(o1,y1).item()*x.size(0)+nn.CrossEntropyLoss()(o2,y2).item()*x.size(0)\n"
                 "            a1s+=(o1.argmax(1)==y1).sum().item()\n"
                 "            a2s+=(o2.argmax(1)==y2).sum().item()\n"
                 "            n+=x.size(0)\n"
                 "        loss_val/=n;a1=a1s/n;a2=a2s/n\n"
                 "    history.append((e,loss_val,a1,a2))\n"
                 "    print(f'Epoch {e:3d}: loss={loss_val:.4f}, acc1={a1:.4f}, acc2={a2:.4f}')\n"
                 "    if loss_val<best_loss:\n        best_loss=loss_val\n        torch.save(m.state_dict(),'best_model.pth')\n"
                 "    if e%SAVE_EVERY==0:\n        torch.save({'epoch':e,'model':m.state_dict(),'opt':o.state_dict()},f'checkpoints/ckpt_{e:04d}.pth')\n"
                 "torch.save(m.state_dict(),'model.pth')\n"
                 + _FOOTER_MT)
        eval_ = ("import torch,numpy as np\nfrom config import *\nfrom model import {cn}\n"
                 "m={cn}();m.load_state_dict(torch.load('model.pth',weights_only=True));m.eval()\n"
                 "X=np.random.randn(100,INPUT_DIM).astype(np.float32)\n"
                 "with torch.no_grad():\n    o1,o2=m(torch.from_numpy(X))\n    print('Task1 pred shape:'+str(o1.shape)+', Task2:'+str(o2.shape))\n")
        return data, train, eval_

    elif model_type == 'gan':
        data = ("import torch,numpy as np\nclass SynData(torch.utils.data.Dataset):\n"
                "    def __init__(self,n=2000):\n        np.random.seed(42)\n"
                "        theta=torch.randn(n,1)*2*np.pi\n        r=2.0\n"
                "        self.X=torch.cat([r*torch.cos(theta),r*torch.sin(theta)],dim=1).numpy().astype(np.float32)\n"
                "    def __len__(self):return len(self.X)\n    def __getitem__(self,i):return torch.from_numpy(self.X[i])\n")
        train = ("import torch,torch.nn as nn,os,numpy as np\nfrom config import *\nfrom model import {cn}\nfrom data import SynData\n"
                 "ds=SynData();lo=torch.utils.data.DataLoader(ds,BATCH_SIZE,shuffle=True)\n"
                 "m={cn}();total_params=sum(p.numel() for p in m.parameters())\n"
                 "print(f'Model: {total_params} parameters')\n"
                 "og=torch.optim.Adam(m.generator.parameters(),lr=LR)\n"
                 "od=torch.optim.Adam(m.discriminator.parameters(),lr=LR)\n"
                 "history=[]\n"
                 "for e in range(EPOCHS):\n"
                 "    dls=gls=0;n=0\n"
                 "    for real in lo:\n"
                 "        z=torch.randn(real.size(0),INPUT_DIM)\n"
                 "        fake=m.forward_gen(z)\n"
                 "        d_real=m.forward_disc(real)\n"
                 "        d_fake=m.forward_disc(fake.detach())\n"
                 "        ld=nn.BCELoss()(d_real,torch.ones_like(d_real))+nn.BCELoss()(d_fake,torch.zeros_like(d_fake))\n"
                 "        od.zero_grad();ld.backward();od.step()\n"
                 "        z=torch.randn(real.size(0),INPUT_DIM)\n"
                 "        fake=m.forward_gen(z)\n"
                 "        d_fake=m.forward_disc(fake)\n"
                 "        lg=nn.BCELoss()(d_fake,torch.ones_like(d_fake))\n"
                 "        og.zero_grad();lg.backward();og.step()\n"
                 "        dls+=ld.item()*real.size(0);gls+=lg.item()*real.size(0);n+=real.size(0)\n"
                 "    dl=dls/n;gl=gls/n\n"
                 "    history.append((e,dl,gl))\n"
                 "    print(f'Epoch {e:3d}: d_loss={dl:.4f}, g_loss={gl:.4f}')\n"
                 "    if gl<best_loss:\n        best_loss=gl\n        torch.save(m.state_dict(),'best_model.pth')\n"
                 "    if e%SAVE_EVERY==0:\n        torch.save({'epoch':e,'model':m.state_dict()},f'checkpoints/ckpt_{e:04d}.pth')\n"
                 "torch.save(m.state_dict(),'model.pth')\n"
                 + _FOOTER_GAN)
        eval_ = ("import torch,numpy as np\nfrom config import *\nfrom model import {cn}\n"
                 "m={cn}();m.load_state_dict(torch.load('model.pth',weights_only=True));m.eval()\n"
                 "z=torch.randn(10,INPUT_DIM)\n"
                 "with torch.no_grad():\n    s=m.forward_gen(z).numpy()\n    print('Generated samples:\\n'+str(s[:3]))\n")
        return data, train, eval_

    elif model_type == 'contrastive':
        data = ("from config import *\nimport torch,numpy as np\nclass SynData(torch.utils.data.Dataset):\n"
                "    def __init__(self,n=2000):\n        np.random.seed(42)\n"
                "        self.X=np.random.randn(n,INPUT_DIM).astype(np.float32)\n"
                "    def __len__(self):return len(self.X)\n"
                "    def __getitem__(self,i):\n        x=torch.from_numpy(self.X[i])\n"
                "        return x+torch.randn_like(x)*0.05,x+torch.randn_like(x)*0.05\n")
        train = ("import torch,torch.nn as nn,os\nfrom config import *\nfrom model import {cn}\nfrom data import SynData\n"
                 "ds=SynData();lo=torch.utils.data.DataLoader(ds,BATCH_SIZE,shuffle=True)\n"
                 "m={cn}();total_params=sum(p.numel() for p in m.parameters())\n"
                 "print(f'Model: {total_params} parameters')\n"
                 "os.makedirs('checkpoints',exist_ok=True)\n"
                 "best_loss=float('inf')\n"
                 "o=torch.optim.Adam(m.parameters(),lr=LR)\n"
                 "history=[]\n"
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
                 "        o.zero_grad();l.backward();o.step()\n"
                 "        epoch_loss+=l.item()*x1.size(0)\n        n+=x1.size(0)\n"
                 "    loss_val=epoch_loss/n\n"
                 "    history.append((e,loss_val))\n"
                 "    print(f'Epoch {e:3d}: loss={loss_val:.4f}')\n"
                 "    if loss_val<best_loss:\n        best_loss=loss_val\n        torch.save(m.state_dict(),'best_model.pth')\n"
                 "    if e%SAVE_EVERY==0:\n        torch.save({'epoch':e,'model':m.state_dict(),'opt':o.state_dict()},f'checkpoints/ckpt_{e:04d}.pth')\n"
                 "torch.save(m.state_dict(),'model.pth')\n"
                 + _FOOTER_REG)
        eval_ = ("import torch,numpy as np\nfrom config import *\nfrom model import {cn}\n"
                 "m={cn}();m.load_state_dict(torch.load('model.pth',weights_only=True));m.eval()\n"
                 "x=torch.randn(10,INPUT_DIM)\n"
                 "with torch.no_grad():\n    e=m(x)\n    print('Embedding shape:'+str(e.shape))\n")
        return data, train, eval_

    elif model_type == 'siamese':
        data = ("from config import *\nimport torch,numpy as np\nclass SynData(torch.utils.data.Dataset):\n"
                "    def __init__(self,n=2000):\n        np.random.seed(42)\n"
                "        self.pairs,self.labels=[],[]\n        for _ in range(n):\n"
                "            if np.random.random()>0.5:\n"
                "                x=np.random.randn(INPUT_DIM).astype(np.float32)\n"
                "                self.pairs.append((x,x+np.random.randn(INPUT_DIM).astype(np.float32)*0.1))\n"
                "                self.labels.append(1)\n"
                "            else:\n"
                "                x=np.random.randn(INPUT_DIM).astype(np.float32)\n"
                "                self.pairs.append((x,np.random.randn(INPUT_DIM).astype(np.float32)))\n"
                "                self.labels.append(0)\n"
                "    def __len__(self):return len(self.pairs)\n"
                "    def __getitem__(self,i):\n        x1,x2=self.pairs[i]\n        return torch.from_numpy(x1),torch.from_numpy(x2),self.labels[i]\n")
        train = ("import torch,torch.nn as nn,os\nfrom config import *\nfrom model import {cn}\nfrom data import SynData\n"
                 "ds=SynData();lo=torch.utils.data.DataLoader(ds,BATCH_SIZE,shuffle=True)\n"
                 "m={cn}();total_params=sum(p.numel() for p in m.parameters())\n"
                 "print(f'Model: {total_params} parameters')\n"
                 "os.makedirs('checkpoints',exist_ok=True)\n"
                 "best_loss=float('inf')\n"
                 "o=torch.optim.Adam(m.parameters(),lr=LR)\n"
                 "history=[]\n"
                 "for e in range(EPOCHS):\n"
                 "    epoch_loss=0;n=0\n"
                 "    for x1,x2,y in lo:\n"
                 "        e1=nn.functional.normalize(m(x1),dim=1)\n"
                 "        e2=nn.functional.normalize(m(x2),dim=1)\n"
                 "        d=torch.norm(e1-e2,dim=1)\n"
                 "        l=torch.mean(y*d**2+(1-y)*torch.clamp(1.0-d,min=0)**2)\n"
                 "        o.zero_grad();l.backward();o.step()\n"
                 "        epoch_loss+=l.item()*x1.size(0)\n        n+=x1.size(0)\n"
                 "    loss_val=epoch_loss/n\n"
                 "    history.append((e,loss_val))\n"
                 "    print(f'Epoch {e:3d}: loss={loss_val:.4f}')\n"
                 "    if loss_val<best_loss:\n        best_loss=loss_val\n        torch.save(m.state_dict(),'best_model.pth')\n"
                 "    if e%SAVE_EVERY==0:\n        torch.save({'epoch':e,'model':m.state_dict(),'opt':o.state_dict()},f'checkpoints/ckpt_{e:04d}.pth')\n"
                 "torch.save(m.state_dict(),'model.pth')\n"
                 + _FOOTER_REG)
        eval_ = ("import torch,numpy as np\nfrom config import *\nfrom model import {cn}\n"
                 "m={cn}();m.load_state_dict(torch.load('model.pth',weights_only=True));m.eval()\n"
                 "x1=torch.randn(5,INPUT_DIM)\n"
                 "with torch.no_grad():\n    e1=m(x1);e2=m(x1+torch.randn_like(x1)*0.1)\n    d=torch.norm(e1-e2,dim=1).mean().item()\n    print('Mean pairwise distance:'+str(d))\n")
        return data, train, eval_

    elif model_type == 'gcn':
        data = ("from config import *\nimport torch,numpy as np\nclass SynData(torch.utils.data.Dataset):\n"
                "    def __init__(self,n=200):\n        np.random.seed(42)\n"
                "        self.features=[]\n        self.labels=[]\n        self.edges=[]\n"
                "        for _ in range(n):\n"
                "            f=np.random.randn(INPUT_DIM).astype(np.float32)\n"
                "            self.features.append(f)\n"
                "            self.labels.append(np.random.randint(0,OUTPUT_DIM))\n"
                "            deg=np.random.randint(1,min(10,n))\n"
                "            neighbors=np.random.choice(n,deg,replace=False)\n"
                "            for nb in neighbors:\n                if _!=nb:\n                    self.edges.append((_,int(nb)))\n"
                "        self.features=np.array(self.features,dtype=np.float32)\n"
                "        self.labels=np.array(self.labels,dtype=np.int64)\n"
                "        self.edges=list(set(self.edges))\n"
                "    def __len__(self):return len(self.features)\n"
                "    def __getitem__(self,i):return torch.from_numpy(self.features[i]),self.labels[i]\n"
                "def build_adj(n_nodes,edges):\n"
                "    A=torch.zeros(n_nodes,n_nodes)\n"
                "    for i,j in edges:\n        A[i,j]=A[j,i]=1\n"
                "    D=A.sum(1).pow(-0.5)\n"
                "    D[torch.isinf(D)]=0\n"
                "    return torch.diag(D)@A@torch.diag(D)\n")
        train = ("import torch,torch.nn as nn,os\nfrom config import *\nfrom model import {cn}\nfrom data import SynData,build_adj\n"
                 "ds=SynData()\n"
                 "X=torch.stack([ds[i][0] for i in range(len(ds))])\n"
                 "y=torch.tensor([ds[i][1] for i in range(len(ds))])\n"
                 "adj=build_adj(len(ds),ds.edges)\n"
                 "m={cn}();total_params=sum(p.numel() for p in m.parameters())\n"
                 "print(f'Model: {total_params} parameters')\n"
                 "os.makedirs('checkpoints',exist_ok=True)\n"
                 "best_loss=float('inf')\n"
                 "o=torch.optim.Adam(m.parameters(),lr=LR)\n"
                 "criterion=nn.CrossEntropyLoss()\n"
                 "history=[]\n"
                 "for e in range(EPOCHS):\n"
                 "    out=m(X,adj)\n"
                 "    l=criterion(out,y)\n"
                 "    o.zero_grad();l.backward();o.step()\n"
                 "    with torch.no_grad():\n"
                 "        pred=m(X,adj).argmax(1)\n"
                 "        acc=(pred==y).float().mean().item()\n"
                 "        loss_val=l.item()\n"
                 "    history.append((e,loss_val,acc))\n"
                 "    print(f'Epoch {e:3d}: loss={loss_val:.4f}, acc={acc:.4f}')\n"
                 "    if loss_val<best_loss:\n        best_loss=loss_val\n        torch.save(m.state_dict(),'best_model.pth')\n"
                 "    if e%SAVE_EVERY==0:\n        torch.save({'epoch':e,'model':m.state_dict(),'opt':o.state_dict()},f'checkpoints/ckpt_{e:04d}.pth')\n"
                 "torch.save(m.state_dict(),'model.pth')\n"
                 + _FOOTER_CLS)
        eval_ = ("import torch,numpy as np\nfrom config import *\nfrom model import {cn}\nfrom data import SynData,build_adj\n"
                 "m={cn}();m.load_state_dict(torch.load('model.pth',weights_only=True));m.eval()\n"
                 "ds=SynData()\n"
                 "X=torch.stack([ds[i][0] for i in range(len(ds))])\n"
                 "y=torch.tensor([ds[i][1] for i in range(len(ds))])\n"
                 "adj=build_adj(len(ds),ds.edges)\n"
                 "with torch.no_grad():\n    a=(m(X,adj).argmax(1)==y).float().mean().item()\n    print('Test Accuracy:'+str(a))\n")
        return data, train, eval_

    else:
        raise ValueError(f'Unknown model_type: {model_type}')
