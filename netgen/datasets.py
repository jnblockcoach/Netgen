"""Dataset providers for real-world and synthetic data sources.

Each dataset provider returns (code_string, input_dim, output_dim).
"""
from typing import Optional, Tuple


def _sklearn_template(dataset_name: str, loader_expr: str,
                      n_features: int, n_classes: int) -> Tuple[str, int, int]:
    """Template for sklearn toy datasets (iris, wine, breast_cancer)."""
    code = (
        f"from config import *\n"
        f"import torch\n"
        f"from sklearn.datasets import {loader_expr}\n"
        f"from sklearn.preprocessing import StandardScaler\n"
        f"\n"
        f"class SynData(torch.utils.data.Dataset):\n"
        f"    def __init__(self):\n"
        f"        data = {loader_expr}()\n"
        f"        X = data.data.astype('float32')\n"
        f"        y = data.target.astype('int64')\n"
        f"        X = StandardScaler().fit_transform(X)\n"
        f"        self.X, self.y = X, y\n"
        f"    def __len__(self):\n"
        f"        return len(self.X)\n"
        f"    def __getitem__(self, i):\n"
        f"        return torch.from_numpy(self.X[i]), self.y[i]\n"
    )
    return code, n_features, n_classes


def _make_classification_template(fn_name: str, fn_args: str,
                                   n_features: int, n_classes: int) -> Tuple[str, int, int]:
    """Template for sklearn generator functions (moons, circles, blobs)."""
    code = (
        f"from config import *\n"
        f"import torch\n"
        f"from sklearn.datasets import {fn_name}\n"
        f"\n"
        f"class SynData(torch.utils.data.Dataset):\n"
        f"    def __init__(self):\n"
        f"        X, y = {fn_name}({fn_args})\n"
        f"        self.X = X.astype('float32')\n"
        f"        self.y = y.astype('int64')\n"
        f"    def __len__(self):\n"
        f"        return len(self.X)\n"
        f"    def __getitem__(self, i):\n"
        f"        return torch.from_numpy(self.X[i]), self.y[i]\n"
    )
    return code, n_features, n_classes


# ── Dataset registry ──

_DATASET_REGISTRY = {
    'iris':          lambda: _sklearn_template('iris', 'load_iris', 4, 3),
    'wine':          lambda: _sklearn_template('wine', 'load_wine', 13, 3),
    'breast_cancer': lambda: _sklearn_template('breast_cancer', 'load_breast_cancer', 30, 2),
    'moons':         lambda: _make_classification_template(
        'make_moons', 'n_samples=2000, noise=0.1, random_state=42', 2, 2),
    'circles':       lambda: _make_classification_template(
        'make_circles', 'n_samples=2000, noise=0.1, random_state=42', 2, 2),
    'blobs':         lambda: _make_classification_template(
        'make_blobs', 'n_samples=2000, n_features=6, centers=5, random_state=42', 6, 5),
    'mnist': lambda: (
        "from config import *\n"
        "import torch\n"
        "from torchvision import datasets, transforms\n"
        "\n"
        "class SynData(torch.utils.data.Dataset):\n"
        "    def __init__(self, train=True):\n"
        "        t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])\n"
        "        data = datasets.MNIST(root='./data', train=train, download=True, transform=t)\n"
        "        self.X, self.y = data.data.float().unsqueeze(1) / 255.0, data.targets\n"
        "    def __len__(self):\n"
        "        return len(self.X)\n"
        "    def __getitem__(self, i):\n"
        "        return self.X[i], self.y[i]\n"
    , 1, 10),
    'cifar10': lambda: (
        "from config import *\n"
        "import torch\n"
        "from torchvision import datasets, transforms\n"
        "\n"
        "class SynData(torch.utils.data.Dataset):\n"
        "    def __init__(self, train=True):\n"
        "        t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])\n"
        "        data = datasets.CIFAR10(root='./data', train=train, download=True, transform=t)\n"
        "        imgs = torch.stack([data[i][0] for i in range(len(data))])\n"
        "        self.X, self.y = imgs, torch.tensor([data[i][1] for i in range(len(data))])\n"
        "    def __len__(self):\n"
        "        return len(self.X)\n"
        "    def __getitem__(self, i):\n"
        "        return self.X[i], self.y[i]\n"
    , 3, 10),
    'text': lambda: (
        "from config import *\n"
        "import torch, numpy as np\n"
        "\n"
        "class SynData(torch.utils.data.Dataset):\n"
        "    def __init__(self, n=2000):\n"
        "        np.random.seed(42)\n"
        "        vocab = list('abcdefghijklmnopqrstuvwxyz ')\n"
        "        self.X, self.y = [], []\n"
        "        for _ in range(n):\n"
        "            seq = [np.random.randint(0, len(vocab)) for _ in range(INPUT_DIM)]\n"
        "            k = np.random.randint(0, OUTPUT_DIM)\n"
        "            seq[0] = k\n"
        "            seq[-1] = k\n"
        "            self.X.append(np.array(seq, dtype=np.int64))\n"
        "            self.y.append(k)\n"
        "        self.X = np.array(self.X)\n"
        "        self.y = np.array(self.y, dtype=np.int64)\n"
        "    def __len__(self):\n"
        "        return len(self.X)\n"
        "    def __getitem__(self, i):\n"
        "        return torch.from_numpy(self.X[i]), self.y[i]\n"
    , 20, 10),
    'line': lambda: (
        "from config import *\n"
        "import torch, numpy as np\n"
        "\n"
        "class SynData(torch.utils.data.Dataset):\n"
        "    def __init__(self, n=2000):\n"
        "        np.random.seed(42)\n"
        "        self.X = np.random.randn(n, 1).astype(np.float32)\n"
        "        self.y = (2 * self.X + 1 + 0.1 * np.random.randn(n, 1)).astype(np.float32)\n"
        "    def __len__(self):\n"
        "        return len(self.X)\n"
        "    def __getitem__(self, i):\n"
        "        return torch.from_numpy(self.X[i]), torch.from_numpy(self.y[i])\n"
    , 1, 1),
}


# Image datasets: input dim when flattened (vector-friendly architectures)
_IMAGE_FLAT_DIMS = {
    'mnist':   1 * 28 * 28,   # 784
    'cifar10': 3 * 32 * 32,   # 3072
}


def get_dataset_code(dataset: str, inp: int, outp: int,
                     flat: bool = False) -> Tuple[Optional[str], int, int]:
    """Get dataset code and dimensions for a named dataset.

    Args:
        dataset: Dataset name (e.g. 'iris', 'mnist', 'syn').
        inp: Default input dimension (used if dataset is 'syn').
        outp: Default output dimension (used if dataset is 'syn').
        flat: For image datasets, flatten each sample to a 1-D vector
            (e.g. MNIST 1x28x28 -> 784) so vector architectures work.

    Returns:
        Tuple of (code_string_or_None, input_dim, output_dim).
        If code is None, use the synthetic data template.
    """
    if dataset == 'syn' or dataset not in _DATASET_REGISTRY:
        return None, inp, outp
    code, in_dim, out_dim = _DATASET_REGISTRY[dataset]()
    if flat and dataset in _IMAGE_FLAT_DIMS:
        # Flatten samples to 1-D vectors in __getitem__
        code = code.replace("        return self.X[i], self.y[i]\n",
                            "        return self.X[i].view(-1), self.y[i]\n")
        in_dim = _IMAGE_FLAT_DIMS[dataset]
    return code, in_dim, out_dim


def list_datasets() -> list:
    """Return list of all available dataset names."""
    return ['syn'] + sorted(_DATASET_REGISTRY.keys())
