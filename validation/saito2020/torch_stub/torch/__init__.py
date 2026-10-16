# Minimal stand-in so that obp 0.5.0 imports without PyTorch.
# No PyTorch code path (NN policy learners) is used by the OPE benchmark.
import sys, types as _t
class _M(type):
    def __getattr__(cls, n):
        if n.startswith('__'): raise AttributeError(n)
        return _D
    def __getitem__(cls, k): return _D
class _D(metaclass=_M):
    def __init__(self, *a, **k): pass
    def __call__(self, *a, **k): return _D()
    def __getattr__(self, n):
        if n.startswith('__'): raise AttributeError(n)
        return _D
def __getattr__(n):
    if n.startswith('__'): raise AttributeError(n)
    return _D
for m in ["torch.nn", "torch.optim", "torch.nn.functional", "torch.utils", "torch.utils.data"]:
    mod = _t.ModuleType(m); mod.__getattr__ = __getattr__; sys.modules[m] = mod
nn = sys.modules["torch.nn"]; optim = sys.modules["torch.optim"]; utils = sys.modules["torch.utils"]
utils.data = sys.modules["torch.utils.data"]; nn.functional = sys.modules["torch.nn.functional"]
