from config import using_intel_npu_acceleration_library, using_statistic
from functools import partial
from collections import OrderedDict
import gc
import time
import numpy as np
import inspect
from torch.nn import functional
import torch

if using_intel_npu_acceleration_library:
    from intel_npu_acceleration_library.device import implements_factory
    from intel_npu_acceleration_library.nn.functional import layer_norm, linear
    from intel_npu_acceleration_library.backend.runtime import run_factory, run_matmul, _model_cache
    from intel_npu_acceleration_library.backend.matmul import MatMul
    from intel_npu_acceleration_library.backend.factory import NNFactory

class Statistic:
    def __init__(self) -> None:
        self.tensor_diff_count = 0
        self.element_diff_count = 0
        self.min_diff = np.inf
        self.max_diff = -np.inf

    def torch_diff(self, c1, c2):
        if not c1.equal(c2):
            self.tensor_diff_count += 1
            diff = c1.sub(c2).reshape((-1,))
            nz = diff.nonzero()
            self.element_diff_count += len(nz)
            min = torch.min(diff[nz])
            max = torch.max(diff[nz])
            self.min_diff = self.min_diff if self.min_diff < min else min
            self.max_diff = self.max_diff if self.max_diff > max else max
            print(f"diff[{c1.shape}<==>{c2.shape}]: count={len(nz)}, min={min}, max={max}")
            # print(f"diff[{c1.shape}<==>{c2.shape}]: idxs={nz}, values={diff[nz]}")
        else:
            print(f"Equal[{c1.shape}<==>{c2.shape}]")

    def show(self):
        print(f"Total: tensor={self.tensor_diff_count}, element={self.element_diff_count}, min={self.min_diff}, max={self.max_diff}")


# implements_factory(torch.device)()
def patching_linear(func):
    def my_linear(x, weight, bias=None):
        # return run_factory(x, weight, partial(MatMul), None)
        base = run_matmul(x, weight)
        if using_statistic:
            statistic.torch_diff(base, func(x, weight))
        return base if bias is None else base + bias
    return my_linear

def patching_matmul(func):
    def _wrapper(input, other, *args, out=None):
        ret = run_matmul(input, other.transpose(-2, -1))
        if using_statistic:
           statistic.torch_diff(ret, func(input, other))
        return ret

    return _wrapper

class LRE:
    def __init__(self) -> None:
        self.queue = OrderedDict()
        self.total = 0

    def hit(self, key, value=None):
        target = self.queue.get(key, None)
        if not target:
            self.queue[key] = value
            shape = [value.batch, value.inC, value.outC]
            self.total += accumulate(shape)
            print(f"Add MATRIX: {shape}, total={self.total}")
        self.queue.move_to_end(key, last=False)
        
    def pop(self):
        key, ret = self.queue.popitem()
        shape = [ret.batch, ret.inC, ret.outC]
        self.total -= accumulate(shape)
        print(f"Remove MATRIX: [{key}]{shape}, total={self.total}")
        return key, ret

    def overflow(self, shape):
        return self.total + accumulate(shape) >= 1000000000


def get_key(self):
    op_class_name, batch, inC, outC, dtype = self.__class__.__name__, self.batch, self.inC, self.outC, "float16"
    return f"{str(op_class_name)}_{batch}_{inC}_x_{outC}_{inC}_{dtype}"

def accumulate(shape):
    return sum([shape[0] * item for item in shape[1:]]) + accumulate(shape[1:]) if len(shape)>1 else 0

def popup():
    global history
    key, value = history.pop()
    v = _model_cache.pop(key)
    assert v[0] is value
    del value
    gc.collect()

def prepare(self):
    global history
    shape = self.batch, self.inC, self.outC
    while True:
        if history.overflow(shape):
            print(f"Overflow for [{self.__class__.__name__}]{shape}")
            popup()
        else:
            break

def patching_compile(func):
    def _wrapper(self, *args, **kwargs):        
        while True:
            retry = False
            try:
                global history
                print(f"{'Retry' if retry else 'Start'} compile[{self.__class__}]: {self.batch}x{self.inC}x{self.outC}")
                prepare(self)
                ret = func(self, *args, **kwargs)

                key = get_key(self)
                history.hit(key, self)

                return ret
            except Exception as e:
                retry = True
                print(f"Exception: {e}")
                popup()

    return _wrapper

def patching_run(func):
    def _wrapper(self, *args, **kwargs):
        global history
        key = get_key(self)
        history.hit(key)

        ret = func(self, *args, **kwargs)
        return ret
    return _wrapper

def patching(container, member, replacement):
    origin = getattr(container, member)
    if inspect.ismethod(origin):
        new = partial(replacement, partial(origin.__func__, origin.__self__))
    else:
        new = partial(replacement, origin)
    setattr(container, member, new)

def wrap_show(container, member):
    def show(caller, *args, **kwargs):
        name = getattr(container, '__name__', container.__class__.__name__)
        start_time = time.asctime()
        print(f">>>>>>>{name}.{member} start: {start_time}")
        ret = caller(*args, **kwargs)
        end_time = time.asctime()
        diff_time = time.mktime(time.strptime(end_time)) - time.mktime(time.strptime(start_time))
        print(f"<<<<<<<{name}.{member} end: {end_time}, consume={diff_time}")
        return ret

    patching(container, member, show)



statistic = Statistic()
history = LRE()

if using_intel_npu_acceleration_library:
    functional.linear = patching_linear(functional.linear)   # linear(matmul)
    torch.matmul = patching_matmul(torch.matmul)
    NNFactory.compile = patching_compile(NNFactory.compile)
