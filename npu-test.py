import torch
from intel_npu_acceleration_library.backend import check_npu_and_driver_version
from intel_npu_acceleration_library.backend.runtime import run_matmul
from intel_npu_acceleration_library.backend.factory import NNFactory
import time



def do():
    a=torch.empty(50000, 5000)
    b=torch.empty(6000, 5000)
    print(a.shape)
    print(b.shape)

    idx = 0
    print("Start")
    def test():
        time_start = time.clock_gettime(0)
        c=torch.matmul(a, b.transpose(0,1))
        # c=run_matmul(a, b)
        print(f"matmul{idx}: {time.clock_gettime(0) - time_start}")
        print(c.shape)
    
    def wrapit(func):
        def _wrapper(*args, **kwargs):
            stime = time.clock_gettime(0)
            ret = func(*args, **kwargs)
            print(f"run{idx}: {time.clock_gettime(0) - stime}")
            return ret
        return _wrapper

    NNFactory.run = wrapit(NNFactory.run)

    for i in range(10):
        test()
        idx += 1

# do()

total_diff = 0
diff_count = 0
def check():
    a = torch.randn(1000, 1000, dtype=torch.float16)
    b = torch.randn(1000, 1000, dtype=torch.float16)

    print("======================================")
    for i in range(1):
        c1 = run_matmul(a, b.transpose(0, 1))
        c2 = torch.matmul(a, b)
        # c1 = torch.nn.functional.linear(a, b)
        # c2 = torch.matmul(a, b.transpose(0,1))   #run_matmul(a, b)
        if not c1.equal(c2):
            global total_diff, diff_count
            total_diff += 1
            diff = c1.sub(c2).reshape((-1,))
            nz = diff.nonzero()
            diff_count += len(nz)
            print(f"diff: idxs={nz}, values={diff[nz]}")
        else:
            print(f"Equal!")

for i in range(10):
    check()
print(f"Total diff: {total_diff}, nz={diff_count}")
