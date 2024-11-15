import torch
# from intel_npu_acceleration_library.backend import check_npu_and_driver_version
# from intel_npu_acceleration_library.backend.runtime import run_matmul
# from intel_npu_acceleration_library.backend.factory import NNFactory
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
    # a = torch.randn(1000, 1000, dtype=torch.float16)
    # b = torch.randn(1000, 1000, dtype=torch.float16)
    a = torch.randn(100, 100, dtype=torch.float16)
    b = torch.randn(100, 100, dtype=torch.float16)

    print("======================================")
    for i in range(1):
        c1 = run_matmul(a, b.transpose(0, 1))
        c2 = torch.matmul(a, b)
        # c1 = torch.nn.functional.linear(a, b)
        # c2 = torch.matmul(a, b.transpose(0,1))   #run_matmul(a, b)
        if not c1.equal(c2):
            global total_diff, diff_count
            total_diff += 1
            diff = c1.sub(c2)
            nz = diff.nonzero(as_tuple=True)
            diff_count += len(nz)
            print(f"c1: {c1[nz].numpy()}")
            print(f"c2: {c2[nz].numpy()}")
            print(f"diff: values={diff[nz].numpy()}, idxs={nz}")
            bb = b.transpose(0,1)
            a1=a[nz[0]]
            b1=bb[nz[1]]
            result = []
            for i in range(len(a1)):
                sum = 0
                for j in range(len(a1[0])):
                    sum += a1[i][j].numpy() * b1[i][j].numpy()
                result.append(sum)
            print(f"manually result: {result}")
        else:
            print(f"Equal!")

import numpy as np

def check_transpose_precise():
    prev = torch.rand(1000, 1000, dtype=torch.float16)
    torch_after = prev.transpose(0, 1).numpy()
    prev = prev.numpy()
    manual_after = torch.zeros(1000, 1000).numpy()
    for i in range(len(prev)):
        for j in range(len(prev[0])):
            manual_after[j][i] = prev[i][j]

    diff = np.subtract(manual_after, torch_after)
    print(np.nonzero(diff))

check_transpose_precise()

# with torch.no_grad():
#     for i in range(1):
#         check()
#     print(f"Total diff: {total_diff}, nz={diff_count}")
# import numpy
# with torch.no_grad():
#     a = torch.rand(100, 100, dtype=torch.float16)
#     b = torch.rand(100, 100, dtype=torch.float16)
#     cc = torch.matmul(a, b).numpy()
#     c1 = run_matmul(a, b.transpose(0,1)).numpy()
#     c2 = [[] for i in range(len(a))]
#     for i in range(len(a)):
#         for j in range(len(a)):
#             sum = 0.0
#             for k in range(len(a)):
#                 sum += a[i][k] * b[k][j]
#             c2[i].append(sum.numpy())

#     diff_intel_manual = abs(c1 - c2)
#     diff_torch_manual = abs(cc - c2)
#     diff_torch_intel = abs(cc - c1)
#     print(f"diff_torch_intel: {len(diff_torch_intel.nonzero()[0])}, max: {diff_torch_intel.max()}, min: {diff_torch_intel.min()}")
#     print(f"diff_torch_manual: {len(diff_torch_manual.nonzero()[0])}, max: {diff_torch_manual.max()}, min: {diff_torch_manual.min()}")
#     print(f"diff_intel_manual: {len(diff_intel_manual.nonzero()[0])}, max: {diff_intel_manual.max()}, min: {diff_intel_manual.min()}")
#     # print(f"c1: {c1}")
#     # print(f"c2: {c2}")

#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

# from intel_npu_acceleration_library.backend import MatMul
# import numpy as np


# def run_matmul(inC, outC, batch):

#     # Create both inputs
#     X1 = np.random.uniform(-1, 1, (batch, inC)).astype(np.float16)
#     X2 = np.random.uniform(-1, 1, (outC, inC)).astype(np.float16)

#     mm = MatMul(inC, outC, batch, profile=False)

#     intel = mm.run(X1, X2)
#     intel = torch.tensor(intel)

#     tor = torch.matmul(torch.tensor(X1), torch.tensor(X2).transpose(-2, -1))
#     print(tor.equal(intel))
#     sub = tor.sub(intel)
#     print(sub.shape)
#     diff_idx = sub.nonzero()
#     print(len(diff_idx))
#     print(diff_idx)
#     print(sub[sub.nonzero(as_tuple=True)])



# if __name__ == "__main__":
#     with torch.no_grad():
#         result = run_matmul(128, 128, 32)
#         print(result)
