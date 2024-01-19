import torch
from pyLiam import LKTimer
import copy
timer=LKTimer.LKTimer()

'''
    Results:
    
    10 matrices 5*5
    normal
    LKTimer : 0.0001 seconds
    streams
    LKTimer : 0.0109 seconds

'''

assert torch.cuda.is_available()

n_matrices=512
matrix_size=(300,300)
matrices_per_stream=1


normal_mats=[]
streams_mats=[]
for i in range(n_matrices):
    mat = torch.ones(matrix_size, device='cuda:0', requires_grad=True) * i
    normal_mats.append(mat)
    mat = torch.ones(matrix_size, device='cuda:0', requires_grad=True) * i
    streams_mats.append(mat)

with timer.time("warmup"):
    for i in range(n_matrices):
        # normal_mats[i] = normal_mats[i]*normal_mats[i]
        normal_mats[i] = normal_mats[i].mm(normal_mats[i])
        # print("ha")

normal_mats=[]
for i in range(n_matrices):
    mat = torch.ones(matrix_size, device='cuda:0', requires_grad=True) * i
    normal_mats.append(mat)

with timer.time("normal"):
    for i in range(n_matrices):
        # normal_mats[i] = normal_mats[i]*normal_mats[i]
        normal_mats[i] = normal_mats[i].mm(normal_mats[i])
        # print("ha")

with timer.time("streams"):
    streams=[]
    for i in range(int(n_matrices/matrices_per_stream)):
        s = torch.cuda.Stream(device='cuda:0')
        streams.append(s)

    multiplied_matrixes=[]
    for i, s in enumerate(streams):
        with torch.cuda.stream(s):
            for k in range(matrices_per_stream):
                # streams_mats[i*matrices_per_stream + k] = streams_mats[i*matrices_per_stream + k]*streams_mats[i*matrices_per_stream + k]
                streams_mats[i*matrices_per_stream + k] = streams_mats[i*matrices_per_stream + k].mm(streams_mats[i*matrices_per_stream + k])
            # print(f"stream {i} done!")

    # synchronize streams before summing
    torch.cuda.synchronize()

for i in range(n_matrices):
    # print(normal_mats[i].sum(), streams_mats[i].sum())
    assert normal_mats[i].sum() == streams_mats[i].sum()
    

cool=torch.stack(normal_mats)
cool2=torch.stack(streams_mats)

cool.sum().backward()
cool2.sum().backward()
