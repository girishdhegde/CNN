import numpy as np
import matplotlib.pyplot as plt
from conv2d import Conv2d, MSELoss
import cv2

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# params
in_ch = 1
out_ch = 1
k_sz = (3, 3)
stride = 1
act = None
optim = 'adam'
lr = 1e-1

out_fldr = './out'

# sobel detection kernel
# my_kernel = [[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]] for _ in range(in_ch)] for _ in range(out_ch)]
# my_kernel = [[[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]] for _ in range(in_ch)] for _ in range(out_ch)]
# my_kernel = np.array(my_kernel, dtype=np.float32)

# layer init
my_conv = Conv2d(in_ch, out_ch, k_sz, stride, bias=False, padding=(0, 0), act=act, optim=optim)

# weight init
my_conv.initWeight(np.random.uniform(-1, 1, size=(out_ch, in_ch, k_sz[0], k_sz[1])))

my_inp = cv2.imread('F:/gitrepo/CNN/sobel.PNG', 0)
my_inp = np.array([[my_inp]]).astype(np.float32)
my_target = np.load('./target.npy').astype(np.float32) 


def train(i):
    my_conv.zeroGrad()
    my_out = my_conv(my_inp[0])
    my_loss, my_grad = MSELoss(my_out, my_target[0])
    my_conv.backward(my_grad)
    my_conv.optimize(lr)
    print(f'iteration: {i+1}, Loss: {my_loss}')
    cv2.imwrite(f'{out_fldr}/itr{i}.png', np.clip(my_out[0], 0, 255).astype(np.uint8))

print('initial weight:\n', my_conv.getWeight())

for i in range(50):
    train(i)

print('trained weight:\n', my_conv.getWeight())