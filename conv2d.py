import numpy as np
import pickle

class _conv:
    def __init__(self, kernel=None, bias=None, size=(1, 1), stride=1, optim='SGD'):
        if kernel is None:
            if len(size) == 1:
                size = (size[0], size[0])
            self.kernel = np.random.randn(*size)
        else:
            self.kernel = kernel

        if bias == None:
            self.req_b = False
            self.bias = 0
        else:
            self.req_b = True
            self.bias = bias

        self.size   = self.kernel.shape
        self.size_0, self.size_1 = self.size
        self.stride = stride

        self.grad  = np.zeros_like(self.kernel)
        self.gradB = 0

        self.vB = 0
        self.mB = 0
        self.vW = np.zeros_like(self.kernel)
        self.mW = np.zeros_like(self.kernel)
        self.t  = 0

        self.optimize  = getattr(self, optim)

    def zeroGrad(self):
        self.grad  = np.zeros_like(self.kernel)
        self.gradB = 0

    def optimZeroGrad(self):
        self.vB = 0
        self.mB = 0
        self.vW = np.zeros_like(self.kernel)
        self.mW = np.zeros_like(self.kernel)

    def SGD(self, lr=0.1, lmda=0.0):
        self.bias   -= lr * self.gradB
        self.kernel  = self.kernel - lr * self.grad - lmda * self.kernel

    def momentumGD(self, lr=0.1, mu=0.9, lmda=0.0):
        self.mW = mu * self.mW - lr * self.grad
        self.mB = mu * self.mB - lr * self.gradB
        self.bias  += self.mB
        self.kernel = self.kernel + self.mW - lmda * self.kernel
        return [self.kernel, self.bias]

    def rmsProp(self, lr=0.1, rho=0.99, eps=1e-8, lmda=0.0):
        self.vW = rho * self.vW + (1 - rho) * (self.grad ** 2)
        self.vB = rho * self.vB + (1 - rho) * (self.gradB ** 2)
        self.bias    = self.bias   - (lr / (self.vB ** 0.5 + eps)) * self.gradB
        self.kernel  = self.kernel - (lr / (self.vW ** 0.5 + eps)) * self.grad - lmda * self.kernel
        return [self.kernel, self.bias]
   
    def adam(self, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8, lmda=0.0):
        self.t += 1
        self.mW = beta1 * self.mW + (1 - beta1) * self.grad
        self.mB = beta1 * self.mB + (1 - beta1) * self.gradB
        self.vW = beta2 * self.vW + (1 - beta2) * (self.grad ** 2)
        self.vB = beta2 * self.vB + (1 - beta2) * (self.gradB ** 2)

        self.bias   = self.bias   - (lr * self.mB / (self.vB ** 0.5 + eps))
        self.kernel = self.kernel - (lr * self.mW / (self.vW ** 0.5 + eps))
   
        return [self.kernel, self.bias]

    def __call__(self, x):
        self.localGrad = x
        h, w  = x.shape
        self.out_h =  ((h - self.size_0) // self.stride) + 1
        self.out_w =  ((w - self.size_1) // self.stride) + 1
        out   = np.zeros((self.out_h, self.out_w))

        for i in range(self.out_h):
            for j in range(self.out_w):
                window = x[i*self.stride: i*self.stride+self.size_0, 
                           j*self.stride: j*self.stride+self.size_1]    
                out[i, j] = np.sum(window * self.kernel)

        self.z = out + self.bias
        return self.z

    def backward(self, in_grad):
        self.gradB += np.sum(in_grad) if self.req_b else 0
        out_grad   = np.zeros_like(self.localGrad, dtype=float)
        m, n       = in_grad.shape

        for i in range(m):
            for j in range(n):
                start_i = i * self.stride
                start_j = j * self.stride
                out_grad[start_i: start_i+self.size_0, start_j: start_j+self.size_1] += self.kernel * in_grad[i, j]
                self.grad += self.localGrad[start_i: start_i+self.size_0, start_j: start_j+self.size_1] * in_grad[i, j]

        return out_grad

  

class Conv2d:
    def __init__(self, in_channels=1, out_channels=1, kernel_size=(1, 1), stride=1, 
                 padding=(0, 0), bias=True, pad_mode='constant', act='relu', optim='SGD', learn=True):
        '''
        in_channels:
        out_channels:
        kernel_size:
        stride:
        padding:
        bias:
        pad_mode:
        act:
        optim:
        '''
        self.learn = learn
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = [[padding[0], padding[0]], [padding[1], padding[1]]]
        self.p1, self.p2 = padding
        self.bias = bias
        self.pad_mode = pad_mode

        self.structure =  [[_conv(size=kernel_size, stride=stride, bias=1 if bias else False, optim=optim) 
                         for _ in range(in_channels)] for _ in range(out_channels)]
        
        self.kernel = np.array([[ch.kernel for ch in kernel] for kernel in self.structure])
        
        self.act = act
        self.optimize = getattr(self, optim)
        if not act:
            self.activation = lambda x: x
            self.activation_grad = self.no_act_grad
        else:
            self.activation = getattr(self, act)
            self.activation_grad = getattr(self, act+'_grad')

    def pad(self, x):
        out = []
        for ch in x:
            out.append(np.pad(ch, self.padding, self.pad_mode))
        return np.array(out)

    def initWeight(self, w):
        for kernel, w_kernel in zip(self.structure, w):
            for ch, w_ch in zip(kernel, w_kernel):
                ch.kernel = w_ch

    def initBias(self, b):
        for kernel, w_kernel in zip(self.structure, b):
            for ch, w_ch in zip(kernel, w_kernel):
                ch.bias = w_ch

    def getWeight(self):
        return np.array([[ch.kernel for ch in kernel] for kernel in self.structure])
    
    def getWeightGrad(self):
        return np.array([[ch.grad for ch in kernel] for kernel in self.structure])
   
    def getBias(self):
        return np.array([[ch.bias for ch in kernel] for kernel in self.structure])

    def getBiasGrad(self):
        return np.array([[ch.gradB for ch in kernel] for kernel in self.structure])

    def zeroGrad(self):
        for kernel in self.structure:
            for ch in kernel:
                ch.zeroGrad()

    def optimZeroGrad(self):
        for kernel in self.structure:
            for ch in kernel:
                ch.optimZeroGrad()

    def relu(self, x):
        return x * (x > 0)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def relu_grad(self):
        return 1. * (self.a > 0)
    
    def sigmoid_grad(self):
        return self.a * (1 - self.a)

    def no_act_grad(self):
        return np.ones_like(self.a)

    def SGD(self, *args, **kwargs):
        for kernel in self.structure:
            for ch in kernel:
                ch.optimize(*args, **kwargs)

    def momentumGD(self, *args, **kwargs):
        for kernel in self.structure:
            for ch in kernel:
                ch.optimize(*args, **kwargs)

    def rmsProp(self, *args, **kwargs):
        for kernel in self.structure:
            for ch in kernel:
                ch.optimize(*args, **kwargs)

    def adam(self, *args, **kwargs):
        for kernel in self.structure:
            for ch in kernel:
                ch.optimize(*args, **kwargs)

    def __call__(self, x):
        x = self.pad(x)
        self.localGrad = x
        out = []
        for kernel in self.structure:
            out_channel = kernel[0](x[0])
            if self.in_channels > 1:
                for channel, k_channel in zip(x[1:], kernel[1:]):
                    out_channel += k_channel(channel)
            self.a = self.activation(out_channel)
            out.append(self.a)
        return np.array(out)
    
    def backward(self, in_grad):
        in_grad = self.activation_grad() * in_grad
        out_grad = np.zeros_like(self.localGrad)
        for kernel, channel in zip(self.structure, in_grad):
            for i, k_channel in enumerate(kernel):
                out_grad[i] += k_channel.backward(channel)
        if self.p1 > 0 and self.p2 > 0:
            return out_grad[:, self.p1: -self.p1, self.p2:- self.p2]
        return out_grad

class flatten:
    def __init__(self):
        self.learn = False

    def __call__(self, x):
        # no learnable parameters
        self.shape = x.shape
        return np.reshape(x, (-1))

    def backward(self, in_grad):
        return np.reshape(in_grad, self.shape)


class view:
    def __init__(self, shape):
        # no learnable parameters
        self.learn = False
        self.newshape = shape
    
    def __call__(self, x):
        self.in_shape = x.shape
        return np.reshape(x, self.newshape)
    
    def backward(self, in_grad):
        return np.reshape(in_grad, self.in_shape)


def MSELoss(pred, target):
    grad = 2 * (pred - target) / np.prod(pred.shape)
    return np.mean((pred - target)**2), grad





