import numpy as np
import pickle
import matplotlib.pyplot as plt

import nn

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

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
class MaxPool2d:
    def __init__(self, size=(2, 2), stride=2):
        # no learnable parameters
        self.learn = False
        self.s_0, self.s_1 = size
        self.stride = stride
    
    def __call__(self, x):
        bs, chn, h, w = x.shape
        outh, outw = ((h - self.s_0) // self.stride) + 1, ((w - self.s_1) // self.stride) + 1
        out = np.zeros((bs, chn, outh, outw))
        self.mask = np.zeros_like(x)
        for img_idx, img in enumerate(x):
            for ch_idx, ch in enumerate(img):
                for i, start_i in enumerate(range(0, x.shape[0], self.s_0)):
                    for j, start_j in enumerate(range(0, x.shape[1], self.s_1)):
                        idx_i, idx_j = np.argmax(ch[start_i: start_i+self.s_0, start_i: start_i+self.s_0])
                        self.mask[img_idx, ch_idx, start_i+idx_i, start_j+idx_j] = 1.0
                        out[img_idx, ch_idx, i, j] = ch[start_i+idx_i, start_j+idx_j]
        return out
    
    def backward(self, in_grad):
        out_grad = np.zeros_like(self.mask)
        for img_idx, img in enumerate(self.mask):
            for ch_idx, ch in enumerate(img):
                for i, start_i in enumerate(range(0, out_grad.shape[0], self.s_0)):
                    for j, start_j in enumerate(range(0, out_grad.shape[1], self.s_1)):
                        idx_i, idx_j = np.argmax(ch[start_i: start_i+self.s_0, start_i: start_i+self.s_0])
                        out_grad[img_idx, ch_idx, start_i+idx_i, start_j+idx_j] = in_grad[img_idx, ch_idx, i, j]      
        return out_grad
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


class Upsample:
    def __init__(self, interpolatioin='nn'):
        pass

    def __call__(self, x):
        pass

    def backward(self, in_grad):
        pass

class CNN:
    def __init__(self, criterion='MSELoss', optim='SGD'):
        self.structure = []
        self.criterion = getattr(self, criterion)
        self.optimize  = getattr(self, optim)

    def __call__(self, x, target):
        self.b_size = len(x)
        pred = []
        loss = 0
        for xi, yi in zip(x, target):
            out = xi.copy()
            for L in self.structure[:-1]:
                out = L(out)
            if isinstance(self.structure[-1], Conv2d):
                # if last layer is cnn
                out = self.structure[-1](out)
                loss += self.criterion(out, yi)
                self.backward()
            else:
                # if last layer is nn
                out, loss_nn = self.structure[-1](np.array([out]), target)
                loss += loss_nn * self.b_size
                in_grad = self.structure[-1].out_grad
                self.backward(in_grad)
            pred.append(out)
        return np.array(pred), loss / self.b_size

    def MSELoss(self, pred, target):
        ''' For regression/prediction '''
        shape = target.shape
        # image shape -> ch, h, w
        n = shape[0] * shape[1] * shape[2] 
        self.loss_grad = 2 * (pred - target) / n
        return np.mean((pred - target)**2)

    def BCELoss(self, pred, target):
        eps = 1e-9
        ''' For Binary Classification with Sigmoid output layer '''
        shape = target.shape
        n = shape[2] * shape[3]
        self.loss_grad = (pred - target) / (n * self.structure[-1].sigmoid_grad())
        loss = np.sum(target * np.log(pred + eps)) + np.sum((1 - target) * np.log(1 - pred + eps))
        return -np.mean(loss)
    
    def CrossEntropyLoss(self, *args, **kwargs):
        ''' implemented only for nn'''
        pass

    def backward(self, in_grad=None):
        if in_grad is not None:
            self.loss_grad = in_grad
        grad = self.loss_grad.copy()
        struct = self.structure[::-1]
        if isinstance(struct[0], Conv2d):
            grad = struct[0].backward(grad)
        for L in struct[1:]:
            grad = L.backward(grad)
        return grad

    def zeroGrad(self):
        for L in self.structure:
            if hasattr(L, 'learn'):
                if L.learn:
                    L.zeroGrad()
            else:
                L.zeroGrad()

    def optimZeroGrad(self):
        for L in self.structure:
            if hasattr(L, 'learn'):
                if L.learn:
                    L.optimZeroGrad()
            else:
                L.optimZeroGrad()

    def SGD(self, lr=0.1, lmda=0.0):
        '''
        lr  : learning rate
        lmda: weight decay
        '''
        # mini-batch optimizer
        lr /= self.b_size
        for L in self.structure:
            L.SGD(lr, lmda)

    def momentumGD(self, lr=0.1, mu=0.9, lmda=0.0):
        '''
        lr: learning rate
        mu: momentum coefficient
        '''
        # mini-batch optimizer
        lr /= self.b_size
        for L in self.structure:
            L.momentumGD(lr, mu, lmda)

    def rmsProp(self, lr=0.1, rho=0.9, eps=1e-8, lmda=0.0):
        '''
        lr : learning rate
        rho: velocity coefficient
        eps: epsilon
        '''
        # mini-batch optimizer
        lr /= self.b_size
        for L in self.structure:
            L.rmsProp(lr, rho, eps, lmda)

    def adam(self, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8, lmda=0.0):
        '''
        lr   : learning rate
        beta1: first momentum
        beta2: second momentum
        eps  : epsilon
        '''
        # mini-batch optimizer
        lr /= self.b_size
        for L in self.structure:
            if hasattr(L, 'adam'):
                L.adam(lr, beta1, beta2, eps, lmda)

    def predict(self, x, classification=True):
        pred = []
        for xi in x:
            out = xi
            for L in self.structure[:-1]:
                out = L(out)
            if isinstance(self.structure[-1], Conv2d):
                out = self.structure[-1](out)
            pred.append(out)
        if not isinstance(self.structure[-1], Conv2d):
            pred = self.structure[-1].predict(pred)
        pred = np.array(pred)
        return pred

    def save(self, path='model.pkl'):
        weight = []
        for layer in self.structure[:-1]:
            if layer.learn:
                weight.append([layer.getWeight, layer.getBias])
        last = self.structure[-1]
        if isinstance(last, Conv2d):
            weight.append([last.getWeight, last.getBias])
        else:
            wt = last.save(path)
            weight.append(wt)
        with open(path, 'wb') as f:
            pickle.dump(weight, f)

    def load(self, path='model.pkl'):
        with open(path, 'rb') as f:
            weight = pickle.load(f)
        cnt = 0
        for layer in self.structure[:-1]:
            if layer.learn:
                layer.initWeight(weight[cnt][0])
                layer.initBias(weight[cnt][1])
                cnt += 1
        last = self.structure[-1]
        if isinstance(last, Conv2d):
            last.initWeight(weight[cnt][0])
            last.initBias(weight[cnt][1])
        else:
            last.load(weight=weight[cnt])

if __name__ == '__main__':
    class net(CNN):
        def __init__(self, criterion='MSELoss', optim='adam'):
            super().__init__(criterion='MSELoss', optim='adam')
            self.structure = [
                                Conv2d(1, 8, (3, 3), 1, act='relu', optim=optim), 
                                Conv2d(8, 16, (3, 3), 2, act='relu', optim=optim),
                                Conv2d(16, 32, (3, 3), 2, act='relu', optim=optim),
                                Conv2d(32, 4, (1, 1), 1, act='relu', optim=optim),
                                flatten(),
                                nn.nn([100, 100, 10], ['relu', 'softmax'], 'CrossEntropyLoss')
                             ]
        

    model = net()
    # import cv2
    # # img = cv2.imread('./hs.tiff')
    # img = np.random.randn(28, 28, 1)
    # inp = np.transpose(img, (2, 0, 1)).astype(np.float32)
    # inp = np.array([inp])
    # print(inp.shape)
    # for i in range(10):
    #     print(model.predict(inp))

    #     # out = model(inp, np.array([8]))
    #     print(i)

    import pandas as pd

    data = pd.read_csv('./kannada_dataset/train.csv')
    data = np.array(data)
    index = np.arange(len(data))
    np.random.shuffle(index)
    data = data[index]

    x = data[:1000, 1:].reshape(-1, 1, 28, 28) / 255
    y = data[:1000, 0].astype(np.int32)

    print(x.shape)
    print(y.shape)

    bs = 10
    lr = 1e-2
    iterations = len(y) // bs

    # model.load()

    for e in range(1):
        Loss = 0    
        for i in range(iterations):
            start = i * bs
            model.zeroGrad()
            out, loss = model(x[start: start+bs], y[start: start+bs])
            Loss += loss
            model.adam(lr)
            model.save()
            print(f'batch: {i}/{iterations}, loss: {loss}')
        print(f'epoch: {e}, loss: {Loss/iterations}')





    # import cv2
    # img = cv2.imread('./hs.tiff')
    # inp = np.transpose(img, (2, 0, 1)).astype(np.float32)
    # print(inp.shape)

    # stride = 3
    # i_ch = 3
    # o_ch = 2
    # kernel = np.array([[-1.0, -2, -1], [0, 0, 0], [1, 2, 1]])
    # kernel = np.array([kernel for _ in range(i_ch)])
    # kernel = np.array([kernel for _ in range(o_ch)], dtype=np.float32)

    # cnn = Conv2d(i_ch, o_ch, (3, 3), stride, padding=(0, 0), bias=False, act='sigmoid', optim='rmsProp')
    # cnn.initWeight(kernel)
    # out = cnn(inp)
    # grd = cnn.backward(np.ones_like(out, dtype=np.float32))
    # # cnn.optimize(lr=0.1)

    # out = cnn(inp)
    # grd += cnn.backward(np.ones_like(out, dtype=np.float32))
    # cnn.optimize(lr=0.1)
    # print(out.shape, grd.shape)


    # import torch
    # import torch.nn as nn
    # import torch.nn.functional as F


    # tkernel = torch.tensor(kernel, dtype=float, requires_grad=True)
    # # bias = torch.tensor([100], dtype=float, requires_grad=True)
    # timg    = torch.tensor([inp], requires_grad=True, dtype=float)
    # class net(nn.Module):
    #     def __init__(self, ):
    #         super().__init__()
    #         self.conv = nn.Conv2d(i_ch, o_ch, (3, 3), stride, bias=None)
    #         self.conv.weight.data = tkernel
    #     def forward(self, x):
    #         # return torch.relu(self.conv(x))
    #         return torch.sigmoid(self.conv(x))

    # net = net()
    # sgd = torch.optim.SGD(net.parameters(), lr=0.1)
    # rms = torch.optim.RMSprop(net.parameters(), lr=0.1)
    # tout = net(timg)
    # tout.backward(torch.ones_like(tout))
    # # sgd.step()
    # # rms.step()
    # tout = net(timg)
    # tout.backward(torch.ones_like(tout))
    # # sgd.step()
    # rms.step()
    # # tgrad = tkernel.grad.numpy()
    # tgrad = net.conv.weight.grad.numpy()
    # print('kernel close:', np.allclose(net.conv.weight.data.numpy(), cnn.getWeight()))
    # print('input grad close:', np.allclose(grd, timg.grad.numpy()))
    # print('kernel grad close:', np.allclose(cnn.getWeightGrad(), tgrad))

    # plt.subplot(221)
    # plt.imshow(out[0])
    # plt.subplot(222)
    # plt.imshow(tout.detach().numpy()[0, 0])
    # plt.subplot(223)
    # grd = grd + 8
    # grd /= 16
    # plt.imshow(grd[0])
    # plt.subplot(224)
    # tgrd = timg.grad.numpy() + 8
    # tgrd /= 16
    # plt.imshow(tgrd[0, 0])
    # plt.show()




