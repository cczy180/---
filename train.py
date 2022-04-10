# 训练：
# 激活函数
# 反向传播，loss以及梯度的计算
# 学习率下降策略
# L2正则化
# 优化器SGD
# 保存模型

import numpy as np

# 全连接层的构造
class FullConnectionLayer():
    def __init__(self,lamb):
        self.mem = {}
        self.lamb = lamb # L2正则化系数
    def forward(self,X,W):
        '''
        param{
            X: shape(m,d), 前向传播输入矩阵
            W: shape(d,d'), 前向传播权重矩阵 
        }
        return {
            H: shape(m,d'), 前向传播输出矩阵
        }
        '''
        self.mem['X'] = X
        self.mem['W'] = W
        H = np.matmul(X,W)
        return H
    def backward(self,grad_H):
        '''
        param {
            grad_H: shape(m,d'), Loss关于H的梯度
        }
        return {
            grad_X: shape(m,d), Loss关于X的梯度
            grad_W: shape(d,d'), Loss关于W的梯度
        }
        '''
        X = self.mem['X']
        W = self.mem['W']
        grad_X = np.matmul(grad_H, W.T)
        grad_W = np.matmul(X.T, grad_H)+self.lamb*W # L2正则化
        return grad_X, grad_W

# 实现relu函数
class Relu():
    def __init__(self):
        self.mem = {}
    def forward(self,X):
        self.mem['X'] = X
        return np.where(X>0,X,np.zeros_like(X))
    def backward(self,grad_y):
        X = self.mem['X']
        return (X>0).astype(np.float32) * grad_y

# 实现交叉熵损失函数
class crossEntropy():
    def __init__(self):
        self.mem = {}
        self.epsilon = 1e-12
    def forward(self,p,y):
        self.mem['p'] = p
        log_p = np.log(p+self.epsilon)
        return np.mean(np.sum(-y*log_p,axis=1))
    def backward(self,y):
        p = self.mem['p']
        return -y*(1/(p+self.epsilon))

# 实现softmax激活函数
class Softmax():
    def __init__(self):
        self.mem = {}
        self.epsilon = 1e-12
    def forward(self,p):
        p_exp = np.exp(p)
        denominator = np.sum(p_exp, axis=1, keepdims=True)
        s = p_exp/(denominator+self.epsilon)
        self.mem['s'] = s
        self.mem['p_exp'] = p_exp
        return s
    def backward(self,grad_s):
        s = self.mem["s"]
        sisj = np.matmul(np.expand_dims(s, axis=2), np.expand_dims(s, axis=1))
        tmp = np.matmul(np.expand_dims(grad_s, axis=1), sisj)
        tmp = np.squeeze(tmp, axis=1)
        grad_p = -tmp + grad_s * s
        return grad_p

# 搭建全连接神经网络
class FullConnectionModel():
    def __init__(self,latent_dims,L2):
        # latent_dims 连接层的维数
        self.W1 = np.random.normal(loc=0, scale=1,size = [28*28+1,latent_dims])/np.sqrt((28*28+1)/2)
        self.W2 = np.random.normal(loc=0, scale =1,size = [latent_dims,10])/np.sqrt(latent_dims/2)
        self.L2 = L2
        self.mul_h1 = FullConnectionLayer(self.L2)
        self.relu = Relu()
        self.mul_h2 = FullConnectionLayer(self.L2)
        self.softmax = Softmax()
        self.cross_en = crossEntropy()


    def forward(self,X,labels):
        bias = np.ones(shape=[X.shape[0],1])
        X = np.concatenate([X,bias],axis=1)
        # 一层全连接
        self.h1 = self.mul_h1.forward(X,self.W1)
        # 全连接后激活
        self.h1_relu = self.relu.forward(self.h1)
        # 激活后一层全连接输出10维向量
        self.h2 = self.mul_h2.forward(self.h1_relu,self.W2)
        # 根据向量softmax求概率
        self.h2_soft = self.softmax.forward(self.h2)
        # 根据预测概率求损失函数
        self.loss = self.cross_en.forward(self.h2_soft,labels)

    def backward(self,labels):
        self.loss_grad = self.cross_en.backward(labels)
        self.h2_soft_grad = self.softmax.backward(self.loss_grad)
        self.h2_grad,self.W2_grad = self.mul_h2.backward(self.h2_soft_grad)
        self.h1_relu_grad = self.relu.backward(self.h2_grad)
        self.h1_grad,self.W1_grad = self.mul_h1.backward(self.h1_relu_grad)

# 计算精确度
def computeAccuracy(prob, labels):
    predictions = np.argmax(prob,axis=1)
    truth = np.argmax(labels,axis=1)
    return np.mean(predictions == truth)

# 训练一次模型
def trainOneStep(model,train_x,train_y,learning_rate=1e-5):
    model.forward(train_x,train_y)
    model.backward(train_y)
    model.W1 += -learning_rate*(model.W1_grad)
    model.W2 += -learning_rate*(model.W2_grad)
    loss = model.loss
    accuracy = computeAccuracy(model.h2_soft,train_y)
    return loss, accuracy

