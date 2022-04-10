# 文件介绍
train.py:   
  编写了FullConnectionLayer,Relu,crossEntropy,Softmax等构造全连接神经网络需要用到的类，每类中包含对应层前向传播和反向传播的计算函数；编写类FullConnectionModel构造两层神经网络模型；另有computeAccuracy()函数用来计算准确率，trainOneStep()函数用来对每个batch进行一次前向传播和反向传播的参数更新。   
  
test.py:   
上传数据，划分训练集、验证集和测试集，并对数据做归一化处理；train()函数用来做参数查找，其中包含学习率、隐藏层大小、正则化强度等超参数；保存最优参数训练得到的最优模型，并在测试集上进行测试，输出分类精度、accuracy曲线和loss曲线。  
  
data:  
MNIST数据集

# 训练及测试步骤
运行test.py文件，可以完成参数查找、模型训练、测试三个步骤。
