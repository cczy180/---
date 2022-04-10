# 参数查找：学习率，隐藏层大小，正则化强度
from train import *
import tqdm
import os
import gzip

# 1.数据集的下载与处理

# 定义加载数据的函数，data_folder为保存gz数据的文件夹，该文件夹下有4个文件
# 'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
# 't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'

def load_data(data_folder):

  files = [
      'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
      't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
  ]

  paths = []
  for fname in files:
    paths.append(os.path.join(data_folder,fname))

  with gzip.open(paths[0], 'rb') as lbpath:
    y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

  with gzip.open(paths[1], 'rb') as imgpath:
    x_train = np.frombuffer(
        imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

  with gzip.open(paths[2], 'rb') as lbpath:
    y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

  with gzip.open(paths[3], 'rb') as imgpath:
    x_test = np.frombuffer(
        imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

  return (x_train, y_train), (x_test, y_test)

# 60000条训练数据，10000条测试数据
(train_images, train_labels), (test_images, test_labels) = load_data('data/')

# 对数据作预处理，images从28*28处理为784维，labels从一维处理为10维
train_images = train_images.copy()
train_image = []
for i in range(len(train_images)):
    train_image.append(np.hstack(list(train_images[i])))
train_images = np.array(train_image)

test_images = test_images.copy()
test_image = []
for i in range(len(test_images)):
    test_image.append(np.hstack(list(test_images[i])))
test_images = np.array(test_image)/255

# label
train_label = [np.zeros(10) for _ in range(len(train_labels))]
for i in range(len(train_labels)):
    train_label[i][train_labels[i]]=1
train_labels = np.array(train_label)

test_label = [np.zeros(10) for _ in range(len(test_labels))]
for i in range(len(test_labels)):
    test_label[i][test_labels[i]]=1
test_labels = np.array(test_label)


validation_images = train_images[:5000]/255
validation_labels = train_labels[:5000]
train_images = train_images[5000:]/255
train_labels = train_labels[5000:]

# 数据记得归一化



# 训练模型和寻优
def train(x_train, y_train, x_validation, y_validation):
    epochs = 50
    batch_size = 50
    batch_num = len(x_train)//batch_size
    learning_rate = [1e-2,1e-3,1e-4]
    latent_dims_list = [100, 200, 300]
    lambda_ = [0.5,1]
    best_accuracy = 0
    best_latent_dims = 0

    # 在验证集上寻优
    print("Start seaching the best parameter...\n")
    for latent_dims in latent_dims_list:
        for L2 in lambda_:
            for learning_rate_ in learning_rate:
                model = FullConnectionModel(latent_dims,L2)

                bar = tqdm.std.trange(20)  # 使用 tqdm 第三方库，调用 tqdm.std.trange 方法给循环加个进度条
                for epoch in bar:
                    for batch in range(batch_num):
                        batch_x = x_train[batch_size*batch:batch_size*(1+batch)]
                        batch_y = y_train[batch_size*batch:batch_size*(1+batch)]
                        loss, accuracy = trainOneStep(model,batch_x,batch_y, learning_rate_) 
                    bar.set_description(f'Parameter latent_dims={latent_dims: <3}, epoch={epoch + 1: <3}, loss={loss: <10.8}, accuracy={accuracy: <8.6}')  # 给进度条加个描述
                bar.close()

                validation_loss, validation_accuracy = evaluate(model, x_validation, y_validation)
                print(f"Parameter latent_dims={latent_dims: <3},L2 = {L2},learning_rate = {learning_rate_} validation_loss={validation_loss}, validation_accuracy={validation_accuracy}.\n")

                if validation_accuracy > best_accuracy:
                    best_accuracy = validation_accuracy
                    best_latent_dims = latent_dims
                    best_lambda = L2
                    best_learning_rate = learning_rate_

    # 得到最好的参数组合，训练最好的模型
    print(f"The best latent dims is {best_latent_dims}.\n")
    print(f"The best L2 parameter is {best_lambda}.\n")
    print(f"The best learning rate is {best_learning_rate}.\n")
    print("Start training the best model...")
    best_model = FullConnectionModel(best_latent_dims,best_lambda)
    x = np.concatenate([x_train, x_validation], axis=0)
    y = np.concatenate([y_train, y_validation], axis=0)
    bar = tqdm.std.trange(epochs)
    batch_num = len(x)//batch_size
    loss_ = []
    accuracy_ = []
    for epoch in bar:
        for batch in range(batch_num):
            batch_x = x[batch_size*batch:batch_size*(1+batch)]
            batch_y = y[batch_size*batch:batch_size*(1+batch)]
            loss, accuracy = trainOneStep(best_model, batch_x, batch_y, best_learning_rate)
        loss_.append(loss)
        accuracy_.append(accuracy)
        bar.set_description(f'Training the best model, epoch={epoch + 1: <3}, loss={loss: <10.8}, accuracy={accuracy: <8.6}')  # 给进度条加个描述
    bar.close()

    return best_model,loss_,accuracy_


# 评估模型
def evaluate(model, x, y):
    model.forward(x, y)
    loss = model.loss
    accuracy = computeAccuracy(model.h2_soft, y)
    return loss, accuracy

import matplotlib.pyplot as plt
if __name__=='__main__':
    model,loss_,accuracy_ = train(train_images, train_labels, validation_images, validation_labels)
    loss, accuracy = evaluate(model, test_images, test_labels)
    print(f'Evaluate the best model, test loss={loss:0<10.8}, accuracy={accuracy:0<8.6}.')
    plt.plot(loss_)
    plt.plot(accuracy_)