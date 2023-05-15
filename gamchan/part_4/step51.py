if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import math
import numpy as np
import matplotlib.pyplot as plt
import dezero
from dezero import optimizers, DataLoader
import dezero.functions as F
from dezero.models import MLP

# Hyperparameters
max_epoch = 5
batch_size = 100
hidden_size = 1000

train_set = dezero.datasets.MNIST(train=True)
test_set = dezero.datasets.MNIST(train=False)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

model = MLP((hidden_size, 10), activation=F.relu)
optimizer = optimizers.SGD().setup(model)
train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []
for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

    print('epoch {}'.format(epoch + 1))
    print('train loss: {:.4f}, accuracy: {:.4f}'.format(sum_loss / len(train_set), sum_acc / len(train_set)))
    train_loss_list.append(sum_loss / len(train_set))
    train_acc_list.append(sum_acc / len(train_set))
    
    sum_loss, sum_acc = 0, 0
    with dezero.no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)
            
    print('test loss: {:.4f}, accuracy: {:.4f}'.format(sum_loss / len(test_set), sum_acc / len(test_set)))
    test_loss_list.append(sum_loss / len(test_set))
    test_acc_list.append(sum_acc / len(test_set))

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

x_axis = np.arange(0, max_epoch, 1)
# Line graph for y1
axs[0].plot(x_axis, train_loss_list, label='train')

# Line graph for y2
axs[0].plot(x_axis, test_loss_list, label='test')

# Set title and labels
axs[0].set_title('loss')
axs[0].set_xlabel('epoch')
axs[0].set_ylabel('loss')

# Add a legend
axs[0].legend()

axs[1].plot(x_axis, train_acc_list, label='train')

# Line graph for y2
axs[1].plot(x_axis, test_acc_list, label='test')

# Set title and labels
axs[1].set_title('acc')
axs[1].set_xlabel('epoch')
axs[1].set_ylabel('acc')

# Add a legend
axs[1].legend()

# Display the figure
plt.show()