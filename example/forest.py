import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split

import HashDL

epoch = 100
batch_size = 32


cov = datasets.fetch_covtype()
x = cov.data
y = cov.target

input_size = x.shape[1]
nclass = y.max() + 1

y_hot = np.zeros(shape=(y.shape[0], nclass))
y_hot[np.arange(y.shape[0]), y] = 1

x_train, x_test, y_train, y_test = train_test_split(x, y_hot, test_size=0.1)

net = HashDL.Network(input_size, (100, 100, nclass),
                     scheduler = HashDL.ConstantFrequency(50 * x_train.shape[0] // batch_size),
                     L2=0.1)

rng = np.random.default_rng()

loss = HashDL.SoftmaxCrossEntropy()


def accuracy(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    return (y_true.argmax(axis=1) == y_pred.argmax(axis=1)).mean()


idx = np.arange(x_train.shape[0])

for i in range(epoch):
    rng.shuffle(idx)

    train_pred = np.zeros_like(y_train)
    for j in range(0, idx.shape[0], batch_size):
        batch_idx = idx[j:j+batch_size]

        y_pred = net(x_train[batch_idx])
        assert np.isfinite(y_pred).all()

        net.backward(loss.gradient(y_train[batch_idx], y_pred))
        train_pred[batch_idx,:] = y_pred

    train_loss = loss(y_train, train_pred)
    train_acc = accuracy(y_train, train_pred)

    test_pred = net(x_test)
    test_loss = loss(y_test, test_pred)
    test_acc = accuracy(y_test, test_pred)

    print(f"Epoch: {i}, Train (Loss: {train_loss}, Acc: {train_acc}) Test (Loss: {test_loss}, Acc: {test_acc})")
