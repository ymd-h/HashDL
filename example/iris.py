import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

import HashDL

epoch = 5000
batch_size = 32


iris = datasets.load_iris()
x = iris.data
y = iris.target

input_size = x.shape[1]
nclass = y.max() + 1

y_hot = np.zeros(shape=(y.shape[0], nclass))
y_hot[np.arange(y.shape[0]), y] = 1

x_train, x_test, y_train, y_test = train_test_split(x, y_hot, test_size=0.1)


net = HashDL.Network(input_size, (32, nclass))
rng = np.random.default_rng()

def softmax_cross_entropy(y_true, y_pred):
    y_norm = y_pred - y_pred.max(axis=1, keepdims=True)
    y_exp = np.exp(y_norm)
    assert np.isfinite(y_exp).all(), f"{y_pred[np.isfinite(y_exp)]}"

    y_log_soft = y_norm - np.log(y_exp.sum(axis=1, keepdims=True))

    return -(y_true * y_log_soft).sum(axis=1).mean()

def grad_softmax_cross_entropy(y_true, y_pred):
    y_norm = y_pred - y_pred.max(axis=1, keepdims=True)
    y_exp = np.exp(y_norm)
    assert np.isfinite(y_exp).all(), f"{y_pred[np.isfinite(y_exp)]}"

    y_soft = y_exp / y_exp.sum(axis=1, keepdims=True)

    return y_soft - y_true


idx = np.arange(x_train.shape[0])

for i in range(epoch):
    rng.shuffle(idx)

    for j in range(0, idx.shape[0], batch_size):
        batch_idx = idx[j:j+batch_size]

        y_pred = net(x_train[batch_idx])
        assert np.isfinite(y_pred).all()

        net.backward(grad_softmax_cross_entropy(y_train[batch_idx], y_pred))

    train_loss = softmax_cross_entropy(y_train, net(x_train))
    test_loss = softmax_cross_entropy(y_test, net(x_test))

    print(f"Epoch: {i}, Train Loss: {train_loss}, Test Loss: {test_loss}")
