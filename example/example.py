import numpy as np

import HashDL

rng = np.random.default_rng()

# y = a * x + b + noise
a = 2
b = 5

# dimension
d = 5

# sample size
N = 1000

x = rng.uniform(low=-10, high=10, size=(N,d))
y = a * x + b + rng.normal(size=(N,d))


opt = HashDL.Adam()
act = HashDL.ReLU()
sch = HashDL.ExponentialDecay(5, 1e-3)
init = HashDL.GaussInitializer(0.0, 1.0)
hash = HashDL.DWTA(8, 5)
net = HashDL.Network(input_size = d, units = (10, 10, d), L = 5,
                     optimizer = opt, activation = act,
                     scheduler = sch, initializer = init,
                     hash = hash)


batch_size = 32
epoch = 1000

idx = np.arange(N,dtype=int)

for e in range(epoch):
    rng.shuffle(idx)

    for i in range(0, idx.shape[0], batch_size):
        idx_batch = idx[i:i+batch_size]
        yhat = net(x[idx_batch])
        net.backward(yhat - y[idx_batch])

    print(f"{e}: {np.square(net(x) - y).sum()}")
