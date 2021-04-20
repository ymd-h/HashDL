# distutils: language = c++
# cython: linetrace=True

cimport numpy as np
import numpy as np

from HashDL cimport slide


cdef class Optimizer:
    cdef slide.Optimizer[float]* opt

    cdef slide.Optimizer[float]* ptr(self):
        return self.opt


cdef class SGD(Optimizer):
    def __cinit__(self, rl=1e-4, decay=1.0, *args, **kwargs):
        self.opt = new slide.SGD[float](rl, decay)


cdef class Adam(Optimizer):
    def __cinit__(self, rl=1e-4, *args, **kwargs):
        self.opt = new slide.Adam[float](rl)


cdef class Hash:
    cdef slide.HashFunc[float]* hash

    cdef slide.HashFunc[float]* ptr(self):
        return self.hash


cdef class WTA(Hash):
    def __cinit__(self, bin_size, sample_size):
        self.hash = new WTAFunc[float](bin_size, sample_size)


cdef class DWTA(Hash):
    def __cinit__(self, bin_size, sample_size, max_attempt=100):
        self.hash = new DWTAFunc[float](bin_size, sample_size, max_attempt)


cdef class Network:
    cdef slide.Network* net
    def __cinit__(self, input_size, units=(30, 30, 30),
                  hash = None, optimizer = None, *args, **kwargs):

        hash = hash or DWTA(8, 8)
        optimizer = optimizer or Adam()

        self.net = new slide.Network(input_size, units, optimizer.ptr(), hash.ptr())

    def __call__(self, X):
        X = np.array(X, ndmin=2, copy=False, dtype=np.float)

        cdef float[:,:] x = X
        cdef BatchView[float] view = BatchView[float](x.shape[1], x.shape[0], &x[0])

        return self.net(view)

    def backward(self, dL_dy):
        dL_dy = np.array(dL_dy, ndmin=2, copy=False, dtype=np.float)

        cdef float[:,:] dl_dy = dL_dy
        cdef BatchView[float] view = BatchView[float](dl_dy.shape[1],
                                                      dl_dy.shape[0],
                                                      &dl_dy[0])

        self.backward(view)
