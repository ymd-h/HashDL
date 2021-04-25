# distutils: language = c++
# cython: linetrace=True

from cython.operator cimport dereference
from libcpp.vector cimport vector

cimport numpy as np
import numpy as np

from . cimport slide

cdef class Optimizer:
    cdef slide.Optimizer[float] *opt

    cdef slide.Optimizer[float]* ptr(self):
        return self.opt


cdef class SGD(Optimizer):
    def __cinit__(self, rl=1e-4, decay=1.0, *args, **kwargs):
        self.opt = <slide.Optimizer[float]*> new slide.SGD[float](rl, decay)


cdef class Adam(Optimizer):
    def __cinit__(self, rl=1e-4, *args, **kwargs):
        self.opt = <slide.Optimizer[float]*> new slide.Adam[float](rl)


cdef class Hash:
    cdef slide.HashFunc[float] *hash

    cdef slide.HashFunc[float]* ptr(self):
        return self.hash


cdef class WTA(Hash):
    def __cinit__(self, bin_size, sample_size):
        self.hash = <slide.HashFunc[float]*> new slide.WTAFunc[float](bin_size, sample_size)


cdef class DWTA(Hash):
    def __cinit__(self, bin_size, sample_size, max_attempt=100):
        self.hash = <slide.HashFunc[float]*> new slide.DWTAFunc[float](bin_size, sample_size, max_attempt)


cdef class Network:
    cdef slide.Network[float]* net
    def __cinit__(self, input_size, units=(30, 30, 30),
                  hash = None, optimizer = None, *args, **kwargs):

        cdef Hash h = hash or DWTA(8, 8)
        cdef float rl = 1e-4
        cdef Optimizer opt = optimizer or Adam(rl)

        cdef vector[size_t] u = units
        self.net = new slide.Network[float](input_size, u, opt.ptr(), h.ptr())

    def __call__(self, X):
        X = np.array(X, ndmin=2, copy=False, dtype=np.float)

        cdef float[:,:] x = X
        cdef slide.BatchView[float] *view = new slide.BatchView[float](x.shape[1],
                                                                       x.shape[0],
                                                                       &x[0,0])

        return self.net(dereference(view))

    def backward(self, dL_dy):
        dL_dy = np.array(dL_dy, ndmin=2, copy=False, dtype=np.float)

        cdef float[:,:] dl_dy = dL_dy
        cdef slide.BatchView[float] *view = new slide.BatchView[float](dl_dy.shape[1],
                                                                       dl_dy.shape[0],
                                                                       &dl_dy[0,0])

        self.net.backward(dereference(view))
