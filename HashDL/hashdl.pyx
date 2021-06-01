# distutils: language = c++
# cython: linetrace=True

from libc.stdlib cimport malloc, free
from cython.operator cimport dereference
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr

cimport numpy as np
import numpy as np

from . cimport slide

cdef class Optimizer:
    cdef shared_ptr[slide.Optimizer[float]] opt

    cdef shared_ptr[slide.Optimizer[float]] ptr(self):
        return self.opt


cdef class SGD(Optimizer):
    def __cinit__(self, rl=1e-4, decay=1.0, *args, **kwargs):
        if rl < 0:
            raise ValueError(f"Learning Rate (rl) must be positive: {rl}")

        self.opt = shared_ptr[slide.Optimizer[float]](<slide.Optimizer[float]*> new slide.SGD[float](rl, decay))


cdef class Adam(Optimizer):
    def __cinit__(self, rl=1e-4, *args, **kwargs):
        if rl < 0:
            raise ValueError(f"Learning Rate (rl) must be positive: {rl}")

        self.opt = shared_ptr[slide.Optimizer[float]](<slide.Optimizer[float]*> new slide.Adam[float](rl))


cdef class Hash:
    cdef shared_ptr[slide.HashFunc[float]] hash

    cdef shared_ptr[slide.HashFunc[float]] ptr(self):
        return self.hash


cdef class WTA(Hash):
    def __cinit__(self, nbins, sample_size):
        self.hash = shared_ptr[slide.HashFunc[float]](<slide.HashFunc[float]*> new slide.WTAFunc[float](nbins, sample_size))


cdef class DWTA(Hash):
    def __cinit__(self, nbins, sample_size, max_attempt=100):
        self.hash = shared_ptr[slide.HashFunc[float]](<slide.HashFunc[float]*> new slide.DWTAFunc[float](nbins, sample_size, max_attempt))


cdef class Scheduler:
    cdef shared_ptr[slide.Scheduler] sch

    cdef shared_ptr[slide.Scheduler] ptr(self):
        return self.sch

cdef class ConstantFrequency(Scheduler):
    def __cinit__(self, N):
        self.sch = shared_ptr[slide.Scheduler](<slide.Scheduler*> new slide.ConstantFrequency(N))

cdef class ExponentialDecay(Scheduler):
    def __cinit__(self, N, decay):
        self.sch = shared_ptr[slide.Scheduler](<slide.Scheduler*> new slide.ExponentialDecay[float](N,decay))

cdef class Activation:
    cdef shared_ptr[slide.Activation[float]] act
    cdef shared_ptr[slide.Activation[float]] ptr(self):
        return self.act

    def __call__(self, x):
        cdef float _x = x
        return dereference(self.act).call(_x)

cdef class Linear(Activation):
    def __cinit__(self):
        self.act = shared_ptr[slide.Activation[float]](<slide.Activation[float]*> new slide.Linear[float]())

cdef class ReLU(Activation):
    def __cinit__(self):
        self.act = shared_ptr[slide.Activation[float]](<slide.Activation[float]*> new slide.ReLU[float]())

cdef class Sigmoid(Activation):
    def __cinit__(self):
        self.act = shared_ptr[slide.Activation[float]](<slide.Activation[float]*> new slide.Sigmoid[float]())

cdef class Initializer:
    cdef shared_ptr[slide.Initializer[float]] init

    cdef shared_ptr[slide.Initializer[float]] ptr(self):
        return self.init

    def __call__(self):
        return dereference(self.init)()

cdef class ConstantInitializer(Initializer):
    def __cinit__(self, v):
        self.init = shared_ptr[slide.Initializer[float]](<slide.Initializer[float]*> new slide.ConstantInitializer[float](v))

cdef class GaussInitializer(Initializer):
    def __cinit__(self, mu, sigma):
        self.init = shared_ptr[slide.Initializer[float]](<slide.Initializer[float]*> new slide.GaussInitializer[float](mu, sigma))

cdef class BatchWrapper:
    cdef slide.BatchData[float]* ptr
    cdef size_t itemsize
    cdef Py_ssize_t* shape
    cdef Py_ssize_t* strides

    def __cinit__(self):
        self.itemsize = sizeof(float)
        self.shape   = <Py_ssize_t*>malloc(sizeof(Py_ssize_t) * 2)
        self.strides = <Py_ssize_t*>malloc(sizeof(Py_ssize_t) * 2)

    cdef void set(self, slide.BatchData[float]* p):
        self.ptr = p

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        self.shape[0] = self.ptr.get_batch_size()
        self.shape[1] = self.ptr.get_data_size()
        self.strides[0] = self.ptr.get_data_size() * self.itemsize
        self.strides[1] = self.itemsize

        buffer.buf = <char *> &dereference(self.ptr.begin())
        buffer.format = 'f'
        buffer.len = (self.ptr.end() - self.ptr.begin()) * self.itemsize
        buffer.readonly = 0
        buffer.ndim = 2
        buffer.shape = self.shape
        buffer.strides = self.strides
        buffer.suboffsets = NULL
        buffer.itemsize = self.itemsize
        buffer.internal = NULL
        buffer.obj = self

    def __dealloc__(self):
        free(self.shape)
        free(self.strides)

cdef class Network:
    cdef slide.Network[float]* net
    cdef slide.BatchData[float] Y
    cdef BatchWrapper y

    def __cinit__(self, input_size, units=(30, 30, 30), L = 50,
                  hash = None, optimizer = None, scheduler = None,
                  activation = None, initializer = None, *args, **kwargs):

        if input_size <= 0:
            raise ValueError(f"input_size must be positive: {input_size}")

        if (np.asarray(units) <= 0).any():
            raise ValueError(f"units must be positive: {units}")

        if L <= 0:
            raise ValueError(f"L must be positive: {L}")

        cdef size_t nbins = 8
        cdef size_t sample_size = min(8, input_size)
        cdef Hash h = hash or DWTA(nbins, input_size)

        cdef float rl = 1e-4
        cdef Optimizer opt = optimizer or Adam(rl)

        cdef size_t N = 50
        cdef float decay = 1e-3
        cdef Scheduler sch = scheduler or ExponentialDecay(N, decay)

        cdef Activation act = activation or ReLU()

        cdef float mu = 0
        cdef float sigma = 1.0
        cdef Initializer init = initializer or GaussInitializer(mu, sigma)

        cdef vector[size_t] u = units
        self.net = new slide.Network[float](input_size, u, L,
                                            h.ptr(), opt.ptr(), sch.ptr(),
                                            act.ptr(), init.ptr())

        self.y = BatchWrapper()

    def __del__(self):
        del self.net

    def __call__(self, X):
        X = np.array(X, ndmin=2, copy=False, dtype=np.single, order="C")

        cdef float[:,:] x = X
        cdef slide.BatchView[float] *view = new slide.BatchView[float](x.shape[1],
                                                                       x.shape[0],
                                                                       &x[0,0])

        self.Y = dereference(self.net)(dereference(view))
        self.y.set(&self.Y)

        del view
        return np.asarray(self.y)

    def backward(self, dL_dy):
        dL_dy = np.array(dL_dy, ndmin=2, copy=False, dtype=np.single, order="C")

        cdef float[:,:] dl_dy = dL_dy
        cdef slide.BatchView[float] *view = new slide.BatchView[float](dl_dy.shape[1],
                                                                       dl_dy.shape[0],
                                                                       &dl_dy[0,0])

        self.net.backward(dereference(view))
        del view
