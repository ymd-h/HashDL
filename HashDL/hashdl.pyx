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


@cython.embedsignature(True)
cdef class SGD(Optimizer):
    def __cinit__(self, lr=1e-4, decay=1.0, *args, **kwargs):
        if lr < 0:
            raise ValueError(f"Learning Rate (lr) must be positive: {lr}")

        self.opt = shared_ptr[slide.Optimizer[float]](<slide.Optimizer[float]*> new slide.SGD[float](lr, decay))


@cython.embedsignature(True)
cdef class Adam(Optimizer):
    def __cinit__(self, lr=1e-4, *args, **kwargs):
        if lr < 0:
            raise ValueError(f"Learning Rate (lr) must be positive: {lr}")

        self.opt = shared_ptr[slide.Optimizer[float]](<slide.Optimizer[float]*> new slide.Adam[float](lr))


cdef class Hash:
    cdef shared_ptr[slide.HashFunc[float]] hash

    cdef shared_ptr[slide.HashFunc[float]] ptr(self):
        return self.hash


@cython.embedsignature(True)
cdef class WTA(Hash):
    def __cinit__(self, K_hashes, sample_size):
        self.hash = shared_ptr[slide.HashFunc[float]](<slide.HashFunc[float]*> new slide.WTAFunc[float](K_hashes, sample_size))

    def __init__(self, K_hashes, sample_size):
        """Initialize WTA hash

        Parameters
        ----------
        K_hashes : int
            Number of LSH bin (aka. hash) in single table.
        sample_size : int
            Number of samples from input in single bin.
            i.e. `permute(index)[:sample_size]` is checked.
        """
        pass


@cython.embedsignature(True)
cdef class DWTA(Hash):
    def __cinit__(self, K_hashes, sample_size, max_attempt=100):
        self.hash = shared_ptr[slide.HashFunc[float]](<slide.HashFunc[float]*> new slide.DWTAFunc[float](K_hashes, sample_size, max_attempt))

    def __init__(self, K_hashes, sample_size, max_attempt=100):
        """Initialize Densified WTA hash

        Parameters
        ----------
        K_hashes : int
            Number of LSH bin (aka. hash) in single table.
        sample_size : int
            Number of samples from input in single bin.
            i.e. `permute(index)[:sample_size]` is checked.
        max_attempt : int, optional
           Number of attempt to densification trial
        """
        pass


cdef class Scheduler:
    cdef shared_ptr[slide.Scheduler] sch

    cdef shared_ptr[slide.Scheduler] ptr(self):
        return self.sch

@cython.embedsignature(True)
cdef class ConstantFrequency(Scheduler):
    def __cinit__(self, N):
        self.sch = shared_ptr[slide.Scheduler](<slide.Scheduler*> new slide.ConstantFrequency(N))

@cython.embedsignature(True)
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

@cython.embedsignature(True)
cdef class Linear(Activation):
    def __cinit__(self):
        self.act = shared_ptr[slide.Activation[float]](<slide.Activation[float]*> new slide.Linear[float]())

@cython.embedsignature(True)
cdef class ReLU(Activation):
    def __cinit__(self):
        self.act = shared_ptr[slide.Activation[float]](<slide.Activation[float]*> new slide.ReLU[float]())

@cython.embedsignature(True)
cdef class Sigmoid(Activation):
    def __cinit__(self):
        self.act = shared_ptr[slide.Activation[float]](<slide.Activation[float]*> new slide.Sigmoid[float]())

cdef class Initializer:
    cdef shared_ptr[slide.Initializer[float]] init

    cdef shared_ptr[slide.Initializer[float]] ptr(self):
        return self.init

    def __call__(self):
        return dereference(self.init)()

@cython.embedsignature(True)
cdef class ConstantInitializer(Initializer):
    def __cinit__(self, v):
        self.init = shared_ptr[slide.Initializer[float]](<slide.Initializer[float]*> new slide.ConstantInitializer[float](v))

@cython.embedsignature(True)
cdef class GaussInitializer(Initializer):
    def __cinit__(self, mu, sigma):
        self.init = shared_ptr[slide.Initializer[float]](<slide.Initializer[float]*> new slide.GaussInitializer[float](mu, sigma))

@cython.embedsignature(True)
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

@cython.embedsignature(True)
cdef class Network:
    cdef slide.Network[float]* net
    cdef slide.BatchData[float] Y
    cdef BatchWrapper y

    def __cinit__(self, input_size, units=(30, 30, 30), L_tables = 50,
                  hash = None, optimizer = None, scheduler = None,
                  activation = None, initializer = None, *args, **kwargs):

        if input_size <= 0:
            raise ValueError(f"input_size must be positive: {input_size}")

        if (np.asarray(units) <= 0).any():
            raise ValueError(f"units must be positive: {units}")

        if L_tables <= 0:
            raise ValueError(f"L must be positive: {L_tables}")

        cdef size_t K_hashes = 8
        cdef size_t sample_size = 8
        cdef Hash h = hash or DWTA(K_hashes, input_size)

        cdef float lr = 1e-4
        cdef Optimizer opt = optimizer or Adam(lr)

        cdef size_t N = 50
        cdef float decay = 1e-3
        cdef Scheduler sch = scheduler or ExponentialDecay(N, decay)

        cdef Activation act = activation or ReLU()

        cdef float mu = 0
        cdef float sigma = 1.0
        cdef Initializer init = initializer or GaussInitializer(mu, sigma)

        cdef vector[size_t] u = units
        self.net = new slide.Network[float](input_size, u, L_tables,
                                            h.ptr(), opt.ptr(), sch.ptr(),
                                            act.ptr(), init.ptr())

        self.y = BatchWrapper()

    def __init__(self, input_size, units=(30, 30, 30), L_tables = 50,
                 hash = None, optimizer = None, scheduler = None,
                 activation = None, initializer = None, *args, **kwargs):
        """
        Initialize SLIDE network

        Parameters
        ----------
        input_size : int
            Input data size. The input data must be flattened.
        units : list of int, optional
            Number of units at hidden layers. The default is `(30, 30, 30)`
        L_tables : int, optional
            Number of hash buckets in a single dense layer. The default is `50`
        hash : HashDL.Hash, optional
            Locality sensitive hash function. The default is `HashDL.DWTA(8, 8)`
        optimizer : HashDL.Optimizer, optional
            Neural network optimizer. The default `HashDL.Adam(1e-4)`
        scheduler : HashDL.Scheduler, optional
            Scheduler for re-hash. The default is `HashDL.ExponentialDecay(50, 1e-3)`
        activation : HashDL.Activation, optional
            Activation function for hidden layer.
            The default is `HashDL.ReLU()`
        initializer : HashDL.initializer, optional
            Weight initializer for hidden layers.
            The default is `HashDL.GaussInitializer(0, 1.0)`
        """
        pass

    def __del__(self):
        del self.net

    def __call__(self, X):
        """
        Forward calculation over batch input

        Parameters
        ----------
        X : array-like of float
            Input batch data. The shape must be [batch_size, input_size]

        Returns
        -------
        Y : np.ndarray
            Output layer's value (aka. activated last hidden layer's value)
        """
        X = np.array(X, ndmin=2, copy=False, dtype=np.single, order="C")

        cdef float[:,:] x = X
        cdef slide.BatchView[float] *view = new slide.BatchView[float](x.shape[1],
                                                                       x.shape[0],
                                                                       &x[0,0])

        self.Y = dereference(self.net)(dereference(view))
        self.y.set(&self.Y)

        del view
        return np.asarray(self.y)

    def backward(self, dL_dY):
        """
        Backward propagation of gradient.

        Parameters
        ----------
        dL_dY : array-like
            Gradient of Loss against network output.
        """
        dL_dY = np.array(dL_dY, ndmin=2, copy=False, dtype=np.single, order="C")

        cdef float[:,:] dl_dy = dL_dY
        cdef slide.BatchView[float] *view = new slide.BatchView[float](dl_dy.shape[1],
                                                                       dl_dy.shape[0],
                                                                       &dl_dy[0,0])

        self.net.backward(dereference(view))
        del view
