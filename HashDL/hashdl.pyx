# distutils: language = c++
# cython: linetrace=True

from HashDL cimport slide

cdef class Optimizer:
    cdef slide.Optimizer* opt


cdef class SGD(Optimizer):
    def __cinit__(self, rl=1e-4, decay=1.0, *args, **kwargs):
        self.opt = (slide.Optimizer*) = new slide.SGD(rl, decay)


cdef class Adam(Optimizer):
    def __cinit__(self, rl=1e-4, *args, **kwargs):
        self.opt = (slide.Optimizer*) = new slide.Adam(rl)


cdef class Network:
    cdef slide.Network* net
    def __cinit__(self, input_size, units=(30, 30, 30),
                  optimizer = None, *args, **kwargs):
        optimizer = optimizer or Adam()

        self.net = new slide.Network()


    def __call__(self, X):
        pass

    def backprop(self, dL_dy):
        pass
