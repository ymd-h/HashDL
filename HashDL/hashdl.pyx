# distutils: language = c++
# cython: linetrace=True

from HashDL cimport slide


cdef class SGD:
    def __cinit__(self, *args, **kwargs):
        pass

cdef class Adam:
    def __cinit__(self, *args, **kwargs):
        pass

cdef class Network:
    def __cinit__(self, *args, **kwargs):
        pass

    def __call__(self, X):
        pass

    def backprop(self, dL_dy):
        pass
