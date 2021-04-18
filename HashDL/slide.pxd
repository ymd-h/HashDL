from libcpp.vector cimport vector

cdef extern from "slide.hh" namespace "HashDL":
    cdef cppclass BatchData[T]
    cdef cppclass DataView[T]
    cdef cppclass SGD[T]:
        SGD(T, T) except +
    cdef cppclass Adam[T]:
        Adam(T) except +
    cdef cppclass Network[T]:
        Network(size_t, vector[T], Optimizer*, function[]) except +
        BatchData[T] operator()(const DataView[T]&) except +
        void backward(const DataView[T]&) except +
