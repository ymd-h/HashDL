from libcpp.vector cimport vector

cdef extern from "slide.hh" namespace "HashDL":
    cdef cppclass BatchData[T]:
        BatchData() except +
    cdef cppclass BatchView[T]:
        BatchView() except +
    cdef cppclass SGD[T]:
        SGD(T, T) except +
    cdef cppclass Adam[T]:
        Adam(T) except +
    cdef cppclass Network[T]:
        Network(size_t, vector[T], Optimizer*, function[]) except +
        BatchData[T] operator()(const DataView[T]&) except +
        void backward(const DataView[T]&) except +
