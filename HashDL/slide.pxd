from libcpp.vector cimport vector

cdef extern from "slide.hh" namespace "HashDL":
    cdef cppclass BatchData[T]:
        BatchData() except +
    cdef cppclass BatchView[T]:
        BatchView(size_t, size_t, T*) except +
    cdef cppclass HashFunc[T]
    cdef cppclass WTAFunc[T]:
        WTAFunc(size_t, size_t) except +
    cdef cppclass DWTAFunc[T]:
        DWTAFunc(size_t, size_t, size_t) except +
    cdef cppclass Optimizer[T]
    cdef cppclass SGD[T]:
        SGD(T, T) except +
    cdef cppclass Adam[T]:
        Adam(T) except +
    cdef cppclass Network[T]:
        Network(size_t, vector[size_t], Optimizer[T]*, HashFunc[T]*) except +
        BatchData[T] operator()(const BatchView[T]&) except +
        void backward(const BatchView[T]&) except +
