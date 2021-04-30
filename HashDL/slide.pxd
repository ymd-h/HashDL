from libcpp.vector cimport vector

cdef extern from "slide.hh" namespace "HashDL":
    cdef cppclass BatchData[T]:
        BatchData() except +
        T* begin()
        T* end()
        size_t get_batch_size()
        size_t get_data_size()
    cdef cppclass BatchView[T]:
        BatchView(size_t, size_t, T*) except +
    cdef cppclass HashFunc[T]:
        HashFunc() except +
    cdef cppclass WTAFunc[T]:
        WTAFunc(size_t, size_t) except +
    cdef cppclass DWTAFunc[T]:
        DWTAFunc(size_t, size_t, size_t) except +
    cdef cppclass Optimizer[T]:
        Optimizer() except +
    cdef cppclass SGD[T]:
        SGD(T, T) except +
    cdef cppclass Adam[T]:
        Adam(T) except +
    cdef cppclass Scheduler:
        Scheduler() except +
    cdef cppclass ConstantFrequency:
        ConstantFrequency() except +
        ConstantFrequency(size_t) except +
    cdef cppclass ExponentialDecay[T]:
        ExponentialDecay() except +
        ExponentialDecay(size_t, T) except +
    cdef cppclass Network[T]:
        Network(size_t&, vector[size_t]&, size_t&,
                HashFunc[T]*, Optimizer[T]*, Scheduler*) except +
        BatchData[T] operator()(const BatchView[T]&) except +
        void backward(const BatchView[T]&) except +
