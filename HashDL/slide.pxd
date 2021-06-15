from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr

cdef extern from "slide.hh" namespace "HashDL":
    cdef cppclass BatchData[T]:
        BatchData() except +
        T* begin()
        T* end()
        size_t get_batch_size()
        size_t get_data_size()
    cdef cppclass BatchView[T]:
        BatchView(size_t, size_t, T*) except +
        size_t get_batch_size()
        size_t get_data_size()
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
    cdef cppclass Activation[T]:
        Activation() except +
        T call(T) except +
    cdef cppclass Linear[T]:
        Linear() except +
    cdef cppclass ReLU[T]:
        ReLU() except +
    cdef cppclass Sigmoid[T]:
        Sigmoid() except +
    cdef cppclass Initializer[T]:
        Initializer() except +
        T operator()() except +
    cdef cppclass ConstantInitializer[T]:
        ConstantInitializer(size_t) except +
    cdef cppclass GaussInitializer[T]:
        GaussInitializer(T,T) except +
    cdef cppclass Network[T]:
        Network(size_t, vector[size_t], size_t, shared_ptr[HashFunc[T]],
                shared_ptr[Optimizer[T]], shared_ptr[Scheduler]) except +
        Network(size_t, vector[size_t], size_t, shared_ptr[HashFunc[T]],
                shared_ptr[Optimizer[T]], shared_ptr[Scheduler],
                shared_ptr[Activation[T]]) except +
        Network(size_t, vector[size_t], size_t, shared_ptr[HashFunc[T]],
                shared_ptr[Optimizer[T]], shared_ptr[Scheduler],
                shared_ptr[Activation[T]], shared_ptr[Initializer[T]]) except +
        Network(size_t, vector[size_t], size_t, shared_ptr[HashFunc[T]],
                shared_ptr[Optimizer[T]], shared_ptr[Scheduler],
                shared_ptr[Activation[T]], shared_ptr[Initializer[T]], T, T) except +
        Network(size_t, vector[size_t], size_t, shared_ptr[HashFunc[T]],
                shared_ptr[Optimizer[T]], shared_ptr[Scheduler],
                shared_ptr[Activation[T]], shared_ptr[Initializer[T]],T,T,T) except +
        BatchData[T] operator()(const BatchView[T]&) except +
        void backward(const BatchView[T]&) except +
