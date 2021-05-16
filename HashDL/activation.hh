#ifndef ACTIVATION_HH
#define ACTIVATION_HH

#include <cmath>

namespace HashDL {
  template<typename T> class Activation {
  public:
    Activation() = default;
    Activation(const Activation&) = default;
    Activation(Activation&&) = default;
    Activation& operator=(const Activation&) = default;
    Activation& operator=(Activation&&) = default;
    virtual ~Activation() = default;

    virtual T call(T x) const = 0;
    virtual T back(T y, T dL_dy) const = 0;
  };

  template<typename T> class Linear : public Activation<T> {
  public:
    T call(T x) const override { return x; }
    T back(T /* y */, T dL_dy) const override { return dL_dy; }
  };

  template<typename T> class ReLU : public Activation<T> {
  public:
    T call(T x) const override { return (x>0)? x: 0; }
    T back(T y, T dL_dy) const override {  return (y>0)? dL_dy: 0; }
  };

  template<typename T> class Sigmoid : public Activation<T> {
  public:
    T call(T x) const override { return 1.0/(1.0 + std::exp(-x)); }
    T back(T y, T dL_dy) const override { return y*(1-y)*dL_dy; }
  };
}

#endif
