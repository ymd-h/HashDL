#ifndef ACTIVATION_HH
#define ACTIVATION_HH

#include <cmath>

namespace HashDL {
  template<typename T> class Activation {
  public:
    virtual T call(T x) const = 0;
    virtual T back(T y, T dn_dy) const = 0;
  };

  template<typename T> class Linear : public Activation<T> {
  public:
    virtual T call(T x) override const { return x; }
    virtual T back(T y, T dn_dy) override const { return dn_dy; }
  };

  template<typename T> class ReLU : public Activation<T> {
  public:
    virtual T call(T x) override const { return (x>0)? x: 0; }
    virtual T back(T y, T dn_dy) override const {  return (y>0)? 1: 0; }
  };

  template<typename T> class Sigmoid : public Activation<T> {
  public:
    virtual T call(T x) override const { return 1.0/(1.0 + std::exp(-x)); }
    virtual T back(T y, T dn_dy) override const { return y*(1-y); }
  }
}

#endif
