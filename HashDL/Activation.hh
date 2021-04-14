#ifndef ACTIVATION_HH
#define ACTIVATION_HH

namespace HashDL {
  template<typename T> class Activation {
  public:
    virtual T call(T x) const = 0;
    virtual T back(T x, T dn_dy) const = 0;
  };

  template<typename T> class Linear : public Activation<T> {
  public:
    virtual T call(T x) override const noexcept { return x; }
    virtual T back(T x, T dn_dy) override const noexcept { return dn_dy; }
  };

  template<typename T> class ReLU : public Activation<T> {
  public:
    virtual T call(T x) override const noexcept { return (x>0)? x: 0; }
    virtual T back(T x, T dn_dy) override const noexcept {  return (x>0)? dn_dy: 0; }
  };
}

#endif
