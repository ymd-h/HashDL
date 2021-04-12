#ifndef ACTIVATION_HH
#define ACTIVATION_HH

namespace HashDL {
  template<typename T> class Activation{
  public:
    virtual T call(T x) = 0;
    virtual T backprop(T x,T dn_dy) = 0;
  };

  template<typename T> class ReLU : public Activation<T> {
  public:
    virtual T call(T x) override noexcept { return (x>0)? x: 0; }
    virtual T backprop(T x, T dn_dy) override noexcept {  return (x>0)? dn_dy: 0; }
  };
}

#endif
