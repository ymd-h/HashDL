#ifndef INITIALIZER_HH
#define INITIALIZER_HH

#include <random>

namespace HashDL {
  template<typename T> class Initializer {
  public:
    Initializer() = default;
    Initializer(const Initializer&) = default;
    Initializer(Initializer&&) = default;
    Initializer& operator=(const Initializer&) = default;
    Initializer& operator=(Initializer&&) = default;
    virtual ~Initializer() = default;
    virtual T operator()() = 0;
  };

  template<typename T> class ConstantInitializer : public Initializer<T> {
  private:
    T v;
  public:
    ConstantInitializer() = default;
    ConstantInitializer(T v): v{v} {}
    ConstantInitializer(const ConstantInitializer&) = default;
    ConstantInitializer(ConstantInitializer&&) = default;
    ConstantInitializer& operator=(const ConstantInitializer&) = default;
    ConstantInitializer& operator=(ConstantInitializer&&) = default;
    ~ConstantInitializer() = default;
    T operator()() override { return v; }
  };

  template<typename T> class GaussInitializer : public Initializer<T> {
  private:
    std::mt19937 g;
    std::normal_distribution<T> gauss;
  public:
    GaussInitializer(): GaussInitializer{0.0, 1.0} {}
    GaussInitializer(T mu, T sigma): g{std::random_device{}()}, gauss{mu, sigma} {}
    GaussInitializer(const GaussInitializer&) = default;
    GaussInitializer(GaussInitializer&&) = default;
    GaussInitializer& operator=(const GaussInitializer&) = default;
    GaussInitializer& operator=(GaussInitializer&&) = default;
    ~GaussInitializer() = default;
    T operator()() override { return gauss(g); }
  };
}

#endif
