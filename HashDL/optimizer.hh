#ifndef OPTIMIZER_HH
#define OPTIMIZER_HH

#include <cmath>

namespace HashDL {
  template<typename T> class OptimizerClient {
  public:
    virtual T diff(T grad) = 0;
  };

  template<typename T> class Optimizer {
  public:
    virtual OptimizerClient<T>* client() const = 0;
    virtual void step(){}
  };

  template<typename T> class SGD;
  template<typename T> class SGDClient : public OptimizerClient<T> {
  private:
    SGD<T>* sgd;
  public:
    SGDClient() = delete;
    SGDClient(SGD<T>* sgd): sgd{sgd} {}
    SGDClient(const SGDClient&) = default;
    SGDClient(SGDClient&&) = default;
    SGDClient& operator=(const SGDClient&) = default;
    SGDClient& operator=(SGDClient&&) = default;
    ~SGDClient() = default;

    virtual T diff(T grad){
      return - sgd->eta() * grad;
    }
  };

  template<typename T> class SGD : public Optimizer<T> {
  private:
    T _eta;
    T decay;
  public:
    SGD(): SGD{0.001} {}
    SGD(T rl, T decay=1.0): _eta{rl}, decay{decay} {}
    SGD(const SGD&) = default;
    SGD(SGD&&) = default;
    SGD& operator=(const SGD&) = default;
    SGD& operator=(SGD&&) = default;
    ~SGD() = default;

    OptimizerClient<T>* client() const override {
      return new SGDClient<T>{this};
    }
    void step() override { _eta *= decay; }
    const auto eta(){ return _eta; }
  };

  template<typename T> class Adam;
  template<typename T> class AdamClient : public OptimizerClient<T>{
  private:
    T m;
    T v;
    Adam<T>* adam;
  public:
    AdamClient() = delete;
    AdamClient(Adam<T>* adam): m{0}, T{0}, adam{adam} {}
    AdamClient(const AdamClient&) = default;
    AdamClient(AdamClient&&) = default;
    AdamClient& operator=(const AdamClient&) = default;
    AdamClient& operator=(AdamClient&&) = default;
    ~AdamClient() = default;

    T diff(T grad) override {
      const auto beta1 = adam->beta1();
      const auto beta2 = adam->beta2();

      m = beta1 * m + (1 - beta1) * grad;
      v = beta2 * v + (1 - beta2) * grad * grad;

      const auto m_hat = m / (1 - adam->beta1t());
      const auto v_hat = v / (1 - adam->beta2t());

      return - adam->eta() * m_hat / (std::sqrt(v_hat) + adam->eps());
    }
  };

  template<typename T> class Adam : public Optimizer<T> {
  private:
    T _eps;
    T _eta;
    T _beta1;
    T _beta1t;
    T _beta2;
    T _beta2t;
  public:
    Adam(): Adam{1e-3} {}
    Adam(T rl)
      : _eps{1e-8}, _eta{rl}, _beta1{0.9}, _beta1t{1}, _beta2{0.999}, _beta2t{1} {}
    Adam(const Adam&) = default;
    Adam(Adam&&) = default;
    Adam& operator=(const Adam&) = default;
    Adam& operator=(Adam&&) = default;
    ~Adam() = default;
    OptimizerClient<T>* client() const override {
      return new AdamClient<T>{this};
    }
    void step() override {
      _beta1t *= _beta1;
      _beta2t *= _beta2;
    }
    const auto eps() const noexcept { return _eps; }
    const auto eta() const noexcept { return _eta; }
    const auto beta1() const noexcept { return _beta1; }
    const auto beta1t() const noexcept { return _beta1t; }
    const auto beta2() const noexcept { return _beta2; }
    const auto beta2t() const noexcept { return _beta2t; }
  };
}

#endif
