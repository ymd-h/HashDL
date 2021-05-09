#ifndef OPTIMIZER_HH
#define OPTIMIZER_HH

#include <cmath>
#include <string>

namespace HashDL {
  template<typename T> class OptimizerClient {
  public:
    OptimizerClient() = default;
    OptimizerClient(const OptimizerClient&) = default;
    OptimizerClient(OptimizerClient&&) = default;
    OptimizerClient& operator=(const OptimizerClient&) = default;
    OptimizerClient& operator=(OptimizerClient&&) = default;
    virtual ~OptimizerClient() = default;

    virtual T diff(T grad) = 0;
    virtual std::string to_string() const = 0;
  };

  template<typename T> class Optimizer {
  public:
    Optimizer() = default;
    Optimizer(const Optimizer&) = default;
    Optimizer(Optimizer&&) = default;
    Optimizer& operator=(const Optimizer&) = default;
    Optimizer& operator=(Optimizer&&) = default;
    virtual ~Optimizer() = default;

    virtual OptimizerClient<T>* client() const = 0;
    virtual void step(){}
    virtual std::string to_string() const = 0;
  };

  template<typename T> class SGD;
  template<typename T> class SGDClient : public OptimizerClient<T> {
  private:
    const SGD<T>* sgd;
  public:
    SGDClient() = delete;
    SGDClient(const SGD<T>* sgd): sgd{sgd} {}
    SGDClient(const SGDClient&) = default;
    SGDClient(SGDClient&&) = default;
    SGDClient& operator=(const SGDClient&) = default;
    SGDClient& operator=(SGDClient&&) = default;
    ~SGDClient() = default;

    virtual T diff(T grad){
      return - sgd->eta() * grad;
    }

    std::string to_string() const override {
      std::string msg = "Client of " + sgd->to_string();
      return msg;
    }
  };

  template<typename T> class SGD : public Optimizer<T> {
  private:
    T _eta;
    T decay;
  public:
    SGD(): SGD{0.001} {}
    SGD(T lr, T decay=1.0): _eta{lr}, decay{decay} {}
    SGD(const SGD&) = default;
    SGD(SGD&&) = default;
    SGD& operator=(const SGD&) = default;
    SGD& operator=(SGD&&) = default;
    ~SGD() = default;

    OptimizerClient<T>* client() const override {
      return new SGDClient<T>{this};
    }
    void step() override { _eta *= decay; }
    const auto eta() const { return _eta; }

    std::string to_string() const override {
      std::string msg = "SGD<T>(eta=" + std::to_string(_eta) +
	", decay=" + std::to_string(decay) + ")";
      return msg;
    }
  };

  template<typename T> class Adam;
  template<typename T> class AdamClient : public OptimizerClient<T>{
  private:
    T m;
    T v;
    const Adam<T>* adam;
  public:
    AdamClient() = delete;
    AdamClient(const Adam<T>* adam): m{0}, v{0}, adam{adam} {}
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

    std::string to_string() const override {
      std::string msg = "Client of " + adam->to_string();
      return msg;
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
    Adam(T lr): Adam{lr, 0.9, 0.999} {}
    Adam(T lr, T b1, T b2, T e=1e-8)
      : _eps{e}, _eta{lr}, _beta1{b1}, _beta1t{b1}, _beta2{b2}, _beta2t{b2} {}
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

    std::string to_string() const override {
      std::string msg = "Adam<T>(eps=" + std::to_string(_eps)
	+ " ,eta=" + std::to_string(_eta)
	+ " ,beta1=" + std::to_string(_beta1)
	+ " ,beta2=" + std::to_string(_beta2) + ")";
      return msg;
    }
  };


  template<typename T> inline auto to_string(const OptimizerClient<T>& c){
    const auto p = &c;
    return p->to_string();
  }
  template<typename T> inline auto to_string(const Optimizer<T>& c){
    const auto p = &c;
    return p->to_string();
  }
}

#endif
