#ifndef SCHEDULER_HH
#define SCHEDULER_HH

#include <cmath>

namespace HashDL {

  class Scheduler {
  public:
    virtual bool operator()() = 0;
  };

  class ConstantFrequency : public Scheduler {
  private:
    std::size_t counter;
    std::size_t N;
  public:
    ConstantFrequency(): ConstantFrequency{1} {}
    ConstantFrequency(std::size_t N): counter{0}, N{N} {}
    ConstantFrequency(const ConstantFrequency&) = default;
    ConstantFrequency(ConstantFrequency&&) = default;
    ConstantFrequency& operator=(const ConstantFrequency&) = default;
    ConstantFrequency& operator=(ConstantFrequency&&) = default;
    virtual ~ConstantFrequency() = default;

    bool operator()() override {
      auto fulfilled = counter++ >= N;

      if(fulfilled){ counter = 0; }

      return fulfilled;
    }
  };

  template<typename T> class ExponentialDecay : public Scheduler {
  private:
    std::size_t counter;
    std::size_t N;
    T exp_decay;
  public:
    ExponentialDecay(): ExponentialDecay{1, 1.0} {}
    ExponentialDecay(std::size_t N, T decay):
      counter{0}, N{N}, exp_decay{std::exp(decay)} {}
    ExponentialDecay(const ExponentialDecay&) = default;
    ExponentialDecay(ExponentialDecay&&) = default;
    ExponentialDecay& operator=(const ExponentialDecay&) = default;
    ExponentialDecay& operator=(ExponentialDecay&&) = default;
    ~ExponentialDecay() = default;

    bool operator()() override {
      auto fulfilled = counter++ >= N;

      if(fulfilled){
	counter = 0;
	N = static_cast<std::size_t>(std::ceil(N * exp_decay));
      }

      return fulfilled;
    }
  };
}
#endif
