#ifndef SCHEDULER_HH
#define SCHEDULER_HH

#include <iostream>
#include <cmath>
#include <cstdint>

namespace HashDL {

  class Scheduler {
  public:
    virtual bool operator()() = 0;
  };

  class ConstantFrequency : public Scheduler {
  private:
    std::uintmax_t counter;
    std::uintmax_t N;
  public:
    ConstantFrequency(): ConstantFrequency{1} {}
    ConstantFrequency(std::size_t n): counter{0}, N{n} {}
    ConstantFrequency(const ConstantFrequency&) = default;
    ConstantFrequency(ConstantFrequency&&) = default;
    ConstantFrequency& operator=(const ConstantFrequency&) = default;
    ConstantFrequency& operator=(ConstantFrequency&&) = default;
    virtual ~ConstantFrequency() = default;

    bool operator()() override {
      auto fulfilled = ++counter >= N;

      if(fulfilled){ counter = 0; }

      return fulfilled;
    }
  };

  template<typename T> class ExponentialDecay : public Scheduler {
  private:
    std::uintmax_t counter;
    std::uintmax_t N;
    T exp_decay;
  public:
    ExponentialDecay(): ExponentialDecay{1, 1.0} {}
    ExponentialDecay(std::size_t n, T decay):
      counter{0}, N{n}, exp_decay{} {
      constexpr const auto max_decay =
	std::log(std::numeric_limits<std::uintmax_t>::max());
      if(decay > max_decay){
	std::cerr << "WARNING: decay is too large. Use "
		  << std::to_string(max_decay) << " instead." << std::endl;
	exp_decay = std::numeric_limits<std::uintmax_t>::max();
      } else {
	exp_decay = std::exp(decay);
      }
    }
    ExponentialDecay(const ExponentialDecay&) = default;
    ExponentialDecay(ExponentialDecay&&) = default;
    ExponentialDecay& operator=(const ExponentialDecay&) = default;
    ExponentialDecay& operator=(ExponentialDecay&&) = default;
    ~ExponentialDecay() = default;

    bool operator()() override {
      auto fulfilled = ++counter >= N;

      if(fulfilled){
	counter = 0;
	N = static_cast<std::uintmax_t>(std::ceil(N * exp_decay));
      }

      return fulfilled;
    }
  };
}
#endif
