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
    ~ConstantFrequency() = default;

    bool operator()() override {
      auto fulfilled = counter++ >= N;

      if(fulfilled){ counter = 0; }

      return fulfilled;
    }
  };
}
#endif
