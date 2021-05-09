#include <scheduler.hh>

#include "unittest.hh"

int main(int argc, char** argv){
  using namespace HashDL;
  auto test = Test{};

  test.Add([](){
    auto confw = ConstantFrequency{};

    AssertTrue(confw());
    AssertTrue(confw());
    AssertTrue(confw());
    AssertTrue(confw());
  }, "1-step Constant Frequency");

  test.Add([](){
    auto confw = ConstantFrequency{2};

    AssertFalse(confw());
    AssertTrue(confw());
    AssertFalse(confw());
    AssertTrue(confw());
  }, "2-step Constant Frequency");

  test.Add([](){
    auto exp = ExponentialDecay<float>{1, 0};

    AssertTrue(exp());
    AssertTrue(exp());
    AssertTrue(exp());
    AssertTrue(exp());
  }, "Exp 0-decay");

  test.Add([](){
    auto exp = ExponentialDecay<float>{1, 100};

    AssertTrue(exp());
    AssertFalse(exp());
    AssertFalse(exp());
    AssertFalse(exp());
  }, "Exp large-decay");

  test.Add([](){
    auto exp = ExponentialDecay<float>{1, 1};

    AssertTrue(exp());
    AssertFalse(exp());
    AssertFalse(exp());
    AssertFalse(exp());
    AssertTrue(exp());
  }, "Exp 1-decay");

  return Test.Run();
}
