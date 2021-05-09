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

  return Test.Run();
}
