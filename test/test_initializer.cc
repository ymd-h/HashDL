#include <initializer.hh>

#include "unittest.hh"

int main(int, char**){
  using namespace HashDL;

  auto test = Test{};

  test.Add([](){
    auto init = ConstantInitializer<float>{0};
    AssertEqual(init(), 0);
    AssertEqual(init(), 0);
  }, "Constant Initializer");

  test.Add([](){
    auto init = ConstantInitializer<float>{0.5};
    AssertEqual(init(), 0.5);
    AssertEqual(init(), 0.5);
  }, "Constant Initializer with non-zero");

  test.Add([](){
    auto init = GaussInitializer<float>{0, 1};
    init();
  }, "Gauss Initializer");

  return test.Run();
}
