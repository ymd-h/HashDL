#include <initializer.hh>

#include "unittest.hh"

int main(int, char**){
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

  return test.Run();
}
