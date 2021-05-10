#include <slide.hh>

#include "unittest.hh"

int main(int argc, char** argv){
  using namespace HashDL;

  auto test = Test{};

  test.Add([](){
    auto opt = std::unique_ptr<Optimizer<float>>(new SGD<float>{1});
    auto p = Param<float>{opt};

    AssertEqual(p(), 0);

    p.add_grad(0.5);
    AssertEqual(p(), 0);

    p.update();
    AssertEqual(p(), -0.5);
  }, "Param");

  return test.Run();
}
