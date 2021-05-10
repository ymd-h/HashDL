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

  test.Add([](){
    auto opt = std::unique_ptr<Optimizer<float>>(new SGD<float>{1});
    auto p = Param<float>{opt, 0.5};

    AssertEqual(p(), 0.5);

    p.add_grad(0.5);
    AssertEqual(p(), 0.5);

    p.update();
    AssertEqual(p(), 0);
  }, "Param with initialization");

  test.Add([](){
    auto opt = std::unique_ptr<Optimizer<float>>{new SGD{1}};
    auto w = Weight<float>{1, o};

    AssertEqual(w.weight(), std::vector<float>{0.0});
    AssertEqual(w.weight(0), 0);
    AssertEqual(w.bias(), 0);
    AssertEqual(w.affine(Data<float>{1}, std::vector<std::size_t>{0}), 0);
    AssertEqual(w.affine(Data<float>{1}, std::vector<std::size_t>{}), 0);
  }, "Weight");

  return test.Run();
}
