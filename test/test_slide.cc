#include <slide.hh>

#include "unittest.hh"

int main(int argc, char** argv){
  using namespace HashDL;

  auto test = Test{};
  auto opt = std::unique_ptr<Optimizer<float>>(new SGD<float>{1});

  test.Add([&](){
    auto p = Param<float>{opt};

    AssertEqual(p(), 0);

    p.add_grad(0.5);
    AssertEqual(p(), 0);

    p.update();
    AssertEqual(p(), -0.5);
  }, "Param");

  test.Add([&](){
    auto p = Param<float>{opt, 0.5};

    AssertEqual(p(), 0.5);

    p.add_grad(0.5);
    AssertEqual(p(), 0.5);

    p.update();
    AssertEqual(p(), 0);
  }, "Param with initialization");

  test.Add([&](){
    auto w = Weight<float>{1, opt};

    AssertEqual(w.weight(), std::vector<float>{0.0});
    AssertEqual(w.weight(0), 0);
    AssertEqual(w.bias(), 0);
    AssertEqual(w.affine(Data<float>{1}, std::vector<std::size_t>{0}), 0);
    AssertEqual(w.affine(Data<float>{1}, std::vector<std::size_t>{}), 0);
  }, "Weight");

  test.Add([&](){
    auto w = Weight<float>{1, opt};

    AssertEqual(w.weight(), std::vector<float>{0.0});
    AssertEqual(w.weight(0), 0);
    AssertEqual(w.bias(), 0);
    AssertEqual(w.affine(Data<float>{1}, std::vector<std::size_t>{0}), 0);
    AssertEqual(w.affine(Data<float>{1}, std::vector<std::size_t>{}), 0);

    w.add_weight_grad(0, 0.5);
    w.add_bias_grad(0.2);
    AssertEqual(w.weight(), std::vector<float>{0.0});
    AssertEqual(w.weight(0), 0);
    AssertEqual(w.bias(), 0);
    AssertEqual(w.affine(Data<float>{1}, std::vector<std::size_t>{0}), 0);
    AssertEqual(w.affine(Data<float>{1}, std::vector<std::size_t>{}), 0);

    w.update();
    AssertEqual(w.weight(), std::vector<float>{-0.5});
    AssertEqual(w.weight(0), -0.5);
    AssertEqual(w.bias(), -0.2);
    AssertEqual(w.affine(Data<float>{1}, std::vector<std::size_t>{0}), -0.2);
    AssertEqual(w.affine(Data<float>{1}, std::vector<std::size_t>{}), -0.2);
    AssertEqual(w.affine(Data<float>{std::vector<float>{1}},
			 std::vector<std::size_t>{0}), -0.7);
    AssertEqual(w.affine(Data<float>{std::vector<float>{1}},
			 std::vector<std::size_t>{}), -0.2);
  }, "Weight update");

  test.Add([&](){
    auto w = Weight<float>{2, opt};

    AssertEqual(w.weight(), std::vector<float>{0.0, 0.0});
    AssertEqual(w.weight(0), 0);
    AssertEqual(w.weight(1), 0);
    AssertEqual(w.bias(), 0);
    AssertEqual(w.affine(Data<float>{2}, std::vector<std::size_t>{0, 1}), 0);
    AssertEqual(w.affine(Data<float>{2}, std::vector<std::size_t>{}), 0);
  }, "Multi dimension");

  test.Add([&](){
    auto w = Weight<float>{1, opt, [](){ return 0.5; }};

    AssertEqual(w.weight(), std::vector<float>{0.5});
    AssertEqual(w.weight(0), 0.5);
    AssertEqual(w.bias(), 0);
  }, "Weight initialization");

  test.Add([&](){
    auto N = Neuron<float>{1, opt};
    auto a = std::unique_ptr<Activation<float>>{new Linear<float>{}};

    AssertEqual(N.w(), Data<float>{1});
    AssertEqual(N.forward(Data<float>{1}, std::vector<std::size_t>{} , a), 0);
    AssertEqual(N.forward(Data<float>{1}, std::vector<std::size_t>{0}, a), 0);
  }, "Neuron");

  test.Add([&](){
    auto N = Neuron<float>{3, opt};
    auto a = std::unique_ptr<Activation<float>>{new Linear<float>{}};

    AssertEqual(N.w(), Data<float>{3});
    AssertEqual(N.forward(Data<float>{1}, std::vector<std::size_t>{} , a), 0);
    AssertEqual(N.forward(Data<float>{1}, std::vector<std::size_t>{0, 1}, a), 0);
  }, "Neuron multi prev");

  test.Add([&](){
    auto N = Neuron<float>{1, opt};
    auto a = std::unique_ptr<Activation<float>>{new Linear<float>{}};

    AssertEqual(N.w(), Data<float>{1});
    AssertEqual(N.forward(Data<float>{1}, std::vector<std::size_t>{} , a), 0);
    AssertEqual(N.forward(Data<float>{1}, std::vector<std::size_t>{0}, a), 0);

    auto dL_dy = Data<float>{std::vector<float>{1.0}};
    N.backward(Data<float>{1}, 0, 0.0, dL_dy,
	       std::vector<std::size_t>{0}, a);
    AssertEqual(N.forward(Data<float>{1}, std::vector<std::size_t>{0}, a), 0);
    N.update();
    AssertEqual(N.forward(Data<float>{1}, std::vector<std::size_t>{0}, a), -1.0);
    AssertEqual(N.w(), Data<float>{std::vector<float>{-1.0}});
  }, "Neuron backward");

  return test.Run();
}
