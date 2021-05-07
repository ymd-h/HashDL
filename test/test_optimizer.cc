#include <optimizer.hh>

#include "unittest.hh"

int main(int argc, char** argv){
  using namespace HashDL;

  auto test = Test{};

  test.Add([](){
    auto sgd = SGD<float>{};

    auto eta = sgd.eta();
    sgd.step();
    AssertEqual(sgd.eta(), eta);
  }, "SGD with no-decay");

  test.Add([](){
    auto decay = 0.1;
    auto sgd = SGD<float>{0.1, decay};

    auto eta = sgd.eta();

    sgd.step();
    AssertEqual(sgd.eta(), eta * decay);

    sgd.step();
    AssertEqual(sgd.eta(), eta * decay * decay);
  }, "SGD with decay");

  test.Add([](){
    auto rl = 0.1;
    auto sgd = SGD<float>{rl};

    auto c = sgd.client();
    AssertTrue(c);

    AssertEqual(c->diff(0), c->diff(0));
    AssertEqual(c->diff(0.5) * 2, c->diff(1.0));

    AssertEqual(c->diff(2.0), (sgd.step(), c->diff(2.0)));

    AssertEqual(c->diff(0.5), - rl * 0.5);

    if(c){
      delete c;
      c = nullptr;
    }
  }, "SGD Client");

  test.Add([](){
    auto rl = 0.1;
    auto decay = 0.01;
    auto sgd = SGD<float>{rl, decay};

    auto c = sgd.client();
    AssertTrue(c);

    auto y = 0.7;
    auto z = 0.9;
    AssertEqual(c->diff(y), c->diff(y));
    AssertEqual(c->diff(z), - rl * z);

    auto x = 2.3;
    auto eta = c->diff(x);
    sgd.step();

    AssertEqual(c->diff(x), eta * decay);

    if(c){
      delete c;
      c = nullptr;
    }
  }, "SGD Client with decay");

  test.Add([](){
    auto adam = Adam<float>{};

    AssertEqual(adam.eta(), 1e-3);
    AssertEqual(adam.eps(), 1e-8);
    AssertEqual(adam.beta1(), 0.9);
    AssertEqual(adam.beta2(), 0.999);

    AssertEqual(adam.beta1t(), 1);
    AssertEqual(adam.beta2t(), 1);

    adam.step();
    AssertEqual(adam.beta1t(), std::pow(adam.beta1(), 1));
    AssertEqual(adam.beta2t(), std::pow(adam.beta2(), 1));

    adam.step();
    AssertEqual(adam.beta1t(), std::pow(adam.beta1(), 2));
    AssertEqual(adam.beta2t(), std::pow(adam.beta2(), 2));
  }, "Adam");

  return test.Run();
}
