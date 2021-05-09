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

    AssertEqual(adam.beta1t(), 0.9);
    AssertEqual(adam.beta2t(), 0.99);

    adam.step();
    AssertEqual(adam.beta1t(), std::pow(adam.beta1(), 2));
    AssertEqual(adam.beta2t(), std::pow(adam.beta2(), 2));

    adam.step();
    AssertEqual(adam.beta1t(), std::pow(adam.beta1(), 3));
    AssertEqual(adam.beta2t(), std::pow(adam.beta2(), 3));
  }, "Adam");

  test.Add([](){
    auto rl = 1e-5;
    auto beta1 = 0.95;
    auto beta2 = 0.995;
    auto adam = Adam<float>{rl, beta1, beta2};

    AssertEqual(adam.eta(), rl);
    AssertEqual(adam.eps(), 1e-8);
    AssertEqual(adam.beta1(), beta1);
    AssertEqual(adam.beta2(), beta2);

    AssertEqual(adam.beta1t(), beta1);
    AssertEqual(adam.beta2t(), beta2);

    adam.step();
    AssertEqual(adam.beta1t(), std::pow(adam.beta1(), 2));
    AssertEqual(adam.beta2t(), std::pow(adam.beta2(), 2));

    adam.step();
    AssertEqual(adam.beta1t(), std::pow(adam.beta1(), 3));
    AssertEqual(adam.beta2t(), std::pow(adam.beta2(), 3));
  }, "Adam with hyperparameter");

  test.Add([](){
    auto rl = 1e-4;
    auto beta1 = 0.9;
    auto beta2 = 0.99;
    auto adam = Adam<float>{rl, beta1, beta2};

    auto c = adam.client();
    AssertTrue(c);

    auto x = 0.7;
    AssertTrue(std::isfinite(c->diff(x)));

    adam.step();
    AssertTrue(std::isfinite(c->diff(x)));
    AssertNotEqual(c->diff(x), c->diff(x));

    if(c){
      delete c;
      c = nullptr;
    }
  }, "Adam Client");

  test.Add([](){
    auto rl = 1e-5;
    auto beta1 = 0;
    auto beta2 = 0;
    auto eps = 0;
    auto adam = Adam<float>{rl, beta1, beta2, eps};

    auto c = adam.client();

    auto x = 0.5;
    AssertEqual(c->diff(x), -rl * x);

    if(c){
      delete c;
      c = nullptr;
    }
  }, "Adam 0-beta");

  test.Add([](){
    auto adam = Adam<float>{};

    auto c = adam.client();
    auto d = adam.client();

    auto x = 0.8;
    AssertEqual(c->diff(x), d->diff(x));
    AssertEqual(c->diff(x), d->diff(x));

    if(c){
      delete c;
      c = nullptr;
    }
    if(d){
      delete d;
      d = nullptr;
    }
  }, "Adam multi client");

  return test.Run();
}
