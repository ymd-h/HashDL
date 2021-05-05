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

  return test.Run();
}
