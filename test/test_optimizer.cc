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



  return test.Run();
}
