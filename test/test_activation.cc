#include "unittest.hh"

#include <activation.hh>

using namespace HashDL;

int main(int argc, char** argv){
  auto test = Test{};

  test.Add([](){
    auto L = Linear<float>{};

    AssertEqual(L.call(0),0);
    AssertEqual(L.call(10), 10);
    AssertEqual(L.call(0.5), 0.5);
    AssertEqual(L.call(-10), -10);
    AssertEqual(L.call(-0.5), -0.5);
  }, "Linear call");

  test.Add([](){
    auto L = Linear<float>{};

    AssertEqual(L.back(0, 0), 0);
    AssertEqual(L.back(0, 10), 10);
    AssertEqual(L.back(0.2, 0.7), 0.7);
  }, "Linear back");

  test.Add([](){
    auto R = ReLU<float>{};

    AssertEqual(R.call(10), 10);
    AssertEqual(R.call(0.5), 0.5);
    AssertEqual(R.call(0.2), 0.2);

    AssertEqual(R.call(-0.3), 0);
    AssertEqual(R.call(-10), 0);
  }, "ReLU call");

  test.Add([](){
    auto R = ReLU<float>{};

    AssertEqual(R.back(0.5, 0.5), 0.5);
    AssertEqual(R.back(10, 7), 7);
    AssertEqual(R.back(1, -0.5), -0.5);

    AssertEqual(R.back(-0.5, 0.5), 0);
    AssertEqual(R.back(-10, 0.5), 0);
    AssertEqual(R.back(-1, 0.5), 0);
  }, "ReLU back");

  return test.Run();
}
