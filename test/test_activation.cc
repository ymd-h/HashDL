#include "unittest.hh"

using namespace HashDL;

int main(int argc, char** argv){
  auto test = Test{};

  test.Add([](){
    auto L = Linear{};

    AssertEqual(L.call(0),0);
    AssertEqual(L.call(10), 10);
    AssertEqual(L.call(0.5), 0.5);
    AssertEqual(L.call(-10), -10);
    AssertEqual(L.call(-0.5), -0.5);
  },"Linear call");

  test.Add([](){
    auto L = Linear{};

    AssertEqual(L.back(0, 0), 1);
    AssertEqual(L.back(0, 10), 1);
    AssertEqual(L.back(0.2, 0.7), 1);
  }, "Linear back");

  test.Add([](){
    auto R = ReLU{};

    AssertEqual(R.call(10), 10);
    AssertEqual(R.call(0.5), 0.5);
    AssertEqual(R.call(0.2), 0.2);

    AssertEqual(R.call(-0.3), 0);
    AssertEqual(R.call(-10), 0);
  }, "ReLU call");

  test.Add([](){
    auto R = ReLU{};

    AssertEqual(R.back(0.5, 0.5), 1);
  },"ReLU back");

  return test.Run();
}
