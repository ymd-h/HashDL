#include <data.hh>

#include "unittest.hh"

int main(int argc, char** argv){
  auto test = Test{};
  auto v = std::vector<float>{0.1, 0.2, 0.3, 0.4, 0.5};

  test.Add([](){
    auto data = Data<float>{10};
    AssertEqual(data.size(), 10);
  }, "Data.size()");

  test.Add([=](){
    auto data = Data<float>(v.begin(), v.end());

    AssertEqual(data, v);
  }, "Data construction");

  test.Add([=](){
    auto data = Data<float>(v.begin(), v.end());

    AssertEqual(data[2], v[2]);
    AssertEqual(data[4], v[4]);
    AssertEqual(data[6], v[6]);
  }, "Data access");

  test.Add([=](){
    auto data = Data<float>(v.begin(), v.end(), [](auto vi){ return 2*vi; });

    AssertEqual(data[2], v[2] * 2);
    AssertEqual(data[4], v[4] * 2);
    AssertEqual(data[6], v[6] * 2);
  }, "Data transform");

  test.Add([=](){
    auto data = Data<float>(v);

    AssertEqual(data, v);
  }, "Data vector construction");

  return test.Run();
}
