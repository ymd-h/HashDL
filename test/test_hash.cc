#include <hash.hh>

#include "unittest.hh"

int main(int argc, char** argv){
  using namespace HashDL;

  auto test = Test{};

  test.Add([](){
    auto wta = WTA<float>{};

    auto d = Data<float>{16};
    AssertEqual(wta.encode(d), wta.encode(d));
  }, "WTA");

  test.Add([](){
    using WTA_t = WTA<float>;
    AssertRaises<std::runtime_error>([](){ WTA_t{8, 2, 10}; },
				     "data size < sample size");
    AssertRaises<std::runtime_error>([](){ WTA_t{64, 16, 2}; },
				     "bin size * sample bits > 64");
    AssertRaises<std::runtime_error>([=](){
      auto wta = WTA_t{8, 8, 4};
      auto d = Data<float>{10};
    }, "Mis mutch data size");
  }, "WTA error");

  return test.Run():
}
