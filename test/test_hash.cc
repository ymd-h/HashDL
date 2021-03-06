#include <hash.hh>

#include "unittest.hh"

int main(int, char**){
  using namespace HashDL;

  auto test = Test{};

  test.Add([](){
    auto wta = WTA<float>{};

    auto d = Data<float>{16};
    AssertEqual(wta.encode(d), wta.encode(d));
    AssertEqual(wta.encode(d), hashcode_t{0});
  }, "WTA");

  test.Add([](){
    using WTA_t = WTA<float>;
    AssertRaises<std::runtime_error>([](){ WTA_t{8, 2, 10}; },
				     "data size < sample size");
    AssertRaises<std::runtime_error>([](){ WTA_t{64, 16, 8}; },
				     "bin size * sample bits > 64");
    AssertRaises<std::runtime_error>([=](){
      auto wta = WTA_t{8, 8, 4};
      auto d = Data<float>{10};
      wta.encode(d);
    }, "Mis mutch data size");
  }, "WTA error");

  test.Add([](){
    auto dwta = DWTA<float>{};
    auto d = Data<float>{16};

    AssertEqual(dwta.encode(d), dwta.encode(d));
    AssertEqual(dwta.encode(d), hashcode_t{0});
  }, "DWTA");

  test.Add([](){
    using DWTA_t = DWTA<float>;
    using Assert_t = AssertRaises<std::runtime_error>;

    Assert_t([](){ DWTA_t{8, 2, 10}; }, "data size < sample size");
    Assert_t([](){ DWTA_t{64, 16, 8}; }, "bin size * sample bits > 64");
    Assert_t([=](){
      auto dwta = DWTA_t{8, 8, 4};
      auto d = Data<float>{16};
      dwta.encode(d);
    }, "Mis much data size");
  }, "DWTA error");

  return test.Run();
}
