#ifndef UNITTEST_HH
#define UNITTEST_HH

#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <functional>
#include <vector>
#include <string>
#include <stdexcept>
#include <type_traits>

class TestCase {
private:
  std::function<void(void)> test;
  std::string name;
  std::string msg;
  bool success;
public:
  TestCase() = default;
  template<typename F> TestCase(F&& f, std::string name)
    : test{f}, name{name}, msg{}, success{false} {}
  TestCase(const TestCase&) = default;
  TestCase(TestCase&&) = default;
  TestCase& operator=(const TestCase&) = default;
  TestCase& operator=(TestCase&&) = default;
  ~TestCase() = default;

  auto operator()(){
    try {
      test();
      success = true;
    } catch (std::exception& e) {
      msg = e.what();
    } catch (...) {
      msg = "No description.";
    }
  }

  explicit operator bool() const { return success; }
  void describe() const {
    std::cout << "Fail: " << name << "\n" << msg << "\n" << std::endl;
  }
};

class Test {
private:
  std::vector<TestCase> cases;
  bool fail;
public:
  Test() = default;
  Test(const Test&) = default;
  Test(Test&&) = default;
  Test& operator=(const Test&) = default;
  Test& operator=(Test&&) = default;
  ~Test() = default;

  template<typename Case> void Add(Case&& test_case, std::string test_name = ""){
    if(test_name.empty()){
      test_name += "Test " + std::to_string(cases.size());
    }
    cases.emplace_back(std::forward<Case>(test_case), test_name);
  }

  int Run(){
    for(auto& c : cases){ c(); }

    Summary();

    return fail ? EXIT_FAILURE: EXIT_SUCCESS;
  }

  void Summary(){
    for(auto& c : cases){ if(!c){ c.describe(); fail = true; } }
  }
};

namespace unittest {
  template<typename T> inline constexpr auto size(T&& v){ return v.size(); }

  template<typename T, std::size_t N>
  inline constexpr auto size(const T(&)[N]){ return N; }

  template<typename T>
  inline constexpr auto to_string(T&& v){
    using std::to_string;

    std::string msg = "[";
    for(auto& vi : v){
      msg += to_string(vi);
      msg += ",";
    }
    msg += "]";
  }

  template<typename L, typename R>
  inline constexpr auto Equal(L&& lhs, R&& rhs, int){
    using std::begin;
    using std::end;

    return std::equal(begin(lhs), end(lhs), begin(rhs), end(rhs));
  }

  template<typename L, typename R>
  inline constexpr auto Equal(L&& lhs, R&& rhs, char){
    return lhs == rhs;
  }
}

template<typename L, typename R>
inline constexpr void AssertEqual(L&& lhs, R&& rhs){
  using namespace unittest;
  using LL = std::remove_reference_t<L>;
  using RR = std::remove_reference_t<R>;
  using LR = std::common_type_t<LL, RR>;

  bool not_equal = true;

  if constexpr (std::is_floating_point_v<LR>){
    using std::abs;

    // When float and double, float has larger error (epsilon).
    // However, float is promoted to double.
    // Without taking care, large error would be compared with smaller threshold.
    constexpr auto eps = std::max<LR>((std::is_floating_point_v<LL> ?
				       std::numeric_limits<LL>::epsilon() : LL{0}),
				      (std::is_floating_point_v<RR> ?
				       std::numeric_limits<RR>::epsilon() : RR{0}));

    not_equal = !(abs(lhs - rhs) <= eps * std::max<LR>({1.0, abs(lhs), abs(rhs)}));
  } else {
    // Literal 0 is considered as int first, then as char.
    // 1st: Equal(L&&, R&&, int)  for iterable (fail for non-iterable)
    // 2nd: Equal(L&&, R&&, char) for non-iterable
    not_equal = !Equal(lhs, rhs, 0);
  }

  if(not_equal){
    using std::to_string;
    using unittest::to_string;
    throw std::runtime_error(to_string(lhs) + " != " + to_string(rhs));
  }
}

#endif
