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
  template<typename T> struct is_iterable {
  private:
    template<typename U>
    static constexpr auto ADL(U&& v)
      -> decltype(begin(std::declval<std::remove_reference_t<U>>()),
		  end  (std::declval<std::remove_reference_t<U>>()),
		  std::true_type());
    static constexpr std::false_type ADL(...);

    template<typename U>
    static constexpr auto STD(U&& v)
      -> decltype(std::begin(std::declval<std::remove_reference_t<U>>()),
		  std::end  (std::declval<std::remove_reference_t<U>>()),
		  std::true_type());
    static constexpr std::false_type STD(...);
  public:
    static constexpr bool value = (decltype(ADL(std::declval<T>()))::type::value ||
				   decltype(STD(std::declval<T>()))::type::value);
  };


  template<typename T> inline constexpr auto size(T&& v){ return v.size(); }

  template<typename T, std::size_t N>
  inline constexpr auto size(const T(&)[N]){ return N; }

  template<typename T>
  inline constexpr auto to_string(T&& v){
    using std::to_string;
    using std::begin;
    using std::end;

    std::string msg = "[";
    for(auto& vi : v){
      msg += to_string(vi);
      msg += ",";
    }
    msg += "]";

    return msg;
  }

  template<typename L, typename R>
  inline constexpr bool Equal(L&& lhs, R&& rhs){
    constexpr const auto L_iterable = is_iterable<L>::value;
    constexpr const auto R_iterable = is_iterable<R>::value;

    if constexpr (L_iterable && R_iterable) {
      using std::begin;
      using std::end;
      return std::equal(begin(lhs), end(lhs), begin(rhs), end(rhs));
    } else if constexpr ((!L_iterable) && (!R_iterable)){
      return lhs == rhs;
    } else {
      static_assert(L_iterable == R_iterable,
		    "Cannot compare iterable and non-iterable");
    }
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

    // epsilon is the difference between 1.0 and the next value.
    // Relative comparison (|X-Y| < eps      ) is preferred for large value.
    // Absolute comparison (|X-Y| < eps * |X|) is preferred for small value.
    not_equal = !(abs(lhs - rhs) <= eps * std::max<LR>({1.0, abs(lhs), abs(rhs)}));
  } else {
    // Literal 0 is considered as int first, then as char.
    // 1st: Equal(L&&, R&&, int)  for iterable (fail for non-iterable)
    // 2nd: Equal(L&&, R&&, char) for non-iterable
    not_equal = !Equal(lhs, rhs);
  }

  if(not_equal){
    using std::to_string;
    using unittest::to_string;
    throw std::runtime_error(to_string(lhs) + " != " + to_string(rhs));
  }
}

template<typename Cond>
inline constexpr void AssertTrue(Cond&& c){
  using namespace unittest;
  if constexpr (is_iterable<Cond>::value){
    using std::begin;
    using std::end;
    for(auto& ci : c){ AssertTrue(ci); }
  } else {
    using std::to_string;
    if(!c){ throw std::runtime_error(to_string(c) + " != true"); }
  }
}

template<typename Cond>
inline constexpr void AssertFalse(Cond&& c){
  using namespace unittest;
  if constexpr (is_iterable<Cond>::value){
    using std::begin;
    using std::end;
    for(auto& ci : c){ AssertFalse(ci); }
  } else {
    using std::string;
    if(!!c){ throw std::runtime_error(to_string(c) + " != false"); }
  }
}

#endif
