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
#include <memory>

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
      -> decltype(begin(v), end(v), std::true_type());
    static constexpr std::false_type ADL(...);

    template<typename U>
    static constexpr auto STD(U&& v)
      -> decltype(std::begin(v), std::end(v), std::true_type());
    static constexpr std::false_type STD(...);

    template<typename U>
    static constexpr auto Member(U&& v)
      -> decltype(v.begin(), v.end(), std::true_type());
    static constexpr std::false_type Member(...);
  public:
    static constexpr bool value = (decltype(ADL(std::declval<T>()))::value ||
				   decltype(STD(std::declval<T>()))::value ||
				   decltype(Member(std::declval<T>()))::value);
    template<typename U>
    static auto begin(U&& v){
      using U_t = std::remove_reference_t<U>;
      static_assert(std::is_same_v<U_t, std::remove_reference_t<T>>,
		    "Call is_iterable<T>::begin() with wrong type argument");
      static_assert(value,
		    "is_iterable<T>::begin() is called for non-iterable type.");

      if constexpr (decltype(Member(std::declval<U_t>()))::value){
	return v.begin();
      } else {
	using std::begin;
	return begin(std::forward<U>(v));
      }
    }

    template<typename U>
    static auto end(U&& v){
      using U_t = std::remove_reference_t<U>;
      static_assert(std::is_same_v<U_t, std::remove_reference_t<T>>,
		    "Call is_iterable<T>::end() with wron type argument");
      static_assert(value,
		    "is_iterable<T>::end() is called for non-iterable type.");

      if constexpr (decltype(Member(std::declval<U_t>()))::value){
	return v.end();
      } else {
	using std::end;
	return end(std::forward<U>(v));
      }
    }
  };


  template<typename T> inline constexpr auto size(T&& v){ return v.size(); }

  template<typename T, std::size_t N>
  inline constexpr auto size(const T(&)[N]){ return N; }

  template<typename T> struct is_smart_pointer: std::false_type {};
  template<typename T> struct is_smart_pointer<std::unique_ptr<T>>: std::true_type {};
  template<typename T> struct is_smart_pointer<std::shared_ptr<T>>: std::true_type {};

  template<typename T>
  inline constexpr auto to_string(T&& v){
    using std::to_string;

    std::string msg = "";
    if constexpr (is_iterable<T>::value) {
      msg += "[";
      for(auto& vi : v){
	msg += to_string(vi);
	msg += ",";
      }
      msg += "]";
    } else if constexpr (std::is_pointer_v<std::remove_reference_t<T>> ||
			 is_smart_pointer<std::remove_reference_t<T>>::value) {
      if(v){
	msg += "&(" + to_string(*v) + ")";
      }else{
	msg += "nullptr";
      }
    } else if constexpr (std::is_arithmetic_v<std::remove_reference_t<T>>) {
      msg += std::to_string(v);
    } else {
      static_assert(is_iterable<T>::value, "Cannot convert to std::string.");
    }

    return msg;
  }

  template<typename L, typename R>
  inline constexpr bool Equal(L&& lhs, R&& rhs){
    constexpr const auto L_iterable = is_iterable<L>::value;
    constexpr const auto R_iterable = is_iterable<R>::value;

    if constexpr (L_iterable && R_iterable) {
      return std::equal(is_iterable<L>::begin(lhs), is_iterable<L>::end(lhs),
			is_iterable<R>::begin(rhs), is_iterable<R>::end(rhs),
			[](auto& l, auto& r){ return Equal(l,r); });
    } else if constexpr ((!L_iterable) && (!R_iterable)){
      using LL = std::remove_reference_t<L>;
      using RR = std::remove_reference_t<R>;
      using LR = std::common_type_t<LL, RR>;

      if constexpr (std::is_floating_point_v<LR>){
	using std::abs;

	// When float and double, float has larger error (epsilon).
	// However, float is promoted to double.
	// Without taking care, large error would be compared with smaller threshold.
	constexpr auto eps = std::max<LR>(std::is_floating_point_v<LL> ?
					  std::numeric_limits<LL>::epsilon() : LL{0},
					  std::is_floating_point_v<RR> ?
					  std::numeric_limits<RR>::epsilon() : RR{0});
	constexpr const LR one = 1.0;

	// epsilon is the difference between 1.0 and the next value.
	// Relative comparison (|X-Y| < eps      ) is preferred for large value.
	// Absolute comparison (|X-Y| < eps * |X|) is preferred for small value.
	return abs(lhs - rhs) <= eps * std::max<LR>({one,abs((LR)lhs),abs((LR)rhs)});
      } else {
	return (LR)lhs == (LR)rhs;
      }
    } else {
      static_assert(L_iterable == R_iterable,
		    "Cannot compare iterable and non-iterable");
    }
  }
}

template<typename L, typename R>
inline constexpr void AssertEqual(L&& lhs, R&& rhs){
  using namespace unittest;
  if(!Equal(lhs, rhs)){
    using std::to_string;
    using unittest::to_string;
    throw std::runtime_error(to_string(lhs) + " != " + to_string(rhs));
  }
}

template<typename L, typename R>
inline constexpr void AssertNotEqual(L&& lhs, R&& rhs){
  using namespace unittest;
  if(!!Equal(lhs, rhs)){
    using std::to_string;
    using unittest::to_string;
    throw std::runtime_error(to_string(lhs) + " == " + to_string(rhs));
  }
}

template<typename Cond>
inline constexpr void AssertTrue(Cond&& c){
  using namespace unittest;
  using std::to_string;
  using unittest::to_string;
  if constexpr (is_iterable<Cond>::value){
    for(auto& ci : c){ AssertTrue(ci); }
  } else {
    if(!c){ throw std::runtime_error(to_string(c) + " != true"); }
  }
}

template<typename Cond>
inline constexpr void AssertFalse(Cond&& c){
  using namespace unittest;
  using std::string;
  using unittest::to_string;
  if constexpr (is_iterable<Cond>::value){
    for(auto& ci : c){ AssertFalse(ci); }
  } else {
    if(!!c){ throw std::runtime_error(to_string(c) + " != false"); }
  }
}

template<typename E> struct AssertRaises{
  template<typename F>
  AssertRaises(F&& f, const std::string& msg){
    auto correct_error = false;
    try {
      f();
    } catch (const E& e){
      correct_error = true;
    } catch (...){
      throw std::runtime_error(msg + " throws wrong exception");
    }

    if(!correct_error){
      throw std::runtime_error(msg + " doesn't throw exception");
    }
  }
};

#endif
