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
    } catch (...) {
      auto e = std::current_exception();
      msg = e ? e.what() : "No description."
    }
  }

  explicit operator bool() const { return success; }
  void describe() const {
    std::cout << "Fail: " << name << "\n" << msg << std::endl;
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

    return fail ? std::EXIT_FAILURE: std::EXIT_SUCCESS;
  }

  void Summary(){
    for(auto& c : cases){ if(!c){ c.describe(); fail = true; } }
  }
};

template<typename T> inline constexpr auto size(T&& v){ return v.size(); }

template<typename T, std::size_t N>
inline constexpr auto size(const T[N]&){ return N; }

template<typename L, typename R>
inline constexpr void AssertEqual(L&& lhs, R&& rhs){
  if(lhs != rhs){
    using std::to_string;
    throw std::runtime_error(to_string(lhs) + " != " + to_string(rhs));
  }
}

template<typename L, typename R>
inline constexpr void AssertEqual(L&& lhs, R&& rhs){
  using std::begin;
  using std::end;

  try {
    if(lhs.size() != rhs.size()){ throw ""; }

    std::for_each(begin(lhs), end(lhs),
		  [it=begin(rhs)](auto& v) mutable { AssertEqual(v, *(it++)); });
  } catch (...){
    using std::to_string;

    throw std::runtime_error();
  }
}

#endif
