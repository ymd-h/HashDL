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
  bool fail;
public:
  TestCase() = default;
  template<typename F> TestCase(F&& f, std::string name)
    : test{f}, name{name}, fail{false} {}
  TestCase(const TestCase&) = default;
  TestCase(TestCase&&) = default;
  TestCase& operator=(const TestCase&) = default;
  TestCase& operator=(TestCase&&) = default;
  ~TestCase() = default;

  auto operator()(){
    try { test(); } catch (...) { fail = true; }
  }
  explicit operator bool() const { return !fail; }
};

class Test {
private:
  std::vector<TestCase> cases;
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

  }
};

template<typename T> inline constexpr auto size(T&& v){ return v.size(); }

template<typename T, std::size_t N>
inline constexpr auto size(const T[N]&){ return N; }

template<typename L, typename R>
inline constexpr void AssertEqual(L&& lhs, R&& rhs){
  if(lhs != rhs){ throw std::runtime_error(); }
}

template<typename L, typename R>
inline constexpr void AssertEqual(L&& lhs, R&& rhs){
  using std::begin;
  using std::end;

  try {
    if(lhs.size() != rhs.size()){ throw std::runtime_error(); }

    std::for_each(begin(lhs), end(lhs),
		  [it=begin(rhs)](auto& v) mutable { AssertEqual(v, *it); });
  } catch (...){
    throw std::runtime_error();
  }
}

#endif
