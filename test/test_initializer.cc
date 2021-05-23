#include <initializer.hh>

#include "unittest.hh"

int main(int, char**){
  auto test = Test{};

  return test.Run();
}
