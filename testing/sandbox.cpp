#include "asgard.hpp"

#include "asgard_test_macros.hpp"

using namespace asgard;

using P = asgard::default_precision;

int main(int argc, char **argv)
{
  std::ignore = argc;
  std::ignore = argv;

  for (int i = 0; i < 10; i++)
    std::cout << i << "   " << fm::intlog2(i) << "   " << fm::ipow2_log2(i) << "\n";

  // keep this file clean for each PR
  // allows someone to easily come here, dump code and start playing
  // this is good for prototyping and quick-testing features/behavior
  return 0;
}
