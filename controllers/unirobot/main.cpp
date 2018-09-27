// Description:   Manage the entree point function

#include "UniRobot.hpp"

#include <cstdlib>

using namespace webots;

int main(int argc, char **argv)
{
  UniRobot *controller = new UniRobot(argc, argv);
  controller->run();
  delete controller;
  return EXIT_FAILURE;
}
