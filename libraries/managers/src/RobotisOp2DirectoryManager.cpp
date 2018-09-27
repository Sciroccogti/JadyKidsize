#include "RobotisOp2DirectoryManager.hpp"

#include <cstdlib>

using namespace std;
using namespace managers;

extern "C" {
  char *wbu_system_getenv(const char *);
}

const string &RobotisOp2DirectoryManager::getDataDirectory() {
#ifdef CROSSCOMPILATION
  static string path = "/robotis/Data/";
#else
  char *WEBOTS_HOME = wbu_system_getenv("WEBOTS_HOME");
  static string path = string(WEBOTS_HOME) + "/projects/robots/robotis/darwin-op/libraries/robotis-op2/robotis/Data/";
#endif
  return path;
}
