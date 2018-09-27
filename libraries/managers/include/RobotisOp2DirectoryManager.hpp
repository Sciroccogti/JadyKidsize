// Description:   Helper class allowing to retrieve directories

#ifndef ROBOTISOP2_DIRECTORY_MANAGER_HPP
#define ROBOTISOP2_DIRECTORY_MANAGER_HPP

#include <string>

namespace managers {
  class RobotisOp2DirectoryManager {
    public:
      static const std::string &getDataDirectory();
  };
}

#endif
