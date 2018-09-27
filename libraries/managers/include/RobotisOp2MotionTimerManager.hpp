// Description:   Facade between webots and the robotis-op2 framework
//                allowing to to start the LinuxMotionTimer in order
//                to play the Robotis motion files and the walking algorithm.

#ifndef ROBOTISOP2_MOTION_TIMER_MANAGER_HPP
#define ROBOTISOP2_MOTION_TIMER_MANAGER_HPP

namespace Robot {
  class LinuxMotionTimer;
}

namespace managers {
  using namespace Robot;
  class RobotisOp2MotionTimerManager {
    public:
                       RobotisOp2MotionTimerManager();
      virtual         ~RobotisOp2MotionTimerManager();
      static void      MotionTimerInit();

    private:
      static bool      mStarted;

  };
}

#endif
