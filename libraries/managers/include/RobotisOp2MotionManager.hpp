// Description:   Facade between webots and the robotis-op2 framework
//                allowing to play the Robotis motion files

#ifndef ROBOTISOP2_MOTION_MANAGER_HPP
#define ROBOTISOP2_MOTION_MANAGER_HPP

#include <string>

#define DMM_NMOTORS 20

namespace webots {
  class Robot;
  class Motor;
  class PositionSensor;
}

namespace Robot {
  class Action;
}

namespace managers {
  using namespace Robot;
  class RobotisOp2MotionManager {
    public:
                       RobotisOp2MotionManager(webots::Robot *robot);
      virtual         ~RobotisOp2MotionManager();
      bool             isCorrectlyInitialized() { return mCorrectlyInitialized; }
      void             playPage(int id, bool sync = true);
      void             step(int duration);
      bool             isMotionPlaying() { return mMotionPlaying; }

    private:
      webots::Robot   *mRobot;
      bool             mCorrectlyInitialized;
      Action          *mAction;
      int              mBasicTimeStep;
      bool             mMotionPlaying;

#ifndef CROSSCOMPILATION
      void             myStep();
      void             wait(int duration);
      void             achieveTarget(int timeToAchieveTarget);
      double           valueToPosition(unsigned short value);
      void             InitMotionAsync();

      webots::Motor           *mMotors[DMM_NMOTORS];
      webots::PositionSensor  *mPositionSensors[DMM_NMOTORS];
      double                   mTargetPositions[DMM_NMOTORS];
      double                   mCurrentPositions[DMM_NMOTORS];
      int                      mRepeat;
      int                      mStepnum;
      int                      mWait;
      int                      mStepNumberToAchieveTarget;
      void                    *mPage;
#else
      static void     *MotionThread(void *param);// thread function

      pthread_t        mMotionThread;// thread structure
#endif
  };
}

#endif
