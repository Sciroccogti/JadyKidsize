// Description:   Facade between webots and the robotis-op2 framework
//                allowing to handle the gait generator

#ifndef ROBOTISOP2_GAIT_MANAGER_HPP
#define ROBOTISOP2_GAIT_MANAGER_HPP

#include <string>

#define DGM_NMOTORS 20
#define DGM_BOUND(x,a,b) (((x)<(a))?(a):((x)>(b))?(b):(x))

namespace webots {
  class Robot;
  class Motor;
}

namespace Robot {
  class Walking;
}

namespace managers {
  using namespace Robot;
  class RobotisOp2GaitManager {
    public:
                       RobotisOp2GaitManager(webots::Robot *robot, const std::string &iniFilename);
      virtual         ~RobotisOp2GaitManager();
      bool             isCorrectlyInitialized() { return mCorrectlyInitialized; }

      void             setXAmplitude(double x) { mXAmplitude = DGM_BOUND(x, -1.0, 1.0) * 20.0; }
      void             setYAmplitude(double y) { mYAmplitude = DGM_BOUND(y, -1.0, 1.0) * 40.0; }
      void             setAAmplitude(double a) { mAAmplitude = DGM_BOUND(a, -1.0, 1.0) * 50.0; }
      void             setMoveAimOn(bool q) { mMoveAimOn = q; }
      void             setBalanceEnable(bool q) { mBalanceEnable = q; }

      void             start();
      void             step(int duration);
      void             stop();

    private:
      webots::Robot   *mRobot;
      bool             mCorrectlyInitialized;
      Walking         *mWalking;
      int              mBasicTimeStep;
      double           mXAmplitude;
      double           mAAmplitude;
      double           mYAmplitude;
      bool             mMoveAimOn;
      bool             mBalanceEnable;
      bool             mIsWalking;

#ifndef CROSSCOMPILATION
      void             myStep();
      double           valueToPosition(unsigned short value);
      webots::Motor   *mMotors[DGM_NMOTORS];
#endif
  };
}

#endif
