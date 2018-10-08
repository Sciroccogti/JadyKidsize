// Description:   Simple unirobot player showing how to use the middleware between webots and
//                the robotis-op2 framework

#ifndef UNIROBOT_HPP
#define UNIROBOT_HPP

#define NMOTORS 20

#include <opencv2/opencv.hpp>
#include <webots/Robot.hpp>
#include <string>
#include <vector>
#include <map>
#include <list>

namespace managers {
  class RobotisOp2MotionManager;
  class RobotisOp2GaitManager;
  class RobotisOp2VisionManager;
}

namespace webots {
  class Motor;
  class LED;
  class Camera;
  class Accelerometer;
  class PositionSensor;
  class Gyro;
  class Display;
};

class UniRobot : public webots::Robot {
  public:
                                     UniRobot(int argc, char **argv);
    virtual                         ~UniRobot();
    void                             run();
    
  private:
    void                             myStep();
    void                             wait(int ms);
    void                             showImage(const unsigned char *rgb);
    void                             imageProcess();
    unsigned char*                   getRGBImage();
    cv::Mat                          getRGBMat(); 

  private:
    enum RunMode
    {
      MODE_BALL = 1,
      MODE_LINE = 2
    };
    
    struct ResultsInfo
    {
      bool ball_found;
      double ball_x, ball_y;
	  double direction;
    };
    ResultsInfo resInfo;
    
    RunMode mode;
    int                              mTimeStep;
    int                              image_w, image_h;
    webots::Motor                    *mMotors[NMOTORS];
    webots::PositionSensor           *mPositionSensors[NMOTORS];
    webots::LED                      *mEyeLED;
    webots::LED                      *mHeadLED;
    webots::LED                      *mBackLedRed;
    webots::LED                      *mBackLedGreen;
    webots::LED                      *mBackLedBlue;
    webots::Camera                   *mCamera;
    webots::Accelerometer            *mAccelerometer;
    webots::Gyro                     *mGyro;
    webots::Display                  *mDisplay;
    managers::RobotisOp2MotionManager  *mMotionManager;
    managers::RobotisOp2GaitManager    *mGaitManager;
};

#endif
