#include "UniRobot.hpp"
#include <webots/Motor.hpp>
#include <webots/LED.hpp>
#include <webots/Camera.hpp>
#include <webots/Accelerometer.hpp>
#include <webots/PositionSensor.hpp>
#include <webots/Gyro.hpp>
#include <webots/Display.hpp>
#include <RobotisOp2MotionManager.hpp>
#include <RobotisOp2GaitManager.hpp>
#include <RobotisOp2VisionManager.hpp>

#include <cassert>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>

using namespace webots;
using namespace managers;
using namespace std;
using namespace cv;

static double clamp(double value, double min, double max) 
{
  if (min > max) 
  {
    assert(0);
    return value;
  }
  return value < min ? min : value > max ? max : value;
}

static double minMotorPositions[NMOTORS];
static double maxMotorPositions[NMOTORS];

void UniRobot::imageProcess()
{
  unsigned char *rgb = getRGBImage();  //get raw data, format: RGB
  Mat rgbMat = getRGBMat(); //get rgb data to cv::Mat
  
  showImage(rgb);
  if(mode == MODE_BALL)
  {
    //TODO Write down your code
    //update the resInfo
    resInfo.ball_found = false;
    resInfo.ball_x = 0.0;
    resInfo.ball_y = 0.0;
  }
  else if(mode == MODE_LINE)
  {
    //TODO Write down your code
    //update the resInfo
  }
  rgbMat.release();
  delete []rgb;
}

// function containing the main feedback loop
void UniRobot::run() 
{

  cout << "---------------SEU-UniRobot-2018---------------" << endl;
  vector<string> modes;
  modes.push_back("1. Ball");
  modes.push_back("2. Line");
  cout << "Run Mode: " <<modes[(int)mode-1]<< endl;

  // First step to update sensors values
  myStep();

  // set eye led to green
  mEyeLED->set(0x00FF00);

  // play the hello motion
  mMotionManager->playPage(1); // init position
  mMotionManager->playPage(9); // walkready position
  wait(200);

  // play the motion preparing the robot to walk
  mGaitManager->start();
  mGaitManager->step(mTimeStep);

  // main loop
  int fup = 0;
  int fdown = 0;
  const double acc_tolerance = 80.0;
  const double acc_step = 20;

  while (true) 
  {
    double neckPosition, headPosition;
    const double *acc = mAccelerometer->getValues();
    // count how many steps the accelerometer
    // says that the robot is down
    if (acc[1] < 512.0 - acc_tolerance)
      fup++;
    else
      fup = 0;

    if (acc[1] > 512.0 + acc_tolerance)
      fdown++;
    else
      fdown = 0;

    // the robot face is down
    if (fup > acc_step) 
    {
      mMotionManager->playPage(1); // init position
      mMotionManager->playPage(10); // f_up
      mMotionManager->playPage(9); // walkready position
      fup = 0;
    }
    // the back face is down
    else if (fdown > acc_step) 
    {
      mMotionManager->playPage(1); // init position
      mMotionManager->playPage(11); // b_up
      mMotionManager->playPage(9); // walkready position
      fdown = 0;
    }
    else //*********************************************************//
    {
      imageProcess();
      //TODO control the robot according to the resInfo you updated in imageProcess
      //demo
      //kick ball
      if(mode == MODE_BALL) // mode ball 
      {
        if(resInfo.ball_found)
        {
          if (resInfo.ball_y > 0.35) 
          {
            mGaitManager->stop();
            wait(500);
            if (resInfo.ball_x<0.0)
              mMotionManager->playPage(13); // left kick
            else
              mMotionManager->playPage(12); // right kick
            mMotionManager->playPage(9); // walkready position
            mGaitManager->start();
          }
        }
        //walk control
        mGaitManager->setXAmplitude(0.0); //x -1.0 ~ 1.0
        mGaitManager->setYAmplitude(0.0); //y -1.0 ~ 1.0
        mGaitManager->setAAmplitude(0.0); //dir -1.0 ~ 1.0
        mGaitManager->step(mTimeStep);
        //head control
        neckPosition = clamp(0.0, minMotorPositions[18], maxMotorPositions[18]); //head yaw position
        headPosition = clamp(0.5, minMotorPositions[19], maxMotorPositions[19]); //head pitch position
        mMotors[18]->setPosition(neckPosition);
        mMotors[19]->setPosition(headPosition);
      }
      else if(mode == MODE_LINE) //mode line
      {
        //walk control
        mGaitManager->setXAmplitude(0.0); //x -1.0 ~ 1.0
        mGaitManager->setYAmplitude(0.0); //y -1.0 ~ 1.0
        mGaitManager->setAAmplitude(0.0); //dir -1.0 ~ 1.0
        mGaitManager->step(mTimeStep);
        //head control
        neckPosition = clamp(0.0, minMotorPositions[18], maxMotorPositions[18]); //head yaw position
        headPosition = clamp(0.0, minMotorPositions[19], maxMotorPositions[19]); //head pitch position
        mMotors[18]->setPosition(neckPosition);
        mMotors[19]->setPosition(headPosition);
      }
    }
    myStep();
  }
}

static const char *motorNames[NMOTORS] = {
  "ShoulderR" /*ID1 */, "ShoulderL" /*ID2 */, "ArmUpperR" /*ID3 */, "ArmUpperL" /*ID4 */,
  "ArmLowerR" /*ID5 */, "ArmLowerL" /*ID6 */, "PelvYR"    /*ID7 */, "PelvYL"    /*ID8 */,
  "PelvR"     /*ID9 */, "PelvL"     /*ID10*/, "LegUpperR" /*ID11*/, "LegUpperL" /*ID12*/,
  "LegLowerR" /*ID13*/, "LegLowerL" /*ID14*/, "AnkleR"    /*ID15*/, "AnkleL"    /*ID16*/,
  "FootR"     /*ID17*/, "FootL"     /*ID18*/, "Neck"      /*ID19*/, "Head"      /*ID20*/
};

UniRobot::UniRobot(int argc, char **argv):
    Robot()
{
  mTimeStep = getBasicTimeStep();

  mEyeLED = getLED("EyeLed");
  mHeadLED = getLED("HeadLed");
  mHeadLED->set(0x00FF00);
  mBackLedRed = getLED("BackLedRed");
  mBackLedGreen = getLED("BackLedGreen");
  mBackLedBlue = getLED("BackLedBlue");
  mCamera = getCamera("Camera");
  mCamera->enable(2*mTimeStep);
  image_w = mCamera->getWidth();
  image_h = mCamera->getHeight();
  mAccelerometer = getAccelerometer("Accelerometer");
  mAccelerometer->enable(mTimeStep);
  mGyro = getGyro("Gyro");
  mGyro->enable(mTimeStep);
  mDisplay = getDisplay("Display");

  for (int i=0; i<NMOTORS; i++) {
    mMotors[i] = getMotor(motorNames[i]);
    string sensorName = motorNames[i];
    sensorName.push_back('S');
    mPositionSensors[i] = getPositionSensor(sensorName);
    mPositionSensors[i]->enable(mTimeStep);
    minMotorPositions[i] = mMotors[i]->getMinPosition();
    maxMotorPositions[i] = mMotors[i]->getMaxPosition();
  }

  mMotionManager = new RobotisOp2MotionManager(this);
  mGaitManager = new RobotisOp2GaitManager(this, "config.ini");
  
  string s(argv[1]);
  mode = (RunMode)(stoi(s));
}

UniRobot::~UniRobot() 
{
}

void UniRobot::myStep() {
  int ret = step(mTimeStep);
  if (ret == -1)
    exit(EXIT_SUCCESS);
}

void UniRobot::wait(int ms) 
{
  double startTime = getTime();
  double s = (double) ms / 1000.0;
  while (s + startTime >= getTime())
    myStep();
}

//param @rgb raw data in format of RGB 
void UniRobot::showImage(const unsigned char *rgb)
{
  int w = mDisplay->getWidth();
  int h = mDisplay->getHeight();
  Mat src(image_h, image_w, CV_8UC3, (unsigned char*)rgb);
  Mat dst;
  resize(src, dst, Size(w,h));
  ImageRef *img = mDisplay->imageNew(w, h, dst.data, Display::RGB);
  src.release();
  dst.release();
  mDisplay->imagePaste(img, 0, 0);
}

unsigned char* UniRobot::getRGBImage()
{
  unsigned char *im = (unsigned char*)(mCamera->getImage()); //image format: BGRA
  unsigned char *rgb = new unsigned char[3*image_w*image_h];
  unsigned int oidx=0, aidx=0;
  
  for(int y=0;y<image_h;y++)
  {
    for(int x=0;x<image_w;x++)
    {
      rgb[aidx+0] = im[oidx+2]; //R
      rgb[aidx+1] = im[oidx+1]; //G
      rgb[aidx+2] = im[oidx+0]; //B
      oidx += 4;
      aidx += 3;
    }
  }
  return rgb;
}

Mat UniRobot::getRGBMat()
{
  unsigned char *rgb = getRGBImage();
  Mat res(image_h, image_w, CV_8UC3, rgb);
  delete []rgb;
  return res;
}