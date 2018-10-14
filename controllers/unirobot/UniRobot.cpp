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
/*
inline uchar* Mat2uchar(const Mat & src)
{
	int i = 0, j = 0;
	int row = src.rows;
	int col = src.cols;

	uchar **dst = (uchar **)malloc(row * sizeof(uchar *));//二维数组dst[][]
	for (i = 0; i < row; i++)
		dst[i] = (uchar *)malloc(col * sizeof(uchar));

	for (i = 0; i < row; i++)
	{
		for (j = 0; j < col; j++)
		{
			dst[i][j] = src.at<uchar>(i, j);//src的矩阵数据传给二维数组dst[][]
		}
	}
	return *dst;
}
*/


bool polynomial_curve_fit(std::vector<cv::Point>& key_point, int n, cv::Mat& A)//https://blog.csdn.net/guduruyu/article/details/72866144
{
	//Number of key points
	int N = key_point.size();
	//构造矩阵X
	cv::Mat X = cv::Mat::zeros(n + 1, n + 1, CV_64FC1);
	for (int i = 0; i < n + 1; i++)
	{
		for (int j = 0; j < n + 1; j++)
		{
			for (int k = 0; k < N; k++)
			{
				X.at<double>(i, j) = X.at<double>(i, j) +
					std::pow(key_point[k].x, i + j);
			}
		}
	}

	//构造矩阵Y
	cv::Mat Y = cv::Mat::zeros(n + 1, 1, CV_64FC1);
	for (int i = 0; i < n + 1; i++)
	{
		for (int k = 0; k < N; k++)
		{
			Y.at<double>(i, 0) = Y.at<double>(i, 0) +
				std::pow(key_point[k].x, i) * key_point[k].y;
		}
	}

	A = cv::Mat::zeros(n + 1, 1, CV_64FC1);
	//求解矩阵A
	cv::solve(X, Y, A, cv::DECOMP_LU);
	return true;
}


void UniRobot::imageProcess()
{
    unsigned char* rgb = getRGBImage(); //get raw data, format: RGB
    Mat rgbMat = getRGBMat(); //get rgb data to cv::Mat
    //240*320, CV_8UC3
	//array: i, j;    CVpoint: j/3, i;    x: nCols, y: nRows;

    if (mode == MODE_BALL) {
        //TODO Write down your code
        //update the resInfo
        resInfo.ball_found = false;
        resInfo.ball_x = 0.0;
        resInfo.ball_y = 0.0;
    } else if (mode == MODE_LINE) {
        //TODO Write down your code

		// Binarization
        Mat binMat = rgbMat.clone();
		uchar *p = binMat.ptr();
		int i, j;
		int nRows = rgbMat.rows;
		int nCols = rgbMat.cols * 3;

		for (i = 0; i < nRows; i ++) {
            for (j = 0; j < nCols; j += 3) {
				if (p[i * nCols + j +2] > 240 && p[i * nCols + j] < 64 && p[i * nCols + j + 1] < 64) {
					resInfo.bluecount++;
					//circle(binMat, Point(j / 3, i), 5, Scalar(0, 0, 255));
				}
                if (p[i * nCols + j] < 200 && p[i * nCols + j + 1] < 200 && p[i * nCols + j + 2] < 200) {  // TODO: modify diametres
                    p[i * nCols + j] = p[i * nCols + j + 1] = p[i * nCols + j + 2] = 0;
                } else {
                    p[i * nCols + j] = p[i * nCols + j + 1] = p[i * nCols + j + 2] = 255;
                }
            }
        }

		if (!resInfo.bluelast && resInfo.bluelastlast && resInfo.bluecount < 200) {
			resInfo.blueline++;
		}
		//cout << resInfo.bluecount << "\tlast:" << resInfo.bluelast << "\tline" << resInfo.blueline <<"\tstep"<< resInfo.stepcount<< endl;

		resInfo.bluelastlast = resInfo.bluelast;
		resInfo.bluelast = resInfo.bluecount > 200;
		resInfo.bluecount = 0;

		morphologyEx(binMat, binMat, MORPH_OPEN, getStructuringElement(0, Size(10, 10), Point(0, 0)));
		/*****************************************************************************************

		// rectify the tilted pic
        // assuming that the angle betwenn the middle of the vision and the plumb line is 60°
		//Mat dstMat = dstMat.clone();
		int nrows = rgbMat.rows;
		int ncols = rgbMat.cols;
		float ratio = cos(60 - 23) / cos(60 + 23);
		vector<Point2f> corners(4);
		corners[0] = Point2f(0, 0);
		corners[1] = Point2f(ncols-1, 0);
		corners[2] = Point2f(ncols * (1 - ratio) / 2, nrows-1);
		corners[3] = Point2f(ncols * (1 + ratio) / 2, nrows-1);
		vector<Point2f> corners_dst(4);
		corners_dst[0] = Point2f(0, 0);
		corners_dst[1] = Point2f(ncols-1, 0);
		corners_dst[2] = Point2f(0, nrows-1);
		corners_dst[3] = Point2f(ncols-1, nrows - 1);

		Mat transform = getPerspectiveTransform(corners, corners_dst);
		warpPerspective(binMat, binMat, transform, binMat.size(), INTER_LINEAR, BORDER_CONSTANT);

		*******************************************************************************************************

		// CannyThreshold
		Mat binGray, edge, dstMat(binMat.size(), binMat.type());
		cvtColor(binMat, binGray, CV_RGB2GRAY);  // try BGR
		blur(binGray, edge, Size(3, 3));
		Canny(edge, edge, 30, 90, 3);
		dstMat = Scalar::all(0);
		binMat.copyTo(dstMat, edge);

		*******************************************************************************************************/

		// get the mid line
		bool leftfound, rightfound;  // black is false, white is true
		cv::Point lastwhite, lastleft = cv::Point(-1, -1);
		vector<cv::Point> left, right, mid;

		for (i = 0; i < nRows; i ++) {
			leftfound = rightfound = false;
			lastwhite = cv::Point(-1, -1);  // TODO: ??

            for (j = 0; j < nCols; j += 3) {
				
				if (binMat.at<uchar>(i, j) && (j == 0 || !binMat.at<uchar>(i, j - 1)) && j < nCols / 2) {  // touch b2w before mid
					lastleft = cv::Point(j / 3, i);
				}
				else if (j > nCols / 2) {
					if (lastleft.x >= 0 && !leftfound) {
						left.push_back(lastleft);
						leftfound = true;
					}
					else if (!leftfound) {
						left.push_back(cv::Point(j / 3, i));
						leftfound = true;
					}
				}

				if (leftfound && !rightfound) {
					if (!binMat.at<uchar>(i, j) || (nCols - j <= 3 && binMat.at<uchar>(i, j))) {
						right.push_back(cv::Point(j / 3 - 1, i));
						rightfound = true;
					}
				}
            }

			if (!rightfound) {  // no white this row
				right.push_back(cv::Point(right[i-1].x, i));  // TODO: change the method of error handling
			}/*
			if (!leftfound) {  // no white this row
				left.push_back(cv::Point(0, i));  // TODO: change the method of error handling
			}*/
        }
		
		for (i = 0; i < right.size() && i < left.size(); i++) {
			mid.push_back(cv::Point((right[i].y + left[i].y) / 2, (right[i].x + left[i].x) / 2));
		}
		for (i = 0; i < left.size(); i++)
		{
			cv::circle(binMat, left[i], 1, cv::Scalar(255, 0, 0));
		}
		for (i = 0; i < right.size(); i++)
		{
			cv::circle(binMat, right[i], 1, cv::Scalar(0, 0, 255));
		}

		/*********************************************************************************************************/
		
		// polyfit 
		cv::Mat A;
		polynomial_curve_fit(mid, 3, A);
		std::vector<cv::Point> points_fitted;
		for (int x = 0; x < 400; x++) {
			double y = A.at<double>(0, 0) + A.at<double>(1, 0) * x + A.at<double>(2, 0)*std::pow(x, 2) + A.at<double>(3, 0)*std::pow(x, 3);
			points_fitted.push_back(cv::Point(x, y));
		}
		
		for (i = 0; i < points_fitted.size(); i++) {
			swap(points_fitted[i].x, points_fitted[i].y);
		}

		cv::polylines(binMat, points_fitted, false, cv::Scalar(0, 255, 255), 1, 8, 0);

		/*******************************************************************************************************/

		// direction control
		resInfo.direction = (nCols / 6.0 - points_fitted[11 * nRows / 12.0].x) / 200.0;
		/*
		if (nCols / 2 > points_fitted[11 * nRows / 12].x) {
			resInfo.direction = 0.03;
		}
		else if (nCols / 2 < points_fitted[11 * nRows / 12].x) {
			resInfo.direction = -0.03;
		}
		else {
			resInfo.direction = 0;
		}
		*/
		/*******************************************************************************************************/
		cv::Point indicator;
		int ar1=1,ar2=2;
		int ac1=1,ac2=8;
		int br1=1,br2=2;
		int bc1=7,bc2=8;
		int cr1=5,cr2=6;
		int cc1=1,cc2=2;
		int dr1=1,dr2=3;
		int dc1=1,dc2=2;
		if (!binMat.at<uchar>(cr1*nRows/cr2,cc1*nCols /cc2/ 3))//C白
		{
			//拟合曲线运行
			indicator.x = cc1*nCols /cc2/ 3;
			indicator.y = cr1*nRows/cr2;
			cv::circle(binMat, indicator, 3, cv::Scalar(255, 0, 0));
		}
		else//C黑
		{
			indicator.x = nCols / 6;
			indicator.y = 5 * nRows / 6;
			cv::circle(binMat, indicator, 3, cv::Scalar(255, 0, 0));
			if(!binMat.at<uchar>(dr1*nRows/dr2,dc1*nCols /dc2/ 3))//D白
			{
				indicator.x = dc1*nCols /dc2/ 3;
				indicator.y = dr1*nRows/dr2;
				cv::circle(binMat, indicator, 3, cv::Scalar(255, 0, 0));
				resInfo.direction = 0;//直行
			}
			else//D黑
			{
				indicator.x = dc1*nCols /dc2/ 3;
				indicator.y = dr1*nRows/dr2;
				cv::circle(binMat, indicator, 3, cv::Scalar(255, 0, 0));
				if(binMat.at<uchar>(ar1 *nCols/ ac1, ar1 *nRows/ ar2/3))//A白
				{
					indicator.x = ar1 *nCols/ ac1;
					indicator.y = ar1 *nRows/ ar2/3;
					cv::circle(binMat, indicator, 3, cv::Scalar(255, 0, 0));
					if (binMat.at<uchar>(br1*nRows / br2, bc1 * nCols / bc2/3))//B黑
					{
					    indicator.x = bc1 * nCols / bc2/3;
					    indicator.y = br1*nRows / br2;
					    cv::circle(binMat, indicator, 3, cv::Scalar(255, 0, 0));
					   // resInfo.direction = 0.6;//左转
	                    //拟合曲线运行
					}
					else//B白
					{
					    indicator.x = bc1 * nCols / bc2/3;
					    indicator.y = br1*nRows / br2;
					    cv::circle(binMat, indicator, 3, cv::Scalar(255, 0, 0));
						//拟合曲线运行
					}
				}
				else //A黑
				{
					indicator.x = ar1 *nCols/ ac1;
					indicator.y = ar1 *nRows/ ar2/3;
					cv::circle(binMat, indicator, 3, cv::Scalar(255, 0, 0));
					if (binMat.at<uchar>(br1*nRows / br2, bc1 * nCols / bc2/3))//B黑
					{
					           indicator.x = bc1 * nCols / bc2/3;
					           indicator.y = br1*nRows / br2;
					           cv::circle(binMat, indicator, 3, cv::Scalar(255, 0, 0));
						//拟合曲线运行
					}
					else//B白
					{
					           indicator.x = bc1 * nCols / bc2/3;
					           indicator.y = br1*nRows / br2;
						cv::circle(binMat, indicator, 3, cv::Scalar(255, 0, 0));//右转
					}
				}
			}
		}
		showImage(binMat.data);
		binMat.release();
		A.release();
        //update the resInfo
    }
    rgbMat.release();
    delete[] rgb;
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
        headPosition = clamp(0.0, minMotorPositions[19], maxMotorPositions[19]); //head pitch position
        mMotors[18]->setPosition(neckPosition);
        mMotors[19]->setPosition(headPosition);
      }
      else if(mode == MODE_LINE) //mode line
      {
		  if (resInfo.blueline > 1) {
			  resInfo.stepcount++;
		}
        //walk control
		  if (resInfo.stepcount < 300) {
			  //cout << resInfo.stepcount << endl;
		   mGaitManager->setXAmplitude(1.0); //x -1.0 ~ 1.0
		   mGaitManager->setYAmplitude(0.0); //y -1.0 ~ 1.0
		   mGaitManager->setAAmplitude(resInfo.direction); //dir -1.0 ~ 1.0
		  }
		  else {
			  mGaitManager->setXAmplitude(0.0); //x -1.0 ~ 1.0
			  mGaitManager->setYAmplitude(0.0); //y -1.0 ~ 1.0
			  mGaitManager->setAAmplitude(0.0); //dir -1.0 ~ 1.0
		  }

        mGaitManager->step(mTimeStep);
        //head control
        neckPosition = clamp(0.0, minMotorPositions[18], maxMotorPositions[18]); //head yaw position
        headPosition = clamp(0.35, minMotorPositions[19], maxMotorPositions[19]); //head pitch position
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
