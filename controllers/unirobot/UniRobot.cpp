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
	//array: i, j; CVpoint: j/3, i

    if (mode == MODE_BALL) {
		//TODO Write down your code


		//Mat binMat = rgbMat.clone();

		RobotisOp2VisionManager Vision(320, 240, 120, 15, 100, 10, 50, 100);
		//double ballx = -1, bally = -1;

		//BallTracker tracker;


		//showImage(binMat.data);
		//binMat.release();

		/*Mat greyMat;
		int i, j, k;
		int nRows = rgbMat.rows;
		int nCols = rgbMat.cols * 3;
		uchar  *p = greyMat.ptr();
		uchar  *q = rgbMat.ptr();

		for (i = 0; i < nRows; i++) {
			for (j = 0; j < nCols; j += 3) {
				// TODO: modify diametres
				//p[i * nCols + j] = p[i * nCols + j + 1] = p[i * nCols + j + 2] = q[i * nCols + j]*0.299+q[i * nCols + j + 1]*0.587+ q[i * nCols + j + 1]*0.114;
				//cout << q[i * nCols + j] * 0.299<<"ttt";//p[i * nCols + j] = p[i * nCols + j + 1] = p[i * nCols + j + 2]=100;
			}
		}*/

		/*************************/
		/*Mat binMat = rgbMat.clone();
		uchar  *p = binMat.ptr();
		int i, j, k;
		int nRows = rgbMat.rows;
		int nCols = rgbMat.cols * 3;
		for (i = 0; i < nRows; i++) {
			for (j = 0; j < nCols; j += 3) {
				if (p[i * nCols + j] < 200 && p[i * nCols + j + 1] < 200 && p[i * nCols + j + 2] < 200) {  // TODO: modify diametres
					p[i * nCols + j] = p[i * nCols + j + 1] = p[i * nCols + j + 2] = 0;
				}
				else {
					p[i * nCols + j] = p[i * nCols + j + 1] = p[i * nCols + j + 2] = 255;
				}
			}
		}*/
		/*************************/

		/****************************/
		Mat binMat = rgbMat.clone();
		//morphologyEx(rgbMat, binMat, MORPH_OPEN, getStructuringElement(0, Size(6, 6), Point(0, 0)));//开闭
		uchar  *p = binMat.ptr();
		int i, j;
		int Rows = rgbMat.rows;
		int Cols = rgbMat.cols * 3;

		for (i = 0; i < Rows; i++) {
			for (j = 0; j < Cols; j += 3) {
				if ((p[i * Cols + j] < 250 && p[i * Cols + j + 1] < 250 && p[i * Cols + j + 2] < 250)/*|| i > 60*/) {  // TODO: modify diametres
					p[i * Cols + j] = p[i * Cols + j + 1] = p[i * Cols + j + 2] = 0;
				}
				else {
					p[i * Cols + j] = p[i * Cols + j + 1] = p[i * Cols + j + 2] = 255;
				}
			}
		}//二值化

		//霍夫圆变换
		Mat src_gray, src_rgb, edge, dstMat(rgbMat.size(), rgbMat.type());
		cvtColor(rgbMat, src_gray, CV_RGB2GRAY);
		//GaussianBlur(src_gray, src_gray, Size(9, 9), 2, 2);


		// CannyThreshold
		//cvtColor(binMat, binGray, CV_RGB2GRAY);  // try BGR
		blur(src_gray, edge, Size(3, 3));
		Canny(edge, edge, 30, 90, 3);
		dstMat = Scalar::all(0);
		src_gray.copyTo(dstMat, edge);


		vector<Vec3f> circles;
		HoughCircles(edge, circles, CV_HOUGH_GRADIENT, 1, edge.rows / 3, 100, 35, 2, 58);

		cvtColor(edge, src_rgb, CV_GRAY2RGB);
		//cout << circles.size() << endl;
		int nCols = edge.cols;
		//cout << nCols << endl;
		resInfo.direction = 0.0;
		for (size_t i = 0; i < circles.size(); i++)
		{
			Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
			int radius = cvRound(circles[i][2]);
			//cout << radius <<"\t";

			// circle center
			circle(src_rgb, center, 3, Scalar(0, 255, 0), -1, 8, 0);
			// circle outline
			circle(src_rgb, center, radius, Scalar(0, 0, 255), 3, 8, 0);
		}
		//cout << resInfo.direction <<  endl;

		showImage(src_rgb.data);
		//waitKey(0);

		/****************************/

		//update the resInfo

		if (circles.size() == 1 && circles[0][2] > 1 && circles[0][1] > 1) {
			resInfo.ball_x = cvRound(circles[0][0]);
			resInfo.ball_y = cvRound(circles[0][1]);
			resInfo.direction = -(circles[0][0] - 158.0) / 500.0;
			resInfo.ball_found = true;
		}
		else {
			resInfo.ball_x = -1;
			resInfo.ball_y = -1;
			resInfo.ball_found = false;
		}

		//resInfo.ball_x = ballx;
		//resInfo.ball_y = bally;
		binMat.release();
		src_rgb.release();
		src_gray.release();
		circles.clear();
    } else if (mode == MODE_LINE) {
        //TODO Write down your code
		
		// Binarization
        Mat binMat = rgbMat.clone();
        uchar  *p = binMat.ptr(); 
        int i, j;
        int nRows = rgbMat.rows;
        int nCols = rgbMat.cols * 3;
        
		for (i = 0; i < nRows; i ++) {
            for (j = 0; j < nCols; j += 3) {/*
				if (p[i * nCols + j + 2] > 240 && p[i * nCols + j] < 64 && p[i * nCols + j + 1] < 64) {
					resInfo.bluecount++;
				}*/
                if (p[i * nCols + j] < 200 && p[i * nCols + j + 1] < 200 && p[i * nCols + j + 2] < 200) {  // TODO: modify diametres
                    p[i * nCols + j] = p[i * nCols + j + 1] = p[i * nCols + j + 2] = 0;
                } else {
                    p[i * nCols + j] = p[i * nCols + j + 1] = p[i * nCols + j + 2] = 255;
                }
            }
        }
		/*
		if (!resInfo.bluelast && resInfo.bluelastlast && resInfo.bluecount < 320) {
			resInfo.blueline++;
		}
		//cout << resInfo.bluecount << "\tlast:" << resInfo.bluelast << "\tline" << resInfo.blueline <<"\tstep"<< resInfo.stepcount<< endl;
		resInfo.bluelastlast = resInfo.bluelast;
		resInfo.bluelast = resInfo.bluecount > 320;
		resInfo.bluecount = 0;
		*/
		morphologyEx(binMat, binMat, MORPH_OPEN, getStructuringElement(0, Size(10, 10), Point(0, 0)));
		morphologyEx(binMat, binMat, MORPH_CLOSE, getStructuringElement(0, Size(10, 10), Point(0, 0)));
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

		//*******************************************************************************************************

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
		cv::Point lastwhite;
		vector<cv::Point> left, right, mid;

		for (i = 0; i < nRows; i ++) {
			leftfound = rightfound = false;
			lastwhite = cv::Point(-1, -1);  // TODO: ??

            for (j = 0; j < nCols; j += 3) {
				if (!leftfound) {
					if (binMat.at<uchar>(i, j)) {  // touch white
						left.push_back(cv::Point(j / 3, i));
						leftfound = true;
					}
				}
				
				if (!rightfound) {
					if (binMat.at<uchar>(i, j)) {  // touch white
						lastwhite = cv::Point(j / 3, i);  // store the lastwhite
					}

					if (nCols - j <= 3) {  // reach the end of the row
						if (binMat.at<uchar>(i, j)) {  // end with white
							right.push_back(cv::Point(j / 3, i));
							rightfound = true;
						}
						else if (lastwhite.x >= 0) {
							right.push_back(lastwhite);
							rightfound = true;
						}
					}
				}
            }

			if (!rightfound) {  // no white this row
				right.push_back(cv::Point(nCols / 3, i));  // TODO: change the method of error handling
			}
			if (!leftfound) {  // no white this row
				left.push_back(cv::Point(0, i));  // TODO: change the method of error handling
			}
        }
		
		for (i = 0; i < right.size() && i < left.size(); i++) {
			mid.push_back(cv::Point((right[i].x + left[i].x) / 2, (right[i].y + left[i].y) / 2));
		}
		
		for (i = 0; i < left.size(); i++)
		{
			cv::circle(binMat, left[i], 1, cv::Scalar(0, 255, 0));
		}
		for (i = 0; i < right.size(); i++)
		{
			cv::circle(binMat, right[i], 1, cv::Scalar(0, 255, 0));
		}
		for (i = 0; i < mid.size(); i++)
		{
			cv::circle(binMat, mid[i], 1, cv::Scalar(0, 255, 0));
		}
		
		/*********************************************************************************************************/
		
		resInfo.direction = (nCols / 6 - mid[11 * nRows / 12].x) / 190.0;
		/*******************************************************************************************************/
		cv::Point indicator;
	            int ar1 = 1, ar2 = 2;
		int ac1 = 2, ac2 = 8;
		int br1 = 1, br2 = 2;
		int bc1 = 3, bc2 = 4;
		int cr1 = 3, cr2 = 4;
		int cc1 = 1, cc2 = 2;
		int dr1 = 1, dr2 = 2;
		int dc1 = 1, dc2 = 3;
		int er1 = 1, er2 = 3;
		int ec1 = 1, ec2 = 3;
		int fr1=1,fr2=3;
		int fc1=2,fc2=3;
		int gr1=1,gr2=2;
		int gc1=2,gc2=11;
		int hr1=1,hr2=2;
		int hc1=9,hc2=11;
		int ir1=13,ir2=14;
		int ic1=9,ic2=10;
		int jr1=13,jr2=14;
		int jc1=1,jc2=10;
		int c1=1,c2=2;
		int r11=1,r12=2;
		int r21=1,r22=3;
		int r31=1,r32=4;
		
		if (binMat.at<uchar>(cr1*nRows / cr2, cc1*nCols / cc2 / 3))//C白
		{
			//拟合曲线运行
			/*if (binMat.at<uchar>(dr1*nRows / dr2, dc1*nCols / dc2 / 3))//D白
			{
			    indicator.x = dc1 * nCols / dc2 / 3;
			    indicator.y = dr1 * nRows / dr2;
			    cv::circle(binMat, indicator, 3, cv::Scalar(255, 0, 0));
		               resInfo.direction = 0;//直行
			}
			else
			{
			    indicator.x = dc1 * nCols / dc2 / 3;
			    indicator.y = dr1 * nRows / dr2;
			    cv::circle(binMat, indicator, 3, cv::Scalar(255, 0, 0));
			    if(binMat.at<uchar>(er1*nRows/er2,ec1*nCols/ec2/3))
			    {
			        indicator.x = ec1 * nCols / ec2 / 3;
			        indicator.y = er1 * nRows / er2;
			        cv::circle(binMat, indicator, 3, cv::Scalar(255, 0, 0));
		                    //resInfo.direction = 0.1;//直行
			    }if(binMat.at<uchar>(fr1*nRows/fr2,fc1*nCols/fc2/3))
			    {
			        indicator.x = fc1 * nCols / fc2 / 3;
			        indicator.y = fr1 * nRows / fr2;
			        cv::circle(binMat, indicator, 3, cv::Scalar(255, 0, 0));
		                    //resInfo.direction = -0.1;//直行
			    }
			    
			}*/
			indicator.x = cc1 * nCols / cc2 / 3;
			indicator.y = cr1 * nRows / cr2;
			cv::circle(binMat, indicator, 3, cv::Scalar(255, 0, 0));
		}
		else//C黑
		{
			indicator.x = nCols / 6;
			indicator.y = 5 * nRows / 6;
			cv::circle(binMat, indicator, 3, cv::Scalar(255, 0, 0));
			if (binMat.at<uchar>(dr1*nRows / dr2, dc1*nCols / dc2 / 3))//D白
			{
				indicator.x = dc1 * nCols / dc2 / 3;
				indicator.y = dr1 * nRows / dr2;
				cv::circle(binMat, indicator, 3, cv::Scalar(255, 0, 0));
				//resInfo.direction = 0;//直行
			}
			else//D黑
			{
				indicator.x = dc1 * nCols / dc2 / 3;
				indicator.y = dr1 * nRows / dr2;
				cv::circle(binMat, indicator, 3, cv::Scalar(255, 0, 0));
				if (binMat.at<uchar>(ar1 *nRows / ar2, ac1 *nCols / ac2 / 3))//A白
				{
					indicator.x = ac1 *nCols / ac2 / 3;
					indicator.y = ar1 *nRows / ar2;
					cv::circle(binMat, indicator, 3, cv::Scalar(255, 0, 0));
					if (!binMat.at<uchar>(br1*nRows / br2, bc1 * nCols / bc2 / 3))//B黑
					{
						indicator.x = bc1 * nCols / bc2 / 3;
						indicator.y = br1 * nRows / br2;
						cv::circle(binMat, indicator, 3, cv::Scalar(255, 0, 0));
						resInfo.direction = 0.1;//左转
						 //拟合曲线运行
					}
					else//B白
					{
						indicator.x = bc1 * nCols / bc2 / 3;
						indicator.y = br1 * nRows / br2;
						cv::circle(binMat, indicator, 3, cv::Scalar(255, 0, 0));
						//拟合曲线运行
					}
				}
				else //A黑
				{
					indicator.x = ac1 *nCols / ac2 / 3;
					indicator.y = ar1 *nRows / ar2;
					cv::circle(binMat, indicator, 3, cv::Scalar(255, 0, 0));
					if (!binMat.at<uchar>(br1*nRows / br2, bc1 * nCols / bc2 / 3))//B黑
					{
						indicator.x = bc1 * nCols / bc2 / 3;
						indicator.y = br1 * nRows / br2;
						cv::circle(binMat, indicator, 3, cv::Scalar(255, 0, 0));
						//拟合曲线运行
					}
					else//B白
					{
						indicator.x = bc1 * nCols / bc2 / 3;
						indicator.y = br1 * nRows / br2;
						cv::circle(binMat, indicator, 3, cv::Scalar(255, 0, 0));
						//resInfo.direction = -0.1;//右转
					}
				}
			}
		}
		if(binMat.at<uchar>(gr1*nRows / gr2,gc1*nCols / gc2 / 3)&&!binMat.at<uchar>(hr1*nRows / hr2,hc1*nCols / hc2 / 3)&&!binMat.at<uchar>(ir1*nRows / ir2,ic1*nCols / ic2 / 3)&&!binMat.at<uchar>(jr1*nRows / jr2,jc1*nCols / jc2 / 3))
		{
		    //indicator.x = ac1 *nCols / ac2 / 3;
		    //indicator.y = ar1 *nRows / ar2;
		    cv::circle(binMat, indicator, 3, cv::Scalar(255, 0, 0));
		    resInfo.direction=-0.025;
		}
		if(!binMat.at<uchar>(gr1*nRows / gr2,gc1*nCols / gc2 / 3)&&binMat.at<uchar>(hr1*nRows / hr2,hc1*nCols / hc2 / 3)&&!binMat.at<uchar>(ir1*nRows / ir2,ic1*nCols / ic2 / 3)&&!binMat.at<uchar>(jr1*nRows / jr2,jc1*nCols / jc2 / 3))
		{
		    //indicator.x = ac1 *nCols / ac2 / 3;
		    //indicator.y = ar1 *nRows / ar2;
		    cv::circle(binMat, indicator, 3, cv::Scalar(255, 0, 0));
		    resInfo.direction=0.025;
		}
		if(binMat.at<uchar>(jr1*nRows / jr2,jc1*nCols / jc2 / 3)&&!binMat.at<uchar>(ir1*nRows / ir2,ic1*nCols / ic2 / 3))
		{
		    indicator.x = jc1 *nCols / jc2 / 3;
		    indicator.y = jr1 *nRows / jr2;
		    cv::circle(binMat, indicator, 3, cv::Scalar(255, 0, 0));
		    indicator.x = ic1 *nCols / ic2 / 3;
		    indicator.y = ir1 *nRows / ir2;
		    cv::circle(binMat, indicator, 3, cv::Scalar(0, 255, 0));
		    resInfo.direction=1;
		}
		else if(!binMat.at<uchar>(jr1*nRows / jr2,jc1*nCols / jc2 / 3)&&binMat.at<uchar>(ir1*nRows / ir2,ic1*nCols / ic2 / 3))
		{
		    indicator.x = jc1 *nCols / jc2 / 3;
		    indicator.y = jr1 *nRows / jr2;
		    cv::circle(binMat, indicator, 3, cv::Scalar(0, 255, 0));
		    indicator.x = ic1 *nCols / ic2 / 3;
		    indicator.y = ir1 *nRows / ir2;
		    cv::circle(binMat, indicator, 3, cv::Scalar(255, 0, 0));
		    resInfo.direction=-1;
		}
		if (binMat.at<uchar>(c1*nRows/c2,r11*nCols/r12/3)||binMat.at<uchar>(c1*nRows/c2,r21*nCols/r22/3)||binMat.at<uchar>(c1*nRows/c2,r31*nCols/r32/3))
		{
		    resInfo.direction = (nCols / 6 - mid[11 * nRows / 12].x) / 190.0;
		}
		
		showImage(binMat.data);
		binMat.release();
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
		  if (resInfo.ball_found)
		  {
			  cout << "Found!\t" << resInfo.ball_y << "\t" << 0.235 - (resInfo.ball_y - 50) / 800.0 << endl;
			  resInfo.stepcount = resInfo.stepcount * 11 / 12;
			  if (resInfo.ball_y >= 203)
			  {
				  cout << "kick!" << resInfo.ball_y << "\t" << resInfo.ball_x << endl;
				  mGaitManager->stop();
				  wait(500);
				  //mGaitManager->setMoveAimOn(1);
				  if (resInfo.ball_x < 160)
					  mMotionManager->playPage(13); // left kick
				  else
					  mMotionManager->playPage(12); // right kick
				  mMotionManager->playPage(9); // walkready position
				  mGaitManager->start();
			  }
			  /*
			  if (abs(resInfo.direction) < 0.05 && abs(resInfo.direction) > 0.00001 && resInfo.ball_y < 201) {
				  mGaitManager->setXAmplitude(0.85); //x -1.0 ~ 1.0
			  }
			  else {*/
			  mGaitManager->setXAmplitude(1.0);
			  //}

			  headPosition = clamp(0.18 - (resInfo.ball_y - 50) / 1000.0, minMotorPositions[19], maxMotorPositions[19]); //head pitch position
		  /*
		  if (resInfo.ball_y < 100 && resInfo .ball_y > 50) {
			  headPosition = clamp(0.2, minMotorPositions[19], maxMotorPositions[19]); //head pitch position
		  }
		  else if(resInfo.ball_y > 200){
			  headPosition = clamp(0.1, minMotorPositions[19], maxMotorPositions[19]); //head pitch position
		  }
		  else {
			  headPosition = clamp(0.0, minMotorPositions[19], maxMotorPositions[19]); //head pitch position
		  }
		  */
			  mGaitManager->setYAmplitude(0.0); //y -1.0 ~ 1.0
			  mGaitManager->setAAmplitude(resInfo.direction); //dir -1.0 ~ 1.0

			  //headPosition = clamp(resInfo.stepcount / 5000.0, minMotorPositions[19], maxMotorPositions[19]); //head pitch position
		  }
		  else {
			  resInfo.stepcount++;
			  //cout << resInfo.stepcount<<"\t"<< 400.0 / resInfo.stepcount << endl;
			  mGaitManager->setXAmplitude(1.0);
			  mGaitManager->setYAmplitude(0.0);
			  mGaitManager->setAAmplitude(400.0 / resInfo.stepcount);

			  headPosition = clamp(0.35, minMotorPositions[19], maxMotorPositions[19]); //head pitch position
		  }

		  mGaitManager->step(mTimeStep);
		  //head control
		  neckPosition = clamp(0.0, minMotorPositions[18], maxMotorPositions[18]); //head yaw position
		  //headPosition = clamp(0.0, minMotorPositions[19], maxMotorPositions[19]); //head pitch position
		  mMotors[19]->setPosition(headPosition);
      }
      else if(mode == MODE_LINE) //mode line
      {/*
		  if (resInfo.blueline > 1) {
			  resInfo.stepcount++;
		  }*/
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
        headPosition = clamp(0.38, minMotorPositions[19], maxMotorPositions[19]); //head pitch position
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