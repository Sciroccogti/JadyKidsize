#include <RobotisOp2MotionTimerManager.hpp>

#include <webots/Robot.hpp>

#include <LinuxMotionTimer.h>
#include <MotionManager.h>

using namespace Robot;
using namespace managers;
using namespace webots;
using namespace std;

RobotisOp2MotionTimerManager::RobotisOp2MotionTimerManager() {
}

void RobotisOp2MotionTimerManager::MotionTimerInit() {
  if (!mStarted) {
    LinuxMotionTimer *motion_timer = new LinuxMotionTimer(MotionManager::GetInstance());
    motion_timer->Start();
    mStarted = true;
  }
}

RobotisOp2MotionTimerManager::~RobotisOp2MotionTimerManager() {
}

bool RobotisOp2MotionTimerManager::mStarted = false;
