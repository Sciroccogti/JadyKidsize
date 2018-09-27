// Description:   Facade between webots and the robotis-op2 framework
//                allowing to used the main image processing tools

#include <ImgProcess.h>
#include <ColorFinder.h>
#include <Image.h>
#include <Point.h>

#ifndef ROBOTISOP2_VISION_MANAGER_HPP
#define ROBOTISOP2_VISION_MANAGER_HPP

namespace Robot {
  class ColorFinder;
  class FrameBuffer;
}

namespace managers {
  using namespace Robot;
  class RobotisOp2VisionManager {
    public:
                   RobotisOp2VisionManager(int width, int height, int hue, int hueTolerance, int minSaturation, int minValue, int minPercent, int maxPercent);
      virtual     ~RobotisOp2VisionManager();

      bool         getBallCenter(double &x, double &y, const unsigned char *image);
      bool         isDetected(int x, int y);
      void         setHue(int hue)                     { mFinder->m_hue = hue; }
      void         setHueTolerance(int hueTolerance)   { mFinder->m_hue_tolerance = hueTolerance; }
      void         setMinSaturation(int minSaturation) { mFinder->m_min_saturation = minSaturation; }
      void         setMinValue(int minValue)           { mFinder->m_min_value = minValue; }
      void         setMinPercent(int minPercent)       { mFinder->m_min_percent = minPercent; }
      void         setmaxPercent(int maxPercent)       { mFinder->m_max_percent = maxPercent; }

    private:
      ColorFinder *mFinder;
      FrameBuffer *mBuffer;

  };
}

#endif
