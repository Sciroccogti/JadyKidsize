/*
 * Description:  Add a new tab at CategoryViewer to interact with Robotis robots
 */

#ifndef VIEWER_HPP
#define VIEWER_HPP

#include <gui/GenericWindow.hpp>

class Viewer : public webotsQtUtils::GenericWindow
{
  public:
             Viewer();
    virtual ~Viewer();
};

#endif
