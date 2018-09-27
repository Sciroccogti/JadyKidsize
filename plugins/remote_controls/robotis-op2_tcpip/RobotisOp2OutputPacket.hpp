/*
 * Description:  Defines a packet sending from the remote control library to the ROBOTIS OP2
 */

#ifndef ROBOTISOP2_OUTPUT_PACKET_HPP
#define ROBOTISOP2_OUTPUT_PACKET_HPP

#include "Packet.hpp"

class Device;

class RobotisOp2OutputPacket : public Packet {
  public:
                 RobotisOp2OutputPacket();
    virtual     ~RobotisOp2OutputPacket();
    virtual void clear();
    void         apply(int simulationTime);
    bool         isAccelerometerRequested() const { return mAccelerometerRequested; }
    bool         isGyroRequested() const { return mGyroRequested; }
    bool         isCameraRequested() const { return mCameraRequested; }
    bool         isPositionSensorRequested(int at) const { return mPositionSensorRequested[at]; }
    bool         isMotorForceFeedback(int at) const { return mMotorTorqueFeedback[at]; }

  private:
    bool         mAccelerometerRequested;
    bool         mGyroRequested;
    bool         mCameraRequested;
    bool         mPositionSensorRequested[20];
    bool         mMotorTorqueFeedback[20];
};

#endif
