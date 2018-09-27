/*
 * Description:  Defines a packet sending from the real ROBOTIS OP2 to the remote control library
 */

#ifndef ROBOTISOP2_INPUT_PACKET_HPP
#define ROBOTISOP2_INPUT_PACKET_HPP

#include "Packet.hpp"

class RobotisOp2OutputPacket;

class RobotisOp2InputPacket : public Packet {
  public:
                   RobotisOp2InputPacket();
    virtual       ~RobotisOp2InputPacket();

    void           decode(int simulationTime, const RobotisOp2OutputPacket &outputPacket);

  private:
    unsigned char *readJpegImage(const unsigned char *data, unsigned int length);
    int            mCameraWidth;
    int            mCameraHeight;
};

#endif
