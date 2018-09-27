/*
 * Description:  Implement communication throught tcpip with the ROBOTIS OP2
 */

#ifndef COMMUNICATION_HPP
#define COMMUNICATION_HPP

#include <webots/types.h>

class Packet;

class Communication {
  public:
                Communication();
    virtual    ~Communication();

    bool        initialize(const char *ip, int port);
    void        close();

    bool        isInitialized() const { return mSocket != -1; }

    bool        sendPacket(const Packet *packet);
    bool        receivePacket(Packet *packet);

  private:
    int         mSocket;
};

#endif
