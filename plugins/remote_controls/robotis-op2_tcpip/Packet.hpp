/*
 * Description:   Defines a packet
 */

#ifndef PACKET_HPP
#define PACKET_HPP

class Packet {
  public:
                         Packet(int maxSize);
    virtual             ~Packet();

    const unsigned char *data() const { return mData; }
    virtual void         clear() { mIndex = 0; }
    int                  size() const { return mSize; }
    int                  maxSize() const { return mMaxSize; }
    void                 append(const unsigned char *data, int size);
    void                 append(const char *data, int size) { append((const unsigned char *)data, size); }
    void                 appendInt(int value);
    int                  readIntAt(int pos) const;
    const unsigned char *getBufferFromPos(int pos) const;
    bool                 readFromSocket(int socket, int n);
  protected:
    int                  mMaxSize;
    int                  mSize;
    unsigned char       *mData;
    int                  mIndex;
};

#endif
