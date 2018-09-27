#include "Packet.hpp"

#include <iostream>
#include <cassert>
#include <cstring>
#include <sys/types.h>
#ifdef _WIN32
#include <winsock.h>
#else
 #include <sys/socket.h>
#endif

using namespace std;

Packet::Packet(int maxSize) :
  mMaxSize(maxSize), mSize(0), mIndex(0)
{
  mData = new unsigned char[mMaxSize];
}

Packet::~Packet() {
  delete [] mData;
}

void Packet::append(const unsigned char *data, int size) {
  assert(mIndex + size <= mMaxSize);
  memcpy(mData + mIndex, data, size);
  mIndex += size;
  mSize += size;
}

void Packet::appendInt(int value) {
  unsigned char array[4];
  array[0] = (value >> 24) & 0xff;
  array[1] = (value >> 16) & 0xff;
  array[2] = (value >> 8) & 0xff;
  array[3] = value & 0xff;
  append(array, 4);
}

int Packet::readIntAt(int pos) const {
  assert(pos + 3 < mSize);
  unsigned char c1 = mData[pos + 3];
  unsigned char c2 = mData[pos + 2];
  unsigned char c3 = mData[pos + 1];
  unsigned char c4 = mData[pos];
  int r = c1 + (c2 << 8) + (c3 << 16) + (c4 << 24);
  return r;
}

const unsigned char *Packet::getBufferFromPos(int pos) const {
  assert(pos < mSize);
  return &mData[pos];
}

bool Packet::readFromSocket(int socket, int n) {
  n += mIndex;
  assert(n <= mMaxSize);
  do {
    int r = recv(socket, (char *)&mData[mIndex], n - mIndex, 0);
    if (r == -1)
      return false;
    mIndex += r;
  } while (mIndex < n);
  mSize += n;
  return true;
}
