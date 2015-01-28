#ifndef __ARENA_H__
#define __ARENA_H__

#include <cassert>

/** The Arena2d class is a data structure to store information about the network arena.
 */
template <typename T>
class Arena2d {
public:

  /** Construct a two-dimensional arena with rectangular bounds.
   *
   * @param xmin x-coordinate of the bottom-left corner of the rectangular bound (meters).
   * @param xmax x-coordinate of the top-right corner of the rectangular bound (meters). 
   * @param ymin y-coordinate of the bottom-left corner of the rectangular bound (meters).
   * @param ymax y-coordinate of the top-right corner of the rectangular bound (meters).
   */
  Arena2d(T xmin, T xmax, T ymin, T ymax) {
    this->setBounds(xmin, xmax, ymin, ymax);
  }
  Arena2d(T xlen, T ylen) {
    this->setBounds(0.0, xlen, 0.0, ylen);
  }
  Arena2d(T xlen) {
    this->setBounds(0.0, xlen, 0.0, xlen);
  }
  ~Arena2d() {;}

  /* Accessors */
  void getBounds(T &_xmin, T &_xmax, T &_ymin, T &_ymax) const {
    _xmin = xmin; _xmax = xmax; _ymin = ymin; _ymax = ymax;
  }
  unsigned int getVolume() const {
    return (xmax-xmin)*(ymax-ymin);
  }

  /* Mutators */
  void setBounds(T _xmin, T _xmax, T _ymin, T _ymax) {
    assert(_xmin <= _xmax);
    assert(_ymin <= _ymax);
    xmin = _xmin; xmax = _xmax; ymin = _ymin; ymax = _ymax;
  }

private:

  T xmin;                                /* X-coordinate of the bottom-left corner of the rectangular bound (meters). */
  T xmax;                                /* X-coordinate of the top-right corner of the rectangular bound (meters). */
  T ymin;                                /* Y-coordinate of the bottom-left corner of the rectangular bound (meters). */
  T ymax;                                /* Y-coordinate of the top-right corner of the rectangular bound (meters). */ 
};

#endif /* __ARENA_H__ */
