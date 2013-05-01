#ifndef __TYPES_H__
#define __TYPES_H__

//#include <iostream>
//#include <sstream>

//#include <thrust/tuple.h>
//#include <cuda_runtime_api.h>     /* for __host__, __device__ */

//template <typename T>
//class Point2d : public thrust::tuple<T,T> {
//
//  __host__ __device__ Point2d() : thrust::tuple<T,T>(0.0,0.0) {;}
//  __host__ __device__ Point2d(T x, T y) : thrust::tuple<T,T>(x,y) {;}
//
//  friend std::ostream& operator<<(std::ostream &out, const Point2d<T> &p) {
//    std::ostringstream output;
//    output<<"("<<thrust::get<0>(p)<<", "<<thrust::get<1>(p)<<")";
//    return out<<output.str();
//  }
//};

//template <typename T>
//struct Point2d {
//  T x, y;
//
//  __host__ __device__ Point2d() : x(0.0), y(0.0) {;}
//  __host__ __device__ Point2d(T _x, T _y): x(_x), y(_y) {;}
//
//  friend std::ostream& operator<<(std::ostream &out, const Point2d<T> &p) {
//    std::ostringstream output;
//    output<<"("<<p.x<<", "<<p.y<<")";
//    return out<<output.str();
//  }
//};


typedef int nid_t;   /**< Node ID type. */
typedef int cid_t;   /**< Channel ID type. */


typedef enum assocZoneRule {
  AZR_SINR = 0,
  AZR_POWER,
  AZR_SINR_BIASED,
  AZR_POWER_BIASED,
} assocZoneRule_t;


typedef enum radPatternType {
  RPT_ROSEPETAL = 0,
  RPT_ELLIPSE = 1,
} radPatternType_t;


typedef enum channelType {
  CT_TIMESLOT = 0,
  CT_FREQBAND = 1,
} channelType_t;


template <typename T>
struct RadPattern {
  radPatternType_t type;  /* radiation patterns */
  T p1;                   /* rad pattern parameter 1 */
  T p2;                   /* rad pattern parameter 2 */ 

  RadPattern() : type(RPT_ELLIPSE), p1(0.0), p2(1.0) {;}
  RadPattern(radPatternType_t _type, T _p1, T _p2) : type(_type), p1(_p1), p2(_p2) {;}

  /** @todo Add operator<<() */
};


#endif /* __TYPES_H__ */
