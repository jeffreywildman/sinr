#ifndef __SPATIALDENSITY_CUH__
#define __SPATIALDENSITY_CUH__

#include <thrust/functional.h>    /* for thrust::unary_function<> */
#include <thrust/tuple.h>         /* for thrust::tuple<> */

#include <sinr/arena.h>           /* for Arena2d<> */


/** Unary functor to compute the uniform density function within a given arena.
 */
template <typename T>
struct DensityUniform2d : public thrust::unary_function<T, thrust::tuple<T,T> > {
public:
  typedef typename thrust::tuple<T,T> Point2d; 
  T xmin;
  T xmax;
  T ymin;
  T ymax;
  T prob;

  /** Construct the uniform density functor.
   *
   * @param xmin 
   * @param xmax 
   * @param ymin 
   * @param ymax 
   *
   * @todo enable assertions on input parameters
   */
  __host__ __device__ DensityUniform2d(T _xmin, T _xmax, T _ymin, T _ymax) :
    xmin(_xmin), xmax(_xmax), ymin(_ymin), ymax(_ymax), prob(T(1.0)/((_xmax-_xmin)*(_ymax - _ymin))) {
      //assert(xmin <= xmax && ymin <= ymax && a >= 0.0);
      ;
    }

  DensityUniform2d(Arena2d<T> arena) {
    arena.getBounds(xmin, xmax, ymin, ymax);
    prob = T(1.0)/((xmax-xmin)*(ymax - ymin)); 
  }

  __host__ __device__ DensityUniform2d(const DensityUniform2d &d) {
    xmin = d.xmin;
    xmax = d.xmax;
    ymin = d.ymin;
    ymax = d.ymax;
    prob = d.prob;
  }

  __host__ __device__ ~DensityUniform2d() {;}

  __host__ __device__ T operator()(Point2d p) {
    using thrust::get;
    return (get<0>(p) >= xmin && 
            get<0>(p) <= xmax && 
            get<1>(p) >= ymin && 
            get<1>(p) <= ymax) 
      ? prob : T(0.0);
  }
};


/** Unary functor to compute a density function centered within an arena.
 */
template <typename T>
struct DensityCentered2d : public thrust::unary_function<T, thrust::tuple<T,T> > {
public:
  typedef typename thrust::tuple<T,T> Point2d; 
  T xmin;
  T xmax;
  T ymin;
  T ymax;
  T xlen;
  T ylen;
  T constant;
  T a;

  /** Construct the centered density functor.
   *
   * @param xmin 
   * @param xmax 
   * @param ymin 
   * @param ymax 
   * @param a     Parameter controlling the concentration of the density function about the center of the arena, with
   * valid range [0,Inf).  a == 0 recovers the uniform distribution.  
   *
   * @todo enable assertions on input parameters
   */
  __host__ __device__ DensityCentered2d(T _xmin, T _xmax, T _ymin, T _ymax, T _a = T(1.0)) : xmin(_xmin), xmax(_xmax), 
  ymin(_ymin), ymax(_ymax), xlen(_xmax-_xmin), ylen(_ymax-_ymin), a(_a) {
    //assert(xmin <= xmax && ymin <= ymax && a >= 0.0);
    constant = T(1.0)/((xmax-xmin)*(ymax-ymin)*pow(tgamma(a + T(1.0)),T(4.0))/pow(tgamma(T(2.0)*a + T(2.0)),T(2.0)));
  }

  DensityCentered2d(Arena2d<T> arena, T _a = T(1.0)) : a(_a) {
    arena.getBounds(xmin, xmax, ymin, ymax);
    xlen = xmax - xmin;
    ylen = ymax - ymin;
    constant = T(1.0)/((xmax-xmin)*(ymax-ymin)*pow(tgamma(a + T(1.0)),T(4.0))/pow(tgamma(T(2.0)*a + T(2.0)),T(2.0)));
  }

  __host__ __device__ DensityCentered2d(const DensityCentered2d &d) {
    xmin      = d.xmin;
    xmax      = d.xmax;
    ymin      = d.ymin;
    ymax      = d.ymax;
    xlen      = d.xlen;
    ylen      = d.ylen;
    constant  = d.constant;
    a         = d.a;
  }

  __host__ __device__ ~DensityCentered2d() {;}

  __host__ __device__ T operator()(Point2d p) {
    using thrust::get;
    if (get<0>(p) >= xmin && 
        get<0>(p) <= xmax && 
        get<1>(p) >= ymin && 
        get<1>(p) <= ymax) {
      T xhat = (get<0>(p) - xmin)/(xmax-xmin);
      T yhat = (get<1>(p) - ymin)/(ymax-ymin);
      return constant*pow(xhat*yhat*(T(1.0)-xhat)*(T(1.0)-yhat),a);
    } else { 
      return T(0.0);
    }
  }
};

#endif /* __SPATIALDENSITY_CUH__ */
