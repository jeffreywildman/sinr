#ifndef __RADIATIONPATTERN_CUH__
#define __RADIATIONPATTERN_CUH__

#include <cassert>

#include <thrust/functional.h>                    
#include <thrust/iterator/transform_iterator.h>   
#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_vector.h>

// assert() is only supported for devices of compute capability 2.0 and higher
//#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
//#undef assert
//#define assert(arg)
//#endif


/** Unary functor to compute discrete interpolation points.
 *
 * @todo Find a good home for this.
 */
template <typename T>
struct Interpolate : public thrust::unary_function<T,T> {
  T a;
  T b;
  T N;

  /** Construct the discrete interpolation functor.
   *
   * @param a Start of the range to interpolate.
   * @param b End of the range to interpolate.
   * @param N Number of discrete interpolation points.
   */
  __host__ __device__ Interpolate(T _a, T _b, T _N) : a(_a), b(_b), N(_N) {;}

  __host__ __device__ T operator()(T n) {
    return a + (b-a)*n/N;
  }
};


/** Compute the total radiated power (TRP) of a radiation pattern.
 *
 * @param f Unary functor representing the radiation pattern.
 * @param N Number of sample points to use in computing TRP.
 */
template <typename T, typename UnaryFunction>
T totalRadiatedPower(UnaryFunction f, T N = 10000) {
  T theta_a = 0.0;
  T theta_b = 2.0*M_PI;

  return thrust::transform_reduce(thrust::make_transform_iterator(thrust::counting_iterator<T>(0), Interpolate<T>(theta_a, theta_b, N)),
                                  thrust::make_transform_iterator(thrust::counting_iterator<T>(N), Interpolate<T>(theta_a, theta_b, N)),
                                  f,
                                  0.0,
                                  thrust::plus<T>())*((theta_b-theta_a)/N);
}


/** Unary functor to compute an elliptical radiation pattern.
 */
template <typename T>
struct RadPatternEllipse : public thrust::unary_function<T,T> {
  T a, b;
  T constant;

  /** Construct an elliptical radiation pattern functor.
   *
   * @param ecc   Parameter controlling the eccentricity of the ellipse, with valid range [0,1].  ecc == 0 is the
   * standard circle.
   * @param norm  Normalization factor.  This can be determined from ecc via the static normalize function.
   *
   * @todo enable assertions on input parameters
   */
  __host__ __device__ RadPatternEllipse(T ecc, T norm = T(1.0)) {
    //assert(0.0 <= ecc && ecc <= 1.0 && norm > 0.0);

    a         = T(1.0)/sqrt(M_PI*sqrt(T(1.0)-pow(ecc, T(2.0))));
    b         = T(1.0)/(M_PI*a);
    constant  = a*b*norm;
  }

  /** Compute the normalization factor for a given eccentricity.  
   *
   * @param ecc The eccentricity used to compute the normalization factor.
   * @param N   Number of sample points used to determine the normalization factor.
   *
   * @return The normalization factor.
   */
  static T normalize(T ecc, T N = 10000) {
    assert(0.0 <= ecc && ecc <= 1.0 && N > 0);
    
    T theta_a = 0.0;
    T theta_b = 2.0*M_PI;

    thrust::device_vector<T> rad(N);
    thrust::transform(thrust::make_transform_iterator(thrust::counting_iterator<T>(0), Interpolate<T>(theta_a, theta_b, N)),
                      thrust::make_transform_iterator(thrust::counting_iterator<T>(N), Interpolate<T>(theta_a, theta_b, N)),
                      rad.begin(),
                      RadPatternEllipse<T>(ecc));
    return T(1.0)/(thrust::reduce(rad.begin(),rad.end(),T(0.0))*((theta_b-theta_a)/N));
  }

  __host__ __device__ RadPatternEllipse(const RadPatternEllipse &ant) {
    a         = ant.a;
    b         = ant.b;
    constant  = ant.constant;
  }

  __host__ __device__ ~RadPatternEllipse() {;}

  __host__ __device__ T operator()(T angle) {
    return constant/sqrt(pow(b*cos(angle), T(2.0))+pow(a*sin(angle), T(2.0)));   
  }
};


/** Unary functor to compute a rose petal radiation pattern.
 */
template <typename T>
struct RadPatternRosePetal : public thrust::unary_function<T,T> {
  T width, power;
  T constant;

  /** Construct a rose petal radiation pattern functor.
   *
   * @param width Parameter controlling the first-null beamwidth of the pattern, with valid range [0,2*Pi]. 
   * @param power Parameter controlling the sharpness of the rose petal edge, with valid range, [0,Inf).  power == 1 is 
   * the standard rose petal curve.  power >= 1 tends toward a highly directed beam.  power <= tends toward a standard 
   * sector of a circle.
   *
   * @todo enable assertion on input parameters
   */
  __host__ __device__ RadPatternRosePetal(T _width, T _power = T(1.0)) : width(_width), power(_power) {
    //assert(0.0 <= width && width <= 2.0*M_PI && power > 0.0);

    constant = sqrt(M_PI)/width*tgamma(1.0+0.5*power)/tgamma(0.5*(1.0+power));
  }

  __host__ __device__ RadPatternRosePetal(const RadPatternRosePetal &ant) {
    width = ant.width;
    power = ant.power;
    constant = ant.constant;
  }

  __host__ __device__ ~RadPatternRosePetal() {;}

  __host__ __device__ T operator()(T angle) {
    T angle2 = remainder(T(angle + M_PI), T(2.0*M_PI));
    angle2 < T(0.0) ? angle2 += M_PI : angle2 -= M_PI;               /* wrap to [-M_PI,M_PI] */
    //return (abs(angle2) < width/T(2.0)) ? constant*pow(T(cos(M_PI*angle2/width)),T(1.0/power)) : T(0.0);
    return (abs(angle2) < width/T(2.0)) ? constant*pow(T(cos(M_PI*angle2/width)),T(power)) : T(0.0);
  }
};

#endif /* __RADIATIONPATTERN_CUH__ */
