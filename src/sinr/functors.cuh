#ifndef __FUNCTORS_CUH__
#define __FUNCTORS_CUH__

#include <thrust/functional.h>


namespace sinr {
  namespace functors {

    /** Unary functor to convert linear values to dB scale.
     */
    template <typename T>
      struct Lin2dB : public thrust::unary_function<T,T> {

        __host__ __device__ T operator()(T val) const {
          return T(10.0)*log10(val);
        }
      };


    /** Unary functor to convert dB values to linear scale.
     */
    template <typename T>
      struct dB2Lin : public thrust::unary_function<T,T> {

        __host__ __device__ T operator()(T valdB) const {
          return pow(T(10.0),valdB/T(10.0));
        }
      };


    /** Unary functor to increment by a constant value.
     */
    template <typename T>
      struct Plus : public thrust::unary_function<T,T> {
        T c;

        Plus(T _c) : c(_c) {;}

        __host__ __device__ T operator()(T val) const {
          return val+T(1.0);
        }
      };


    /** Unary functor to multiply by a constant value. */
    template <typename T>
      struct Multiply : public thrust::unary_function<T,T> { 
        T c;

        Multiply(T _c) : c(_c) {;}

        __host__ __device__ T operator()(T val) const {
          return c*val;
        }
      };

  }; /* namespace functors */
}; /* namespace sinr */

#endif /* __FUNCTORS_CUH__ */
