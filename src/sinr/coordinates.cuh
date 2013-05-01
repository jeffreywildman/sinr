#ifndef __COORDINATES_CUH__
#define __COORDINATES_CUH__

#include <cassert>

//#include <iostream>     /* for ostream */

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/random/uniform_real_distribution.h>
#include <thrust/tuple.h>

#include <sinr/arena.h>

namespace sinr {

  namespace coordinates {

    /** Unary functor to compute a grid of coordinates.
     *
     * We perform some extra indexing work to reconcile the fact that elements are traditionally indexed in a 2D
     * array/matrix by row first then column with the fact that Cartesian coordinates are indexed by x-coordinate first then
     * y-coordinate.  
     */
    template <typename T>
      struct ComputeCoordGrid2d : public thrust::unary_function<unsigned int, thrust::tuple<T,T> > {
        typedef typename thrust::tuple<T,T> Point2d; 
        T _xmin;               
        T _xmax;               
        T _xlen;               
        T _ymin;               
        T _ymax;               
        T _ylen;               
        unsigned int _C;
        unsigned int _R;

        /** 
         * @param xmin x-coordinate of bottom-left corner of arena (meters)
         * @param xmax x-coordinate of top-right corner of arena (meters)
         * @param ymin y-coordinate of bottom-left corner (meters) 
         * @param ymax y-coordinate of top-right corner of arena (meters) 
         * @param w width of arena (meters) 
         * @param h width of arena (meters) 
         */
        ComputeCoordGrid2d(T xmin, T xmax, T ymin, T ymax, unsigned int w, unsigned int h) : 
          _xmin(xmin), _xmax(xmax), _xlen(xmax-xmin), _ymin(ymin), _ymax(ymax), _ylen(ymax-ymin), _C(w), _R(h) {;}

        /** Generate the ith grid coordinate in a pseudo-twodimensional array of coordinates.
         *
         * The coordinate is generated in the center of the ith grid rectangle.
         *
         * @return 
         */
        __host__ __device__ Point2d operator()(unsigned int index) {
          /* array element indexing (row-column) */
          unsigned int r = index/_C;
          unsigned int c = index - r*_C;
          //return Point2d(_xmin + _xlen*c/T(_C),_ymin + _ylen*r/T(_R));
          /* middle pixel sampling */
          return Point2d(_xmin + _xlen*(c+T(0.5))/T(_C), _ymin + _ylen*(r+T(0.5))/T(_R));
        }
      };


    /** Generator to create uniformly random coordinates.
     */
    template <typename T>
      struct ComputeCoordRandom2d : public thrust::unary_function<unsigned int, thrust::tuple<T,T> > {
        typedef typename thrust::tuple<T,T> Point2d; 
        thrust::random::uniform_real_distribution<T> xdist;
        thrust::random::uniform_real_distribution<T> ydist;
        unsigned int seed;

        /** See Thrust Monte Carlo example: https://github.com/thrust/thrust/blob/master/examples/monte_carlo.cu.
         */
        __host__ __device__
          unsigned int hash(unsigned int a) {
            a = (a+0x7ed55d16) + (a<<12);
            a = (a^0xc761c23c) ^ (a>>19);
            a = (a+0x165667b1) + (a<<5);
            a = (a+0xd3a2646c) ^ (a<<9);
            a = (a+0xfd7046c5) + (a<<3);
            a = (a^0xb55a4f09) ^ (a>>16);
            return a;
          }

        /** 
         * @param xmin x-coordinate of bottom-left corner of arena (meters)
         * @param xmax x-coordinate of top-right corner of arena (meters)
         * @param ymin y-coordinate of bottom-left corner (meters) 
         * @param ymax y-coordinate of top-right corner of arena (meters) 
         * @param seed Seed for the random number generator.
         */
        ComputeCoordRandom2d(T xmin, T xmax, T ymin, T ymax, unsigned int _seed) : 
          xdist(xmin,xmax), ydist(ymin,ymax), seed(_seed) {;}

        __host__ __device__ Point2d operator()(unsigned int thread_id) {
          /* @note must compute seed inside thread, or else all threads will generate the same RVs! */
          /* @note we do not like seed == 0 */
          thrust::default_random_engine rng(hash(thread_id)*(seed+1));
          /** @note We garauntee order of generating x and y.  Using Point2d<T>(xdist(rng), ydist(rng)) actually generates
           * coordinates with reversed x,y values on device vs. host. 
           * See: http://stackoverflow.com/questions/2934904/order-of-evaluation-in-c-function-parameters 
           */
          T x = xdist(rng);
          T y = ydist(rng);
          return Point2d(x, y);
        }
      };


    /** @p generate applies a unary function to a set of indices and stores the result in the corresponding
     * position in an output sequence.
     *
     * @param f
     * @param first
     * @param size
     * @return The end of the output sequence.
     *
     * @tparam UnaryF
     * @tparam 
     */
    template <typename UnaryFunction, typename OutputIterator>
      OutputIterator generate(UnaryFunction f, OutputIterator first, OutputIterator last) {
        typedef thrust::counting_iterator<unsigned int> CUIntIterator;
        unsigned int size = last-first;
        return thrust::transform(CUIntIterator(0), CUIntIterator(size), first, f);
      }

    template <typename OutputIterator, typename T>
      OutputIterator generateRandom(OutputIterator first, OutputIterator last, Arena2d<T> arena, unsigned int samples, unsigned int seed = 1) {
        assert(last - first == samples);
        T xmin, xmax, ymin, ymax;
        arena.getBounds(xmin,xmax,ymin,ymax);
        return coordinates::generate(ComputeCoordRandom2d<T>(xmin,xmax,ymin,ymax,seed), first, last);
      }

    template <typename OutputVector, typename T>
      void generateRandom(OutputVector &coords, Arena2d<T> arena, unsigned int samples, unsigned int seed = 1) {
        coords.resize(samples);
        coordinates::generateRandom(coords.begin(), coords.end(), arena, samples, seed);
      }

    template <typename OutputIterator, typename T>
      OutputIterator generateGrid(OutputIterator first, OutputIterator last, Arena2d<T> arena, unsigned int w, unsigned int h) {
        assert(last - first == w*h);
        T xmin, xmax, ymin, ymax;
        arena.getBounds(xmin,xmax,ymin,ymax);
        return coordinates::generate(ComputeCoordGrid2d<T>(xmin,xmax,ymin,ymax,w,h), first, last);
      }

    template <typename OutputVector, typename T>
      void generateGrid(OutputVector &coords, Arena2d<T> arena, unsigned int w, unsigned int h) {
        coords.resize(w*h);
        coordinates::generateGrid(coords.begin(), coords.end(), arena, w, h);
      }

  }; /* namespace coordinates */

}; /* namespace sinr */


//#include <iostream>
//#include <sstream>
//#include <iomanip>
//std::ostream& operator<<(std::ostream &out, const Coordinates &coords) {
//  out<<"Coordinates:"<<std::endl<<std::endl;
//
//  out<<"Bounding Box:"<<std::endl; 
//  out<<"Bottom-Left Corner: ("<<coords.xmin<<","<<coords.ymin<<")"<<std::endl; 
//  out<<"Top-Right Corner: ("<<coords.xmax<<","<<coords.ymax<<")"<<std::endl; 
//
//  out<<"Coordinates: "  <<coords.c.size()<<std::endl;
//  out<<"Width: "        <<coords.w<<std::endl;
//  out<<"Height: "       <<coords.h<<std::endl;
//  out<<"Method: "       <<coords.cgm<<std::endl;
//
//  thrust::host_vector<Point2d<Real> > host_c = coords.c;
//
//  unsigned int index;
//  for (unsigned int i = 0; i < coords.w; i++) {
//    for (unsigned int j = 0; j < coords.h; j++) {
//      index = j*coords.w + i;
//      std::ostringstream oss(std::ostringstream::out);
//      oss<<"("<<host_c[index].x<<","<<host_c[index].y<<")";
//      out<<std::setw(10)<<oss.str();
//    }
//    out<<std::endl;
//  }
//  return out; 
//}

#endif /* __COORDINATES_CUH__ */
