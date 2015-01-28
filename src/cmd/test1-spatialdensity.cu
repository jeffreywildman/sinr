#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>

//#include <sinr/gui.h>
//#include <sinr/types.h>
#include <sinr/arena.h>
#include <sinr/spatialdensity.cuh>
#include <sinr/coordinates.cuh>
#include <sinr/visualizer.cuh>
#include <sinr/util.h>

using namespace std;

typedef double Real;
typedef thrust::tuple<Real,Real> Point2d;

/** This test program plots spatial density functions and computes their integral over the defined domain.  The integral
 * should be at or very close to 1.
 */
int main(int argc __attribute__((unused)), char** argv __attribute__((unused))) {

  /* common parameters */
  Real xmin = 10.0;
  Real xmax = 20.0;
  Real ymin = 10.0;
  Real ymax = 30.0;
  Arena2d<Real> arena(xmin,xmax,ymin,ymax);
  unsigned int w = 250;
  unsigned int h = 500;
  Real max;
  Real prob;

  /* construct grid coordinates */
  thrust::device_vector<Point2d> coords(w*h);
  sinr::coordinates::generateGrid(coords, arena, w, h);

  /* more setup */
  thrust::device_vector<Real> density(coords.size());
  thrust::device_vector<uchar4> rgba(w*h);



  /* uniform density function */
  thrust::transform(coords.begin(), 
                    coords.end(),
                    density.begin(),
                    DensityUniform2d<Real>(arena));
  max = *thrust::max_element(density.begin(),density.end());
  sinr::visualize::grayscale(density, rgba, 0.0, max);
  prob = arena.getVolume()/Real(coords.size())*thrust::reduce(density.begin(),density.end(),0.0);
  cout<<"Uniform Density Integral: "<<prob<<endl;

  sinr::visualize::outputBMP(rgba, w, h, "./uniformdensity.bmp");



  /* centered density function */
  Real a = 1.0;

  thrust::transform(coords.begin(),
                    coords.end(),
                    density.begin(),
                    DensityCentered2d<Real>(arena, a));
  max = *thrust::max_element(density.begin(),density.end());
  sinr::visualize::grayscale(density, rgba, 0.0, max);
  prob = arena.getVolume()/Real(coords.size())*thrust::reduce(density.begin(),density.end(),0.0);
  cout<<"Centered Density Integral: "<<prob<<endl;
  
  sinr::visualize::outputBMP(rgba, w, h, "./centereddensity.bmp");

  return 0;
}
