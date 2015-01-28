#include <iostream>
#include <vector>

#include <thrust/host_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <sinr/radiationpattern.cuh>
#include <sinr/util.h>

using namespace std;

typedef double Real;


/** This test program plots radiation patterns and their total radiated power (TRP).  The total radiated power should be
 * at or very close to 1.
 */
int main(int argc __attribute__((unused)), char** argv __attribute__((unused))) {
  
  /* TRP parameters */
  Real trp;
  Real theta_a = 0.0;
  Real theta_b = 2.0*M_PI;
  Real N = 10000;
  thrust::counting_iterator<Real> cbeg(0);
  thrust::counting_iterator<Real> cend(N);
 
  /* plotting variables */
  thrust::host_vector<Real> rad(N);
  thrust::host_vector<Real> the(N);
  vector<Real> theta;
  vector<Real> radius;
 
  thrust::transform(thrust::make_transform_iterator(cbeg, Interpolate<Real>(0, 360, N)),
                    thrust::make_transform_iterator(cend, Interpolate<Real>(0, 360, N)),
                    the.begin(),
                    thrust::identity<Real>());
  theta = vector<Real>(the.begin(), the.end());



  /* ellipse parameters */
  Real ecc  = 0.9;
  Real norm = RadPatternEllipse<Real>::normalize(ecc);
 
  thrust::transform(thrust::make_transform_iterator(cbeg, Interpolate<Real>(theta_a, theta_b, N)),
                    thrust::make_transform_iterator(cend, Interpolate<Real>(theta_a, theta_b, N)),
                    rad.begin(),
                    RadPatternEllipse<Real>(ecc, norm));
  radius = vector<Real>(rad.begin(), rad.end());
  

  /* integrate radiation pattern from [0,2*pi] and make sure area is near 1. */
  trp = totalRadiatedPower<Real>(RadPatternEllipse<Real>(ecc, norm));
  cout<<"Ellipse Area: "<<trp<<endl;



  /* rose petal parameters */
  Real width = M_PI/4.0;
  Real power = 1;
 
  thrust::transform(thrust::make_transform_iterator(cbeg, Interpolate<Real>(theta_a, theta_b, N)),
                    thrust::make_transform_iterator(cend, Interpolate<Real>(theta_a, theta_b, N)),
                    rad.begin(),
                    RadPatternRosePetal<Real>(width, power));
  radius = vector<Real>(rad.begin(), rad.end());
 
  
  /* integrate radiation pattern from [0,2*pi] and make sure area is near 1. */
  trp = totalRadiatedPower<Real>(RadPatternRosePetal<Real>(width, power));
  cout<<"Rose Petal Area: "<<trp<<endl;

  return 0;
}
