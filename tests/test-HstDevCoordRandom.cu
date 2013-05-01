#include <iostream>
#include <vector>
#include <math.h>       /* for M_PI */

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

//#include <cuda.h>
//#include "cuPrintf.cu"

#include <sinr/network.h>
#include <sinr/coordinates.h>
#include <sinr/networkmetrics.h>
#include <sinr/visualizer.h>
#include <sinr/gui.h>
#include <sinr/types.h>

using namespace std;

typedef double Real;

int main(int argc __attribute__((unused)), char** argv __attribute__((unused))) {

  thrust::device_vector<Point2d<double> > d1;
  thrust::host_vector  <Point2d<double> > h1,h2;

  int n = 10;

  h1.resize(n);
  h2.resize(n);
  d1.resize(n);

  thrust::transform(thrust::counting_iterator<unsigned int>(0),
                    thrust::counting_iterator<unsigned int>(d1.size()),
                    d1.begin(),
                    ComputeCoordRandom<double>(0,100,
                                          100,200,
                                          1));
  thrust::transform(thrust::counting_iterator<unsigned int>(0),
                    thrust::counting_iterator<unsigned int>(h2.size()),
                    h2.begin(),
                    ComputeCoordRandom<double>(0,100,
                                          100,200,
                                          1));
  thrust::copy(d1.begin(),d1.end(),h1.begin());

  for (unsigned int i = 0; i < h1.size(); i++) {
    std::cout<<"h1["<<i<<"] "<<h1[i]<<std::endl;
  }
  std::cout<<std::endl;
  for (unsigned int i = 0; i < h2.size(); i++) {
    std::cout<<"h2["<<i<<"] "<<h2[i]<<std::endl;
  }

  /** @todo compare h1 with h2 and assert! */

  return 0;
}
