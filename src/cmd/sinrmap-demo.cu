#include <iostream>
#include <vector>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <sinr/arena.h>
#include <sinr/network.h>
#include <sinr/networkmetrics.cuh>
#include <sinr/visualizer.cuh>
#include <sinr/util.h>

using namespace std;

typedef double Real;
typedef thrust::tuple<Real,Real> Point2d;


/** This test program plots the sinr map of a randomly generated network.
 */
int main(int argc __attribute__((unused)), char** argv __attribute__((unused))) {

  /* size of arena (meters) */
  Real xlen = 100.0;
  Real ylen = 100.0;
  Arena2d<Real> arena(xlen,ylen);

  /* pixel size of sinr map */
  unsigned int w = 1000; // width
  unsigned int h = 1000; // height

  Real N      = 16;     // number of nodes
  Real betadB = 3;      // SINR threshold (dB)
  Real widthdB = 5;     // Display range of SINR values

  /* construct grid coordinates */
  thrust::device_vector<Point2d> coords(w*h);
  sinr::coordinates::generateGrid(coords, arena, w, h);

  /* construct network */
  Network<Real> net(N);

  /* construct node coordinates */
  thrust::host_vector<Point2d> positions(w*h);
  //unsigned int wpos = ceil(sqrt(N));
  //unsigned int hpos = ceil(sqrt(N));
  //sinr::coordinates::generateGrid(positions, arena, wpos, hpos);
  unsigned int seed = 1;
  sinr::coordinates::generateRandom(positions, arena, N, seed);
  for (unsigned int n = 0; n < net.getSize(); n++) {
    net.setPosition(n, positions[n]);
  }

  NetworkMetricsDev<Real> nm(&net, &arena, &coords);
  thrust::device_vector<uchar4> rgba(w*h);
  const thrust::device_vector<Real> *max_sinr = nm.computeMapMaxSINR();
  sinr::visualize::grayscaledB(*max_sinr, rgba, pow(10.0,(betadB-widthdB)/10.0), pow(10.0,betadB/10.0)); 
  sinr::visualize::outputBMP(rgba, w, h, "./sinrmap-demo-maxsinr.bmp"); 

  return 0;
}
