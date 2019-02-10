#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <sinr/gui.h>
#include <sinr/arena.h>
#include <sinr/network.h>
#include <sinr/networkmetrics.cuh>
#include <sinr/visualizer.cuh>

using namespace std;

typedef double Real;
typedef thrust::tuple<Real,Real> Point2d;

/**
 * Render the max SINR metric of a 10-node network with random positions.
 */
int main(int argc __attribute__((unused)), char** argv __attribute__((unused))) {

  unsigned int w    = 1024;    /* (pixels) */
  unsigned int h    = 1024;    /* (pixels) */

  /* random seed for positions */
  unsigned int seed = 1;

  /* size of arena (meters) */
  Real xlen = 100.0;
  Real ylen = 100.0;
  Arena2d<Real> arena(xlen,ylen);

  Real N      = 10;     // number of nodes
  Real widthdB = 5;     // display range of SINR values

  /* construct network */
  Network<Real> net(N);

  /* construct and set random node coordinates */
  std::vector<Point2d> positions(N);
  sinr::coordinates::generateRandom(positions, arena, N, seed);
  net.setPosition(positions);

  /* construct grid coordinates for max SINR metric */
  thrust::device_vector<Point2d> coords(w*h);
  sinr::coordinates::generateGrid(coords, arena, w, h);

  /* compute the max SINR produced by network nodes measured over the set of coords */
  NetworkMetricsDev<Real> nm(&net, &arena, &coords);
  const thrust::device_vector<Real> *max_sinr = nm.computeMapMaxSINR();

  /* convert the max SINR vector to RGBA values for display to file and screen */
  thrust::device_vector<uchar4> rgba(w*h);
  sinr::visualize::grayscaledB(*max_sinr, rgba, Real(net.getSINRThresholddB()-widthdB), net.getSINRThresholddB());
  sinr::visualize::outputBMP(rgba, w, h, "./ex0-network.bmp"); 

  Gui g(w,h);
  g.display(&rgba,w,h);

  return 0;
}
