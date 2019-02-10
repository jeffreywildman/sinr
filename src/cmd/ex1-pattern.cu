#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <sinr/gui.h>
#include <sinr/arena.h>
#include <sinr/network.h>
#include <sinr/networkmetrics.cuh>
#include <sinr/radiationpattern.cuh>
#include <sinr/visualizer.cuh>

using namespace std;

typedef double Real;
typedef thrust::tuple<Real,Real> Point2d;

/**
 * Render the SINR metric of a single-node network with varying elliptical radiation patterns.
 */
unsigned int renderEllipse(unsigned int tick, unsigned int w, unsigned int h, uchar4 *dev_rgba) {
  /* size of arena (meters) */
  Real xlen = 1000.0;
  Real ylen = 1000.0;
  Arena2d<Real> arena(xlen,ylen);

  Real N       = 1;     // number of nodes
  Real widthdB = 5;     // display range of SINR values
  Real power   = 0.01;  // transmit power
  Real ecc     = 1.0 - 1.0/((tick*2 % 500) + 1); // eccentricity of ellipse
  Real norm    = RadPatternEllipse<Real>::normalize(ecc); // constant for normalized total radiated power

  /* construct network */
  Network<Real> net(N);

  /* construct and set single node params */
  net.setPosition(0, Point2d(xlen/2.0,ylen/2.0));
  net.setRadPattern(0, RadPattern<Real>(RPT_ELLIPSE, ecc, norm));
  net.setPower(0, power);

  /* construct grid coordinates for SINR metric */
  thrust::device_vector<Point2d> coords(w*h);
  sinr::coordinates::generateGrid(coords, arena, w, h);

  /* compute the SINR produced by the node over the set of coords */
  NetworkMetricsDev<Real> nm(&net, &arena, &coords);
  const thrust::device_vector<Real> *sinr = nm.computeMapSINR(0);

  /* convert the SINR vector to RGBA values for display to file and screen */
  thrust::device_ptr<uchar4> tdp_rgba = thrust::device_pointer_cast(dev_rgba);
  sinr::visualize::grayscaledB(*sinr, tdp_rgba, Real(net.getSINRThresholddB()-widthdB), net.getSINRThresholddB());
  //sinr::visualize::outputBMP(tdb_rgba, w, h, "./ex1-pattern.bmp"); 

  return 1;
}




/**
 * Render the SINR metric of a single-node network with varying elliptical radiation patterns.
 */
int main(int argc __attribute__((unused)), char** argv __attribute__((unused))) {

  unsigned int w    = 1024;    /* (pixels) */
  unsigned int h    = 1024;    /* (pixels) */

  /* size of arena (meters) */
  Real xlen = 1000.0;
  Real ylen = 1000.0;
  Arena2d<Real> arena(xlen,ylen);

  Real N       = 1;     // number of nodes
  Real widthdB = 5;     // display range of SINR values
  Real power   = 0.05;  // transmit power
  Real ecc     = 0.9;   // eccentricity of ellipse
  Real norm    = RadPatternEllipse<Real>::normalize(ecc); // constant for normalized total radiated power

  /* construct network */
  Network<Real> net(N);
  net.setSINRThresholddB(10);

  /* construct and set single node params */
  net.setPosition(0, Point2d(xlen/2.0,ylen/2.0));
  net.setRadPattern(0, RadPattern<Real>(RPT_ELLIPSE, ecc, norm));
  net.setPower(0, power);

  /* construct grid coordinates for SINR metric */
  thrust::device_vector<Point2d> coords(w*h);
  sinr::coordinates::generateGrid(coords, arena, w, h);

  /* compute the SINR produced by the node over the set of coords */
  NetworkMetricsDev<Real> nm(&net, &arena, &coords);
  const thrust::device_vector<Real> *sinr = nm.computeMapSINR(0);

  /* convert the SINR vector to RGBA values for display to file and screen */
  thrust::device_vector<uchar4> rgba(w*h);
  sinr::visualize::grayscaledB(*sinr, rgba, Real(net.getSINRThresholddB()-widthdB), net.getSINRThresholddB());
  sinr::visualize::outputBMP(rgba, w, h, "./ex1-pattern.bmp"); 

  Gui g(w,h);
  //g.display(&rgba,w,h);
  g.run(renderEllipse,w,h);

  return 0;
}
