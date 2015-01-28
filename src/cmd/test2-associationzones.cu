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


/** This test program plots association zones with and without biasing weights.
 */
int main(int argc __attribute__((unused)), char** argv __attribute__((unused))) {

  /* common parameters */
  Real xmin = 1000.0;
  Real xmax = 2000.0;
  Real ymin = 1000.0;
  Real ymax = 3000.0;
  Arena2d<Real> arena(xmin,xmax,ymin,ymax);
  unsigned int w = 250;
  unsigned int h = 500;
  Real N      = 16;
  Real betadB = 3;
  Real widthdB = 5;

  /* construct grid coordinates */
  thrust::device_vector<Point2d> coords(w*h);
  sinr::coordinates::generateGrid(coords, arena, w, h);

  Network<Real> net(N);

  /* construct node coordinates */
  unsigned int wpos = ceil(sqrt(N));
  unsigned int hpos = ceil(sqrt(N));
  thrust::host_vector<Point2d> positions(w*h);
  sinr::coordinates::generateGrid(positions, arena, wpos, hpos);
  for (unsigned int n = 0; n < net.getSize(); n++) {
    net.setPosition(n, positions[n]);
  }

  NetworkMetricsDev<Real> nm(&net, &arena, &coords);
  thrust::device_vector<uchar4> rgba(w*h);
  const thrust::device_vector<Real> *max_sinr = nm.computeMapMaxSINR();
  sinr::visualize::grayscaledB(*max_sinr, rgba, pow(10.0,(betadB-widthdB)/10.0), pow(10.0,betadB/10.0)); 
  sinr::visualize::outputBMP(rgba, w, h, "./maxsinr_cells.bmp"); 
 
  const thrust::device_vector<nid_t> *zone;
  thrust::host_vector<Real> prob;
  Real sumprob;


  /* compute unbiased association zones */
  nm.setAssocZoneRule(AZR_SINR);
  zone = nm.computeMapAssocZone();
  sinr::visualize::assoczone(*zone, rgba, N);
  sinr::visualize::outputBMP(rgba, w, h, "./unbiasedsinr_zones.bmp"); 

  prob = nm.computeAssocZoneProb(DensityUniform2d<Real>(arena));
  sumprob = thrust::reduce(prob.begin(),prob.end());
  cout<<"SINR Zones Sum Prob: "<<sumprob<<endl;


  /* compute unbiased association zones */
  nm.setAssocZoneRule(AZR_POWER);
  zone = nm.computeMapAssocZone();
  sinr::visualize::assoczone(*zone, rgba, N);
  sinr::visualize::outputBMP(rgba, w, h, "./unbiasedpower_zones.bmp"); 

  prob = nm.computeAssocZoneProb(DensityUniform2d<Real>(arena));
  sumprob = thrust::reduce(prob.begin(),prob.end());
  cout<<"Power Zones Sum Prob: "<<sumprob<<endl;


  /* compute biased association zones */
  net.setBias(7,10);
  nm.setAssocZoneRule(AZR_SINR_BIASED);
  zone = nm.computeMapAssocZone();
  sinr::visualize::assoczone(*zone, rgba, N);
  sinr::visualize::outputBMP(rgba, w, h, "./biasedsinr_zones.bmp"); 

  prob = nm.computeAssocZoneProb(DensityUniform2d<Real>(arena));
  sumprob = thrust::reduce(prob.begin(),prob.end());
  cout<<"SINR Biased Zones Sum Prob: "<<sumprob<<endl;


  /* compute biased association zones */
  net.setBias(7,10);
  nm.setAssocZoneRule(AZR_POWER_BIASED);
  zone = nm.computeMapAssocZone();
  sinr::visualize::assoczone(*zone, rgba, N);
  sinr::visualize::outputBMP(rgba, w, h, "./biasedpower_zones.bmp"); 

  prob = nm.computeAssocZoneProb(DensityUniform2d<Real>(arena));
  sumprob = thrust::reduce(prob.begin(),prob.end());
  cout<<"Power Biased Zones Sum Prob: "<<sumprob<<endl;

  return 0;
}
