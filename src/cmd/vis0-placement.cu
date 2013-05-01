#include <iostream>
#include <vector>
#include <math.h>       /* for M_PI */

#include <oman/general/omandirs.h>            /* non c++11 */

#include <sinr/network.h>
#include <sinr/coordinates.cuh>
#include <sinr/networkmetrics.cuh>
#include <sinr/visualizer.cuh>
#include <sinr/util.h>
#include <sinr/gui.h>
#include <sinr/types.h>

typedef double Real;
typedef thrust::tuple<Real,Real> Point2d;

using namespace std;


unsigned int renderer(unsigned int tick, unsigned int w, unsigned int h, uchar4 *dev_rgba) {
  static unsigned int N   = 10;                     /* number of nodes */
  static Real sidelength  = 1000.0;                 /* length of arena side */
  static Real widthdB     = 5;
  
  static Network<Real> net(N);
  static Arena2d<Real> arena(sidelength);
 
  /* coordinates to compute metrics over */
  static thrust::device_vector<Point2d> coords(w*h);
  static NetworkMetricsDev<Real> nm(&net, &arena, &coords);

  /* coordinates to move node over */
  static thrust::host_vector<Point2d> coords2((w/10)*(h/10));
  
  if (tick == 0) {
    for (unsigned int n = 0; n < N; n++) {
      net.setPosition(n, Point2d(Util::uniform_double(0,sidelength),Util::uniform_double(0,sidelength)));
    }
    sinr::coordinates::generateGrid(coords2, arena, w/10, h/10);
    net.setPosition(1,coords2[tick]);
    
    sinr::coordinates::generateGrid(coords, arena, w, h); 
    nm = NetworkMetricsDev<Real>(&net, &arena, &coords);
  } else if (tick < coords2.size()) {
    net.setPosition(1,coords2[tick]);
  } else {
    return 0;
  }

  const thrust::device_vector<Real> *maxsinr = nm.computeMapMaxSINR();
  thrust::device_ptr<uchar4> tdp_rgba = thrust::device_pointer_cast(dev_rgba);
  sinr::visualize::grayscaledB(*maxsinr, tdp_rgba, Real(net.getSINRThresholddB()-widthdB), net.getSINRThresholddB());

  return 1;
}



/**
 *
 */
int main(int argc __attribute__((unused)), char** argv __attribute__((unused))) {

  /** @note if you only have one GPU, free memory depends on the number of windows you have open */
  /** @todo generalize this code so that other programs can use it */
  cudaSetDevice(0);
  cudaDeviceReset();
  size_t free, total;
  cudaMemGetInfo(&free, &total);
  std::cout<<"free: "<<free<<"\t total: "<<total<<std::endl;

  /* User-controlled setup for experiment */
  Util::deleteDirContents(OmanDirs::temp());
  Util::deleteDirContents(OmanDirs::images());
  Util::deleteDirContents(OmanDirs::videos());
  Util::deleteDirContents(OmanDirs::logs());

  Util::seedRandomGenerator(0);

  unsigned int dev  = 0;       /* TODO: set the cuda device more smartly. */
  unsigned int w    = 500;                    /* (pixels) */
  unsigned int h    = 500;                    /* (pixels) */
  
  Gui g(w,h,dev);
  g.run(renderer,w,h);

  return 0;
}
