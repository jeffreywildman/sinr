#include <iostream>
#include <vector>
#include <math.h>       /* for M_PI */

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <oman/general/omandirs.h>            /* non c++11 */

#include <sinr/network.h>
#include <sinr/coordinates.cuh>
#include <sinr/networkmetrics.cuh>
#include <sinr/visualizer.cuh>
#include <sinr/gui.h>
#include <sinr/types.h>
#include <sinr/spatialdensity.cuh>
#include <sinr/util.h>

typedef double Real;
typedef thrust::tuple<Real,Real> Point2d;

using namespace std;


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

  //unsigned int dev  = 0;       /* TODO: set the cuda device more smartly. */
  unsigned int w      = 500;                    /* (pixels) */
  unsigned int h      = 500;                    /* (pixels) */
  unsigned int N      = 10;                     /* number of nodes */
  Real sidelength     = 1000;                   /* arena side length */

  Arena2d<Real> arena(sidelength);
  Network<Real> net(N);
  for (unsigned int n = 0; n < N; n++) {
    net.setPosition(n, Point2d(Util::uniform_double(0,sidelength),Util::uniform_double(0,sidelength)));
  }

  thrust::device_vector<Point2d> coords(w*h);
  sinr::coordinates::generateGrid(coords, arena, w, h);
  NetworkMetricsDev<Real> nm(&net,&arena,&coords);

  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  double actcap = nm.computeActSpatialCapacity();

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop); 
  cudaEventElapsedTime(&time, start, stop);
  printf ("Time for actual: %f ms\n", time);

  unsigned int M = coords.size();

  cudaEventRecord(start, 0);
  double expcap = nm.computeExpSpatialCapacity(M);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop); 
  cudaEventElapsedTime(&time, start, stop);
  printf ("Time for expected, old: %f ms\n", time);
  
  
  cudaEventRecord(start, 0);
  //double expcap2 = nm.computeExpSpatialCapacity2(M);
  double expcap2 = nm.computeExpSpatialCapacity2(M, DensityCentered2d<Real>(arena, 1.0));
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop); 
  cudaEventElapsedTime(&time, start, stop);
  printf ("Time for expected, new: %f ms\n", time);

  std::cout<<"actcap:  "<<actcap<<std::endl;
  std::cout<<"expcap:  "<<expcap<<std::endl;
  std::cout<<"expcap2:  "<<expcap2<<std::endl;

  return 0;
}
