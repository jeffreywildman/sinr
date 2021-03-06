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

  /// now you don't have to flush the buffer: http://stackoverflow.com/a/1716621/627517
  setbuf(stdout, NULL);
  
  //unsigned int dev  = 0;       /* TODO: set the cuda device more smartly. */
  unsigned int w          = 10;                    /* (pixels) */
  unsigned int h          = 10;                    /* (pixels) */
  unsigned int f          = 1;                     /* reduction in sampling for change metric */
  unsigned int w2         = w/f;                    /* (pixels) */
  unsigned int h2         = h/f;                    /* (pixels) */
  unsigned int N          = 10;                     /* number of nodes */
  double arenaSideLength  = 1000.0;                 /* (m) */
  Real widthdB            = 5.0;                    /* visual display width of SINR cells (dB) */

  Arena2d<Real> arena(arenaSideLength);
  Network<Real> net(N);
  cout<<"Node, Position"<<endl;
  for (unsigned int n = 0; n < N; n++) {
    Point2d p2d(Util::uniform_double(0,arenaSideLength), Util::uniform_double(0,arenaSideLength));
    net.setPosition(n, p2d);
    cout<<n<<" "<<"("<<thrust::get<0>(p2d)<<","<<thrust::get<1>(p2d)<<")"<<endl;
  }
  cout<<endl;

  /* coordinates to compute metrics over */
  thrust::device_vector<Point2d> coords(w*h);
  sinr::coordinates::generateGrid(coords, arena, w, h); 
  NetworkMetricsDev<Real> nm(&net, &arena, &coords);
  
  /* common variables for display */
  const thrust::device_vector<Real> *maxsinr;
  thrust::device_vector<uchar4> rgba;

  /* coordinates to move node over */
  static thrust::host_vector<Point2d> coords2(w2*h2);
  sinr::coordinates::generateGrid(coords2, arena, w2, h2);

  /* storage of expected capacities */
  thrust::host_vector<Real> expcap_delta(w2*h2);

  /* save original snapshot of network */
  maxsinr = nm.computeMapMaxSINR();

  rgba.resize(coords.size());
  sinr::visualize::grayscaledB(*maxsinr, rgba, pow(10.0,(net.getSINRThresholddB()-widthdB)/10.0), pow(10.0,net.getSINRThresholddB()/10.0));
  sinr::visualize::outputBMP(rgba, w, h, OmanDirs::images() + "/original.bmp");

  Real power_orig = net.getPower(0);
  net.setPower(0,0.0);

  /* save original snapshot of network */
  maxsinr = nm.computeMapMaxSINR();
  rgba.resize(coords.size());
  sinr::visualize::grayscaledB(*maxsinr, rgba, pow(10.0,(net.getSINRThresholddB()-widthdB)/10.0), pow(10.0,net.getSINRThresholddB()/10.0));
  sinr::visualize::outputBMP(rgba, w, h, OmanDirs::images() + "/original2.bmp");

  thrust::device_vector<Real> zone_probabilities(N);
  zone_probabilities = nm.computeAssocZoneProb(DensityUniform2d<Real>(arena));

  cout<<N-1<<"-node Uniform Case"<<endl;
  cout<<"Cell, Prob"<<endl;
  for (unsigned int n = 0; n < net.getSize(); n++) {
    cout<<n<<", "<<zone_probabilities[n]<<endl;
  }
  cout<<endl;

  zone_probabilities = nm.computeAssocZoneProb(DensityCentered2d<Real>(arena));

  cout<<N-1<<"-node Centered Case"<<endl;
  cout<<"Cell, Prob"<<endl;
  for (unsigned int n = 0; n < net.getSize(); n++) {
    cout<<n<<", "<<zone_probabilities[n]<<endl;
  }
  cout<<endl;

  /* wonky coordinate to matrix row/column conversion */
  thrust::host_vector<Point2d> coords_hst(coords);
  thrust::host_vector<unsigned int> assoc_index_hst(*nm.computeMapAssocZone());
  thrust::host_vector<Real> maxsinr_hst(*nm.computeMapMaxSINR());
  thrust::host_vector<Real> maxcap_hst(*nm.computeMapMaxShannonCapacity());
  cout<<"Coordinate (meters), AssocIndex, MaxSINR (mW/mW), MaxCap (Mbps)"<<endl;
  for (unsigned int i = 0; i < w; i++) {
    for (unsigned int j = 0; j < h; j++) {
      cout<<"("<<thrust::get<0>(coords_hst[j+i*h])<<","<<thrust::get<1>(coords_hst[j+i*h])<<") \t"<<assoc_index_hst[j+i*h]<<" \t"<<maxsinr_hst[j+i*h]<<"    \t"<<maxcap_hst[j+i*h]<<endl;
    }
  }
  cout<<endl;

  Real avgsinr_orig = nm.computeAvgMaxSINR();
  Real avgcap_orig = nm.computeAvgMaxCapacity();
  Real expcap_orig = nm.computeExpSpatialCapacity2(MANY_USERS, DensityUniform2d<Real>(arena));
  cout<<"Uniform Case: Initial Average Max SINR: "<<avgsinr_orig<<" mW/mW"<<endl;
  cout<<"Uniform Case: Initial Average Max Capacity: "<<avgcap_orig<<" Mbps"<<endl;
  cout<<"Uniform Case: Initial Expected Spatial Capacity: "<<expcap_orig<<" Mbps/m^2"<<endl;
  net.setPower(0,power_orig);
  
  cout<<"Uniform Case: Processing...";
  int tenths = 0;
  for(unsigned int i = 0; i < coords2.size(); i++) {
    if (i % int(coords2.size()/10) == 0) {
      cout<<(tenths++)*10<<"%...";
    }
    net.setPosition(0,coords2[i]);
    expcap_delta[i] = nm.computeExpSpatialCapacity2(MANY_USERS, DensityUniform2d<Real>(arena)) - expcap_orig;
  } 
  cout<<endl;

  thrust::device_vector<Real> expcap_delta_dev(expcap_delta);
  rgba.resize(expcap_delta_dev.size());
  sinr::visualize::twotone(expcap_delta_dev, rgba);
  sinr::visualize::outputBMP(rgba, w2, h2, OmanDirs::images() + "/spatialcapacity_delta_uniform.bmp");



  power_orig = net.getPower(0);
  net.setPower(0,0.0);
  avgsinr_orig = nm.computeAvgMaxSINR();
  avgcap_orig = nm.computeAvgMaxCapacity();
  expcap_orig = nm.computeExpSpatialCapacity2(MANY_USERS, DensityCentered2d<Real>(arena));
  cout<<"Centered Case: Initial Average Max SINR: "<<avgsinr_orig<<" mW/mW"<<endl;
  cout<<"Centered Case: Initial Average Max Capacity: "<<avgcap_orig<<" Mbps"<<endl;
  cout<<"Centered Case: Initial Expected Spatial Capacity: "<<expcap_orig<<" Mbps/m^2"<<endl;
  net.setPower(0,power_orig);

  cout<<"Centered Case: Processing...";
  tenths = 0;
  for(unsigned int i = 0; i < coords2.size(); i++) {
    if (i % int(coords2.size()/10) == 0) {
      cout<<(tenths++)*10<<"%...";
    }
    net.setPosition(0,coords2[i]);
    expcap_delta[i] = nm.computeExpSpatialCapacity2(MANY_USERS, DensityCentered2d<Real>(arena)) - expcap_orig;
  } 
  cout<<endl;

  expcap_delta_dev = thrust::device_vector<Real>(expcap_delta);
  rgba.resize(expcap_delta_dev.size());
  sinr::visualize::twotone(expcap_delta_dev, rgba);
  sinr::visualize::outputBMP(rgba, w2, h2, OmanDirs::images() + "/spatialcapacity_delta_centered.bmp");


  return 0;
}
