#include <iostream>

#include <oman/visualizer/plot.h> /* Must include before other - looks like a clash of namespaces between OMAN::Utility and Magick::_ImageInfo? */
#include <oman/general/omandirs.h>            /* non c++11 */
#include <oman/general/iterationfunction.h>   /* non c++11 */
#include <oman/general/templatedefines.h>     /* non c++11 */

#include <sinr/network.h> /* Must include before other - looks like a clash of namespaces between OMAN::Utility and Thrust::Utility? */
#include <sinr/coordinates.cuh>
#include <sinr/visualizer.cuh>
#include <sinr/networkmetrics.cuh>
#include <sinr/util.h>
#include <sinr/types.h>

typedef double Real;
typedef thrust::tuple<Real,Real> Point2d;


/** This program computes averaged metrics versus the number of data points used to sample each metric over a
 * 2d arena, parameteried by the number of nodes in the network.  Each resulting data point is averaged over sampleCount
 * independent runs.
 *
 * A main comparison is made between performing the spatial metric calculations on the host CPU or device GPU.
 */
int main(int argc __attribute__((unused)), char **argv __attribute__((unused))) {

  /** @note if you only have one GPU, free memory depends on the number of windows you have open */
  /** @todo generalize this code so that other programs can use it */
  cudaSetDevice(0);
  cudaDeviceReset();
  size_t free, total;
  cudaMemGetInfo(&free, &total);
  std::cout<<"free: "<<free<<"\t total: "<<total<<std::endl;

  Util::seedRandomGenerator(0);

  vector<double> nodeIter      = IterationFunction(10, 10, 1).getVector();        /* number of nodes to iterate over */
  vector<double> sidePixelIter = IterationFunction(25, 500, 10).getVector();   /* image sizes to iterate over */
  unsigned int sampleCount = 1;                           /* samples to average over for each param and indVar value*/

  Real arenaSideLength = 1000.0;            /* (m) */
  Real widthdB         = 5.0;               /* visual display width of SINR cells (dB) */
  Arena2d<Real> arena(arenaSideLength);

  bool paramStatus = true;
  bool indVarStatus = true;
  bool sampleStatus = true;
  bool saveImage = true;
  bool witherrorbars = true;


  /* Map iterators to parameter and independent variables 
   * Note: remember this mapping to make the right variable assignments inside the nested for-loops */ 
  vector<double> paramIter(nodeIter);
  string paramName = "nodes";
  string paramLegend = "n=";
  vector<double> indVarIter(sidePixelIter);
  string indVarName = "pixels";
  string xlabel = "Image Side Length (pixels)";
  string imagePrename = "vs" + indVarName + "-samples" + Util::to_string(sampleCount) + "-"; 



  /* Automatic setup for experiment */
  vd3 cov_dev_grid(paramIter.size(), vd2(indVarIter.size(), vd1(sampleCount, 0.0)));  /* Coverage */
  vd3 snr_dev_grid(paramIter.size(), vd2(indVarIter.size(), vd1(sampleCount, 0.0)));  /* SINR Max */
  vd3 cap_dev_grid(paramIter.size(), vd2(indVarIter.size(), vd1(sampleCount, 0.0)));  /* Capacity */
  vd3 tme_dev_grid(paramIter.size(), vd2(indVarIter.size(), vd1(sampleCount, 0.0)));  /* Time */

  vd3 cov_hst_grid(paramIter.size(), vd2(indVarIter.size(), vd1(sampleCount, 0.0)));  /* Coverage */
  vd3 snr_hst_grid(paramIter.size(), vd2(indVarIter.size(), vd1(sampleCount, 0.0)));  /* SINR Max */
  vd3 cap_hst_grid(paramIter.size(), vd2(indVarIter.size(), vd1(sampleCount, 0.0)));  /* Capacity */
  vd3 tme_hst_grid(paramIter.size(), vd2(indVarIter.size(), vd1(sampleCount, 0.0)));  /* Time */

  vd3 cov_dev_rand(paramIter.size(), vd2(indVarIter.size(), vd1(sampleCount, 0.0)));  /* Coverage */
  vd3 snr_dev_rand(paramIter.size(), vd2(indVarIter.size(), vd1(sampleCount, 0.0)));  /* SINR Max */
  vd3 cap_dev_rand(paramIter.size(), vd2(indVarIter.size(), vd1(sampleCount, 0.0)));  /* Capacity */
  vd3 tme_dev_rand(paramIter.size(), vd2(indVarIter.size(), vd1(sampleCount, 0.0)));  /* Time */

  vd3 cov_hst_rand(paramIter.size(), vd2(indVarIter.size(), vd1(sampleCount, 0.0)));  /* Coverage */
  vd3 snr_hst_rand(paramIter.size(), vd2(indVarIter.size(), vd1(sampleCount, 0.0)));  /* SINR Max */
  vd3 cap_hst_rand(paramIter.size(), vd2(indVarIter.size(), vd1(sampleCount, 0.0)));  /* Capacity */
  vd3 tme_hst_rand(paramIter.size(), vd2(indVarIter.size(), vd1(sampleCount, 0.0)));  /* Time */

  uint64_t start, stop;

  /// now you don't have to flush the buffer: http://stackoverflow.com/a/1716621/627517
  setbuf(stdout, NULL);



  for(unsigned int sample=0; sample<sampleCount; sample++) {

    if(sampleStatus) {std::cout<<"s"<<sample<<" "<<std::endl;}



    for(unsigned int i_param=0; i_param<paramIter.size(); i_param++) {
      /* NOTE: make sure we are assigning elements of paramIter to the right environment variable */
      unsigned int N = paramIter.at(i_param);

      if(paramStatus) {std::cout<<"  "<<paramName<<i_param<<" ";}

      Network<Real> net_dev_grid(N);
      /* Set up network for CUDA */
      for (unsigned int n = 0; n < N; n++) {
        net_dev_grid.setPosition(n, Point2d(Util::uniform_double(0,arenaSideLength), Util::uniform_double(0,arenaSideLength)));
      }

      Network<Real> net_hst_grid = net_dev_grid;
      Network<Real> net_dev_rand = net_dev_grid;
      Network<Real> net_hst_rand = net_dev_grid;



      for(unsigned int i_indVar=0; i_indVar<indVarIter.size(); i_indVar++) {
        /* NOTE: make sure we are assigning elements of indVarIter to the right environment variable */
        unsigned int sidePixelCount = indVarIter.at(i_indVar);

        if(indVarStatus) {std::cout<<indVarName<<i_indVar<<" ";}

        /* device-based grid */
        start = Util::getTimeNS();
        thrust::device_vector<Point2d> coords_dev_grid(sidePixelCount*sidePixelCount);
        sinr::coordinates::generateGrid(coords_dev_grid, 
                                        arena, 
                                        sidePixelCount, 
                                        sidePixelCount);
        NetworkMetricsDev<Real> nm_dev_grid(&net_dev_grid, &arena, &coords_dev_grid);
        cov_dev_grid.at(i_param).at(i_indVar).at(sample) = nm_dev_grid.computeAvgMaxCoverage();
        snr_dev_grid.at(i_param).at(i_indVar).at(sample) = nm_dev_grid.computeAvgMaxSINR();
        cap_dev_grid.at(i_param).at(i_indVar).at(sample) = nm_dev_grid.computeAvgMaxCapacity();
        stop = Util::getTimeNS();
        tme_dev_grid.at(i_param).at(i_indVar).at(sample) = (stop-start)/double(Util::nanoPerSec);

        /* host-based grid */
        start = Util::getTimeNS();
        thrust::host_vector<Point2d> coords_hst_grid(sidePixelCount*sidePixelCount);
        sinr::coordinates::generateGrid(coords_hst_grid, 
                                        arena, 
                                        sidePixelCount, 
                                        sidePixelCount);
        NetworkMetricsHst<Real> nm_hst_grid(&net_hst_grid, &arena, &coords_hst_grid);
        cov_hst_grid.at(i_param).at(i_indVar).at(sample) = nm_hst_grid.computeAvgMaxCoverage();
        snr_hst_grid.at(i_param).at(i_indVar).at(sample) = nm_hst_grid.computeAvgMaxSINR();
        cap_hst_grid.at(i_param).at(i_indVar).at(sample) = nm_hst_grid.computeAvgMaxCapacity();
        stop = Util::getTimeNS();
        tme_hst_grid.at(i_param).at(i_indVar).at(sample) = (stop-start)/double(Util::nanoPerSec);

        /* device-based random */
        start = Util::getTimeNS();
        thrust::device_vector<Point2d> coords_dev_rand(sidePixelCount*sidePixelCount);
        sinr::coordinates::generateRandom(coords_dev_rand, 
                                          arena, 
                                          sidePixelCount*sidePixelCount,
                                          sample);
        NetworkMetricsDev<Real> nm_dev_rand(&net_dev_rand, &arena, &coords_dev_rand);
        cov_dev_rand.at(i_param).at(i_indVar).at(sample) = nm_dev_rand.computeAvgMaxCoverage();
        snr_dev_rand.at(i_param).at(i_indVar).at(sample) = nm_dev_rand.computeAvgMaxSINR();
        cap_dev_rand.at(i_param).at(i_indVar).at(sample) = nm_dev_rand.computeAvgMaxCapacity();
        stop = Util::getTimeNS();
        tme_dev_rand.at(i_param).at(i_indVar).at(sample) = (stop-start)/double(Util::nanoPerSec);

        /* host-based random */
        start = Util::getTimeNS();
        thrust::host_vector<Point2d> coords_hst_rand(sidePixelCount*sidePixelCount);
        sinr::coordinates::generateRandom(coords_hst_rand, 
                                          arena, 
                                          sidePixelCount*sidePixelCount,
                                          sample);
        NetworkMetricsHst<Real> nm_hst_rand(&net_hst_rand, &arena, &coords_hst_rand);
        cov_hst_rand.at(i_param).at(i_indVar).at(sample) = nm_hst_rand.computeAvgMaxCoverage();
        snr_hst_rand.at(i_param).at(i_indVar).at(sample) = nm_hst_rand.computeAvgMaxSINR();
        cap_hst_rand.at(i_param).at(i_indVar).at(sample) = nm_hst_rand.computeAvgMaxCapacity();
        stop = Util::getTimeNS();
        tme_hst_rand.at(i_param).at(i_indVar).at(sample) = (stop-start)/double(Util::nanoPerSec);

        if (saveImage && (sample == 0)) {
          string imagePostname = paramName + Util::to_string((int)paramIter.at(i_param)) + "-" + indVarName + Util::to_string((int)indVarIter.at(i_indVar)) + "-" + "s" + Util::to_string((int)sample);

          thrust::device_vector<uchar4> rgba_dev(sidePixelCount*sidePixelCount);
          const thrust::device_vector<Real> *maxsinr_dev = nm_dev_grid.computeMapMaxSINR();
          sinr::visualize::grayscaledB(*maxsinr_dev, rgba_dev, net_dev_grid.getSINRThresholddB()-widthdB, net_dev_grid.getSINRThresholddB());
          string imagePath_dev = OmanDirs::images() + "/" + imagePrename + imagePostname + "-dev.bmp";
          sinr::visualize::outputBMP(rgba_dev, sidePixelCount, sidePixelCount, imagePath_dev);

          thrust::host_vector<uchar4> rgba_hst(sidePixelCount*sidePixelCount);
          const thrust::host_vector<Real> *maxsinr_hst = nm_hst_grid.computeMapMaxSINR();
          sinr::visualize::grayscaledB(*maxsinr_hst, rgba_hst, net_hst_grid.getSINRThresholddB()-widthdB, net_dev_grid.getSINRThresholddB());
          string imagePath_hst = OmanDirs::images() + "/" + imagePrename + imagePostname + "-hst.bmp";
          sinr::visualize::outputBMP(rgba_hst, sidePixelCount, sidePixelCount, imagePath_hst);
        }
      }
      if (paramStatus) {std::cout<<std::endl;}
    }
  }

  if(paramStatus || indVarStatus || sampleStatus) {
    std::cout << std::endl;
  }


  /// Here we use the computed coverage, sinrmax, and capacity to determine the "error" in the lower resolution images.

  /* Device-based grid samples */
  vd2 cov_dev_grid_mu(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 snr_dev_grid_mu(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 cap_dev_grid_mu(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 tme_dev_grid_mu(paramIter.size(), vd1(indVarIter.size(), 0));

  vd2 cov_dev_grid_std(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 snr_dev_grid_std(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 cap_dev_grid_std(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 tme_dev_grid_std(paramIter.size(), vd1(indVarIter.size(), 0));

  vd2 cov_dev_grid_error_mu(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 snr_dev_grid_error_mu(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 cap_dev_grid_error_mu(paramIter.size(), vd1(indVarIter.size(), 0));

  vd2 cov_dev_grid_error_std(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 snr_dev_grid_error_std(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 cap_dev_grid_error_std(paramIter.size(), vd1(indVarIter.size(), 0));

  /* Host-based grid samples */
  vd2 cov_hst_grid_mu(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 snr_hst_grid_mu(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 cap_hst_grid_mu(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 tme_hst_grid_mu(paramIter.size(), vd1(indVarIter.size(), 0));

  vd2 cov_hst_grid_std(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 snr_hst_grid_std(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 cap_hst_grid_std(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 tme_hst_grid_std(paramIter.size(), vd1(indVarIter.size(), 0));

  vd2 cov_hst_grid_error_mu(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 snr_hst_grid_error_mu(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 cap_hst_grid_error_mu(paramIter.size(), vd1(indVarIter.size(), 0));

  vd2 cov_hst_grid_error_std(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 snr_hst_grid_error_std(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 cap_hst_grid_error_std(paramIter.size(), vd1(indVarIter.size(), 0));

  /* Device-based random samples */
  vd2 cov_dev_rand_mu(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 snr_dev_rand_mu(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 cap_dev_rand_mu(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 tme_dev_rand_mu(paramIter.size(), vd1(indVarIter.size(), 0));

  vd2 cov_dev_rand_std(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 snr_dev_rand_std(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 cap_dev_rand_std(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 tme_dev_rand_std(paramIter.size(), vd1(indVarIter.size(), 0));

  vd2 cov_dev_rand_error_mu(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 snr_dev_rand_error_mu(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 cap_dev_rand_error_mu(paramIter.size(), vd1(indVarIter.size(), 0));

  vd2 cov_dev_rand_error_std(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 snr_dev_rand_error_std(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 cap_dev_rand_error_std(paramIter.size(), vd1(indVarIter.size(), 0));

  /* Host-based random samples */
  vd2 cov_hst_rand_mu(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 snr_hst_rand_mu(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 cap_hst_rand_mu(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 tme_hst_rand_mu(paramIter.size(), vd1(indVarIter.size(), 0));

  vd2 cov_hst_rand_std(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 snr_hst_rand_std(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 cap_hst_rand_std(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 tme_hst_rand_std(paramIter.size(), vd1(indVarIter.size(), 0));

  vd2 cov_hst_rand_error_mu(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 snr_hst_rand_error_mu(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 cap_hst_rand_error_mu(paramIter.size(), vd1(indVarIter.size(), 0));

  vd2 cov_hst_rand_error_std(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 snr_hst_rand_error_std(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 cap_hst_rand_error_std(paramIter.size(), vd1(indVarIter.size(), 0));

  for(unsigned int i=0; i<paramIter.size(); i++) {
    for(unsigned int j=0; j<indVarIter.size(); j++) {

      /* Device-based grid samples */
      cov_dev_grid_mu.at(i).at(j) = Util::mean(cov_dev_grid.at(i).at(j));
      snr_dev_grid_mu.at(i).at(j) = Util::mean(snr_dev_grid.at(i).at(j));
      cap_dev_grid_mu.at(i).at(j) = Util::mean(cap_dev_grid.at(i).at(j)); 
      tme_dev_grid_mu.at(i).at(j) = Util::mean(tme_dev_grid.at(i).at(j)); 

      cov_dev_grid_std.at(i).at(j) = Util::stddev(cov_dev_grid.at(i).at(j));
      snr_dev_grid_std.at(i).at(j) = Util::stddev(snr_dev_grid.at(i).at(j));
      cap_dev_grid_std.at(i).at(j) = Util::stddev(cap_dev_grid.at(i).at(j)); 
      tme_dev_grid_std.at(i).at(j) = Util::stddev(tme_dev_grid.at(i).at(j)); 

      cov_dev_grid_error_mu.at(i).at(j) = Util::mean(Util::relErr(cov_dev_grid.at(i).at(j), cov_dev_grid.at(i).back()));
      snr_dev_grid_error_mu.at(i).at(j) = Util::mean(Util::relErr(snr_dev_grid.at(i).at(j), snr_dev_grid.at(i).back()));
      cap_dev_grid_error_mu.at(i).at(j) = Util::mean(Util::relErr(cap_dev_grid.at(i).at(j), cap_dev_grid.at(i).back()));

      cov_dev_grid_error_std.at(i).at(j) = Util::stddev(Util::relErr(cov_dev_grid.at(i).at(j), cov_dev_grid.at(i).back()));
      snr_dev_grid_error_std.at(i).at(j) = Util::stddev(Util::relErr(snr_dev_grid.at(i).at(j), snr_dev_grid.at(i).back()));
      cap_dev_grid_error_std.at(i).at(j) = Util::stddev(Util::relErr(cap_dev_grid.at(i).at(j), cap_dev_grid.at(i).back()));

      /* Host-based grid samples */
      cov_hst_grid_mu.at(i).at(j) = Util::mean(cov_hst_grid.at(i).at(j));
      snr_hst_grid_mu.at(i).at(j) = Util::mean(snr_hst_grid.at(i).at(j));
      cap_hst_grid_mu.at(i).at(j) = Util::mean(cap_hst_grid.at(i).at(j)); 
      tme_hst_grid_mu.at(i).at(j) = Util::mean(tme_hst_grid.at(i).at(j)); 

      cov_hst_grid_std.at(i).at(j) = Util::stddev(cov_hst_grid.at(i).at(j));
      snr_hst_grid_std.at(i).at(j) = Util::stddev(snr_hst_grid.at(i).at(j));
      cap_hst_grid_std.at(i).at(j) = Util::stddev(cap_hst_grid.at(i).at(j)); 
      tme_hst_grid_std.at(i).at(j) = Util::stddev(tme_hst_grid.at(i).at(j)); 

      cov_hst_grid_error_mu.at(i).at(j) = Util::mean(Util::relErr(cov_hst_grid.at(i).at(j), cov_hst_grid.at(i).back()));
      snr_hst_grid_error_mu.at(i).at(j) = Util::mean(Util::relErr(snr_hst_grid.at(i).at(j), snr_hst_grid.at(i).back()));
      cap_hst_grid_error_mu.at(i).at(j) = Util::mean(Util::relErr(cap_hst_grid.at(i).at(j), cap_hst_grid.at(i).back()));

      cov_hst_grid_error_std.at(i).at(j) = Util::stddev(Util::relErr(cov_hst_grid.at(i).at(j), cov_hst_grid.at(i).back()));
      snr_hst_grid_error_std.at(i).at(j) = Util::stddev(Util::relErr(snr_hst_grid.at(i).at(j), snr_hst_grid.at(i).back()));
      cap_hst_grid_error_std.at(i).at(j) = Util::stddev(Util::relErr(cap_hst_grid.at(i).at(j), cap_hst_grid.at(i).back()));

      /* Device-based random samples */
      cov_dev_rand_mu.at(i).at(j) = Util::mean(cov_dev_rand.at(i).at(j));
      snr_dev_rand_mu.at(i).at(j) = Util::mean(snr_dev_rand.at(i).at(j));
      cap_dev_rand_mu.at(i).at(j) = Util::mean(cap_dev_rand.at(i).at(j)); 
      tme_dev_rand_mu.at(i).at(j) = Util::mean(tme_dev_rand.at(i).at(j)); 

      cov_dev_rand_std.at(i).at(j) = Util::stddev(cov_dev_rand.at(i).at(j));
      snr_dev_rand_std.at(i).at(j) = Util::stddev(snr_dev_rand.at(i).at(j));
      cap_dev_rand_std.at(i).at(j) = Util::stddev(cap_dev_rand.at(i).at(j)); 
      tme_dev_rand_std.at(i).at(j) = Util::stddev(tme_dev_rand.at(i).at(j)); 

      cov_dev_rand_error_mu.at(i).at(j) = Util::mean(Util::relErr(cov_dev_rand.at(i).at(j), cov_dev_rand.at(i).back()));
      snr_dev_rand_error_mu.at(i).at(j) = Util::mean(Util::relErr(snr_dev_rand.at(i).at(j), snr_dev_rand.at(i).back()));
      cap_dev_rand_error_mu.at(i).at(j) = Util::mean(Util::relErr(cap_dev_rand.at(i).at(j), cap_dev_rand.at(i).back()));

      cov_dev_rand_error_std.at(i).at(j) = Util::stddev(Util::relErr(cov_dev_rand.at(i).at(j), cov_dev_rand.at(i).back()));
      snr_dev_rand_error_std.at(i).at(j) = Util::stddev(Util::relErr(snr_dev_rand.at(i).at(j), snr_dev_rand.at(i).back()));
      cap_dev_rand_error_std.at(i).at(j) = Util::stddev(Util::relErr(cap_dev_rand.at(i).at(j), cap_dev_rand.at(i).back()));      

      /* Host-based random samples */
      cov_hst_rand_mu.at(i).at(j) = Util::mean(cov_hst_rand.at(i).at(j));
      snr_hst_rand_mu.at(i).at(j) = Util::mean(snr_hst_rand.at(i).at(j));
      cap_hst_rand_mu.at(i).at(j) = Util::mean(cap_hst_rand.at(i).at(j)); 
      tme_hst_rand_mu.at(i).at(j) = Util::mean(tme_hst_rand.at(i).at(j)); 

      cov_hst_rand_std.at(i).at(j) = Util::stddev(cov_hst_rand.at(i).at(j));
      snr_hst_rand_std.at(i).at(j) = Util::stddev(snr_hst_rand.at(i).at(j));
      cap_hst_rand_std.at(i).at(j) = Util::stddev(cap_hst_rand.at(i).at(j)); 
      tme_hst_rand_std.at(i).at(j) = Util::stddev(tme_hst_rand.at(i).at(j)); 

      cov_hst_rand_error_mu.at(i).at(j) = Util::mean(Util::relErr(cov_hst_rand.at(i).at(j), cov_hst_rand.at(i).back()));
      snr_hst_rand_error_mu.at(i).at(j) = Util::mean(Util::relErr(snr_hst_rand.at(i).at(j), snr_hst_rand.at(i).back()));
      cap_hst_rand_error_mu.at(i).at(j) = Util::mean(Util::relErr(cap_hst_rand.at(i).at(j), cap_hst_rand.at(i).back()));

      cov_hst_rand_error_std.at(i).at(j) = Util::stddev(Util::relErr(cov_hst_rand.at(i).at(j), cov_hst_rand.at(i).back()));
      snr_hst_rand_error_std.at(i).at(j) = Util::stddev(Util::relErr(snr_hst_rand.at(i).at(j), snr_hst_rand.at(i).back()));
      cap_hst_rand_error_std.at(i).at(j) = Util::stddev(Util::relErr(cap_hst_rand.at(i).at(j), cap_hst_rand.at(i).back()));      
    } 
  }


  Plot plot;
  //plot.constants.logscale_x = true;
  //plot.constants.logscale_y = false;

  /* Coverage */
  plot.create(PT_LINE_POINT, "", xlabel, "Coverage");
  for (unsigned int i=0; i<paramIter.size(); i++) {
    if (witherrorbars) {
      plot.addData_withErrorBars(indVarIter, cov_dev_grid_mu.at(i), cov_dev_grid_std.at(i), "d,g," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData_withErrorBars(indVarIter, cov_hst_grid_mu.at(i), cov_hst_grid_std.at(i), "h,g," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData_withErrorBars(indVarIter, cov_dev_rand_mu.at(i), cov_dev_rand_std.at(i), "d,r," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData_withErrorBars(indVarIter, cov_hst_rand_mu.at(i), cov_hst_rand_std.at(i), "h,r," + paramLegend + Util::to_string((int)paramIter.at(i)));
    } else {
      plot.addData(indVarIter, cov_dev_grid_mu.at(i), "d,g," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData(indVarIter, cov_hst_grid_mu.at(i), "h,g," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData(indVarIter, cov_dev_rand_mu.at(i), "d,r," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData(indVarIter, cov_hst_rand_mu.at(i), "h,r," + paramLegend + Util::to_string((int)paramIter.at(i)));
    }
  }
  plot.save(imagePrename + "coverage"); 

  //plot.constants.logscale_y = true;
  plot.create(PT_LINE_POINT, "", xlabel, "Relative Error of Coverage");
  for (unsigned int i=0; i<paramIter.size(); i++) {
    if (witherrorbars) {
      plot.addData_withErrorBars(indVarIter, cov_dev_grid_error_mu.at(i), cov_dev_grid_error_std.at(i), "d,g," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData_withErrorBars(indVarIter, cov_hst_grid_error_mu.at(i), cov_hst_grid_error_std.at(i), "h,g," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData_withErrorBars(indVarIter, cov_dev_rand_error_mu.at(i), cov_dev_rand_error_std.at(i), "d,r," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData_withErrorBars(indVarIter, cov_hst_rand_error_mu.at(i), cov_hst_rand_error_std.at(i), "h,r," + paramLegend + Util::to_string((int)paramIter.at(i)));
    } else {
      plot.addData(indVarIter, cov_dev_grid_error_mu.at(i), "d,g," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData(indVarIter, cov_hst_grid_error_mu.at(i), "h,g," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData(indVarIter, cov_dev_rand_error_mu.at(i), "d,r," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData(indVarIter, cov_hst_rand_error_mu.at(i), "h,r," + paramLegend + Util::to_string((int)paramIter.at(i)));
    }
  }
  plot.save(imagePrename + "coverage-error"); 

  /* SINR Max */
  plot.create(PT_LINE_POINT, "", xlabel, "Average Max SINR (W/W)");
  for (unsigned int i=0; i<paramIter.size(); i++) {
    if (witherrorbars) {
      plot.addData_withErrorBars(indVarIter, snr_dev_grid_mu.at(i), snr_dev_grid_std.at(i), "d,g," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData_withErrorBars(indVarIter, snr_hst_grid_mu.at(i), snr_hst_grid_std.at(i), "h,g," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData_withErrorBars(indVarIter, snr_dev_rand_mu.at(i), snr_dev_rand_std.at(i), "d,r," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData_withErrorBars(indVarIter, snr_hst_rand_mu.at(i), snr_hst_rand_std.at(i), "h,r," + paramLegend + Util::to_string((int)paramIter.at(i)));
    } else {
      plot.addData(indVarIter, snr_dev_grid_mu.at(i), "d,g," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData(indVarIter, snr_hst_grid_mu.at(i), "h,g," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData(indVarIter, snr_dev_rand_mu.at(i), "d,r," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData(indVarIter, snr_hst_rand_mu.at(i), "h,r," + paramLegend + Util::to_string((int)paramIter.at(i)));
    }
  }
  plot.save(imagePrename + "sinrmax"); 

  plot.create(PT_LINE_POINT, "", xlabel, "Relative Error of Average Max SINR");
  for (unsigned int i=0; i<paramIter.size(); i++) {
    if (witherrorbars) {
      plot.addData_withErrorBars(indVarIter, snr_dev_grid_error_mu.at(i), snr_dev_grid_error_std.at(i), "d,g," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData_withErrorBars(indVarIter, snr_hst_grid_error_mu.at(i), snr_hst_grid_error_std.at(i), "h,g," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData_withErrorBars(indVarIter, snr_dev_rand_error_mu.at(i), snr_dev_rand_error_std.at(i), "d,r," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData_withErrorBars(indVarIter, snr_hst_rand_error_mu.at(i), snr_hst_rand_error_std.at(i), "h,r," + paramLegend + Util::to_string((int)paramIter.at(i)));
    } else {
      plot.addData(indVarIter, snr_dev_grid_error_mu.at(i), "d,g," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData(indVarIter, snr_hst_grid_error_mu.at(i), "h,g," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData(indVarIter, snr_dev_rand_error_mu.at(i), "d,r," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData(indVarIter, snr_hst_rand_error_mu.at(i), "h,r," + paramLegend + Util::to_string((int)paramIter.at(i)));
    }
  }
  plot.save(imagePrename + "sinrmax-error"); 

  /* Capacity */
  plot.create(PT_LINE_POINT, "", xlabel, "Average Max Capacity (bps)");
  for (unsigned int i=0; i<paramIter.size(); i++) {
    if (witherrorbars) {
      plot.addData_withErrorBars(indVarIter, cap_dev_grid_mu.at(i), cap_dev_grid_std.at(i), "d,g," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData_withErrorBars(indVarIter, cap_hst_grid_mu.at(i), cap_hst_grid_std.at(i), "h,g," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData_withErrorBars(indVarIter, cap_dev_rand_mu.at(i), cap_dev_rand_std.at(i), "d,r," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData_withErrorBars(indVarIter, cap_hst_rand_mu.at(i), cap_hst_rand_std.at(i), "h,r," + paramLegend + Util::to_string((int)paramIter.at(i)));
    } else {
      plot.addData(indVarIter, cap_dev_grid_mu.at(i), "d,g," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData(indVarIter, cap_hst_grid_mu.at(i), "h,g," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData(indVarIter, cap_dev_rand_mu.at(i), "d,r," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData(indVarIter, cap_hst_rand_mu.at(i), "h,r," + paramLegend + Util::to_string((int)paramIter.at(i)));
    }
  }
  plot.save(imagePrename + "capacity"); 

  //plot.constants.logscale_y = true;
  plot.create(PT_LINE_POINT, "", xlabel, "Relative Error of Average Max Capacity");
  for (unsigned int i=0; i<paramIter.size(); i++) {
    if (witherrorbars) {
      plot.addData_withErrorBars(indVarIter, cap_dev_grid_error_mu.at(i), cap_dev_grid_error_std.at(i), "d,g," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData_withErrorBars(indVarIter, cap_hst_grid_error_mu.at(i), cap_hst_grid_error_std.at(i), "h,g," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData_withErrorBars(indVarIter, cap_dev_rand_error_mu.at(i), cap_dev_rand_error_std.at(i), "d,r," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData_withErrorBars(indVarIter, cap_hst_rand_error_mu.at(i), cap_hst_rand_error_std.at(i), "h,r," + paramLegend + Util::to_string((int)paramIter.at(i)));
    } else {
      plot.addData(indVarIter, cap_dev_grid_error_mu.at(i), "d,g," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData(indVarIter, cap_hst_grid_error_mu.at(i), "h,g," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData(indVarIter, cap_dev_rand_error_mu.at(i), "d,r," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData(indVarIter, cap_hst_rand_error_mu.at(i), "h,r," + paramLegend + Util::to_string((int)paramIter.at(i)));
    }
  }
  plot.save(imagePrename + "capacity-error"); 

  /* Running Time */
  plot.create(PT_LINE_POINT, "", xlabel, "Running Time (seconds)");
  for (unsigned int i=0; i<paramIter.size(); i++) {
    if (witherrorbars) {
      plot.addData_withErrorBars(indVarIter, tme_dev_grid_mu.at(i), tme_dev_grid_std.at(i), "d,g," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData_withErrorBars(indVarIter, tme_hst_grid_mu.at(i), tme_hst_grid_std.at(i), "h,g," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData_withErrorBars(indVarIter, tme_dev_rand_mu.at(i), tme_dev_rand_std.at(i), "d,r," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData_withErrorBars(indVarIter, tme_hst_rand_mu.at(i), tme_hst_rand_std.at(i), "h,r," + paramLegend + Util::to_string((int)paramIter.at(i)));
    } else {
      plot.addData(indVarIter, tme_dev_grid_mu.at(i), "d,g," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData(indVarIter, tme_hst_grid_mu.at(i), "h,g," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData(indVarIter, tme_dev_rand_mu.at(i), "d,r," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData(indVarIter, tme_hst_rand_mu.at(i), "h,r," + paramLegend + Util::to_string((int)paramIter.at(i)));
    }
  }
  plot.save(imagePrename + "running-time"); 

  return 0;
}
