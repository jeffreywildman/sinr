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

typedef thrust::tuple<double,double> Point2d_dbl;
typedef thrust::tuple<float,float> Point2d_flt;


/** This program computes averaged metrics versus the size of the network, parameterized over the number of
 * data points used to sample each metric over a 2d arena.  Each resulting data point is averaged over sampleCount
 * independent runs.
 *
 * A main comparison is made between using floats or doubles as the underlying data type in the averaged metric
 * calculations, as GPUs seem to be faster at evaluating single-precision floating point operations.
 */
int main(int argc __attribute__((unused)), char **argv __attribute__((unused))) {

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

  vector<double> nodeIter      = IterationFunction(10, 50, 5).getVector();        /* number of nodes to iterate over */
  vector<double> sidePixelIter = IterationFunction(500, 500, 1).getVector();   /* image sizes to iterate over */
  unsigned int sampleCount = 100;                           /* samples to average over for each param and indVar value*/

  double arenaSideLength = 1000.0;              /* (m) */
  double widthdB    = 5.0;                    /* visual display width of SINR cells (dB) */
  Arena2d<double> arena_dbl(arenaSideLength);
  Arena2d<float>  arena_flt(arenaSideLength);

  bool paramStatus = true;
  bool indVarStatus = true;
  bool sampleStatus = true;
  bool saveImage = true;
  bool witherrorbars = true;


  /* Map iterators to parameter and independent variables 
   * Note: remember this mapping to make the right variable assignments inside the nested for-loops */ 
  vector<double> paramIter(sidePixelIter);
  string paramName = "pixels";
  string paramLegend = "p=";
  vector<double> indVarIter(nodeIter);
  string indVarName = "nodes";
  string xlabel = "Number of Nodes";
  string imagePrename = "vs" + indVarName + "-samples" + Util::to_string(sampleCount) + "-"; 



  /* Automatic setup for experiment */
  vd3 cov_flt_grid(paramIter.size(), vd2(indVarIter.size(), vd1(sampleCount, 0.0)));  /* Coverage */
  vd3 snr_flt_grid(paramIter.size(), vd2(indVarIter.size(), vd1(sampleCount, 0.0)));  /* SINR Max */
  vd3 cap_flt_grid(paramIter.size(), vd2(indVarIter.size(), vd1(sampleCount, 0.0)));  /* Capacity */
  vd3 tme_flt_grid(paramIter.size(), vd2(indVarIter.size(), vd1(sampleCount, 0.0)));  /* Time */

  vd3 cov_dbl_grid(paramIter.size(), vd2(indVarIter.size(), vd1(sampleCount, 0.0)));  /* Coverage */
  vd3 snr_dbl_grid(paramIter.size(), vd2(indVarIter.size(), vd1(sampleCount, 0.0)));  /* SINR Max */
  vd3 cap_dbl_grid(paramIter.size(), vd2(indVarIter.size(), vd1(sampleCount, 0.0)));  /* Capacity */
  vd3 tme_dbl_grid(paramIter.size(), vd2(indVarIter.size(), vd1(sampleCount, 0.0)));  /* Time */

  vd3 cov_flt_rand(paramIter.size(), vd2(indVarIter.size(), vd1(sampleCount, 0.0)));  /* Coverage */
  vd3 snr_flt_rand(paramIter.size(), vd2(indVarIter.size(), vd1(sampleCount, 0.0)));  /* SINR Max */
  vd3 cap_flt_rand(paramIter.size(), vd2(indVarIter.size(), vd1(sampleCount, 0.0)));  /* Capacity */
  vd3 tme_flt_rand(paramIter.size(), vd2(indVarIter.size(), vd1(sampleCount, 0.0)));  /* Time */

  vd3 cov_dbl_rand(paramIter.size(), vd2(indVarIter.size(), vd1(sampleCount, 0.0)));  /* Coverage */
  vd3 snr_dbl_rand(paramIter.size(), vd2(indVarIter.size(), vd1(sampleCount, 0.0)));  /* SINR Max */
  vd3 cap_dbl_rand(paramIter.size(), vd2(indVarIter.size(), vd1(sampleCount, 0.0)));  /* Capacity */
  vd3 tme_dbl_rand(paramIter.size(), vd2(indVarIter.size(), vd1(sampleCount, 0.0)));  /* Time */

  uint64_t start, stop;

  /// now you don't have to flush the buffer: http://stackoverflow.com/a/1716621/627517
  setbuf(stdout, NULL);



  for(unsigned int i_param=0; i_param<paramIter.size(); i_param++) {
    /* NOTE: make sure we are assigning elements of paramIter to the right environment variable */
    unsigned int sidePixelCount = paramIter.at(i_param);

    if(paramStatus) {std::cout<<paramName<<i_param<<" "<<std::endl;}



    for(unsigned int i_indVar=0; i_indVar<indVarIter.size(); i_indVar++) {
      /* NOTE: make sure we are assigning elements of indVarIter to the right environment variable */
      unsigned int N = indVarIter.at(i_indVar);

      if(indVarStatus) {std::cout<<"  "<<indVarName<<i_indVar<<" ";}

      Network<float> net_flt_grid(N);
      Network<float> net_flt_rand = net_flt_grid; 
      
      Network<double> net_dbl_grid(N);
      Network<double> net_dbl_rand = net_dbl_grid;



      for(unsigned int sample=0; sample<sampleCount; sample++) {

        if(sampleStatus) {std::cout<<"s"<<sample<<" ";}

        /* Set up network for CUDA */
        for (unsigned int n = 0; n < N; n++) {
          Point2d_dbl p(Util::uniform_double(0,arenaSideLength), Util::uniform_double(0,arenaSideLength));
          net_dbl_grid.setPosition(n,p);
          net_dbl_rand.setPosition(n,p);
          net_flt_grid.setPosition(n,p);
          net_flt_rand.setPosition(n,p);
        }

        /* float-based grid */
        start = Util::getTimeNS();
        thrust::device_vector<Point2d_flt> coords_flt_grid(sidePixelCount*sidePixelCount);
        sinr::coordinates::generateGrid(coords_flt_grid, 
                                        arena_flt,
                                        sidePixelCount, 
                                        sidePixelCount);
        NetworkMetricsDev<float> nm_flt_grid(&net_flt_grid, &arena_flt, &coords_flt_grid);
        cov_flt_grid.at(i_param).at(i_indVar).at(sample) = nm_flt_grid.computeAvgMaxCoverage();
        snr_flt_grid.at(i_param).at(i_indVar).at(sample) = nm_flt_grid.computeAvgMaxSINR();
        cap_flt_grid.at(i_param).at(i_indVar).at(sample) = nm_flt_grid.computeAvgMaxCapacity();
        stop = Util::getTimeNS();
        tme_flt_grid.at(i_param).at(i_indVar).at(sample) = (stop-start)/double(Util::nanoPerSec);

        /* double-based grid */
        start = Util::getTimeNS();
        thrust::device_vector<Point2d_dbl> coords_dbl_grid(sidePixelCount*sidePixelCount);
        sinr::coordinates::generateGrid(coords_dbl_grid, 
                                        arena_dbl,
                                        sidePixelCount, 
                                        sidePixelCount);
        NetworkMetricsDev<double> nm_dbl_grid(&net_dbl_grid, &arena_dbl, &coords_dbl_grid);
        cov_dbl_grid.at(i_param).at(i_indVar).at(sample) = nm_dbl_grid.computeAvgMaxCoverage();
        snr_dbl_grid.at(i_param).at(i_indVar).at(sample) = nm_dbl_grid.computeAvgMaxSINR();
        cap_dbl_grid.at(i_param).at(i_indVar).at(sample) = nm_dbl_grid.computeAvgMaxCapacity();
        stop = Util::getTimeNS();
        tme_dbl_grid.at(i_param).at(i_indVar).at(sample) = (stop-start)/double(Util::nanoPerSec);

        /* float-based random */
        start = Util::getTimeNS();
        thrust::device_vector<Point2d_flt> coords_flt_rand(sidePixelCount*sidePixelCount);
        sinr::coordinates::generateRandom(coords_flt_rand, 
                                          arena_flt,
                                          sidePixelCount*sidePixelCount,
                                          sample);
        NetworkMetricsDev<float> nm_flt_rand(&net_flt_rand, &arena_flt, &coords_flt_rand);
        cov_flt_rand.at(i_param).at(i_indVar).at(sample) = nm_flt_rand.computeAvgMaxCoverage();
        snr_flt_rand.at(i_param).at(i_indVar).at(sample) = nm_flt_rand.computeAvgMaxSINR();
        cap_flt_rand.at(i_param).at(i_indVar).at(sample) = nm_flt_rand.computeAvgMaxCapacity();
        stop = Util::getTimeNS();
        tme_flt_rand.at(i_param).at(i_indVar).at(sample) = (stop-start)/double(Util::nanoPerSec);

        /* double-based random */
        start = Util::getTimeNS();
        thrust::device_vector<Point2d_dbl> coords_dbl_rand(sidePixelCount*sidePixelCount);
        sinr::coordinates::generateRandom(coords_dbl_rand, 
                                          arena_dbl,
                                          sidePixelCount*sidePixelCount,
                                          sample);
        NetworkMetricsDev<double> nm_dbl_rand(&net_dbl_rand, &arena_dbl, &coords_dbl_rand);
        cov_dbl_rand.at(i_param).at(i_indVar).at(sample) = nm_dbl_rand.computeAvgMaxCoverage();
        snr_dbl_rand.at(i_param).at(i_indVar).at(sample) = nm_dbl_rand.computeAvgMaxSINR();
        cap_dbl_rand.at(i_param).at(i_indVar).at(sample) = nm_dbl_rand.computeAvgMaxCapacity();
        stop = Util::getTimeNS();
        tme_dbl_rand.at(i_param).at(i_indVar).at(sample) = (stop-start)/double(Util::nanoPerSec);

        if (saveImage && (sample == 0)) {
          string imagePostname = paramName + Util::to_string((int)paramIter.at(i_param)) + "-" + indVarName + Util::to_string((int)indVarIter.at(i_indVar)) + "-" + "s" + Util::to_string((int)sample);

          thrust::device_vector<uchar4> rgba_flt(sidePixelCount*sidePixelCount);
          const thrust::device_vector<float> *maxsinr_flt = nm_flt_grid.computeMapMaxSINR();
          sinr::visualize::grayscaledB(*maxsinr_flt, rgba_flt, net_flt_grid.getSINRThresholddB()-widthdB, net_flt_grid.getSINRThresholddB());
          string imagePath_flt = OmanDirs::images() + "/" + imagePrename + imagePostname + "-flt.bmp";
          sinr::visualize::outputBMP(rgba_flt, sidePixelCount, sidePixelCount, imagePath_flt);

          thrust::device_vector<uchar4> rgba_dbl(sidePixelCount*sidePixelCount);
          const thrust::device_vector<double> *maxsinr_dbl = nm_dbl_grid.computeMapMaxSINR();
          sinr::visualize::grayscaledB(*maxsinr_dbl, rgba_dbl, net_flt_grid.getSINRThresholddB()-widthdB, net_flt_grid.getSINRThresholddB());
          string imagePath_dbl = OmanDirs::images() + "/" + imagePrename + imagePostname + "-dbl.bmp";
          sinr::visualize::outputBMP(rgba_dbl, sidePixelCount, sidePixelCount, imagePath_dbl);
        }
      }
      if (indVarStatus) {std::cout<<std::endl;}
    }
  }

  if(paramStatus || indVarStatus || sampleStatus) {
    std::cout << std::endl;
  }


  /// Here we use the computed coverage, sinrmax, and capacity to determine the "error" in the lower resolution images.

  /* float-based grid samples */
  vd2 cov_flt_grid_mu(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 snr_flt_grid_mu(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 cap_flt_grid_mu(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 tme_flt_grid_mu(paramIter.size(), vd1(indVarIter.size(), 0));

  vd2 cov_flt_grid_std(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 snr_flt_grid_std(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 cap_flt_grid_std(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 tme_flt_grid_std(paramIter.size(), vd1(indVarIter.size(), 0));

  /* double-based grid samples */
  vd2 cov_dbl_grid_mu(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 snr_dbl_grid_mu(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 cap_dbl_grid_mu(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 tme_dbl_grid_mu(paramIter.size(), vd1(indVarIter.size(), 0));

  vd2 cov_dbl_grid_std(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 snr_dbl_grid_std(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 cap_dbl_grid_std(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 tme_dbl_grid_std(paramIter.size(), vd1(indVarIter.size(), 0));

  /* float-based random samples */
  vd2 cov_flt_rand_mu(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 snr_flt_rand_mu(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 cap_flt_rand_mu(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 tme_flt_rand_mu(paramIter.size(), vd1(indVarIter.size(), 0));

  vd2 cov_flt_rand_std(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 snr_flt_rand_std(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 cap_flt_rand_std(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 tme_flt_rand_std(paramIter.size(), vd1(indVarIter.size(), 0));

  /* double-based random samples */
  vd2 cov_dbl_rand_mu(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 snr_dbl_rand_mu(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 cap_dbl_rand_mu(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 tme_dbl_rand_mu(paramIter.size(), vd1(indVarIter.size(), 0));

  vd2 cov_dbl_rand_std(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 snr_dbl_rand_std(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 cap_dbl_rand_std(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 tme_dbl_rand_std(paramIter.size(), vd1(indVarIter.size(), 0));

  /* Error Measurements - Grid */
  vd2 cov_grid_error_mu(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 snr_grid_error_mu(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 cap_grid_error_mu(paramIter.size(), vd1(indVarIter.size(), 0));

  vd2 cov_grid_error_std(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 snr_grid_error_std(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 cap_grid_error_std(paramIter.size(), vd1(indVarIter.size(), 0));

  /* Error Measurements - Random */
  vd2 cov_rand_error_mu(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 snr_rand_error_mu(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 cap_rand_error_mu(paramIter.size(), vd1(indVarIter.size(), 0));

  vd2 cov_rand_error_std(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 snr_rand_error_std(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 cap_rand_error_std(paramIter.size(), vd1(indVarIter.size(), 0));


  for(unsigned int i=0; i<paramIter.size(); i++) {
    for(unsigned int j=0; j<indVarIter.size(); j++) {

      /* float-based grid samples */
      cov_flt_grid_mu.at(i).at(j) = Util::mean(cov_flt_grid.at(i).at(j));
      snr_flt_grid_mu.at(i).at(j) = Util::mean(snr_flt_grid.at(i).at(j));
      cap_flt_grid_mu.at(i).at(j) = Util::mean(cap_flt_grid.at(i).at(j)); 
      tme_flt_grid_mu.at(i).at(j) = Util::mean(tme_flt_grid.at(i).at(j)); 

      cov_flt_grid_std.at(i).at(j) = Util::stddev(cov_flt_grid.at(i).at(j));
      snr_flt_grid_std.at(i).at(j) = Util::stddev(snr_flt_grid.at(i).at(j));
      cap_flt_grid_std.at(i).at(j) = Util::stddev(cap_flt_grid.at(i).at(j)); 
      tme_flt_grid_std.at(i).at(j) = Util::stddev(tme_flt_grid.at(i).at(j)); 

      /* double-based grid samples */
      cov_dbl_grid_mu.at(i).at(j) = Util::mean(cov_dbl_grid.at(i).at(j));
      snr_dbl_grid_mu.at(i).at(j) = Util::mean(snr_dbl_grid.at(i).at(j));
      cap_dbl_grid_mu.at(i).at(j) = Util::mean(cap_dbl_grid.at(i).at(j)); 
      tme_dbl_grid_mu.at(i).at(j) = Util::mean(tme_dbl_grid.at(i).at(j)); 

      cov_dbl_grid_std.at(i).at(j) = Util::stddev(cov_dbl_grid.at(i).at(j));
      snr_dbl_grid_std.at(i).at(j) = Util::stddev(snr_dbl_grid.at(i).at(j));
      cap_dbl_grid_std.at(i).at(j) = Util::stddev(cap_dbl_grid.at(i).at(j)); 
      tme_dbl_grid_std.at(i).at(j) = Util::stddev(tme_dbl_grid.at(i).at(j)); 

      /* grid-based absolute error */
      cov_grid_error_mu.at(i).at(j) = Util::mean(Util::absErr(cov_flt_grid.at(i).at(j), cov_dbl_grid.at(i).at(j)));
      snr_grid_error_mu.at(i).at(j) = Util::mean(Util::absErr(snr_flt_grid.at(i).at(j), snr_dbl_grid.at(i).at(j)));
      cap_grid_error_mu.at(i).at(j) = Util::mean(Util::absErr(cap_flt_grid.at(i).at(j), cap_dbl_grid.at(i).at(j)));

      cov_grid_error_std.at(i).at(j) = Util::stddev(Util::absErr(cov_flt_grid.at(i).at(j), cov_dbl_grid.at(i).at(j)));
      snr_grid_error_std.at(i).at(j) = Util::stddev(Util::absErr(snr_flt_grid.at(i).at(j), snr_dbl_grid.at(i).at(j)));
      cap_grid_error_std.at(i).at(j) = Util::stddev(Util::absErr(cap_flt_grid.at(i).at(j), cap_dbl_grid.at(i).at(j)));

      /* float-based random samples */
      cov_flt_rand_mu.at(i).at(j) = Util::mean(cov_flt_rand.at(i).at(j));
      snr_flt_rand_mu.at(i).at(j) = Util::mean(snr_flt_rand.at(i).at(j));
      cap_flt_rand_mu.at(i).at(j) = Util::mean(cap_flt_rand.at(i).at(j)); 
      tme_flt_rand_mu.at(i).at(j) = Util::mean(tme_flt_rand.at(i).at(j)); 

      cov_flt_rand_std.at(i).at(j) = Util::stddev(cov_flt_rand.at(i).at(j));
      snr_flt_rand_std.at(i).at(j) = Util::stddev(snr_flt_rand.at(i).at(j));
      cap_flt_rand_std.at(i).at(j) = Util::stddev(cap_flt_rand.at(i).at(j)); 
      tme_flt_rand_std.at(i).at(j) = Util::stddev(tme_flt_rand.at(i).at(j)); 

      /* double-based random samples */
      cov_dbl_rand_mu.at(i).at(j) = Util::mean(cov_dbl_rand.at(i).at(j));
      snr_dbl_rand_mu.at(i).at(j) = Util::mean(snr_dbl_rand.at(i).at(j));
      cap_dbl_rand_mu.at(i).at(j) = Util::mean(cap_dbl_rand.at(i).at(j)); 
      tme_dbl_rand_mu.at(i).at(j) = Util::mean(tme_dbl_rand.at(i).at(j)); 

      cov_dbl_rand_std.at(i).at(j) = Util::stddev(cov_dbl_rand.at(i).at(j));
      snr_dbl_rand_std.at(i).at(j) = Util::stddev(snr_dbl_rand.at(i).at(j));
      cap_dbl_rand_std.at(i).at(j) = Util::stddev(cap_dbl_rand.at(i).at(j)); 
      tme_dbl_rand_std.at(i).at(j) = Util::stddev(tme_dbl_rand.at(i).at(j)); 

      /* random-based absolute error */
      cov_rand_error_mu.at(i).at(j) = Util::mean(Util::absErr(cov_flt_rand.at(i).at(j), cov_dbl_rand.at(i).at(j)));
      snr_rand_error_mu.at(i).at(j) = Util::mean(Util::absErr(snr_flt_rand.at(i).at(j), snr_dbl_rand.at(i).at(j)));
      cap_rand_error_mu.at(i).at(j) = Util::mean(Util::absErr(cap_flt_rand.at(i).at(j), cap_dbl_rand.at(i).at(j)));

      cov_rand_error_std.at(i).at(j) = Util::stddev(Util::absErr(cov_flt_rand.at(i).at(j), cov_dbl_rand.at(i).at(j)));
      snr_rand_error_std.at(i).at(j) = Util::stddev(Util::absErr(snr_flt_rand.at(i).at(j), snr_dbl_rand.at(i).at(j)));
      cap_rand_error_std.at(i).at(j) = Util::stddev(Util::absErr(cap_flt_rand.at(i).at(j), cap_dbl_rand.at(i).at(j)));
    } 
  }


  Plot plot;
  //plot.constants.logscale_x = true;
  //plot.constants.logscale_y = false;

  /* Coverage */
  plot.create(PT_LINE_POINT, "", xlabel, "Coverage");
  for (unsigned int i=0; i<paramIter.size(); i++) {
    if (witherrorbars) {
      plot.addData_withErrorBars(indVarIter, cov_flt_grid_mu.at(i), cov_flt_grid_std.at(i), "f,g," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData_withErrorBars(indVarIter, cov_dbl_grid_mu.at(i), cov_dbl_grid_std.at(i), "d,g," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData_withErrorBars(indVarIter, cov_flt_rand_mu.at(i), cov_flt_rand_std.at(i), "f,r," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData_withErrorBars(indVarIter, cov_dbl_rand_mu.at(i), cov_dbl_rand_std.at(i), "d,r," + paramLegend + Util::to_string((int)paramIter.at(i)));
    } else {
      plot.addData(indVarIter, cov_flt_grid_mu.at(i), "f,g," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData(indVarIter, cov_dbl_grid_mu.at(i), "d,g," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData(indVarIter, cov_flt_rand_mu.at(i), "f,r," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData(indVarIter, cov_dbl_rand_mu.at(i), "d,r," + paramLegend + Util::to_string((int)paramIter.at(i)));
    }
  }
  plot.save(imagePrename + "coverage"); 

  plot.create(PT_LINE_POINT, "", xlabel, "Absolute Error of Coverage");
  for (unsigned int i=0; i<paramIter.size(); i++) {
    if (witherrorbars) {
      plot.addData_withErrorBars(indVarIter, cov_grid_error_mu.at(i), cov_grid_error_std.at(i), "g," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData_withErrorBars(indVarIter, cov_rand_error_mu.at(i), cov_rand_error_std.at(i), "r," + paramLegend + Util::to_string((int)paramIter.at(i)));
    } else {
      plot.addData(indVarIter, cov_grid_error_mu.at(i), "g," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData(indVarIter, cov_rand_error_mu.at(i), "r," + paramLegend + Util::to_string((int)paramIter.at(i)));
    }
  }
  plot.save(imagePrename + "coverage-error"); 

  /* SINR Max */
  plot.create(PT_LINE_POINT, "", xlabel, "Average Max SINR (W/W)");
  for (unsigned int i=0; i<paramIter.size(); i++) {
    if (witherrorbars) {
      plot.addData_withErrorBars(indVarIter, snr_flt_grid_mu.at(i), snr_flt_grid_std.at(i), "f,g," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData_withErrorBars(indVarIter, snr_dbl_grid_mu.at(i), snr_dbl_grid_std.at(i), "d,g," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData_withErrorBars(indVarIter, snr_flt_rand_mu.at(i), snr_flt_rand_std.at(i), "f,r," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData_withErrorBars(indVarIter, snr_dbl_rand_mu.at(i), snr_dbl_rand_std.at(i), "d,r," + paramLegend + Util::to_string((int)paramIter.at(i)));
    } else {
      plot.addData(indVarIter, snr_flt_grid_mu.at(i), "f,g," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData(indVarIter, snr_dbl_grid_mu.at(i), "d,g," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData(indVarIter, snr_flt_rand_mu.at(i), "f,r," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData(indVarIter, snr_dbl_rand_mu.at(i), "d,r," + paramLegend + Util::to_string((int)paramIter.at(i)));
    }
  }
  plot.save(imagePrename + "sinrmax"); 

  plot.create(PT_LINE_POINT, "", xlabel, "Absolute Error of Average Max SINR");
  for (unsigned int i=0; i<paramIter.size(); i++) {
    if (witherrorbars) {
      plot.addData_withErrorBars(indVarIter, snr_grid_error_mu.at(i), snr_grid_error_std.at(i), "g," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData_withErrorBars(indVarIter, snr_rand_error_mu.at(i), snr_rand_error_std.at(i), "r," + paramLegend + Util::to_string((int)paramIter.at(i)));
    } else {
      plot.addData(indVarIter, snr_grid_error_mu.at(i), "g," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData(indVarIter, snr_rand_error_mu.at(i), "r," + paramLegend + Util::to_string((int)paramIter.at(i)));
    }
  }
  plot.save(imagePrename + "sinrmax-error"); 

  /* Capacity */
  plot.create(PT_LINE_POINT, "", xlabel, "Average Max Capacity (bps)");
  for (unsigned int i=0; i<paramIter.size(); i++) {
    if (witherrorbars) {
      plot.addData_withErrorBars(indVarIter, cap_flt_grid_mu.at(i), cap_flt_grid_std.at(i), "f,g," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData_withErrorBars(indVarIter, cap_dbl_grid_mu.at(i), cap_dbl_grid_std.at(i), "d,g," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData_withErrorBars(indVarIter, cap_flt_rand_mu.at(i), cap_flt_rand_std.at(i), "f,r," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData_withErrorBars(indVarIter, cap_dbl_rand_mu.at(i), cap_dbl_rand_std.at(i), "d,r," + paramLegend + Util::to_string((int)paramIter.at(i)));
    } else {
      plot.addData(indVarIter, cap_flt_grid_mu.at(i), "f,g," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData(indVarIter, cap_dbl_grid_mu.at(i), "d,g," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData(indVarIter, cap_flt_rand_mu.at(i), "f,r," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData(indVarIter, cap_dbl_rand_mu.at(i), "d,r," + paramLegend + Util::to_string((int)paramIter.at(i)));
    }
  }
  plot.save(imagePrename + "capacity"); 

  plot.create(PT_LINE_POINT, "", xlabel, "Absolute Error of Average Max Capacity");
  for (unsigned int i=0; i<paramIter.size(); i++) {
    if (witherrorbars) {
      plot.addData_withErrorBars(indVarIter, cap_grid_error_mu.at(i), cap_grid_error_std.at(i), "g," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData_withErrorBars(indVarIter, cap_rand_error_mu.at(i), cap_rand_error_std.at(i), "r," + paramLegend + Util::to_string((int)paramIter.at(i)));
    } else {
      plot.addData(indVarIter, cap_grid_error_mu.at(i), "g," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData(indVarIter, cap_rand_error_mu.at(i), "r," + paramLegend + Util::to_string((int)paramIter.at(i)));
    }
  }
  plot.save(imagePrename + "capacity-error"); 

  /* Running Time */
  plot.create(PT_LINE_POINT, "", xlabel, "Running Time (seconds)");
  for (unsigned int i=0; i<paramIter.size(); i++) {
    if (witherrorbars) {
      plot.addData_withErrorBars(indVarIter, tme_flt_grid_mu.at(i), tme_flt_grid_std.at(i), "f,g," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData_withErrorBars(indVarIter, tme_dbl_grid_mu.at(i), tme_dbl_grid_std.at(i), "d,g," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData_withErrorBars(indVarIter, tme_flt_rand_mu.at(i), tme_flt_rand_std.at(i), "f,r," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData_withErrorBars(indVarIter, tme_dbl_rand_mu.at(i), tme_dbl_rand_std.at(i), "d,r," + paramLegend + Util::to_string((int)paramIter.at(i)));
    } else {
      plot.addData(indVarIter, tme_flt_grid_mu.at(i), "f,g," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData(indVarIter, tme_dbl_grid_mu.at(i), "d,g," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData(indVarIter, tme_flt_rand_mu.at(i), "f,r," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData(indVarIter, tme_dbl_rand_mu.at(i), "d,r," + paramLegend + Util::to_string((int)paramIter.at(i)));
    }
  }
  plot.save(imagePrename + "running-time"); 

  return 0;
}
