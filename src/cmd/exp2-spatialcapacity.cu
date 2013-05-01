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


/** This program computes the actual and expected spatial capacity of a network versus the number of users (receivers),
 * parameterized by the number of nodes in the network.  Each resulting data point is averaged over sampleCount
 * independent runs.
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

  vector<double> nodeIter  = IterationFunction(10, 10, 1, IFT_LINEAR).getVector();        /* number of nodes to iterate over */
  vector<double> usersIter = IterationFunction(1, 10000, 20, IFT_LOGARITHMIC).getVector();   /* image sizes to iterate over */
  unsigned int sampleCount = 100;                           /* samples to average over for each param and indVar value*/

  unsigned int w  = 500;                    /* (pixels) */
  unsigned int h  = 500;                    /* (pixels) */
  double arenaSideLength = 1000.0;              /* (m) */
  Real widthdB    = 5.0;                    /* visual display width of SINR cells (dB) */
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
  vector<double> indVarIter(usersIter);
  string indVarName = "users";
  string xlabel = "Number of Users";
  string imagePrename = "vs" + indVarName + "-samples" + Util::to_string(sampleCount) + "-"; 



  /* Automatic setup for experiment */
  vd3 cap_act(paramIter.size(), vd2(indVarIter.size(), vd1(sampleCount, 0.0)));  /* Actual Spatial Capacity */

  vd3 cap_exp_Minf(paramIter.size(), vd2(indVarIter.size(), vd1(sampleCount, 0.0)));  /* Exp. Spatial Capacity with M=MANY_USERS*/
  vd3 cap_exp_M(paramIter.size(), vd2(indVarIter.size(), vd1(sampleCount, 0.0)));  /* Exp. Spatial Capacity with M*/

  vd3 tme(paramIter.size(), vd2(indVarIter.size(), vd1(sampleCount, 0.0)));  /* Time */

  uint64_t start, stop;

  /// now you don't have to flush the buffer: http://stackoverflow.com/a/1716621/627517
  setbuf(stdout, NULL);



  for(unsigned int i_param=0; i_param<paramIter.size(); i_param++) {
    /* NOTE: make sure we are assigning elements of paramIter to the right environment variable */
    unsigned int N = paramIter.at(i_param);

    if(paramStatus) {std::cout<<paramName<<i_param<<" "<<std::endl;}

    Network<Real> net_grid(N);
    /* Set up network for CUDA */
    for (unsigned int n = 0; n < N; n++) {
      net_grid.setPosition(n, Point2d(Util::uniform_double(0,arenaSideLength), Util::uniform_double(0,arenaSideLength)));
    }
    Network<Real> net_rand = net_grid;

    /* grid coordinates */
    thrust::device_vector<Point2d> coords_grid(w*h);
    sinr::coordinates::generateGrid(coords_grid, arena, w, h);
    NetworkMetricsDev<Real> nm_grid(&net_grid, &arena, &coords_grid); 

    /* random coordinates */
    thrust::device_vector<Point2d> coords_rand(w*h);
    NetworkMetricsDev<Real> nm_rand(&net_rand, &arena, &coords_rand);


    for(unsigned int i_indVar=0; i_indVar<indVarIter.size(); i_indVar++) {
      /* NOTE: make sure we are assigning elements of indVarIter to the right environment variable */
      unsigned int M = indVarIter.at(i_indVar);

      if(indVarStatus) {std::cout<<"  "<<indVarName<<i_indVar<<" ";}



      for(unsigned int sample=0; sample<sampleCount; sample++) {

        if(sampleStatus) {std::cout<<"s"<<sample<<" ";}

        start = Util::getTimeNS();

        /* grid coordinates */
        cap_exp_Minf.at(i_param).at(i_indVar).at(sample) = nm_grid.computeExpSpatialCapacity(MANY_USERS);
        cap_exp_M.at(i_param).at(i_indVar).at(sample) = nm_grid.computeExpSpatialCapacity(M);

        /* random coordinates */
        unsigned int seed = sample;
        sinr::coordinates::generateRandom(coords_rand, arena, w*h, seed);
        nm_rand = NetworkMetricsDev<Real>(&net_rand, &arena, &coords_rand);
        cap_act.at(i_param).at(i_indVar).at(sample) = nm_rand.computeActSpatialCapacity();

        /* timing */
        stop = Util::getTimeNS();
        tme.at(i_param).at(i_indVar).at(sample) = (stop-start)/double(Util::nanoPerSec);

        if (saveImage && (sample == 0) && (i_indVar == 0)) {
          string imagePostname = paramName + Util::to_string((int)paramIter.at(i_param)) + "-" + indVarName + Util::to_string((int)indVarIter.at(i_indVar)) + "-" + "s" + Util::to_string((int)sample);

          thrust::device_vector<uchar4> rgba(w*h);
          const thrust::device_vector<Real> *maxsinr = nm_grid.computeMapMaxSINR();
          sinr::visualize::grayscaledB(*maxsinr, rgba, net_grid.getSINRThresholddB()-widthdB, net_grid.getSINRThresholddB());
          string imagePath = OmanDirs::images() + "/" + imagePrename + imagePostname + ".bmp";
          sinr::visualize::outputBMP(rgba, w, h, imagePath);
        }
      }
      if (indVarStatus) {std::cout<<std::endl;}
    }
  }

  if(paramStatus || indVarStatus || sampleStatus) {
    std::cout << std::endl;
  }


  /// Here we use the computed coverage, sinrmax, and capacity to determine the "error" in the lower resolution images.

  /* metrics */
  vd2 cap_act_mu(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 cap_exp_Minf_mu(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 cap_exp_M_mu(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 tme_mu(paramIter.size(), vd1(indVarIter.size(), 0));

  vd2 cap_act_std(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 cap_exp_Minf_std(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 cap_exp_M_std(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 tme_std(paramIter.size(), vd1(indVarIter.size(), 0));

  vd2 cap_act_min(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 cap_act_max(paramIter.size(), vd1(indVarIter.size(), 0));

  /* relative error */
  vd2 cap_exp_Minf_error_mu(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 cap_act_error_mu(paramIter.size(), vd1(indVarIter.size(), 0));

  vd2 cap_exp_Minf_error_std(paramIter.size(), vd1(indVarIter.size(), 0));
  vd2 cap_act_error_std(paramIter.size(), vd1(indVarIter.size(), 0)); 

  for(unsigned int i=0; i<paramIter.size(); i++) {
    for(unsigned int j=0; j<indVarIter.size(); j++) {

      /* metrics mean and std dev */
      cap_act_mu.at(i).at(j) = Util::mean(cap_act.at(i).at(j)); 
      cap_exp_Minf_mu.at(i).at(j) = Util::mean(cap_exp_Minf.at(i).at(j)); 
      cap_exp_M_mu.at(i).at(j) = Util::mean(cap_exp_M.at(i).at(j)); 
      tme_mu.at(i).at(j) = Util::mean(tme.at(i).at(j)); 

      cap_act_std.at(i).at(j) = Util::stddev(cap_act.at(i).at(j)); 
      cap_exp_Minf_std.at(i).at(j) = Util::stddev(cap_exp_Minf.at(i).at(j)); 
      cap_exp_M_std.at(i).at(j) = Util::stddev(cap_exp_M.at(i).at(j)); 
      tme_std.at(i).at(j) = Util::stddev(tme.at(i).at(j)); 

      cap_act_min.at(i).at(j) = Util::min(cap_act.at(i).at(j));  
      cap_act_max.at(i).at(j) = Util::max(cap_act.at(i).at(j)); 

      /* error */
      cap_exp_Minf_error_mu.at(i).at(j) = Util::mean(Util::relErr(cap_exp_Minf.at(i).at(j), cap_exp_M.at(i).at(j)));
      cap_act_error_mu.at(i).at(j) = Util::mean(Util::relErr(cap_act.at(i).at(j), cap_exp_M.at(i).at(j)));

      cap_exp_Minf_error_std.at(i).at(j) = Util::stddev(Util::relErr(cap_exp_Minf.at(i).at(j), cap_exp_M.at(i).at(j)));
      cap_act_error_std.at(i).at(j) = Util::stddev(Util::relErr(cap_act.at(i).at(j), cap_exp_M.at(i).at(j)));

    }
  }


  Plot plot;
  plot.constants.logscale_x = true;
  //plot.constants.logscale_y = false;

  /* Capacity */
  plot.create(PT_LINE_POINT, "", xlabel, "Spatial Capacity (bps/m^2)");
  for (unsigned int i=0; i<paramIter.size(); i++) {
    if (witherrorbars) {
      plot.addData_withCandleSticks(indVarIter, cap_act_mu.at(i), 
                                    cap_act_min.at(i), Util::sub(cap_act_mu.at(i), cap_act_std.at(i)), 
                                    Util::add(cap_act_mu.at(i), cap_act_std.at(i)), cap_act_max.at(i), 
                                    "A," + paramLegend + Util::to_string((int)paramIter.at(i))); 
      plot.addData_withErrorBars(indVarIter, cap_exp_Minf_mu.at(i), cap_exp_Minf_std.at(i), "E,M=Inf," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData_withErrorBars(indVarIter, cap_exp_M_mu.at(i),    cap_exp_M_std.at(i),    "E," + paramLegend + Util::to_string((int)paramIter.at(i)));
    } else {
      plot.addData(indVarIter, cap_act_mu.at(i),      "A," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData(indVarIter, cap_exp_Minf_mu.at(i), "E,M=Inf," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData(indVarIter, cap_exp_M_mu.at(i),    "E," + paramLegend + Util::to_string((int)paramIter.at(i)));
    }
  }
  plot.save(imagePrename + "capacity"); 

  plot.create(PT_LINE_POINT, "", xlabel, "Relative Error against Expected Spatial Capacity");
  for (unsigned int i=0; i<paramIter.size(); i++) {
    if (witherrorbars) {
      plot.addData_withErrorBars(indVarIter, cap_act_error_mu.at(i),      cap_act_error_std.at(i),      "A," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData_withErrorBars(indVarIter, cap_exp_Minf_error_mu.at(i), cap_exp_Minf_error_std.at(i), "E,M=Inf," + paramLegend + Util::to_string((int)paramIter.at(i)));
    } else {
      plot.addData(indVarIter, cap_act_error_mu.at(i),      "A," + paramLegend + Util::to_string((int)paramIter.at(i)));
      plot.addData(indVarIter, cap_exp_Minf_error_mu.at(i), "E,M=Inf," + paramLegend + Util::to_string((int)paramIter.at(i)));
    }
  }
  plot.save(imagePrename + "capacity-error"); 

  /* Running Time */
  plot.create(PT_LINE_POINT, "", xlabel, "Running Time (seconds)");
  for (unsigned int i=0; i<paramIter.size(); i++) {
    if (witherrorbars) {
      plot.addData_withErrorBars(indVarIter, tme_mu.at(i), tme_std.at(i), "" + paramLegend + Util::to_string((int)paramIter.at(i)));
    } else {
      plot.addData(indVarIter, tme_mu.at(i), "" + paramLegend + Util::to_string((int)paramIter.at(i)));
    }
  }
  plot.save(imagePrename + "running-time"); 



  return 0;

}
