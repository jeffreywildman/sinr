#include <iostream>
#include <vector>
#include <math.h>

#include <oman/general/omandirs.h>            /* non c++11 */

#include <sinr/network.h>
#include <sinr/coordinates.cuh>
#include <sinr/networkmetrics.cuh>
#include <sinr/visualizer.cuh>
#include <sinr/util.h>
#include <sinr/gui.h>
#include <sinr/types.h>
#include <sinr/optioniterator.h>

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
  unsigned int w          = 300;                    /* (pixels) */
  unsigned int h          = 300;                    /* (pixels) */
  unsigned int N          = 4;                      /* number of nodes */
  unsigned int nchannels  = 2;                      /* number of channels */
  channelType_t cType     = CT_TIMESLOT;            /* channel type */
  optionComputeMethod_t ocm = OCM_BASEB;         /* option compute method */
  double arenaSideLength  = 1000.0;                 /* (m) */
  Real widthdB            = 5.0;                    /* visual display width of SINR cells (dB) */
  unsigned int sampleCount = 1;

  bool snapshots = true;


  unsigned int npatterns  = 2;
  vector<RadPattern<Real> > patterns(npatterns);
  patterns.at(0) = RadPattern<Real>(RPT_ROSEPETAL, 2.0*M_PI/3.0, 1.0);
  patterns.at(1) = RadPattern<Real>(RPT_ROSEPETAL, M_PI/3.0, 1.0);

  Arena2d<Real> arena(arenaSideLength);
  Network<Real> net(N);

  /* coordinates to compute metrics over */
  thrust::device_vector<Point2d> coords(w*h);
  sinr::coordinates::generateGrid(coords, arena, w, h); 
  NetworkMetricsDev<Real> nm(&net, &arena, &coords);

  /* common variables for display */
  const thrust::device_vector<Real> *maxsinr;
  const thrust::device_vector<nid_t> *assoczone;
  thrust::device_vector<uchar4> rgba(coords.size());


  /* Automatic setup for experiment */
  uint64_t start, stop;
  vector<Real> cap_1b1p(sampleCount, 0.0);
  vector<Real> tme_1b1p(sampleCount, 0.0);

  vector<Real> cap_2b1p(sampleCount, 0.0);
  vector<Real> tme_2b1p(sampleCount, 0.0);

  vector<Real> cap_1b2p(sampleCount, 0.0);
  vector<Real> tme_1b2p(sampleCount, 0.0);

  vector<Real> cap_2b2p(sampleCount, 0.0);
  vector<Real> tme_2b2p(sampleCount, 0.0);



  /*** CASE 1: 1 band, 1 mode ***/
  net.setNumberChannels(1);
  net.setChannelType(cType);
  net.setChannel(vector<cid_t>(N,0)); 
  net.setRadPattern(vector<RadPattern<Real> >(N,patterns.at(0)));

  for (unsigned int s = 0; s < sampleCount; s++) {

    cout<<"1b1p, s: " + Util::to_string(s)<<endl;

    start = Util::getTimeNS();

    //    net.setPosition(0,Point2d(250,250));
    //    net.setPosition(1,Point2d(250,750));
    //    net.setPosition(2,Point2d(750,250));
    //    net.setPosition(3,Point2d(750,750));
    //    net.setOrientation(vector<Real>(N,0.0));

    /* set random node positions and random orientations */
    for (unsigned int n = 0; n < N; n++) {
      net.setPosition(n,Point2d(Util::uniform_double(0,arenaSideLength),Util::uniform_double(0,arenaSideLength)));
      net.setOrientation(n,Util::uniform_double(0,2*M_PI));
    }

    /* record expected spatial capacity */
    cap_1b1p.at(s) = nm.computeExpSpatialCapacity2(MANY_USERS, DensityUniform2d<Real>(arena));
    stop = Util::getTimeNS();
    tme_1b1p.at(s) = (stop-start)/double(Util::nanoPerSec);
  }



  /*** CASE 2: 2 bands, 1 mode ***/
  net.setNumberChannels(2);
  net.setChannelType(cType);
  net.setRadPattern(vector<RadPattern<Real> >(N,patterns.at(0)));

  for (unsigned int s = 0; s < sampleCount; s++) {

    cout<<"2b1p, s: " + Util::to_string(s)<<endl;

    start = Util::getTimeNS();

    //    net.setPosition(0,Point2d(250,250));
    //    net.setPosition(1,Point2d(250,750));
    //    net.setPosition(2,Point2d(750,250));
    //    net.setPosition(3,Point2d(750,750));
    //    net.setOrientation(vector<Real>(N,0.0));

    /* set random node positions and random orientations */
    for (unsigned int n = 0; n < N; n++) {
      net.setPosition(n,Point2d(Util::uniform_double(0,arenaSideLength),Util::uniform_double(0,arenaSideLength)));
      net.setOrientation(n,Util::uniform_double(0,2*M_PI));
    }

    Real tmp_cap = 0.0;
    for (OptionIterator<cid_t> channelIterator(nchannels,N,ocm); channelIterator.getDecVal() <= channelIterator.getDecMax(); channelIterator.increment()) {

      net.setChannel(channelIterator.getVal());

      /* record expected spatial capacity */
      tmp_cap = std::max(tmp_cap, nm.computeExpSpatialCapacity2(MANY_USERS, DensityUniform2d<Real>(arena)));

      if (s == 0 && snapshots) {
        /* save original snapshot of network */
        maxsinr = nm.computeMapMaxSINR();
        sinr::visualize::grayscaledB(*maxsinr, rgba, pow(10.0,(net.getSINRThresholddB()-widthdB)/10.0), pow(10.0,net.getSINRThresholddB()/10.0));
        sinr::visualize::outputBMP(rgba, w, h, OmanDirs::images() + "/copt" + Util::to_string(channelIterator.getVal()) + "-sinr.bmp");

        assoczone = nm.computeMapAssocZone();
        sinr::visualize::assoczone(*assoczone, rgba, N);
        sinr::visualize::outputBMP(rgba, w, h, OmanDirs::images() + "/copt" + Util::to_string(channelIterator.getVal()) + "-zone.bmp");

        cout<<"s:" + Util::to_string(s) + " \tcopt:" + Util::to_string(channelIterator.getVal())<<endl;
      }
    }

    /* record expected spatial capacity */
    cap_2b1p.at(s) = tmp_cap;
    stop = Util::getTimeNS();
    tme_2b1p.at(s) = (stop-start)/double(Util::nanoPerSec);
  }



  /*** CASE 3: 1 band, 2 modes ***/
  net.setNumberChannels(1);
  net.setChannelType(cType);
  net.setChannel(vector<cid_t>(N,0));

  for (unsigned int s = 0; s < sampleCount; s++) {

    cout<<"1b2p, s: " + Util::to_string(s)<<endl;

    start = Util::getTimeNS();

    //    net.setPosition(0,Point2d(250,250));
    //    net.setPosition(1,Point2d(250,750));
    //    net.setPosition(2,Point2d(750,250));
    //    net.setPosition(3,Point2d(750,750));
    //    net.setOrientation(vector<Real>(N,0.0));

    /* set random node positions and random orientations */
    for (unsigned int n = 0; n < N; n++) {
      net.setPosition(n,Point2d(Util::uniform_double(0,arenaSideLength),Util::uniform_double(0,arenaSideLength)));
      net.setOrientation(n,Util::uniform_double(0,2*M_PI));
    }

    Real tmp_cap = 0.0;
    for (OptionIterator<> patternIterator(npatterns,N,ocm); patternIterator.getDecVal() <= patternIterator.getDecMax(); patternIterator.increment()) {

      /* set radiation patterns */
      for (unsigned int n = 0; n < N; n++) {
        net.setRadPattern(n,patterns.at(patternIterator.getVal().at(n)));
      }

      /* record expected spatial capacity */
      tmp_cap = std::max(tmp_cap, nm.computeExpSpatialCapacity2(MANY_USERS, DensityUniform2d<Real>(arena)));

      if (s == 0 && snapshots) {
        /* save original snapshot of network */
        maxsinr = nm.computeMapMaxSINR();
        sinr::visualize::grayscaledB(*maxsinr, rgba, pow(10.0,(net.getSINRThresholddB()-widthdB)/10.0), pow(10.0,net.getSINRThresholddB()/10.0));
        sinr::visualize::outputBMP(rgba, w, h, OmanDirs::images() + "/popt" + Util::to_string(patternIterator.getVal()) + "-sinr.bmp");

        assoczone = nm.computeMapAssocZone();
        sinr::visualize::assoczone(*assoczone, rgba, N);
        sinr::visualize::outputBMP(rgba, w, h, OmanDirs::images() + "/popt" + Util::to_string(patternIterator.getVal()) + "-zone.bmp");

        cout<<"s:" + Util::to_string(s) + " \tpopt:" + Util::to_string(patternIterator.getVal())<<endl;
      }
    }

    /* record expected spatial capacity */
    cap_1b2p.at(s) = tmp_cap;
    stop = Util::getTimeNS();
    tme_1b2p.at(s) = (stop-start)/double(Util::nanoPerSec);
  }



  /*** CASE 4: 2 bands, 2 modes ***/
  net.setNumberChannels(2);
  net.setChannelType(cType);

  for (unsigned int s = 0; s < sampleCount; s++) {

    cout<<"2b2p, s: " + Util::to_string(s)<<endl;

    start = Util::getTimeNS();

    //    net.setPosition(0,Point2d(250,250));
    //    net.setPosition(1,Point2d(250,750));
    //    net.setPosition(2,Point2d(750,250));
    //    net.setPosition(3,Point2d(750,750));
    //    net.setOrientation(vector<Real>(N,0.0));

    /* set random node positions and random orientations */
    for (unsigned int n = 0; n < N; n++) {
      net.setPosition(n,Point2d(Util::uniform_double(0,arenaSideLength),Util::uniform_double(0,arenaSideLength)));
      net.setOrientation(n,Util::uniform_double(0,2*M_PI));
    }

    Real tmp_cap = 0.0;
    for (OptionIterator<cid_t> channelIterator(nchannels,N,ocm); channelIterator.getDecVal() <= channelIterator.getDecMax(); channelIterator.increment()) {
      for (OptionIterator<> patternIterator(npatterns,N,ocm); patternIterator.getDecVal() <= patternIterator.getDecMax(); patternIterator.increment()) {

        net.setChannel(channelIterator.getVal());

        /* set radiation patterns */
        for (unsigned int n = 0; n < N; n++) {
          net.setRadPattern(n,patterns.at(patternIterator.getVal().at(n)));
        }

        /* record expected spatial capacity */
        tmp_cap = std::max(tmp_cap, nm.computeExpSpatialCapacity2(MANY_USERS, DensityUniform2d<Real>(arena)));

        if (s == 0 && snapshots) {
          /* save original snapshot of network */
          maxsinr = nm.computeMapMaxSINR();
          sinr::visualize::grayscaledB(*maxsinr, rgba, pow(10.0,(net.getSINRThresholddB()-widthdB)/10.0), pow(10.0,net.getSINRThresholddB()/10.0));
          sinr::visualize::outputBMP(rgba, w, h, OmanDirs::images() + "/copt" + Util::to_string(channelIterator.getVal()) + "-popt" + Util::to_string(patternIterator.getVal()) + "-sinr.bmp");

          assoczone = nm.computeMapAssocZone();
          sinr::visualize::assoczone(*assoczone, rgba, N);
          sinr::visualize::outputBMP(rgba, w, h, OmanDirs::images() + "/copt" + Util::to_string(channelIterator.getVal()) + "-popt" + Util::to_string(patternIterator.getVal()) + "-zone.bmp");

          cout<<"s:" + Util::to_string(s) + " \tcopt:" + Util::to_string(channelIterator.getVal()) + "\tpopt:" + Util::to_string(patternIterator.getVal())<<endl;
        }
      }
    }

    /* record expected spatial capacity */
    cap_2b2p.at(s) = tmp_cap;
    stop = Util::getTimeNS();
    tme_2b2p.at(s) = (stop-start)/double(Util::nanoPerSec);
  }



  cout<<"1b1p:"<<endl;
  cout<<"\tcap: "<<Util::mean(cap_1b1p)<<" +/ "<<Util::stddev(cap_1b1p)<<" Mbps/m^2"<<endl;
  cout<<"\ttime: "<<Util::mean(tme_1b1p)<<" s"<<endl;
  cout<<"2b1p:"<<endl;
  cout<<"\tcap: "<<Util::mean(cap_2b1p)<<" +/ "<<Util::stddev(cap_2b1p)<<" Mbps/m^2"<<endl;
  cout<<"\ttime: "<<Util::mean(tme_2b1p)<<" s"<<endl;
  cout<<"1b2p:"<<endl;
  cout<<"\tcap: "<<Util::mean(cap_1b2p)<<" +/ "<<Util::stddev(cap_1b2p)<<" Mbps/m^2"<<endl;
  cout<<"\ttime: "<<Util::mean(tme_1b2p)<<" s"<<endl;
  cout<<"2b2p:"<<endl;
  cout<<"\tcap: "<<Util::mean(cap_2b2p)<<" +/ "<<Util::stddev(cap_2b2p)<<" Mbps/m^2"<<endl;
  cout<<"\ttime: "<<Util::mean(tme_2b2p)<<" s"<<endl;

  return 0;
}
