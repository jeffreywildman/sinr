#ifndef __NETWORK_H__
#define __NETWORK_H__

#include <vector>
#include <algorithm>

//#include <iostream>
//#include <iomanip>

#include <thrust/tuple.h>

#include <sinr/types.h>               /* for radPattern */
#include <sinr/radiationpattern.cuh>  /* for RadPatternEllipse::normalize */


/** The Network class provides for the storage and modification of a network consisting of wireless nodes.  The class
 * also contains a simple 'caching' scheme that allows calling functions to ascertain if data members have been
 * modified since the last cache query.
 * 
 * Caching: 
 *  cache == true: the relevant value has not changed since last cache access 
 *  cache == false: the relevant value has changed since last cache access 
 */
template <typename T>
class Network {
public:
  typedef thrust::tuple<T,T> Point2d;
  
  /** Construct a network instance.
   *
   * @param N Number of nodes in the network.
   * @todo More cleanly specify system constants.  Also update setSize() defaults.
   */
  Network(nid_t N) : t0(293), B0(10), eta(4.0343e-11) {
    this->setSize(N);

    this->setPathloss(T(3.0));
    this->setSINRThresholddB(T(10.0));
    this->setTemperature(293);
    this->setSystemBandwidth(10);
    this->setNumberChannels(1);
    this->setChannelType(CT_TIMESLOT);
  }
  ~Network() {;}

  /* Accessors */
  nid_t getSize() const {
    return N;
  }

  T getPathloss() const {
    return alpha;
  }
  T getChannelNoise() const {
    return (CType == CT_FREQBAND) ? (eta*(t/t0)*(B/B0))/T(numC) : eta*(t/t0)*(B/B0);
  }
  T getSINRThreshold() const {
    return pow(T(10.0),betadB/T(10.0));
  }
  T getSINRThresholddB() const {
    return betadB;
  }
  T getChannelBandwidth() const {
    return (CType == CT_FREQBAND) ? B/T(numC) : B;
  }
  T getChannelTimeShare() const {
    return (CType == CT_TIMESLOT) ? T(1.0)/T(numC) : T(1.0);
  }
  cid_t getNumberChannels() const {
    return numC;
  }
  channelType_t getChannelType() const {
    return CType;
  }

  Point2d getPosition(nid_t n) const {
    return S.at(n);
  }

  T getPower(nid_t n) const {
    return P.at(n);
  }

  T getBias(nid_t n) const {
    return Bias.at(n);
  }


  cid_t getChannel(nid_t n) const {
    return C.at(n);
  }

  T getOrientation(nid_t n) const {
    return Theta.at(n);
  }

  RadPattern<T> getRadPattern(nid_t n) const {
    return Pattern.at(n);
  }


  /* Cache Accessors */
  unsigned int getSizeCache() {
    return getCache(cache_N);
  }

  unsigned int getPathlossCache() {
    return getCache(cache_alpha);
  }
  unsigned int getChannelNoiseCache() {
    return getCache(cache_eta);
  }
  unsigned int getSINRThresholddBCache() {
    return getCache(cache_betadB);
  }
  unsigned int getChannelBandwidthCache() {
    return getCache(cache_B);
  }
  unsigned int getNumberChannelsCache() {
    return getCache(cache_numC);
  }
  unsigned int getChannelTypeCache() {
    return getCache(cache_CType);
  }

  unsigned int getPositionCache(nid_t n) {
    return getCache(cache_S.at(n));
  }

  unsigned int getPowerCache(nid_t n) {
    return getCache(cache_P.at(n)); 
  }

  unsigned int getBiasCache(nid_t n) {
    return getCache(cache_P.at(n)); 
  }

  unsigned int getChannelCache(nid_t n) {
    return getCache(cache_C.at(n));
  }

  unsigned int getOrientationCache(nid_t n) {
    return getCache(cache_Theta.at(n));
  }

  unsigned int getRadPatternCache(nid_t n) {
    return getCache(cache_Pattern.at(n));
  }


  /* Mutators */
  void setSize(nid_t _N) {
    N = _N; cache_N = false;

    /* new node entries */
    S.resize(N,Point2d(0.0,0.0));
    P.resize(N,20.0);
    Bias.resize(N,1.0);
    C.resize(N,0);
    Theta.resize(N,0.0);
    T ecc = 0.0;
    Pattern.resize(N,RadPattern<T>(RPT_ELLIPSE, ecc, RadPatternEllipse<T>::normalize(ecc)));

    /* new cache entries */
    cache_S.resize(N,false);
    cache_P.resize(N,false);
    cache_Bias.resize(N,false);
    cache_C.resize(N,false);
    cache_Theta.resize(N,false);
    cache_Pattern.resize(N,false);
  }
  void setPathloss(T _alpha) {
    alpha = _alpha; cache_alpha = false;
  }
  void setSINRThresholddB(T _betadB) {
    betadB = _betadB; cache_betadB = false;
  }
  void setTemperature(T _t) {
    t = _t; cache_eta = false;
  }
  void setSystemBandwidth(T _B) {
    B = _B; cache_eta = false;
  }
  void setNumberChannels(cid_t _numChannels) {
    numC = _numChannels; cache_numC = false;
    /** @todo Truncate old channel assignments that go above new maximum */
  }
  void setChannelType(channelType_t _CType) {
    CType = _CType; cache_CType = false;
  }

  void setPosition(nid_t n, Point2d s) {
    S.at(n) = s; cache_S.at(n) = false;
  }
  void setPosition(std::vector<Point2d> _S) {
    S = _S; std::fill(cache_S.begin(), cache_S.end(), false);
  }

  void setPower(nid_t n, T p) {
    P.at(n) = p; cache_P.at(n) = false;
  }
  void setPower(std::vector<T> _P) {
    P = _P; std::fill(cache_P.begin(), cache_P.end(), false);
  }

  void setBias(nid_t n, T bias) {
    Bias.at(n) = bias; cache_Bias.at(n) = false;
  }
  void setBias(std::vector<T> _Bias) {
    Bias = _Bias; std::fill(cache_Bias.begin(), cache_Bias.end(), false);
  }

  void setChannel(nid_t n, cid_t c) {
    C.at(n) = c; cache_C.at(n) = false;
  }
  void setChannel(std::vector<cid_t> _C) {
    C = _C; std::fill(cache_C.begin(), cache_C.end(), false);
    /** @todo assert that channel assignments do not violate max number of channels */
  }

  void setOrientation(nid_t n, T theta) {
    Theta.at(n) = theta; cache_Theta.at(n) = false;
  }
  void setOrientation(std::vector<T> _Theta) {
    Theta = _Theta; std::fill(cache_Theta.begin(), cache_Theta.end(), false);
  }

  void setRadPattern(nid_t n, RadPattern<T> pattern) {
    Pattern.at(n) = pattern; cache_Pattern.at(n) = false;
  } 
  void setRadPattern(std::vector<RadPattern<T> > _Pattern) {
    Pattern = _Pattern; std::fill(cache_Pattern.begin(), cache_Pattern.end(), false);
  }

  //friend std::ostream& operator<<(std::ostream &out, const Network &n);

private:

  /* environmental parameters */
  T alpha;              /* pathloss exponent */
  T eta;                /* nominal noise power (milliWatts) */
  T betadB;             /* SINR threshold (dB) */ 
  T t0;                 /* nominal temperature (Kelvin) */
  T t;                  /* temperature (Kelvin) */

  /* system parameters */
  T B0;                 /* nominal bandwidth (MHz) */
  T B;                  /* total system bandwidth (MHz) */
  cid_t numC;           /* number of orthogonal channels */
  channelType_t CType;  /* type of orthogonal channels */

  /* nodes */
  nid_t                       N;            /* number of nodes */
  std::vector<Point2d>        S;            /* node positions (meters) */
  std::vector<T>              P;            /* node transmission powers (milliWatts) */
  std::vector<T>              Bias;         /* node association bias (unitless) */
  std::vector<cid_t>          C;            /* node channel */
  std::vector<T>              Theta;        /* node orientations (radians) */
  std::vector<RadPattern<T> > Pattern;      /* radiation patterns */

  /** @todo Use keyword mutable on cache variables so that they can still be updated even after all the accessor
   * functions are redefined as const member functions. */
  unsigned int cache_alpha;
  unsigned int cache_eta;
  unsigned int cache_betadB;
  unsigned int cache_B;
  unsigned int cache_numC;
  unsigned int cache_CType;

  /** @note We use unsigned int instead of bool because of STL's wierd handling of bool inside vectors. */
  unsigned int cache_N;
  std::vector<unsigned int> cache_S;
  std::vector<unsigned int> cache_P;
  std::vector<unsigned int> cache_Bias;
  std::vector<unsigned int> cache_C;
  std::vector<unsigned int> cache_Theta;
  std::vector<unsigned int> cache_Pattern;

  /** Query a cache variable's value, then reset the cache's value.
   * 
   * @param Reference to the desired cache to query.
   * @return Current cache value.
   */
  unsigned int getCache(unsigned int &cache) {
    unsigned int orig = cache;
    cache = true;
    return orig;
  }
};


/** @todo get operator working again for templated Network class
 */
//std::ostream& operator<<(std::ostream &out, const Network &net) {
//  out<<"Network:"<<std::endl<<std::endl;
//
//  out<<"Environmental Parameters:"<<std::endl; 
//  out<<"alpha: "  <<net.alpha   <<std::endl; 
//  out<<"eta: "    <<net.eta     <<" mW"<<std::endl; 
//  out<<"beta: "   <<net.betadB  <<" dB"<<std::endl<<std::endl; 
//
//  out<<"Nodes: "  <<net.N       <<std::endl;
//  out<<std::left<<std::setw(10)<<"Node";
//  out<<std::setw(25)<<"Position (m)";
//  out<<std::setw(15)<<"Power (mW)";
//  out<<std::setw(20)<<"Orientation (rad)";
//  out<<std::setw(15)<<"RadPattern";
//  out<<std::setw(15)<<"RadParam1";
//  out<<std::setw(15)<<"RadParam2"<<std::endl;
//  for (nid_t n = 0; n < net.N; n++) {
//    out<<std::left<<std::setw(10)<<n;
//    out<<std::setw(25)<<net.S.at(n);
//    out<<std::setw(15)<<net.P.at(n);
//    out<<std::setw(20)<<net.Theta.at(n);
//    out<<std::setw(15)<<net.Pattern.at(n).type;
//    out<<std::setw(15)<<net.Pattern.at(n).p1;
//    out<<std::setw(15)<<net.Pattern.at(n).p2<<std::endl;
//  }
//  return out; 
//}

#endif /* __NETWORK_CUH__ */
