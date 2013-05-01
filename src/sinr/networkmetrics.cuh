#ifndef __NETWORKMETRICS_CUH__
#define __NETWORKMETRICS_CUH__

#include <cassert>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/inner_product.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <sinr/types.h>
#include <sinr/arena.h>
#include <sinr/network.h>
#include <sinr/functors.cuh>
#include <sinr/radiationpattern.cuh>
#include <sinr/spatialdensity.cuh>


#define MANY_USERS 0
#define NULL_ZONE -1


/** The NetworkMetrics class computes network metrics, either averaged over a set of coordinates, or for a set of
 * coordinates.  
 */
template <typename T, template <typename> class Alloc, template <typename, typename> class ContainerType>
class NetworkMetrics {
  typedef typename thrust::tuple<T,T> Point2d;

public:

  /** @todo Enforce const-ness on input parameters.  They are not changeable by this class.
   */
  NetworkMetrics(Network<T> *net, const Arena2d<T> *arena, const ContainerType<Point2d, Alloc<Point2d> > *coords) {
    setParams(net, arena, coords);
    setAssocZoneRule(AZR_SINR);
  }

  ~NetworkMetrics() {;}

  void setAssocZoneRule(assocZoneRule_t azr);

  /* scalar, individual metrics - per node */
  T computeAvgSINR(nid_t n);
  T computeAvgCoverage(nid_t n);
  T computeAvgCapacity(nid_t n);

  /* scalar, network metrics - maximum over all nodes */
  T computeAvgMaxSINR();
  T computeAvgMaxCoverage();
  T computeAvgMaxCapacity();

  /* scalar, spatial capacity */
  T computeActSpatialCapacity();
  T computeActSpatialCapacityFairness(); 
  T computeExpSpatialCapacity(unsigned int M);
  template <typename SpatialDistribution>
  T computeExpSpatialCapacity2(unsigned int M, SpatialDistribution f);

  /* mapped, individual metrics */
  const ContainerType<T, Alloc<T> > *computeMapPower(nid_t n);
  const ContainerType<T, Alloc<T> > *computeMapSINR(nid_t n);
  const ContainerType<T, Alloc<T> > *computeMapCoverage(nid_t n) {n = n; assert(0); return NULL;}
  const ContainerType<T, Alloc<T> > *computeMapCapacity(nid_t n) {n = n; assert(0); return NULL;}

  /* mapped, network metrics */
  void                                       computeMapPower();
  void                                       computeMapSumPower();
  void                                       computeMapSINR();
  const ContainerType<T, Alloc<T> >         *computeMapMaxSINR();
  const ContainerType<T, Alloc<T> >         *computeMapMaxCoverage() {assert(0); return NULL;}
  const ContainerType<T, Alloc<T> >         *computeMapMaxShannonCapacity();

  const ContainerType<nid_t, Alloc<nid_t> > *computeMapAssocZone();
  const ContainerType<nid_t, Alloc<nid_t> > *computeMapAssocZone_SINR();
  const ContainerType<nid_t, Alloc<nid_t> > *computeMapAssocZone_Power();
  const ContainerType<nid_t, Alloc<nid_t> > *computeMapAssocZone_SINRBiased();
  const ContainerType<nid_t, Alloc<nid_t> > *computeMapAssocZone_PowerBiased();

  /* */
  template <typename SpatialDistribution>
  ContainerType<T, Alloc<T> >                              computeAssocZoneProb(SpatialDistribution f);

private:

  Network<T> *net;
  const Arena2d<T> *arena;
  const ContainerType<Point2d, Alloc<Point2d> > *coords;

  assocZoneRule_t azr;
  
  std::vector<ContainerType<T, Alloc<T> > > sumpowers;        /* (host/device) storage for per-channel total power values */
  ContainerType<T, Alloc<T> >               maxsinr;          /* (host/device) storage for max SINR */
  ContainerType<T, Alloc<T> >               maxcap;           /* (host/device) storage for max Shannon Capacity */
  ContainerType<nid_t, Alloc<nid_t> >       zone;             /* (host/device) storage for association zone indices */
  ContainerType<T, Alloc<T> >               spatialcap;       /* (host/device) storage for spatial capacity */
  std::vector<ContainerType<T, Alloc<T> > > powers;           /* (host/device) storage for individual power values */
  std::vector<ContainerType<T, Alloc<T> > > sinrs;            /* (host/device) storage for individual sinr values */

  void setParams(Network<T> *net, 
                 const Arena2d<T> *arena, 
                 const ContainerType<Point2d, Alloc<Point2d> > *coords);
  
  unsigned int numAssocZone();

  void updateCache();

  /* Caching: 
   *  cache == true: the relevant value has changed since last computation 
   *  cache == false: the relevant value has changed since last computation 
   */
  unsigned int cache_sumpowers;
  unsigned int cache_maxsinr;
  unsigned int cache_zone;
  std::vector<unsigned int> cache_powers;
  std::vector<unsigned int> cache_sinrs;

};

template <typename T>
class NetworkMetricsDev : public NetworkMetrics<T, thrust::device_malloc_allocator, thrust::device_vector> {
public:
  typedef typename thrust::tuple<T,T> Point2d;
  NetworkMetricsDev(Network<T> *net,
                    const Arena2d<T> *arena,
                    const typename thrust::device_vector<Point2d> *coords) :
    NetworkMetrics<T, thrust::device_malloc_allocator, thrust::device_vector>(net, arena, coords) {;}
};

template <typename T>
class NetworkMetricsHst : public NetworkMetrics<T, std::allocator, thrust::host_vector> {
public:
  typedef typename thrust::tuple<T,T> Point2d;
  NetworkMetricsHst(Network<T> *net,
                    const Arena2d<T> *arena,
                    const typename thrust::host_vector<Point2d> *coords) :
    NetworkMetrics<T, std::allocator, thrust::host_vector>(net, arena, coords) {;}
};


/** Unary functor to compute the received power at a point c from point pos.
 *
 */
template <typename T>
struct ComputePower {
  typedef typename thrust::tuple<T,T> Point2d;
  
  Point2d pos;
  T pt, theta, alpha;
  RadPattern<T> pattern;

  ComputePower(Point2d _pos, T _pt, T _theta, RadPattern<T> _pattern, T _alpha) : 
    pos(_pos), pt(_pt), theta(_theta), pattern(_pattern), alpha(_alpha) {;}

  __host__ __device__ T operator()(Point2d c) {
    using thrust::get;
    /* impose minimum distance constraint */
    /** @todo resolve visualization oddities ('overlapping') cells due to minimum distance constraint */
    /** @todo compute dmin based on maximum distance between points in adjacent pixels.  best guess: sqrt((2*pixelwidth)^2 + (2*pixelheight)^2)*/
    T dmin = 1.0;
    T d = max(sqrt(pow(get<0>(c) - get<0>(pos),T(2.0)) + pow(get<1>(c) - get<1>(pos),T(2.0))),dmin);
    //T d = sqrt(pow(get<0>(c) - get<0>(pos),T(2.0)) + pow(get<1>(c) - get<1>(pos),T(2.0)));
    T phi = atan2(get<1>(c) - get<1>(pos), get<0>(c) - get<0>(pos)); 
    T pr = T(0.0);
    /** @todo reinvestigate creating an abstract base class for antenna patterns */
    switch(pattern.type) {
    case RPT_ELLIPSE:
      pr = pt*RadPatternEllipse<T>(pattern.p1, pattern.p2)(phi-theta)*pow(d, -alpha);
      break;
    case RPT_ROSEPETAL:
      pr = pt*RadPatternRosePetal<T>(pattern.p1, pattern.p2)(phi-theta)*pow(d, -alpha);
      break;
    default:
      /* TODO: DANGER Will Robinson! */
      pr = pt*pow(d, -alpha);
      break;
    }
    return pr;
  }
};


/* Binary functor...
 *
 *
 */
template <typename T>
struct ComputeSINR {
  T N;

  ComputeSINR(T _N) : N(_N) {;}

  //__device__ T operator()(T p, T sump) {return p/(sump - p + _N);}
  /* Note: max is to avoid display and computation errors associated with single-precision floating point ops 
   * also see template specialization for floats below */
  __host__ __device__ T operator()(T p, T sump) const {
    return p/(max(sump - p,T(0.0)) + N);
  }
};
/* Note: there seems to be a significant performance hit when using the template specialization for floats,
 * and it doesn't seem to be the extra max() operation's fault. */
//template <>
//__device__ float ComputeSINR<float>::operator()(float p, float sump) {return p/(max(sump - p,0.0) + _N);}


/** Unary functor to compute the time-averaged Shannon capacity of a point-to-point channel with a given bandwidth @p B
 * and SINR @p sinr.  
 *
 * @todo fuse ComputeSINR and ComputeShannonCapacity kernels by making a constructor that allows an SINR binary functor
 * to be passed in.  then add operator()(T p, T sump)
 */
template <typename T>
struct ComputeShannonCapacity {
  T factor;

  /** 
   * @param B bandwidth of the channel.
   * @param frac Fraction of time the channel is used.
   */
  ComputeShannonCapacity(T B = 10.0, T frac = 1.0) {
    assert(B >= 0.0);
    assert(frac >= 0 && frac <= 1.0);
    factor = B*frac;
  }

  /** 
   * @param sinr SINR of the channel.
   * @returns Time-averaged channel capacity (Mbps). 
   */
  __host__ __device__ T operator()(T sinr) const {
    return factor*log1p(sinr)/log(T(2.0));
  }

};


/* Unary functor...
 *
 */
template <typename T>
struct IsGte {
  T t;

  IsGte(T _t) : t(_t) {;}

  __host__ __device__ bool operator()(T x) const {
    return (x >= t);
  }
};


/* Unary functor...
 *
 */
template <typename T>
struct IsEq {
  T y;

  IsEq(T _y) : y(_y) {;}

  __host__ __device__ bool operator()(T x) const {
    return (x == y);
  }
};


/** Binary Functor
 *
 */
template <typename T>
struct SetIfArgsEq {
  T val;

  SetIfArgsEq(T _val) : val(_val) {;}

  __host__ __device__ T operator()(T x, T y) {
    return (x == y) ? val : T(0.0);
  }
};


/* Unary functor...
 *
 */
template <typename T1, typename T2>
struct Arg1IfArg2Eq {
  T2 y2;

  Arg1IfArg2Eq(T2 _y2) : y2(_y2) {;}

  __host__ __device__ T1 operator()(T1 x1, T2 x2) {
    return (x2 == y2) ? x1 : T1(0.0);
  }
};


template <typename T, template <typename> class Alloc, template <typename, typename> class ContainerType>
void NetworkMetrics<T, Alloc, ContainerType>::setParams(Network<T> *_net, 
                                                        const Arena2d<T> *_arena, 
                                                        const ContainerType<Point2d, Alloc<Point2d> > *_coords) {                       
  assert(_net && _arena && _coords);

  net     = _net;
  arena   = _arena;
  coords  = _coords;

  maxsinr.resize(coords->size(), 0.0);
  maxcap.resize(coords->size(), 0.0);
  zone.resize(coords->size(), 0.0);

  powers.resize(net->getSize());
  sinrs.resize(net->getSize());

  for (nid_t n = 0; n < net->getSize(); n++) {
    powers.at(n).resize(coords->size(), 0.0);
    sinrs.at(n).resize(coords->size(), 0.0);
  } 

  sumpowers.resize(net->getNumberChannels());
  for (cid_t c = 0; c < net->getNumberChannels(); c++) {
    sumpowers.at(c).resize(coords->size(), 0.0); 
  }

  cache_sumpowers = false;
  cache_maxsinr = false;
  cache_zone = false;
  cache_powers.resize(net->getSize(), false);
  cache_sinrs.resize(net->getSize(), false);
}


template <typename T, template <typename> class Alloc, template <typename, typename> class ContainerType>
void NetworkMetrics<T, Alloc, ContainerType>::setAssocZoneRule(assocZoneRule_t _azr) {
  azr = _azr;
}


/** Compute the average SINR produced by a given node over the set of coordinates.
 *
 * No assumption is made on the distribution of the coordinates.
 */
template <typename T, template <typename> class Alloc, template <typename, typename> class ContainerType>
T NetworkMetrics<T, Alloc, ContainerType>::computeAvgSINR(nid_t n) {
  this->updateCache();

  this->computeMapSINR(n);

  T sum = thrust::reduce(sinrs.at(n).begin(), 
                         sinrs.at(n).end(), 
                         0.0);
  return sum/(T)(sinrs.at(n).size());
}


/** Compute a given node's coverage over the set of coordinates.
 *
 * No assumption is made on the distribution of the coordinates.
 */
template <typename T, template <typename> class Alloc, template <typename, typename> class ContainerType>
T NetworkMetrics<T, Alloc, ContainerType>::computeAvgCoverage(nid_t n) {
  this->updateCache();

  this->computeMapSINR(n);

  T count = thrust::count_if(sinrs.at(n).begin(), 
                             sinrs.at(n).end(), 
                             IsGte<T>(net->getSINRThreshold()));
  return count/(T)(sinrs.at(n).size());
}


/** Compute the average Shannon capacity produced by a given node over the set of coordinates.
 *
 * No assumption is made on the distribution of the coordinates.
 */
template <typename T, template <typename> class Alloc, template <typename, typename> class ContainerType>
T NetworkMetrics<T, Alloc, ContainerType>::computeAvgCapacity(nid_t n) {
  this->updateCache();

  this->computeMapSINR(n);

  T sum = thrust::transform_reduce(sinrs.at(n).begin(), 
                                   sinrs.at(n).end(), 
                                   ComputeShannonCapacity<T>(net->getChannelBandwidth(),net->getChannelTimeShare()), 
                                   0.0, 
                                   thrust::plus<T>());
  return sum/(T)(sinrs.at(n).size());

}


/** Compute the average maximum SINR.  
 *
 * The maximum is taken across all nodes, and the average is taken across the set of coordinates.  
 *
 * No assumption is made on the distribution of the coordinates.
 */
template <typename T, template <typename> class Alloc, template <typename, typename> class ContainerType> 
T NetworkMetrics<T, Alloc, ContainerType>::computeAvgMaxSINR() { this->updateCache();

  this->computeMapMaxSINR();

  T sum = thrust::reduce(maxsinr.begin(), 
                         maxsinr.end(), 
                         0.0);
  return sum/(T)(maxsinr.size());
}


/** Compute a network's coverage over the set of coordinates. 
 *
 * No assumption is made on the distribution of the coordinates.
 */
template <typename T, template <typename> class Alloc, template <typename, typename> class ContainerType>
T NetworkMetrics<T, Alloc, ContainerType>::computeAvgMaxCoverage() {
  this->updateCache();

  this->computeMapMaxSINR();

  T count = thrust::count_if(maxsinr.begin(), 
                             maxsinr.end(), 
                             IsGte<T>(net->getSINRThreshold()));
  return count/(T)(maxsinr.size());
}


/** Compute the average maximum Shannon capacity.  
 *
 * The maximum is taken across all nodes, and the average is taken across the set of coordinates.
 *
 * No assumption is made on the distribution of the coordinates.
 */

template <typename T, template <typename> class Alloc, template <typename, typename> class ContainerType> 
T NetworkMetrics<T, Alloc, ContainerType>::computeAvgMaxCapacity() { 
  this->updateCache();

  //this->computeMapMaxSINR();
  this->computeMapMaxShannonCapacity();

  //T sum = thrust::transform_reduce(maxsinr.begin(), 
  //                                 maxsinr.end(), 
  //                                 ComputeShannonCapacity<T>(net->getChannelBandwidth(),net->getChannelTimeShare()), 
  //                                 0.0, 
  //                                 thrust::plus<T>());
  T sum = thrust::reduce(maxcap.begin(), 
                         maxcap.end(), 
                         0.0); 
  return sum/(T)(maxsinr.size());
}


/** Compute the actual spatial capacity for users located at a set of coordinates.  
 *
 * Each user's capacity depends on the transmitter it associates with and the number of other users associating with the
 * same transmitter.  The individual user capacities are summed up and divided by the volume of the network arena.
 *
 * No assumption is made on the distribution of the user coordinates. 
 *
 * @todo make number of users an input parameter?  currently number of users controlled by coords->getSize()
 */
template <typename T, template <typename> class Alloc, template <typename, typename> class ContainerType> 
T NetworkMetrics<T, Alloc, ContainerType>::computeActSpatialCapacity() { 
  this->updateCache();

  this->computeMapMaxSINR();
  this->computeMapMaxShannonCapacity();
  this->computeMapAssocZone();

  T actcap = T(0.0);
  for (nid_t n = 0; n < net->getSize(); n++) {

    /* compute the number of user coordinates that reside in the association zone */
    T k_n = thrust::count_if(zone.begin(),
                             zone.end(), 
                             IsEq<nid_t>(n));

    /* compute the sum of the un-scaled user capacities within the association zone */
    T cap = thrust::inner_product(maxcap.begin(),
                                  maxcap.end(),
                                  zone.begin(),
                                  0.0,
                                  thrust::plus<T>(),
                                  Arg1IfArg2Eq<T,nid_t>(n));

    /* k_n == 0, then don't count this cell, otherwise, scale the sum user capacities by 1/k_n before adding */
    actcap += k_n ? T(1.0)/k_n*cap : T(0.0);
  }

  /* divide the sum of the per-user capacities by the volume of the network arena */
  return actcap/arena->getVolume();
}


/** Compute Jain and Chu's fairness index on the actual spatial capacity rates received by users located at a set of
 * coordinates.
 */
template <typename T, template <typename> class Alloc, template <typename, typename> class ContainerType> 
T NetworkMetrics<T, Alloc, ContainerType>::computeActSpatialCapacityFairness() { 
  this->updateCache();

  this->computeMapMaxSINR();
  this->computeMapMaxShannonCapacity();
  this->computeMapAssocZone();

  /* compute the map of square max Shannon capacities */
  ContainerType<T, Alloc<T> > maxcapsq(maxsinr.size());
  thrust::transform(maxcap.begin(),
                    maxcap.end(),
                    maxcap.begin(),
                    maxcapsq.begin(),
                    thrust::multiplies<T>());

  T actcap = T(0.0);
  T actcapsq = T(0.0);
  for (nid_t n = 0; n < net->getSize(); n++) {

    /* compute the number of user coordinates that reside in the association zone */
    T k_n = thrust::count_if(zone.begin(),
                             zone.end(), 
                             IsEq<nid_t>(n));

    /* compute the sum of the un-scaled user capacities within the association zone */
    T cap = thrust::inner_product(maxcap.begin(),
                                  maxcap.end(),
                                  zone.begin(),
                                  0.0,
                                  thrust::plus<T>(),
                                  Arg1IfArg2Eq<T,nid_t>(n));

    /* compute the sum of the un-scaled squared user capacities within the association zone */
    T capsq = thrust::inner_product(maxcapsq.begin(),
                                    maxcapsq.end(),
                                    zone.begin(),
                                    0.0,
                                    thrust::plus<T>(),
                                    Arg1IfArg2Eq<T,nid_t>(n));

    /* k_n == 0, then don't count this cell, otherwise, scale the sum user capacities by 1/k_n before adding */
    actcap    += k_n ? T(1.0)/k_n*cap : T(0.0);
    actcapsq  += k_n ? T(1.0)/pow(k_n,2.0)*capsq : T(0.0);
  }

  /* compute the fairness index */
  return pow(actcap,2.0)/(maxsinr->size()*actcapsq);
}


/** Compute the expected spatial capacity for M users located according to a given distribution.  
 *
 * Each user's capacity depends on the station it associates with and the number of other users associating with
 * the same station.  The arena is split into association zones, and the expected 
 *
 * The user-to-station association rule is assumed to be the max-sinr association rule.  
 *
 * @note We assume that the set of of coordinates must be generated along a uniform grid.
 *
 * @todo Allow an arbitrary user density function to be supplied as input.
 *
 * Notes: 
 *
 * 1) Allow an arbitrary user density function to be supplied as input, but require either a uniform grid or
 * uniform random coordinates.  Provide a shortcut if the user density function is uniform.  
 *
 * 2) Allowing any arbitrary
 * set of coordinates that was generated according to some distribution is equivalent to calling
 * computeActSpatialCapacity().  
 *
 */
template <typename T, template <typename> class Alloc, template <typename, typename> class ContainerType> 
T NetworkMetrics<T, Alloc, ContainerType>::computeExpSpatialCapacity(unsigned int M = 1) { 
  this->updateCache();

  this->computeMapMaxSINR();
  this->computeMapMaxShannonCapacity();

  T expcap = T(0.0);
  if (M == 1) {
    /* single-user expected spatial capacity - integral of capacity over the entire arena (single association zone) */
    /** @todo incorporate nonuniform user densities */
    expcap = T(1.0)/T(maxcap.size())*thrust::reduce(maxcap.begin(),
                                                    maxcap.end());
  } else {
    this->computeMapAssocZone();

    for (nid_t n = 0; n < net->getSize(); n++) {

      /* compute the probability of an arbitrary user residing in the association zone, given the user density function */
      /** @todo this line will change for arbitrary densities */
      T P_n = T(1.0)/T(zone.size())*thrust::count_if(zone.begin(),
                                                     zone.end(),
                                                     IsEq<nid_t>(n));

      //std::cout<<"P["<<n<<"] "<<P_n<<std::endl;

      /* compute the expected sum capacity of users within the association zone, given that at least one user resides in
       * the zone */
      /** @todo incorporate nonuniform user densities */
      /** @note the 1/zone.size() factor is due to the lambda(y)delta(y) = (1/|A|)*(|A|/|samples|) = 1/|samples| that should be inside the integral */
      T cap = T(1.0)/T(zone.size())*thrust::inner_product(maxcap.begin(),
                                                          maxcap.end(),
                                                          zone.begin(),
                                                          0.0,
                                                          thrust::plus<T>(),
                                                          Arg1IfArg2Eq<T,nid_t>(n));

      //std::cout<<"C["<<n<<"] "<<cap<<std::endl;

      /* add this zone's expected sum capacity to the total; if P_n == 0, then don't count this cell */
      /* weight by 1/P_n to complete the above conditional expectation */
      if (M == MANY_USERS) {
        /* many-users assumption, it is garaunteed that one or more users resides in this cell */
        expcap += P_n ? T(1.0)/P_n*cap : T(0.0);
      } else {
        /* further weight by the probability that NO users reside in this zone */
        expcap += P_n ? (T(1.0)-pow(T(1.0)-P_n,T(M)))/P_n*cap : T(0.0);
      }
    }
  }

  /* divide the expected sum of the per-user capacities by the volume of the network arena */
  return expcap/arena->getVolume();
}


template <typename Tuple>
struct TuplePlus : public thrust::binary_function<Tuple, Tuple, Tuple> {
  __host__ __device__ Tuple operator()(const Tuple &lhs, const Tuple &rhs) const {
    using thrust::get;
    return Tuple(get<0>(lhs) + get<0>(rhs), get<1>(lhs) + get<1>(rhs));
  }
};

template <typename T>
struct SpCapProd : public thrust::binary_function<T, T, T> {
  unsigned int M;
  T C;

  /** 
   * @param M Number of users.
   * @param C Factor to finish computation of zone probabilities 
   */
  __host__ __device__ SpCapProd(unsigned int _M, T _C) : M(_M), C(_C) {;}

  __host__ __device__ T operator()(const T &p, const T &cap) const {
    if (M == MANY_USERS) {
      return p ? T(1.0)/(p*C)*cap : T(0.0);
    } else {
      return p ? (T(1.0)-pow(T(1.0)-(p*C),T(M)))/(p*C)*cap : T(0.0);
    }
  }
};



/** Compute the expected spatial capacity for M users located according to a given distribution.  
 *
 * Each user's capacity depends on the station it associates with and the number of other users associating with
 * the same station.  The arena is split into association zones, and the expected 
 *
 * The user-to-station association rule is assumed to be the max-sinr association rule.  
 *
 * @note We assume that the set of of coordinates must be generated along a uniform grid.
 *
 * @todo Allow an arbitrary user density function to be supplied as input.
 *
 * Notes: 
 *
 * 1) Allow an arbitrary user density function to be supplied as input, but require either a uniform grid or
 * uniform random coordinates.  Provide a shortcut if the user density function is uniform.  
 *
 * 2) Allowing any arbitrary
 * set of coordinates that was generated according to some distribution is equivalent to calling
 * computeActSpatialCapacity().  
 *
 */
template <typename T, template <typename> class Alloc, template <typename, typename> class ContainerType>
template <typename SpatialDistribution>
T NetworkMetrics<T, Alloc, ContainerType>::computeExpSpatialCapacity2(unsigned int M, SpatialDistribution f) { 
  this->updateCache();

  this->computeMapMaxSINR();
  this->computeMapMaxShannonCapacity();

  /* construct the map of spatial density values over the arena */
  ContainerType<T, Alloc<T> > density(coords->size());
  thrust::transform(coords->begin(), coords->end(), density.begin(), f);

  T expcap = T(0.0);
  if (M == 1) {
    /* Single-user expected spatial capacity - integral of capacity over the entire arena (single association zone). */
    expcap = T(1.0)/T(maxcap.size())*thrust::inner_product(maxcap.begin(),maxcap.end(),density.begin(),0.0);
  } else {
    this->computeMapAssocZone();

    ContainerType<nid_t, Alloc<nid_t> > ikeys(zone);
    ContainerType<T, Alloc<T> >         ivals1(density);   
    ContainerType<T, Alloc<T> >         ivals2(maxcap.size());   
    thrust::transform(maxcap.begin(),maxcap.end(),density.begin(),ivals2.begin(),thrust::multiplies<T>());

    ContainerType<nid_t, Alloc<nid_t> > okeys(net->getSize()); /* note: we are purposely allocating less than coords->size() */
    ContainerType<T, Alloc<T> >         ovals1(net->getSize()); /* note: we are purposely allocating less than coords->size() */
    ContainerType<T, Alloc<T> >         ovals2(net->getSize()); /* note: we are purposely allocating less than coords->size() */

    /* Sort both densities and maxcap*densities at the same time with a zip iterator.  Sorting is based on association 
     * index-based key values. */
    thrust::sort_by_key(ikeys.begin(),
                        ikeys.end(),
                        thrust::make_zip_iterator(thrust::make_tuple(ivals1.begin(),ivals2.begin())));
    /* Reduce the densities and maxcap*densities at the same time with a zip iterator.  Reduction is based on contiguous
     * sets of key values, which were sorted into contiguous regions the previous step.  The reduced values form the
     * basis of zone probabilities and per-zone expected capacity - they still lack multiplicative factors to 
     * account for Reimann sums numerical integration: \int_A f(x)dx = |A|/|samples| * \sum_i f(x_i). */
    /** @todo: truncate okeys, ovals1, ovals2 (will need to be done if a cell is empty - no keys for a cell).  Do this
     * by grabbing new_end output by reduce_by_key */
    thrust::reduce_by_key(ikeys.begin(),
                          ikeys.end(),
                          thrust::make_zip_iterator(thrust::make_tuple(ivals1.begin(), ivals2.begin())),
                          okeys.begin(),
                          thrust::make_zip_iterator(thrust::make_tuple(ovals1.begin(), ovals2.begin())),
                          thrust::equal_to<nid_t>(),
                          TuplePlus<thrust::tuple<T,T> >());

    //    thrust::host_vector<nid_t> hokeys(okeys);
    //    thrust::host_vector<T> hovals1(ovals1);
    //    thrust::host_vector<T> hovals2(ovals2);
    //    for (nid_t k = 0; k < net->getSize(); k++) {
    //      std::cout<<"key["<<k<<"] "<<hokeys[k]<<std::endl;
    //      std::cout<<"P["<<k<<"] "<<hovals1[k]*T(arena->getVolume())/T(maxcap.size())<<std::endl;
    //      std::cout<<"cap["<<k<<"] "<<hovals2[k]*T(arena->getVolume())/T(maxcap.size())<<std::endl;
    //    }

    /* To compute spatial capacity, we take the inner product of zone-probabilities and per-zone expected capacity and 
     * divide by the volume of the network arena (1/|A|).  Note, the remaining factor to compute zone
     * probabilities is supplied within the binary operator.  Also note, the remaining factor to compute the per-zone
     * expected capacity (|A|/|samples|), partially cancels with dividing by the network arena area, leaving the final
     * factor of 1/|samples| in front of the expression below. */
    /** @todo make sure ovals1 and ovals2 are truncated to new end!!! */
    expcap = T(1.0)/T(maxcap.size())*thrust::inner_product(ovals1.begin(),
                                                           ovals1.end(),
                                                           ovals2.begin(),
                                                           0.0,
                                                           thrust::plus<T>(),
                                                           SpCapProd<T>(M,T(arena->getVolume())/T(maxcap.size())));
  }

  return expcap;
}

/**
 * @return The number of association zones, including the null association zone.
 */
template <typename T, template <typename> class Alloc, template <typename, typename> class ContainerType>
unsigned int NetworkMetrics<T, Alloc, ContainerType>::numAssocZone() {
  return net->getSize() + 1;
}

/**
 * @return A vector of association zone probabilities, excluding the null association zone.
 */
template <typename T, template <typename> class Alloc, template <typename, typename> class ContainerType>
template <typename SpatialDistribution>
ContainerType<T, Alloc<T> > NetworkMetrics<T, Alloc, ContainerType>::computeAssocZoneProb(SpatialDistribution f) {
  this->updateCache();

  this->computeMapAssocZone();
  
  /* construct the map of spatial density values over the arena */
  ContainerType<T, Alloc<T> > density(coords->size());
  thrust::transform(coords->begin(), coords->end(), density.begin(), f);

  ContainerType<nid_t, Alloc<nid_t> > ikeys(zone);
  ContainerType<T, Alloc<T> >         ivals(density);   

  ContainerType<nid_t, Alloc<nid_t> > okeys(this->numAssocZone()); /* note: we are purposely allocating less than coords->size() */
  ContainerType<T, Alloc<T> >         ovals(this->numAssocZone()); /* note: we are purposely allocating less than coords->size() */

  /* Sort densities based on association index key values. */
  thrust::sort_by_key(ikeys.begin(), ikeys.end(), ivals.begin());

  /* Reduce the densities based on contiguous sets of key values, which were sorted into contiguous regions the previous
   * step.  The reduced values form the basis of zone probabilities - they still lack multiplicative factors to account
   * for Reimann sums numerical integration: \int_A f(x)dx = |A|/|samples| * \sum_i f(x_i). */
  typedef typename ContainerType<nid_t, Alloc<nid_t> >::iterator  NIDIterator;
  typedef typename ContainerType<T, Alloc<T> >::iterator          TIterator;
  thrust::pair<NIDIterator, TIterator> newend;

  newend = thrust::reduce_by_key(ikeys.begin(),
                                 ikeys.end(),
                                 ivals.begin(),
                                 okeys.begin(),
                                 ovals.begin());

  /* truncate okeys, ovals (will need to be done if a cell is empty - no keys for a cell) */
  okeys.resize(newend.first-okeys.begin());
  ovals.resize(newend.second-ovals.begin());

  /* if the first key corresponds to the null zone, skip it */
  unsigned int offset = 0;
  if (okeys.front() == NULL_ZONE) {
    offset = 1;
  }

  /* fill non-zero zones probabilities into vector */
  ContainerType<T,Alloc<T> > probabilities(net->getSize(),0.0);
  thrust::transform(ovals.begin()+offset,
                    ovals.end(),
                    thrust::make_permutation_iterator(probabilities.begin(), okeys.begin()+offset),
                    sinr::functors::Multiply<T>(T(arena->getVolume())/T(coords->size())));

  return probabilities;
}


/** Compute the received power at all coordinates from the specified node.
 */
template <typename T, template <typename> class Alloc, template <typename, typename> class ContainerType>
const ContainerType<T, Alloc<T> > *NetworkMetrics<T, Alloc, ContainerType>::computeMapPower(nid_t n) {
  this->updateCache();

  if (cache_powers.at(n) == false) {

    thrust::transform(coords->begin(), 
                      coords->end(), 
                      powers.at(n).begin(), 
                      ComputePower<T>(net->getPosition(n), 
                                      net->getPower(n), 
                                      net->getOrientation(n),
                                      net->getRadPattern(n),
                                      net->getPathloss()));

    cache_powers.at(n) = true; 
  }

  return &powers.at(n);
}


/** Compute the SINR at all coordinates from the specified node.
 *
 * The SINR is computed using co-channel interference.  
 */
template <typename T, template <typename> class Alloc, template <typename, typename> class ContainerType>
const ContainerType<T, Alloc<T> > *NetworkMetrics<T, Alloc, ContainerType>::computeMapSINR(nid_t n) {
  this->updateCache();

  this->computeMapPower(n);
  /** @todo per-channel computeMapSumPower() */
  this->computeMapSumPower();

  if (cache_sinrs.at(n) == false) {

    thrust::transform(powers.at(n).begin(), 
                      powers.at(n).end(), 
                      sumpowers.at(net->getChannel(n)).begin(), 
                      sinrs.at(n).begin(), 
                      ComputeSINR<T>(net->getChannelNoise()));

    cache_sinrs.at(n) = true;
  }

  return &sinrs.at(n);
}


/** Compute the received power at all coordinates from each node.
 *
 * @todo look at thrust zip iterators to compute power for sets of transmitters at a time.
 */
template <typename T, template <typename> class Alloc, template <typename, typename> class ContainerType>
void NetworkMetrics<T, Alloc, ContainerType>::computeMapPower() {
  this->updateCache();

  for (nid_t n = 0; n < net->getSize(); n++) {
    this->computeMapPower(n);
  }
}


/** Compute the sum received power at all coordinates from all nodes.
 *
 * @todo look at thrust zip iterators to see if this can be done faster
 */
template <typename T, template <typename> class Alloc, template <typename, typename> class ContainerType>
void NetworkMetrics<T, Alloc, ContainerType>::computeMapSumPower() {
  this->updateCache();

  this->computeMapPower();

  if (cache_sumpowers == false) {

    for (cid_t c = 0; c < net->getNumberChannels(); c++) {
      thrust::fill(sumpowers.at(c).begin(),sumpowers.at(c).end(),0.0);
    }
    for (nid_t n = 0; n < net->getSize(); n++) { 
      thrust::transform(sumpowers.at(net->getChannel(n)).begin(), 
                        sumpowers.at(net->getChannel(n)).end(), 
                        powers.at(n).begin(), 
                        sumpowers.at(net->getChannel(n)).begin(), 
                        thrust::plus<T>());
    }

    cache_sumpowers = true;
  }
}


/** Compute the SINR at all coordinates from each node.
 * 
 */
template <typename T, template <typename> class Alloc, template <typename, typename> class ContainerType>
void NetworkMetrics<T, Alloc, ContainerType>::computeMapSINR() {
  this->updateCache();

  this->computeMapPower();
  this->computeMapSumPower();

  for (nid_t n = 0; n < net->getSize(); n++) {
    this->computeMapSINR(n);
  }
}


/** Compute the maximum SINR at all coordinates across all nodes.
 *
 */
template <typename T, template <typename> class Alloc, template <typename, typename> class ContainerType>
const ContainerType<T, Alloc<T> > *NetworkMetrics<T, Alloc, ContainerType>::computeMapMaxSINR() {
  this->updateCache();

  this->computeMapSINR();

  if (cache_maxsinr == false) {

    thrust::fill(maxsinr.begin(), maxsinr.end(), 0.0);
    for (nid_t n = 0; n < net->getSize(); n++) {
      thrust::transform(maxsinr.begin(), 
                        maxsinr.end(), 
                        sinrs.at(n).begin(), 
                        maxsinr.begin(), 
                        thrust::maximum<T>());
    }

    cache_maxsinr = true;
  }

  return &maxsinr;
}


/** Compute the maximum Shannon Capacity at all coordinates across all nodes.
 *
 * @note bandwidth not taken into account here.
 * @todo cache this computation
 */
template <typename T, template <typename> class Alloc, template <typename, typename> class ContainerType>
const ContainerType<T, Alloc<T> > *NetworkMetrics<T, Alloc, ContainerType>::computeMapMaxShannonCapacity() {
  this->updateCache();

  this->computeMapMaxSINR();

  //  if (cache_maxcap == false) {

  thrust::transform(maxsinr.begin(),
                    maxsinr.end(),
                    maxcap.begin(),
                    ComputeShannonCapacity<T>(net->getChannelBandwidth(),net->getChannelTimeShare())); 

  //cache_maxcap = true;
  //}

  return &maxcap;
}


/** 
 * Constructor provides index2
 * Given (val1, val2, index1) 
 * Output (index1) if val1 > val2
 * Output (index2) if val1 < val2
 */
template <typename T>
struct AssocZone { 
  typedef typename thrust::tuple<T,T,nid_t> Tuple;
  nid_t n;

  /**
   * @param _n The node whose association zone we are detecting.
   */
  AssocZone(nid_t _n): n(_n) {;}

  /**
   * Examines a tuple of (max sinr, sinr of node n, current node association) values at a particular coordinate. 
   * If the max sinr equals the sinr from node n, then the function returns the new node association, otherwise 
   * return the current node association.
   */
  __host__ __device__ nid_t operator()(const Tuple &t) const {
    using thrust::get;
    /** @note Use with care! equal_to with floating point values! */
    return (get<0>(t) == get<1>(t)) ? n : get<2>(t);
    //return (fabs(get<0>(t) - get<1>(t)) < 1e-10) ? n : get<2>(t);
  }
};


///** 
// * Given (maxsinr, sinr, index) 
// * Output new (index)
// */
//template <typename T>
//struct BiasedSINRMaxEq { 
//  typedef typename thrust::tuple<T,T,nid_t> Tuple;
//  nid_t n;
//  T bias;
//
//  BiasedSINRMaxEq(nid_t _n, T _bias): n(_n), bias(_bias) {;}
//
//  __host__ __device__ nid_t operator()(const Tuple &t) const {
//    using thrust::get;
//    /** @note Use with care! equal_to with floating point values! */
//    return (get<0>(t) == bias*get<1>(t)) ? n : get<2>(t);
//  }
//};


/** Compute the associated node at all coordinates.
 *
 * @todo cache this result
 */
template <typename T, template <typename> class Alloc, template <typename, typename> class ContainerType>
const ContainerType<nid_t, Alloc<nid_t> > *NetworkMetrics<T, Alloc, ContainerType>::computeMapAssocZone() {
  switch(azr) {
  case AZR_SINR:
    return computeMapAssocZone_SINR();
  case AZR_POWER:
    return computeMapAssocZone_Power();
  case AZR_SINR_BIASED:
    return computeMapAssocZone_SINRBiased();
  case AZR_POWER_BIASED:
    return computeMapAssocZone_PowerBiased();
  default:
    assert(0);
    return NULL;
  }
}


/** Compute the associated node at all coordinates.
 *
 * @todo cache this result
 */
template <typename T, template <typename> class Alloc, template <typename, typename> class ContainerType>
const ContainerType<nid_t, Alloc<nid_t> > *NetworkMetrics<T, Alloc, ContainerType>::computeMapAssocZone_SINR() {
  this->updateCache();

  this->computeMapMaxSINR();

  thrust::fill(zone.begin(), zone.end(), NULL_ZONE);
  for (nid_t n = 0; n < net->getSize(); n++) {
    /* compute which user coordinates associate with node n based on max sinr association rules */
    thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(maxsinr.begin(),sinrs.at(n).begin(),zone.begin())),
                      thrust::make_zip_iterator(thrust::make_tuple(maxsinr.end(),sinrs.at(n).end(),zone.end())),
                      zone.begin(),
                      AssocZone<T>(n));
  }
  /* if any max sinr values are strictly zero, then it belongs to the null association zone */
  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(maxsinr.begin(),thrust::constant_iterator<T>(0.0),zone.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(maxsinr.end(),thrust::constant_iterator<T>(0.0),zone.end())),
                    zone.begin(),
                    AssocZone<T>(NULL_ZONE));

  return &zone;
}


/** Compute the associated node at all coordinates.
 *
 * @todo cache this result
 */
template <typename T, template <typename> class Alloc, template <typename, typename> class ContainerType>
const ContainerType<nid_t, Alloc<nid_t> > *NetworkMetrics<T, Alloc, ContainerType>::computeMapAssocZone_Power() {
  this->updateCache();

  this->computeMapPower();

  ContainerType<T, Alloc<T> > maxpower(maxsinr.size(), 0.0);
  for (nid_t n = 0; n < net->getSize(); n++) {
    thrust::transform(maxpower.begin(), 
                      maxpower.end(), 
                      powers.at(n).begin(), 
                      maxpower.begin(), 
                      thrust::maximum<T>());
  }

  thrust::fill(zone.begin(), zone.end(), NULL_ZONE);
  for (nid_t n = 0; n < net->getSize(); n++) {
    /* compute which user coordinates associate with node n based on max sinr association rules */
    thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(maxpower.begin(),powers.at(n).begin(),zone.begin())),
                      thrust::make_zip_iterator(thrust::make_tuple(maxpower.end(),powers.at(n).end(),zone.end())),
                      zone.begin(),
                      AssocZone<T>(n));
  }
  /* if any max power values are strictly zero, then it belongs to the null association zone */
  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(maxpower.begin(),thrust::constant_iterator<T>(0.0),zone.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(maxpower.end(),thrust::constant_iterator<T>(0.0),zone.end())),
                    zone.begin(),
                    AssocZone<T>(NULL_ZONE));

  return &zone;
}

/** Compute the associated node at all coordinates using biasing weights.
 *
 * @todo cache this result
 */
template <typename T, template <typename> class Alloc, template <typename, typename> class ContainerType>
const ContainerType<nid_t, Alloc<nid_t> > *NetworkMetrics<T, Alloc, ContainerType>::computeMapAssocZone_SINRBiased() {
  this->updateCache();

  this->computeMapSINR();

  using sinr::functors::Multiply;
  
  ContainerType<T, Alloc<T> > maxbiasedsinr(maxsinr.size(), 0.0);
  for (nid_t n = 0; n < net->getSize(); n++) {
    thrust::transform(maxbiasedsinr.begin(), 
                      maxbiasedsinr.end(), 
                      thrust::make_transform_iterator(sinrs.at(n).begin(),Multiply<T>(net->getBias(n))), 
                      maxbiasedsinr.begin(), 
                      thrust::maximum<T>());
  }

  thrust::fill(zone.begin(), zone.end(), NULL_ZONE);
  for (nid_t n = 0; n < net->getSize(); n++) {
    /* compute which user coordinates associate with node n based on max sinr association rules */
    thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(maxbiasedsinr.begin(),thrust::make_transform_iterator(sinrs.at(n).begin(),Multiply<T>(net->getBias(n))),zone.begin())),
                      thrust::make_zip_iterator(thrust::make_tuple(maxbiasedsinr.end(),thrust::make_transform_iterator(sinrs.at(n).end(),Multiply<T>(net->getBias(n))),zone.end())),
                      zone.begin(),
                      AssocZone<T>(n));
  }
  /* if any max biased sinr values are strictly zero, then it belongs to the null association zone */
  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(maxbiasedsinr.begin(),thrust::constant_iterator<T>(0.0),zone.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(maxbiasedsinr.end(),thrust::constant_iterator<T>(0.0),zone.end())),
                    zone.begin(),
                    AssocZone<T>(NULL_ZONE));

  return &zone;
}


/** Compute the associated node at all coordinates using biasing weights.
 *
 * @todo cache this result
 */
template <typename T, template <typename> class Alloc, template <typename, typename> class ContainerType>
const ContainerType<nid_t, Alloc<nid_t> > *NetworkMetrics<T, Alloc, ContainerType>::computeMapAssocZone_PowerBiased() {
  this->updateCache();

  this->computeMapPower();

  using sinr::functors::Multiply;
  
  ContainerType<T, Alloc<T> > maxbiasedpower(maxsinr.size(), 0.0);
  for (nid_t n = 0; n < net->getSize(); n++) {
    thrust::transform(maxbiasedpower.begin(), 
                      maxbiasedpower.end(), 
                      thrust::make_transform_iterator(powers.at(n).begin(), Multiply<T>(net->getBias(n))), 
                      maxbiasedpower.begin(), 
                      thrust::maximum<T>());
  }

  thrust::fill(zone.begin(), zone.end(), NULL_ZONE);
  for (nid_t n = 0; n < net->getSize(); n++) {
    /* compute which user coordinates associate with node n based on max sinr association rules */
    thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(maxbiasedpower.begin(),thrust::make_transform_iterator(powers.at(n).begin(),Multiply<T>(net->getBias(n))),zone.begin())),
                      thrust::make_zip_iterator(thrust::make_tuple(maxbiasedpower.end(),thrust::make_transform_iterator(powers.at(n).end(),Multiply<T>(net->getBias(n))),zone.end())),
                      zone.begin(),
                      AssocZone<T>(n));
  }
  /* if any max biased power values are strictly zero, then it belongs to the null association zone */
  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(maxbiasedpower.begin(),thrust::constant_iterator<T>(0.0),zone.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(maxbiasedpower.end(),thrust::constant_iterator<T>(0.0),zone.end())),
                    zone.begin(),
                    AssocZone<T>(NULL_ZONE));

  return &zone;
}


/** Query the stored Network object for changes in parameters.  
 *
 * Update the internal cache as needed.
 *
 * @todo check coords for updates.
 * @todo cache association zone
 */
template <typename T, template <typename> class Alloc, template <typename, typename> class ContainerType>
void NetworkMetrics<T, Alloc, ContainerType>::updateCache() {

  /* check parameters that affect size of internal storage */

  /* network size */
  if (net->getSizeCache() == false) {
    nid_t N = net->getSize();

    powers.resize(N);
    sinrs.resize(N);
    for (nid_t n = 0; n < N; n++) {
      powers.at(n).resize(coords->size(), 0.0);
      sinrs.at(n).resize(coords->size(), 0.0);
    } 

    cache_powers.resize(N, false);
    cache_sinrs.resize(N, false);

    cache_sumpowers = false;
    std::fill(cache_sinrs.begin(), cache_sinrs.end(), false);
    cache_maxsinr = false;
  }

  /* number of orthogonal channels */
  if (net->getNumberChannelsCache() == false) {
    sumpowers.resize(net->getNumberChannels());
    for (cid_t c = 0; c < net->getNumberChannels(); c++) {
      sumpowers.at(c).resize(coords->size(), 0.0);
    }

    cache_sumpowers = false;
    std::fill(cache_sinrs.begin(), cache_sinrs.end(), false);
    cache_maxsinr = false;
  }


  /* check global parameters that affect computation of received power and up */

  /* pathloss */
  if (net->getPathlossCache() == false) {
    /* all received powers need to be recomputed */
    std::fill(cache_powers.begin(),cache_powers.end(),false);
    cache_sumpowers = false;
    std::fill(cache_sinrs.begin(), cache_sinrs.end(), false);
    cache_maxsinr = false;
  }


  /* check global parameters that affect computation of SINR and up */

  /* noise, bandwidth, channel type */
  if (net->getChannelNoiseCache() == false || 
      net->getChannelBandwidthCache() == false || 
      net->getChannelTypeCache() == false) {
    /* all sinrs need to be recomputed */
    std::fill(cache_sinrs.begin(), cache_sinrs.end(), false);
    cache_maxsinr = false;
  }


  /* other global parameters */

  /* SINR threshold */
  if (net->getSINRThresholddBCache() == false) {
    /* nothing new needs to be recomputed (currently)! YAY! */
  }


  /* check per-node parameters */

  /** @todo If transmit power changes, there may be a more efficient recomputation, but only if we know the old transmit
   * power level. 
   */
  /** @todo If position or orientation change, there may be more efficient recomputations, such as rotations and
   * translations of received power values, but only if we know the old position and orientation values. 
   */
  for (nid_t n = 0; n < net->getSize(); n++) {
    if (net->getPositionCache(n) == false || net->getPowerCache(n) == false || 
        net->getOrientationCache(n) == false || net->getRadPatternCache(n) == false) {
      cache_powers.at(n) = false;
      cache_sumpowers = false;
      std::fill(cache_sinrs.begin(), cache_sinrs.end(), false);  /* one node's power affects all nodes' sinrs */
      cache_maxsinr = false;
    }
    if (net->getChannelCache(n) == false) {
      /* power calculation still valid if the channel changes */
      cache_sumpowers = false;
      std::fill(cache_sinrs.begin(), cache_sinrs.end(), false);  /* one node's power affects all nodes' sinrs */
      cache_maxsinr = false; 
    }
    if (net->getBiasCache(n) == false) {
      /* nothing new needs to be recomputed (currently), but association zones, when cached, can depend on biasing */
    }
  }
}

#endif /* __NETWORKMETRICS_CUH__ */
