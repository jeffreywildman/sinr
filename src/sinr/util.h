#ifndef __UTIL_H__
#define __UTIL_H__

#include <cassert>

#include <string>
#include <vector>
#include <numeric>
#include <functional>
#include <algorithm>


namespace Util {

  /* C++11 to_string Wrappers */
  std::string    to_string(int i);
  std::string    to_string(unsigned int i);
  std::string    to_string(double d);
  std::string    to_string(std::vector<unsigned int> v);
  std::string    to_string(std::vector<int> v);

  /* Directory Management */
  bool           trailingSlash(std::string path);
  std::string    slashify(std::string path);
  bool           programInstalled(std::string programName);
  int            deleteDirContents(std::string dir);

  /* Random Number Generation */
  void           seedRandomGenerator(long value);
  void           seedRandomGenerator();
  double         uniform(double a, double b); // uniform float [a,b)
  inline double  uniform_double(double a, double b) {return uniform(a, b);}

  /* Clock/Time Related */
  const uint64_t       nanoPerSec = 1e9;
  uint64_t             getTimeNS();

  /* Statistics Related */
  template <typename T> std::vector<T>   absErr(std::vector<T> const &est, std::vector<T> const &act);
  template <typename T> std::vector<T>   relErr(std::vector<T> const &est, std::vector<T> const &act);
  template <typename T> std::vector<T>   sub(std::vector<T> const &a, std::vector<T> const &b);
  template <typename T> std::vector<T>   add(std::vector<T> const &a, std::vector<T> const &b);
  template <typename T> std::vector<T>   normalize(std::vector<T> const &v);
  template <typename T> T                mean(std::vector<T> const &v);
  template <typename T> T                variance(std::vector<T> const &v);
  template <typename T> T                rms(std::vector<T> const &v); 
  template <typename T> T                stddev(std::vector<T> const &v);
  template <typename T> T                min(std::vector<T> const &v);
  template <typename T> T                max(std::vector<T> const &v);
  

}; /* namespace Util */


/* Functors */
template <typename T>
class AbsErrFunctor {
public:
  const T operator()(const T est, const T act) {
    return std::abs(est-act); 
  }
};


template <typename T>
class RelErrFunctor {
public:
  const T operator()(const T est, const T act) {
    return std::abs(est-act)/act; 
  }
};


template <typename T>
std::vector<T> Util::absErr(std::vector<T> const &est, std::vector<T> const &act) {
  assert(est.size() == act.size() && est.size());

  std::vector<T> err(est.size(), T(0.0));
  std::transform(est.begin(), est.end(), act.begin(), err.begin(), AbsErrFunctor<T>());

  return err;
}


template <typename T>
std::vector<T> Util::relErr(std::vector<T> const &est, std::vector<T> const &act) {
  assert(est.size() == act.size() && est.size());

  std::vector<T> err(est.size(), T(0.0));
  std::transform(est.begin(), est.end(), act.begin(), err.begin(), RelErrFunctor<T>());

  return err;
}


template <typename T> 
std::vector<T> Util::sub(std::vector<T> const &a, std::vector<T> const &b) {
  assert(a.size() == b.size() && a.size());

  std::vector<T> c(a.size(), T(0.0));
  std::transform(a.begin(), a.end(), b.begin(), c.begin(), std::minus<T>());

  return c;
}


template <typename T> 
std::vector<T> Util::add(std::vector<T> const &a, std::vector<T> const &b) {
  assert(a.size() == b.size() && a.size());

  std::vector<T> c(a.size(), T(0.0));
  std::transform(a.begin(), a.end(), b.begin(), c.begin(), std::plus<T>());

  return c;
}


/**
 * Euclidean normalization of vector.
 */
template <typename T> 
std::vector<T> Util::normalize(std::vector<T> const &v) {
  assert(v.size());

  T c = Util::rms(v);

  std::vector<T> vv(v.size(), T(0.0));
  std::transform(vv.begin(), vv.end(), vv.begin(), std::bind2nd(std::divides<T>(), c));

  return vv;
}


template <typename T>
T Util::mean(std::vector<T> const &v) {
  assert(v.size());

  return std::accumulate(v.begin(),v.end(),T(0.0))/v.size();
}


template <typename T> 
T Util::variance(std::vector<T> const &v) {
  assert(v.size());

  T m = Util::mean(v);
  std::vector<T> tmp(v.size());

  /* subtract mean from data */
  std::transform(v.begin(), v.end(), tmp.begin(), std::bind2nd(std::minus<T>(), m));
  /* sum the squares of the zero-mean data and divide by (n-1) */
  return std::inner_product(tmp.begin(), tmp.end(), tmp.begin(), 0.0)/(v.size()-1);
}


template <typename T>
T Util::rms(std::vector<T> const &v) { 
  assert(v.size());

  /* sum the squares of the zero-mean data and divide by (n-1) */
  return sqrt(std::inner_product(v.begin(), v.end(), v.begin(), 0.0)/v.size()); 
}


template <typename T>
T Util::stddev(std::vector<T> const &v) {
  assert(v.size());

  return sqrt(Util::variance(v));
}


template <typename T>
T Util::min(std::vector<T> const &v) {
  assert(v.size());

  return *std::min_element(v.begin(),v.end());
}


template <typename T>
T Util::max(std::vector<T> const &v) {
  assert(v.size());

  return *std::max_element(v.begin(),v.end());
}


#endif /* __UTIL_H__ */
