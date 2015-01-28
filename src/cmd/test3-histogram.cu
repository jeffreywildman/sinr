#include <thrust/host_vector.h>

#include <sinr/histogram.cuh>
#include <sinr/util.h>

using namespace std;

typedef double Real;
typedef thrust::tuple<Real,Real> Point2d;

/** This test program creates random data and plots a histogram.
 */
int main(int argc __attribute__((unused)), char** argv __attribute__((unused))) {

  /* common parameters */
  Real min1 = 10.0;
  Real max1 = 20.0;
  Real min2 = 30.0;
  Real max2 = 40.0;
  unsigned int numbins = 30;
  unsigned int N = 200;

  thrust::host_vector<Real> data(N);
  thrust::host_vector<Real> bincenters(numbins);
  thrust::host_vector<Real> binvals(numbins);
  std::vector<Real> bincenters_std(numbins);
  std::vector<Real> binvals_std(numbins);

  /* create random data */
  for (unsigned int d = 0; d < 100; d++) {
    data[d] = Util::uniform_double(min1,max1);
  }
  for (unsigned int d = 100; d < 200; d++) {
    data[d] = Util::uniform_double(min2,max2);
  }


  /* linear histogram */
  sinr::histogram::histogram(data,bincenters,binvals,numbins);
  thrust::copy(bincenters.begin(),bincenters.end(),bincenters_std.begin());
  thrust::copy(binvals.begin(),binvals.end(),binvals_std.begin());

  cout<<"Linear Histogram"<<endl;
  for (unsigned int bin = 0; bin < numbins; bin++) {
    cout<<"BinCenter: "<<bincenters_std.at(bin)<<", BinValue: "<<binvals_std.at(bin)<<endl;
  }
  cout<<endl<<endl;

  
  /* logarithmic histogram */
  sinr::histogram::histogramdB(data,bincenters,binvals,numbins);
  thrust::copy(bincenters.begin(),bincenters.end(),bincenters_std.begin());
  thrust::copy(binvals.begin(),binvals.end(),binvals_std.begin());

  cout<<"Logarithmic Histogram"<<endl;
  for (unsigned int bin = 0; bin < numbins; bin++) {
    cout<<"BinCenter: "<<bincenters_std.at(bin)<<", BinValue: "<<binvals_std.at(bin)<<endl;
  }
  cout<<endl<<endl;

  return 0;
}
