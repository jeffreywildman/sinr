#ifndef __HISTOGRAM_CUH__
#define __HISTOGRAM_CUH__

#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/adjacent_difference.h>

#include <sinr/functors.cuh>


namespace sinr {
  namespace histogram {

    /** Class to create and maintain linearly-spaced bins of a histogram. */
    template <typename T>
      struct HistBin : public thrust::unary_function<T, unsigned int> {
        T min;
        T max;
        unsigned int numbins;
        T width;

        /** Create @p _numbins linearly-spaced bins with range @p _min to @p max. */
        __host__ __device__ HistBin(T _min, T _max, unsigned int _numbins) :
          min(_min), max(_max), numbins(_numbins) {
            width = (_max - _min)/T(_numbins);
          }

        /** Unary function to convert data values (@p val) to histogram bin indices. */
        __host__ __device__ unsigned int operator()(T val) const {
          return (unsigned int) ((val - min)/width);
        }

        /** Return the set of centers associated with the histogram bins indices. */
        thrust::host_vector<T> getBins() const {
          thrust::host_vector<T> bincenter(numbins);
          thrust::sequence(bincenter.begin(), bincenter.end(), min+width/2.0, width);
          return bincenter;
        }
      };


    /** Class to create and maintain logarithmically-spaced bins of a histogram. */
    template <typename T>
      struct HistBindB : public thrust::unary_function<T, unsigned int> {
        T mindB;
        T maxdB;
        unsigned int numbins;
        T widthdB;
        sinr::functors::Lin2dB<T> lin2dB;

        /** Create @p _numbins logarithmically-spaced bins with linear range @p _min to @p max. */
        __host__ __device__ HistBindB(T _min, T _max, unsigned int _numbins) : lin2dB(), 
          mindB(lin2dB(_min)), maxdB(lin2dB(_max)), numbins(_numbins) {
            widthdB = (maxdB - mindB)/T(numbins);
          }

        /** Unary function to convert linear data values (@p val) to logarithmic histogram bin indices. */
        __host__ unsigned int operator()(T val) const {
          return (unsigned int) ((lin2dB(val) - mindB)/widthdB);
        }

        /** Return the set of centers associated with the histogram bins indices. */
        thrust::host_vector<T> getBins() const {
          thrust::host_vector<T> bincenter(numbins);
          thrust::sequence(bincenter.begin(), bincenter.end(), mindB+widthdB/2.0, widthdB);
          thrust::transform(bincenter.begin(), bincenter.end(), bincenter.begin(), sinr::functors::dB2Lin<T>());
          return bincenter;
        }
      };


    template <typename InputVector, typename OutputVector1, typename OutputVector2>
      void histogram(InputVector &data, OutputVector1 &hbins, OutputVector2 &hvals, unsigned int numbins) {
        typedef typename InputVector::value_type T;

        /* sort a copy of the data */
        InputVector cdata(data);
        thrust::sort(cdata.begin(), cdata.end());

        /* transform data into bin indices */
        T min = cdata.front();
        T max = cdata.back();
        HistBin<T> h(min, max, numbins);
        thrust::transform(cdata.begin(), cdata.end(), cdata.begin(), h);

        hbins = OutputVector1(h.getBins());
        hvals.resize(numbins);

        /* find the end of each bin of values */
        thrust::upper_bound(cdata.begin(),
                            cdata.end(), 
                            thrust::counting_iterator<unsigned int>(0),
                            thrust::counting_iterator<unsigned int>(numbins),
                            hvals.begin());

        /* compute the histogram by taking differences of the cumulative histogram */
        thrust::adjacent_difference(hvals.begin(), hvals.end(), hvals.begin());

        return;
      }


    template <typename InputVector, typename OutputVector1, typename OutputVector2>
      void histogramdB(InputVector &data, OutputVector1 &hbins, OutputVector2 &hvals, unsigned int numbins) {
        typedef typename InputVector::value_type T;

        /* sort a copy of the data */
        InputVector cdata(data);
        thrust::sort(cdata.begin(), cdata.end());

        /* transform data into bin indices */
        T min = cdata.front();
        T max = cdata.back();
        HistBindB<T> h(min, max, numbins);
        thrust::transform(cdata.begin(), cdata.end(), cdata.begin(), h);

        hbins = OutputVector1(h.getBins());
        hvals.resize(numbins);

        /* find the end of each bin of values */
        thrust::upper_bound(cdata.begin(),
                            cdata.end(), 
                            thrust::counting_iterator<unsigned int>(0),
                            thrust::counting_iterator<unsigned int>(numbins),
                            hvals.begin());

        /* compute the histogram by taking differences of the cumulative histogram */
        thrust::adjacent_difference(hvals.begin(), hvals.end(), hvals.begin());

        return;
      }

  }; /* namespace histogram */
}; /* namespace sinr */

#endif /* __HISTOGRAM_CUH__ */
