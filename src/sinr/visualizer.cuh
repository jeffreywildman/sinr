#ifndef __VISUALIZER_CUH__
#define __VISUALIZER_CUH__

#include <cassert>

//#include <QImage>
//#include <QString>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/extrema.h>

#include <sinr/types.h>
#include <sinr/coordinates.cuh>
#include <sinr/bitmap.h>
#include <sinr/functors.cuh>


namespace sinr {

  namespace visualize {

    /** Unary functor to restrict data values to a given range, normalize the values, and convert the values to grayscale.
     */
    template <typename T>
      struct Normalize : public thrust::unary_function<T, uchar4> {
        T _min, _max, _height;

        /**
         * @param min Lower bound data values by min before normalization.
         * @param max Upper bound data values by max before normalization.
         */
        Normalize(T min, T max) : _min(min), _max(max) {
          _height = _max - _min;
        }

        __host__ __device__ uchar4 operator()(T x) {
          unsigned char vis = min(max((x-_min)/_height,T(0.0)),T(1.0))*255;
          uchar4 ptr;
          ptr.x = vis;
          ptr.y = vis;
          ptr.z = vis;
          ptr.w = 255;
          return ptr;
        }
      };


    /** Unary functor to convert values to dB, restrict values to a given range, normalize the values, and convert the 
     * values to grayscale.
     */
    template <typename T>
      struct NormalizedB : public thrust::unary_function<T, uchar4> {
        T _mindB, _maxdB, _heightdB;

        /**
         * @param min Lower bound data values by min before normalization.
         * @param max Upper bound data values by max before normalization.
         */
        NormalizedB(T min, T max) : _mindB(T(10.0)*log10(min)), _maxdB(T(10.0)*log10(max)) {
          _heightdB = _maxdB - _mindB;
        }

        __host__ __device__ uchar4 operator()(T x) {
          unsigned char vis = min(max((T(10.0)*log10(x)-_mindB)/_heightdB,T(0.0)),T(1.0))*255;
          uchar4 ptr;
          ptr.x = vis;
          ptr.y = vis;
          ptr.z = vis;
          ptr.w = 255;
          return ptr;
        }
      };


    /** Unary functor to restrict data values to a given neg/pos range, normalize the values, and convert the values to
     * red/blue-scale based on sign.
     */
    template <typename T>
      struct NormalizeTwoTone : public thrust::unary_function<T, uchar4> {
        T _min, _max, _height;

        /**
         * @param min Lower bound data values by min before normalization.
         * @param max Upper bound data values by max before normalization.
         */
        NormalizeTwoTone(T min, T max) : _min(min), _max(max) {
          _height = std::max(fabs(_max),fabs(_min));
        }

        __host__ __device__ uchar4 operator()(T x) {
          unsigned char vis = min(max(fabs(x)/_height,T(0.0)),T(1.0))*255;
          uchar4 ptr;
          if (x >= 0.0) {
            /* blue */
            ptr.x = 0;
            ptr.z = vis;
          } else {
            /* red */
            ptr.x = vis;
            ptr.z = 0;
          }
          ptr.y = 0;
          ptr.w = 255;
          return ptr;
        }
      };


    template <typename InputIterator, typename OutputIterator>
      OutputIterator grayscale(InputIterator first, 
                               InputIterator last, 
                               OutputIterator result, 
                               typename InputIterator::value_type min, 
                               typename InputIterator::value_type max) {
        return thrust::transform(first, last, result, Normalize<double>(min, max));
      }


    template <typename InputIterator, typename OutputIterator>
      OutputIterator grayscale(InputIterator first, 
                               InputIterator last, 
                               OutputIterator result) {
        InputIterator min = thrust::min_element(first, last);
        InputIterator max = thrust::max_element(first, last);
        return sinr::visualize::grayscale(first, last, result, *min, *max);
      }


    template <typename InputVector, typename OutputVector>
      void grayscale(InputVector &data, OutputVector &result) {
        result.resize(data.size());
        sinr::visualize::grayscale(data.begin(), data.end(), result.begin());
      }


    template <typename InputVector, typename OutputVector>
      void grayscale(InputVector &data, 
                     OutputVector &result, 
                     typename InputVector::value_type min, 
                     typename InputVector::value_type max) {
        result.resize(data.size());
        sinr::visualize::grayscale(data.begin(), data.end(), result.begin(), min, max);
      }


    template <typename InputIterator, typename OutputIterator>
      OutputIterator grayscaledB(InputIterator first, 
                                 InputIterator last, 
                                 OutputIterator result, 
                                 typename InputIterator::value_type min, 
                                 typename InputIterator::value_type max) {
        return thrust::transform(first, last, result, NormalizedB<double>(min, max));
      }


    template <typename InputIterator, typename OutputIterator>
      OutputIterator grayscaledB(InputIterator first, 
                                 InputIterator last, 
                                 OutputIterator result) {
        InputIterator min = thrust::min_element(first, last);
        InputIterator max = thrust::max_element(first, last);
        return sinr::visualize::grayscaledB(first, last, result, *min, *max);
      }


    template <typename InputVector, typename OutputVector>
      void grayscaledB(InputVector &data, OutputVector &result) {
        result.resize(data.size());
        sinr::visualize::grayscaledB(data.begin(), data.end(), result.begin());
      }


    template <typename InputVector, typename OutputVector>
      void grayscaledB(InputVector &data, 
                       OutputVector &result, 
                       typename InputVector::value_type min, 
                       typename InputVector::value_type max) {
        result.resize(data.size());
        sinr::visualize::grayscaledB(data.begin(), data.end(), result.begin(), min, max);
      }


    template <typename InputVector>
      void grayscaledB(InputVector &data, 
                       thrust::device_ptr<uchar4> &result, 
                       typename InputVector::value_type min, 
                       typename InputVector::value_type max) {
        sinr::visualize::grayscaledB(data.begin(), data.end(), result, min, max);
      }

    template <typename InputIterator, typename OutputIterator>
      OutputIterator twotone(InputIterator first, 
                             InputIterator last, 
                             OutputIterator result, 
                             typename InputIterator::value_type min, 
                             typename InputIterator::value_type max) {
        return thrust::transform(first, last, result, NormalizeTwoTone<double>(min, max));
      }


    template <typename InputIterator, typename OutputIterator>
      OutputIterator twotone(InputIterator first, 
                             InputIterator last, 
                             OutputIterator result) {
        InputIterator min = thrust::min_element(first, last);
        InputIterator max = thrust::max_element(first, last);
        return sinr::visualize::twotone(first, last, result, *min, *max);
      }


    template <typename InputVector, typename OutputVector>
      void twotone(InputVector &data, OutputVector &result) {
        result.resize(data.size());
        sinr::visualize::twotone(data.begin(), data.end(), result.begin());
      }


    template <typename InputVector, typename OutputVector>
      void twotone(InputVector &data, 
                   OutputVector &result, 
                   typename InputVector::value_type min, 
                   typename InputVector::value_type max) {
        result.resize(data.size());
        sinr::visualize::twotone(data.begin(), data.end(), result.begin(), min, max);
      }


    template <typename InputVector, typename OutputVector>
      void assoczone(InputVector &data,
                     OutputVector &result,
                     typename InputVector::value_type nodes) {
        typedef typename InputVector::value_type T;
        result.resize(data.size());
        /* add one to the data to shift from -1,N-1 to 0,N */
        sinr::visualize::grayscale(thrust::make_transform_iterator(data.begin(), sinr::functors::Plus<T>(1.0)),
                                   thrust::make_transform_iterator(data.end(), sinr::functors::Plus<T>(1.0)),
                                   result.begin(),
                                   0,
                                   nodes);
      }


    template <typename InputIterator>
      void outputBMP(InputIterator first, InputIterator last, unsigned int w, unsigned int h, const std::string filename) {
        assert(last - first == w*h && w && h);
        /** @todo assert InputIterator::value_type is equivalent to uchar4 */

        thrust::host_vector<uchar4> rgba(first, last);
        uchar4 *temp = (uchar4 *)thrust::raw_pointer_cast(&(rgba[0]));

        CBitmap bmp;
        /* masks due to byte storage */
        bmp.SetBits(temp, w, h, 0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000);
        bmp.Save(filename.c_str());

      }


    template <typename InputVector>
      void outputBMP(InputVector &data, unsigned int w, unsigned int h, const std::string filename) {
        assert(data.size() == w*h && w && h);
        sinr::visualize::outputBMP(data.begin(), data.end(), w, h, filename);
      }

  }; /* namespace visualize */

}; /* namespace sinr */


///* Save an RGBA map as a file using Qt.
// */
//void Visualizer::outputBMP2(const uchar4 *rgba, unsigned int w, unsigned int h, const std::string *filename) {
//  unsigned char* buffer = (unsigned char *)rgba;
//  QImage image((unsigned char *)rgba, w, h, QImage::Format_ARGB32);
//  image = image.mirrored(false, true);  /* flip vertically to account for Qt's coordinate system */
//  image = image.scaled(512, 512, Qt::KeepAspectRatio, Qt::FastTransformation);  /* scale up */
//  image.save(QString::fromStdString(filename));
//}

#endif /* __VISUALIZER_CUH__ */
