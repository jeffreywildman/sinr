#ifndef __OPTIONITERATOR_H__
#define __OPTIONITERATOR_H__

#include <cassert>
#include <cmath>
#include <vector>

typedef enum optionComputeMethod {
  OCM_BASEB = 0,
  OCM_GRAYCODE = 1,
} optionComputeMethod_t;

/**
 * OptionIterator provides methods for iterating through all assignments of @p b values to each of @p n parameters.  
 * Essentially, we are storing and incrementing a decimal number while providing its conversion to an @p n digit number
 * of base @p b.
 */
template <typename Digit = unsigned int>
class OptionIterator {

public:

  /** Initialize the OptionIterator with the base of the number @p _b and number of digits @p _n. */
  OptionIterator(unsigned int _b, unsigned int _n, optionComputeMethod_t _ocm = OCM_BASEB) : b(_b), n(_n), ocm(_ocm) {
    assert(b >= 2 && n > 0);
    
    dmax = computeDecMax(b,n);
    setDecVal(0);
  }

  /** Set the decimal value of the OptionIterator to @p _dval. */ 
  std::vector<Digit> setDecVal(unsigned int _dval) {
    assert(_dval <= dmax);

    dval = _dval;
    return computeVal();
  }

  /** Get the current decimal value of the OptionIterator. */
  unsigned int getDecVal() {
    return dval;
  }

  /** Get the maximum decimal value able to be stored by the OptionIterator. */
  unsigned int getDecMax() {
    return dmax;
  }

  /** Get the current option values stored by the OptionIterator. */
  std::vector<Digit> getVal() {
    return computeVal();
  }

  /** Increment the option value stored by the OptionIterator. */
  void increment() {
    dval++;
  }

private:
 
  /** Compute the maximum decimal value possible for an @p n digit base @p b number. */
  unsigned int computeDecMax (unsigned int b, unsigned int n) {
    return pow(b,n)-1;
  }

  /** Compute the option value. */
  std::vector<Digit> computeVal() {
    switch(ocm) {
      case OCM_BASEB:
        return computeVal_BaseB();
      case OCM_GRAYCODE:
        return computeVal_GrayCode();
      default:
        assert(0);
        break;
    }
  }

  /** Compute the option value from the stored decimal value. */
  std::vector<Digit> computeVal_BaseB() {
    std::vector<Digit> digits(n,0);
    unsigned int _dval = dval;
    for(unsigned int i = 0; i < n; i++) {
      digits.at(i) = _dval % b;
      _dval /= b;
    }
    return digits;
  }

  /** Compute the option value from the stored decimal value using an base b Gray code conversion. 
   * See: http://en.wikipedia.org/wiki/Gray_code#n-ary_Gray_code */
  std::vector<Digit> computeVal_GrayCode() {
    std::vector<Digit> digits = computeVal_BaseB();
    unsigned int shift = 0;
    for (int i = n-1; i >= 0; i--) {
      digits.at(i) = (digits.at(i) + shift) % b;
      shift = shift + b - digits.at(i);
    }
    return digits;
  }

  unsigned int b;             /**< number of values base */
  unsigned int n;             /**< number of digits */
  optionComputeMethod_t ocm;  /**< method of computing the option value */ 
  
  unsigned int dval;  /**< current decimal value stored in object */
  unsigned int dmax;  /**< maximum decimal value able to be stored by object */
};

#endif /* __OPTIONITERATOR_H__ */
