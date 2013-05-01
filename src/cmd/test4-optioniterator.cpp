#include <iostream>
#include <vector>
#include <algorithm>

#include <oman/general/omandirs.h>

#include <sinr/optioniterator.h>
#include <sinr/util.h>

using namespace std;

typedef double Real;

void print(unsigned int i) {
  cout<<i<<" ";
}

/** This test program creates and tests a BaseBUInt object.
 */
int main(int argc __attribute__((unused)), char** argv __attribute__((unused))) {

  Util::deleteDirContents(OmanDirs::images());

  unsigned int b = 3;
  unsigned int n = 3;


  cout<<"Standard Decimal Iterator: "<<endl;
  for (OptionIterator<int> bnum(b,n,OCM_BASEB); bnum.getDecVal() <= bnum.getDecMax(); bnum.increment()) {
    cout<<bnum.getDecVal()<<": \t"<<Util::to_string(bnum.getVal())<<endl;
  }
  cout<<"Max: "<<OptionIterator<int>(b,n,OCM_BASEB).getDecMax()<<endl<<endl;

  
  cout<<"Gray Code Iterator: "<<endl;
  for (OptionIterator<int> bnum(b,n,OCM_GRAYCODE); bnum.getDecVal() <= bnum.getDecMax(); bnum.increment()) {
    cout<<bnum.getDecVal()<<": \t"<<Util::to_string(bnum.getVal())<<endl;
  }
  cout<<"Max: "<<OptionIterator<int>(b,n,OCM_GRAYCODE).getDecMax()<<endl<<endl;

  return 0;
}
