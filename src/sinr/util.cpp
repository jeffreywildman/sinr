#include <sinr/util.h>

#include <ctime>
#include <cassert>
#include <string>
using std::string;
#include <vector>
using std::vector;

#include <gsl/gsl_rng.h>
const gsl_rng_type * T = gsl_rng_default;
gsl_rng * rng = gsl_rng_alloc(T);


string Util::to_string(int i) {
  return std::to_string(i);
}


string Util::to_string(unsigned int i) {
  return std::to_string(i);
}


string Util::to_string(double d) {
  return std::to_string(d);
}


string Util::to_string(vector<unsigned int> v) {
  assert(v.size() > 0);
  string str;
  for (unsigned int i = 0; i < v.size() - 1; i++) {
    str += Util::to_string(v.at(i)) + "-";
  }
  /* last element without trailing dash */
  str += Util::to_string(v.back());
  return str;
}


string Util::to_string(vector<int> v) {
  assert(v.size() > 0);
  string str;
  for (unsigned int i = 0; i < v.size() - 1; i++) {
    str += Util::to_string(v.at(i)) + "-";
  }
  /* last element without trailing dash */
  str += Util::to_string(v.back());
  return str;
}


/** Check if path ends in a forward slash
 */
bool Util::trailingSlash(string path) {
  assert(path.size() > 0);

  return path.back() == '/';
}


/** Check if the string ends with a slash. If it DOES NOT, then add it
 */
string Util::slashify(string path) {
  assert(path.size() > 0);

  if(!trailingSlash(path)) {
    path += '/';
    assert(trailingSlash(path));
  }

  return path;
}


/** Check if a program is installed
 */
bool Util::programInstalled(string programName) {
  assert(programName.size() > 0);

  string whichCommand = "which " + programName + " > /dev/null";
  int result = system(whichCommand.c_str());

  return result == 0;
}


int Util::deleteDirContents(string dir) {
  assert(Util::programInstalled("gvfs-trash"));

  string deleteCommand = "gvfs-trash --force " + Util::slashify(dir) + "*";
  int result = system(deleteCommand.c_str());

  return result;
}


void Util::seedRandomGenerator(long value) {
  gsl_rng_set(rng, value);
  srand(value); 
}


/// @brief Seed the random number generator with the current time.
void Util::seedRandomGenerator() {
  seedRandomGenerator(time(NULL));
}


// uniform float [a,b)
double Util::uniform(double a, double b) {
  assert(a <= b);
  if(a == b)
    return a;
  return gsl_rng_uniform(rng)*(b-a) + a;
}


uint64_t Util::getTimeNS() {
  struct timespec t;
  clock_gettime(CLOCK_REALTIME, &t);
  return t.tv_sec * Util::nanoPerSec + t.tv_nsec;
}
