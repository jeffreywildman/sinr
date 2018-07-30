#ifndef __GUI_H__
#define __GUI_H__

#include <vector_types.h>       /* for uchar4 */

#include <GL/glew.h>          /* must be included before gl.h and glext.h */
#include <GL/glut.h>
#include <GL/freeglut_ext.h>
#include <cuda_runtime_api.h>  /* C API - no need for nvcc */
#include <cuda_gl_interop.h>    /* for cudaGraphicsResource */

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


/** The Gui class handles the display of data to the screen using OpenGL and CUDA's Graphics Library Interoperability.
 */
class Gui {

public:

  /** Initialize a GUI object.  
   *
   * @param w   Width of the GUI window (pixels).
   * @param h   Height of the GUI window (pixels).
   */
  Gui(int w, int h);
  ~Gui();

  /** Display a buffer within the GUI window.
   *
   * @param rgba Buffer containing a pseudo-2D array of RGBA data to display.
   * @param w    Width of the pseudo-2D array.
   * @param h    Height of the pseudo-2D array.
   */
  void display(const thrust::device_vector<uchar4> *const rgba, unsigned int w, unsigned int h);    /* single-frame */
  
  /** Display a buffer within the GUI window.
   *
   * @param rgba Buffer containing a pseudo-2D array of RGBA data to display.
   * @param w    Width of the pseudo-2D array.
   * @param h    Height of the pseudo-2D array.
   */
  void display(const thrust::host_vector<uchar4> *const rgba, unsigned int w, unsigned int h);      /* single-frame */
 
  /** Display a sequence of buffers within the GUI window.
   *
   * @param render  Pointer to function that takes three unsigned ints (frame number, buffer width, buffer height), and 
   * a uchar4 * (buffer) with device memory allocated.  The function returns an unsigned int indicating whether or not
   * the buffer contains a valid frame to display.
   * @param w       Width of the pseudo-2D array.
   * @param h       Height of the pseudo-2D array.
   */
  void run(unsigned int (*render)(unsigned int, unsigned int, unsigned int, uchar4 *), unsigned int w, unsigned int h);    /* movie */

private:

  unsigned int win_w;
  unsigned int win_h;
  static unsigned int bmp_w;
  static unsigned int bmp_h;
  static unsigned int ticks;
  static GLuint bufferObj;
  static cudaGraphicsResource *resource;
  static unsigned int (*render)(unsigned int, unsigned int, unsigned int, uchar4 *);

  /** Open the GUI window.
   */
  void openWindow();

  /** Close the GUI window.
   *
   * @todo Implement this function, if needed.
   */
  static void closeWindow();
 
  /** GLUT callback function to display frame. 
   */
  static void disp_gcb(void);

  /** GLUT callback function to handle key events.
   *
   * @param key 
   * @param x   
   * @param y   
   */
  static void keyb_gcb(unsigned char key, int x __attribute__((unused)), int y __attribute__((unused)));

  /** GLUT callback function for idle time.
   */
  static void idle_gcb(void);

};

#endif /* __GUI_H__ */
