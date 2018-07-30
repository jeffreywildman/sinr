#include "gui.h"

#include <cassert>
#include <iostream>

#include <GL/glew.h>          /* must be included before gl.h and glext.h */
#include <GL/glut.h>
#include <GL/freeglut_ext.h>
#include <cuda_runtime_api.h>  /* C API - no need for nvcc */
#include <cuda_gl_interop.h>    /* for cudaGraphicsResource */

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


static void HandleError(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    std::cout<<cudaGetErrorString(err)<<" in "<<file<<" at line "<<line<<std::endl;
    exit(EXIT_FAILURE);
  }
}


#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


unsigned int Gui::bmp_w;
unsigned int Gui::bmp_h;
GLuint Gui::bufferObj;
cudaGraphicsResource *Gui::resource;
unsigned int Gui::ticks;
unsigned int (*Gui::render)(unsigned int, unsigned int, unsigned int, uchar4 *) = NULL;


Gui::Gui(int _win_w, int _win_h) {
  win_w = _win_w;
  win_h = _win_h;
}


Gui::~Gui() {;}


void Gui::openWindow() {
  /* initialize OpenGL driver */
  int argc = 0;
  char *argv = NULL;
  glutInit(&argc, &argv);
  glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGBA);
  glutInitWindowPosition(100,100);
  glutInitWindowSize(win_w,win_h);
  glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);
  glutCreateWindow("OpenGL Window");
  glewInit();
}


void Gui::display(const thrust::device_vector<uchar4> *const rgba, unsigned int w, unsigned int h) {
  assert(rgba && w*h == rgba->size());

  bmp_w = w;
  bmp_h = h;

  uchar4 *dev_ptr;
  size_t size;

  this->openWindow();
  
  /* create shared resource */
  /** @todo Reallocate buffer if bitmap buffer size changes */
  glGenBuffers(1, &bufferObj);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);
  glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, rgba->size()*sizeof(uchar4), NULL, GL_DYNAMIC_DRAW_ARB);
  HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(&resource, bufferObj, cudaGraphicsMapFlagsNone));
  
  HANDLE_ERROR(cudaGraphicsMapResources(1, &resource, NULL));
  HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&dev_ptr, &size, resource));

  thrust::device_ptr<uchar4> temp_tdp = thrust::device_pointer_cast(dev_ptr);
  thrust::copy(rgba->begin(), rgba->end(), temp_tdp);
  
  HANDLE_ERROR(cudaGraphicsUnmapResources(1, &resource, NULL));

  glPixelZoom(win_w/(float)bmp_w, win_h/(float)bmp_h);

  std::cout<<"Press ESC to close window..."<<std::endl;
 
  glutKeyboardFunc(keyb_gcb);
  glutDisplayFunc(disp_gcb);
  glutIdleFunc(NULL);
  glutMainLoop();
}


void Gui::run(unsigned int (*_render)(unsigned int, unsigned int, unsigned int, uchar4 *), unsigned int w, unsigned int h) {
  assert(_render && w && h);

  render = _render;
  bmp_w = w;
  bmp_h = h;
  ticks = 0;

  this->openWindow();
  
  /* create shared resource */
  /** @todo Reallocate buffer if bitmap buffer size changes */
  glGenBuffers(1, &bufferObj);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);
  glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, bmp_w*bmp_h*sizeof(uchar4), NULL, GL_DYNAMIC_DRAW_ARB);
  HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(&resource, bufferObj, cudaGraphicsMapFlagsNone));
 
  glPixelZoom(win_w/(float)bmp_w, win_h/(float)bmp_h);
  
  std::cout<<"Press ESC to close window..."<<std::endl;
  
  glutKeyboardFunc(keyb_gcb);
  glutDisplayFunc(disp_gcb);
  glutIdleFunc(idle_gcb);
  glutMainLoop();
}


void Gui::display(const thrust::host_vector<uchar4> *const rgba, unsigned int w, unsigned int h) {
  assert(rgba && w*h == rgba->size());

  thrust::device_vector<uchar4> temp_rgba = *rgba;

  this->display(&temp_rgba, w, h);
}


void Gui::disp_gcb(void) {
  glDrawPixels(bmp_w, bmp_h, GL_RGBA, GL_UNSIGNED_BYTE, 0);
  glutSwapBuffers();
}


void Gui::keyb_gcb(unsigned char key, int x __attribute__((unused)), int y __attribute__((unused))) {
  switch(key) {
  case 27:
    /** @todo clean up OpenGL and CUDA */
    /* cudaGraphicsUnregisterResource fails for unknown reasons when called here...
     * perhaps due to the fact that glut has closed by now and the resource doesn't exist? */
    //HANDLE_ERROR(cudaDeviceSynchronize());
    //HANDLE_ERROR(cudaGraphicsUnregisterResource(resource));  
    //glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    //glDeleteBuffers(1, &bufferObj);

    glutLeaveMainLoop();
  }
}


void Gui::idle_gcb(void) {
  uchar4 *dev_ptr;
  size_t size;

  HANDLE_ERROR(cudaGraphicsMapResources(1, &resource, NULL));
  HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&dev_ptr, &size, resource));

  if (!render(ticks++, bmp_w, bmp_h, dev_ptr)) {
    /** @todo clean up OpenGL and CUDA */
    /* cudaGraphicsUnregisterResource fails for unknown reasons when called here...
     * perhaps due to the fact that glut has closed by now and the resource doesn't exist? */
    //HANDLE_ERROR(cudaDeviceSynchronize());
    //HANDLE_ERROR(cudaGraphicsUnregisterResource(resource));  
    //glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    //glDeleteBuffers(1, &bufferObj);
    
    glutLeaveMainLoop();
  }
  
  HANDLE_ERROR(cudaGraphicsUnmapResources(1, &resource, NULL));

  glutPostRedisplay();
}
