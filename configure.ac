dnl Autoconf initialization and config macros
dnl ***************************************************************************

AC_PREREQ([2.69])
AC_INIT([sinr],[0.0.1],[a@b.c],[],[http://oman.ece.drexel.edu/redmine/projects/sinr])

dnl Come up with a nice explanation for this when it is actually used
dnl AC_CONFIG_HEADERS([config.h])

dnl Source directory sanity check using an arbitrary source file
AC_CONFIG_SRCDIR([src/sinr/network.h])

dnl install-sh will be searched (and found) here
AC_CONFIG_AUX_DIR([config])
AC_CONFIG_MACRO_DIR([m4])


dnl Automake initialization and config macros
dnl ***************************************************************************

AM_INIT_AUTOMAKE([color-tests subdir-objects silent-rules no-exeext])

AM_SILENT_RULES([yes])


dnl Doxygen initialization and config macros
dnl ***************************************************************************

DX_DOXYGEN_FEATURE([ON])
DX_PDF_FEATURE([OFF])
DX_PS_FEATURE([OFF])
dnl config and output directory locations are given relative to the doc directory
dnl this is where the Makefile will call doxygen commands from
DX_INIT_DOXYGEN([$PACKAGE_NAME],[./doxygen.cfg],[./doxygen])
AS_IF([test "${DX_FLAG_doc}" = 1],[enable_doxygen=yes],[enable_doxygen=no]) 
AC_CONFIG_FILES([doc/doxygen.cfg])


dnl Other macros
dnl ***************************************************************************


dnl Prevent AC_PROG_C* from messin' wit us
: ${CFLAGS=""}
: ${CXXFLAGS=""}
: ${NVCCFLAGS=""}
save_cflags=$CFLAGS
AC_PROG_CC
AC_PROG_CC_C_O
AC_PROG_CXX
AC_PROG_CXX_C_O
AC_PROG_INSTALL
NVCC="nvcc"
CFLAGS=$save_cflags

dnl http://www.flameeyes.eu/autotools-mythbuster/autoconf/arguments.html
AC_ARG_ENABLE([debug],
              AS_HELP_STRING([--enable-debug],[add '-g' flag to compiler call [default=yes]]),
              [enable_debug=$enableval],
              [enable_debug=yes])
AC_ARG_ENABLE([optimization],
              AS_HELP_STRING([--enable-optimization],[add '-O2' flag to compiler call [default=no]]),
              [enable_optimization=$enableval],
              [enable_optimization=no])
AC_ARG_ENABLE([profile],
              AS_HELP_STRING([--enable-profile],[add '-pg' flag to compiler call [default=no]]),
              [enable_profile=$enableval],
              [enable_profile=no])

if test "x$enable_debug" == "xyes"
then
	CFLAGS="-ggdb3 -fno-inline ${CFLAGS}"
	CXXFLAGS="-ggdb3 -fno-inline ${CXXFLAGS}"
  dnl thrust does not like debugging?
  NVCCFLAGS="-Xcompiler -g -Xcompiler -fno-inline ${NVCCFLAGS}"
else
  CFLAGS="-s -DNDEBUG ${CFLAGS}"
  CXXFLAGS="-s -DNDEBUG ${CXXFLAGS}"
  NVCCFLAGS="-Xcompiler -s -DNDEBUG ${NVCCFLAGS}"
fi

if test "x$enable_optimization" == "xyes"
then
	CFLAGS="-O2 ${CFLAGS}"
	CXXFLAGS="-O2 ${CXXFLAGS}"
  NVCCFLAGS="-O2 ${NVCCFLAGS}"
else
  CFLAGS="-O0 ${CFLAGS}"
  CXXFLAGS="-O0 ${CXXFLAGS}"
  NVCCFLAGS="-O0 ${NVCCFLAGS}"
fi

if test "x$enable_profile" == "xyes"
then
	CFLAGS="-pg ${CFLAGS}"
	CXXFLAGS="-pg ${CXXFLAGS}"
  NVCCFLAGS="-pg ${NVCCFLAGS}"
fi


dnl Check the libraries that use pkg-config
PKG_PROG_PKG_CONFIG([])
PKG_CHECK_MODULES([SINR_DEPS], [gl glu glew gsl QtCore])

dnl Check the libraries that don't use pkg-config
dnl TODO: checks for glut 
dnl TODO: checks for cudart
AC_CHECK_HEADERS_ONCE([thrust/version.h])


OMAN_INCL_DIR="/opt/oman-install-temp/include"

dnl Add these flags to gcc and g++ 
CPPFLAGS="-I${OMAN_INCL_DIR} \
  `pkg-config --cflags-only-other gl glu glew gsl QtCore` \
  `pkg-config --cflags-only-I gl glu glew gsl QtCore`"

CFLAGS="${CFLAGS} -pipe -Wall -Wextra" dnl -Werror"
CXXFLAGS="${CXXFLAGS} -pipe -Wall -Wextra -std=c++0x" dnl -Werror"
NVCCFLAGS="${NVCCFLAGS} -Xcompiler -Wall -Xcompiler -Wno-unused-but-set-variable \
  -I${OMAN_INCL_DIR} \ 
  `pkg-config --cflags QtCore gl glu glew gsl QtCore` \
  -arch=compute_20 -code=compute_20"

dnl -Xcompiler -Wextra -Xcompiler -Werror"

LDFLAGS="${LDFLAGS} -L/usr/lib -L/usr/local/lib -L/opt/oman-install-temp/lib \
  `pkg-config --libs-only-other gl glu glew gsl QtCore`"

LIBS="${LIBS} -Wl,--as-needed -lglut -lcudart \
  `pkg-config --libs-only-l gl glu glew gsl QtCore` \
  -loman"

AC_SUBST(NVCC)
AC_SUBST(NVCCFLAGS)


dnl Libtool related macros
LT_PREREQ([2.4.2])
dnl Turn off shared libraries during development to cut down on compilation time
LT_INIT([disable-shared]) 

AC_CONFIG_FILES([Makefile \
                 src/Makefile \
                 doc/Makefile])
AC_OUTPUT

echo \
"-------------------------------------------------------------------------------

${PACKAGE_NAME} Version ${PACKAGE_VERSION}

Prefix: '${prefix}'
C Compiler: '${CC} ${CPPFLAGS} ${CFLAGS}'
C++ Compiler: '${CXX} ${CPPFLAGS} ${CXXFLAGS}'
NVCC Compiler: '${NVCC} ${NVCCFLAGS}'

Package Features:
    Debugging Symbols:          ${enable_debug}
    Optimization Flags:         ${enable_optimization}
    Profiling Code:             ${enable_profile}
    Shared Libraries:           ${enable_shared}
    Doxygen Documentation:      ${enable_doxygen}

Now type 'make @<:@<target>@:>@' where the optional <target> is:
    all                 - build all binaries
    install             - install everything

-------------------------------------------------------------------------------"
