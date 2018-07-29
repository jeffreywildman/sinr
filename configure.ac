dnl Autoconf initialization and config macros
dnl ***************************************************************************

AC_PREREQ([2.69])
AC_INIT([sinr],[0.0.1],[jeffrey.wildman@gmail.com],[],[])

dnl Come up with a nice explanation for this when it is actually used
dnl AC_CONFIG_HEADERS([config.h])

dnl Source directory sanity check using an arbitrary source file
AC_CONFIG_SRCDIR([src/sinr/network.h])

dnl install-sh will be searched (and found) here
AC_CONFIG_AUX_DIR([config])
AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_FILES([Makefile \
                 src/Makefile \
                 src/sinr/Makefile \
                 doc/Makefile])

# Silence warning: ar: 'u' modifier ignored since 'D' is the default
AC_SUBST(AR_FLAGS, [cr])


dnl Configure compilers and flags
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
  NVCCFLAGS="-g -Xcompiler -fno-inline ${NVCCFLAGS}"
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
PKG_CFGS="cudart-9.2 gl glu glew gsl"
PKG_PROG_PKG_CONFIG([])
PKG_CHECK_MODULES([SINR_DEPS], [${PKG_CFGS}])

dnl Check the libraries that don't use pkg-config
dnl TODO: checks for glut
AC_CHECK_HEADERS_ONCE([thrust/version.h])

dnl Add these flags to gcc and g++
CPPFLAGS="`pkg-config --cflags-only-other ${PKG_CFGS}` \
  `pkg-config --cflags-only-I ${PKG_CFGS}`"

CFLAGS="${CFLAGS} -pipe -Wall -Wextra -Werror"
CXXFLAGS="${CXXFLAGS} -pipe -Wall -Wextra -Werror -std=c++11"
NVCCFLAGS="${NVCCFLAGS} -ccbin ${CXX} -Xcompiler -pipe,-Wall,-Wextra,-Werror -std=c++11 \
  -arch=compute_30 -code=compute_30,sm_30"

LDFLAGS="${LDFLAGS} -L/usr/lib -L/usr/local/lib \
  `pkg-config --libs-only-L ${PKG_CFGS}`"

# pthread not properly pulled in by cudart pc file?
LIBS="${LIBS} -Wl,--as-needed -pthread -lglut \
  `pkg-config --libs-only-l ${PKG_CFGS}`"

AC_SUBST(NVCC)
AC_SUBST(NVCCFLAGS)
AC_SUBST(PKG_CFGS)


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


dnl Automake initialization and config macros
dnl ***************************************************************************

AM_INIT_AUTOMAKE([color-tests subdir-objects silent-rules no-exeext])

AM_SILENT_RULES([yes])

AM_PROG_AR


dnl Libtool related macros
dnl ***************************************************************************

LT_PREREQ([2.4.2])
LT_INIT()


AC_OUTPUT

echo \
"-------------------------------------------------------------------------------

${PACKAGE_NAME} Version ${PACKAGE_VERSION}

Prefix: '${prefix}'
C Compiler: '${CC} ${CPPFLAGS} ${CFLAGS}'
C++ Compiler: '${CXX} ${CPPFLAGS} ${CXXFLAGS}'
NVCC Compiler: '${NVCC} ${CPPFLAGS} ${NVCCFLAGS}'

Package Features:
    Debugging Symbols:          ${enable_debug}
    Optimization Flags:         ${enable_optimization}
    Profiling Code:             ${enable_profile}
    Static Libraries:           ${enable_static}
    Shared Libraries:           ${enable_shared}
    Doxygen Documentation:      ${enable_doxygen}

Now type 'make @<:@<target>@:>@' where the optional <target> is:
    all                 - build all binaries
    install             - install everything

-------------------------------------------------------------------------------"
