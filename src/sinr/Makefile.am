CUDALT=$(top_srcdir)/tools/cudalt

lib_LTLIBRARIES				= libsinr.la

libsinr_la_CPPFLAGS		= -I$(top_srcdir)/src
libsinr_la_LDFLAGS		=
libsinr_la_LIBADD			=
libsinr_la_includedir	= $(includedir)/sinr

libsinr_la_SOURCES = gui.cu \
										 util.cpp

nobase_libsinr_la_include_HEADERS = arena.h \
																		network.h \
																		coordinates.cuh \
																		histogram.cuh \
																		networkmetrics.cuh \
																		visualizer.cuh \
																		radiationpattern.cuh \
																		spatialdensity.cuh \
																		gui.h \
																		types.h \
																		bitmap.h \
																		optioniterator.h \
																		util.h

# Enable silent rules for nvcc
# See: http://www.gnu.org/software/automake/manual/html_node/Automake-silent_002drules-Option.html
nvcc_verbose		= $(nvcc_verbose_@AM_V@)
nvcc_verbose_		= $(nvcc_verbose_@AM_DEFAULT_V@)
nvcc_verbose_0	= @echo "  NVCC     $@";
.cu.lo:
	$(nvcc_verbose)$(CUDALT) $@ $(NVCC) $(NVCCFLAGS) $(libsinr_la_CPPFLAGS) -c $<

MAINTAINERCLEANFILES = Makefile.in
