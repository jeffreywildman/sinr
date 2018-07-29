include $(top_srcdir)/Makefile-common.am

lib_LTLIBRARIES				= libsinr.la

libsinr_la_CPPFLAGS		= -I$(top_srcdir)/src
libsinr_la_LDFLAGS		= -version-number $(LT_VERSION)
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

MAINTAINERCLEANFILES = Makefile.in
