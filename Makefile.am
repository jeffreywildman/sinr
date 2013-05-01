ACLOCAL_AMFLAGS = -I m4

SUBDIRS = doc src .

doxygen-doc:
	(cd doc && $(MAKE) $(AM_MAKEFLAGS) $@) || exit 1

.PHONY: doxygen-doc

# Note: Automake 1.12.6 includes automatic support for CSCOPE
# These rules will be unnecessary when automake is finally upgraded in Ubuntu
cscope: cscopeclean cscope.files
	cscope -b -q -i $(top_srcdir)/cscope.files -f $(top_srcdir)/cscope.out
	
cscopeclean:
	rm -f $(top_srcdir)/cscope.files

cscope.files:
	find $(abs_top_srcdir)/src -name '*.cu' -o -name '*.cpp' -o -name '*.h' > $(top_srcdir)/cscope.files

MAINTAINERCLEANFILES = aclocal.m4 \
					   configure \
					   COPYING \
					   config/depcomp \
					   config/missing \
					   config/config.guess \
					   config/ltmain.sh \
					   config/ltmain.sh.orig \
					   config/config.sub \
					   config/install-sh \
					   INSTALL \
					   m4/ltsugar.m4 \
					   m4/libtool.m4 \
					   m4/ltversion.m4 \
					   m4/lt~obsolete.m4 \
					   m4/ltoptions.m4 \
					   Makefile.in
