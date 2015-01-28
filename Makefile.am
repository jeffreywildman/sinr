ACLOCAL_AMFLAGS = -I m4

SUBDIRS = doc src .

doxygen-doc:
	(cd doc && $(MAKE) $(AM_MAKEFLAGS) $@) || exit 1

.PHONY: doxygen-doc

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
