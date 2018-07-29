include $(top_srcdir)/Makefile-common.am

ACLOCAL_AMFLAGS = -I m4
SUBDIRS = doc src .

doxygen-doc:
	(cd doc && $(MAKE) $(AM_MAKEFLAGS) $@) || exit 1

# Avoid autogen.sh warning about missing README
README: README.md

.PHONY: doxygen-doc

MAINTAINERCLEANFILES = aclocal.m4 \
											 configure \
											 config/ar-lib \
											 config/compile \
											 config/depcomp \
											 config/missing \
											 config/config.guess \
											 config/ltmain.sh \
											 config/ltmain.sh.orig \
											 config/config.sub \
											 config/install-sh \
											 m4/ltsugar.m4 \
											 m4/libtool.m4 \
											 m4/ltversion.m4 \
											 m4/lt~obsolete.m4 \
											 m4/ltoptions.m4 \
											 Makefile.in
