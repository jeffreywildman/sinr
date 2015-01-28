SUBDIRS = .

bin_PROGRAMS = 	sinrvis-test \
								test0-radiationpattern \
								test1-spatialdensity \
								test2-associationzones \
								vis0-placement
#								test3-histogram \
#								test4-optioniterator \
#								exp0-vspixels \
#								exp1-vsnodes \
#								exp2-spatialcapacity \
#								exp3-placement
#								crowncom0-placement \
#								wifius0-modesvbands

# Stop-gap measure until we can get CUDA/Libtool/Automake to play nice with one another.
# Can't get libtool to recognize *.cu targets in *_SOURCES variables.
LIBSINRMAPS_SOURCES = \
									 sinr/arena.h \
									 sinr/network.h \
									 sinr/coordinates.cuh \
									 sinr/histogram.cuh \
									 sinr/networkmetrics.cuh \
									 sinr/visualizer.cuh \
									 sinr/radiationpattern.cuh \
									 sinr/spatialdensity.cuh \
									 sinr/gui.h \
									 sinr/gui.cu \
									 sinr/types.h \
									 sinr/bitmap.h \
									 sinr/optioniterator.h \
									 sinr/util.h \
									 sinr/util.cpp

sinrvis_test_CPPFLAGS = -I$(top_srcdir)/src
sinrvis_test_LDFLAGS  = #-static-libtool-libs
sinrvis_test_LDADD    =
sinrvis_test_SOURCES  = $(LIBSINRMAPS_SOURCES) \
												cmd/sinrvis-test.cu
# If there are no cpp files in the SOURCES variable above, then Automake will 
# choose the wrong linker.  Help Automake choose the C++ linker.
# See: http://www.gnu.org/software/automake/manual/html_node/How-the-Linker-is-Chosen.html#How-the-Linker-is-Chosen
nodist_EXTRA_sinrvis_test_SOURCES = dummy.cxx


# EXPERIMENTS 

exp0_vspixels_CPPFLAGS = -I$(top_srcdir)/src
exp0_vspixels_LDFLAGS  = 
exp0_vspixels_LDADD    = 
exp0_vspixels_SOURCES  = $(LIBSINRMAPS_SOURCES) \
														 cmd/exp0-vspixels.cu
nodist_EXTRA_exp0_vspixels_SOURCES = dummy.cxx


exp1_vsnodes_CPPFLAGS = -I$(top_srcdir)/src
exp1_vsnodes_LDFLAGS  = 
exp1_vsnodes_LDADD    = 
exp1_vsnodes_SOURCES  = $(LIBSINRMAPS_SOURCES) \
														cmd/exp1-vsnodes.cu
nodist_EXTRA_exp1_vsnodes_SOURCES = dummy.cxx


exp2_spatialcapacity_CPPFLAGS = -I$(top_srcdir)/src
exp2_spatialcapacity_LDFLAGS  = 
exp2_spatialcapacity_LDADD    = 
exp2_spatialcapacity_SOURCES  = $(LIBSINRMAPS_SOURCES) \
															cmd/exp2-spatialcapacity.cu
nodist_EXTRA_exp2_spatialcapacity_SOURCES = dummy.cxx


exp3_placement_CPPFLAGS = -I$(top_srcdir)/src
exp3_placement_LDFLAGS  = 
exp3_placement_LDADD    = 
exp3_placement_SOURCES  = $(LIBSINRMAPS_SOURCES) \
															cmd/exp3-placement.cu
nodist_EXTRA_exp3_placement_SOURCES = dummy.cxx


# TESTS

test0_radiationpattern_CPPFLAGS = -I$(top_srcdir)/src
test0_radiationpattern_LDFLAGS  = 
test0_radiationpattern_LDADD    =
test0_radiationpattern_SOURCES  = $(LIBSINRMAPS_SOURCES) \
												cmd/test0-radiationpattern.cu
nodist_EXTRA_test0_radiationpattern_SOURCES = dummy.cxx


test1_spatialdensity_CPPFLAGS = -I$(top_srcdir)/src
test1_spatialdensity_LDFLAGS  = 
test1_spatialdensity_LDADD    =
test1_spatialdensity_SOURCES  = $(LIBSINRMAPS_SOURCES) \
												cmd/test1-spatialdensity.cu
nodist_EXTRA_test1_spatialdensity_SOURCES = dummy.cxx


test2_associationzones_CPPFLAGS = -I$(top_srcdir)/src
test2_associationzones_LDFLAGS  = 
test2_associationzones_LDADD    =
test2_associationzones_SOURCES  = $(LIBSINRMAPS_SOURCES) \
												cmd/test2-associationzones.cu
nodist_EXTRA_test2_associationzones_SOURCES = dummy.cxx


test3_histogram_CPPFLAGS = -I$(top_srcdir)/src
test3_histogram_LDFLAGS  = 
test3_histogram_LDADD    =
test3_histogram_SOURCES  = $(LIBSINRMAPS_SOURCES) \
												cmd/test3-histogram.cu
nodist_EXTRA_test3_histogram_SOURCES = dummy.cxx


test4_optioniterator_CPPFLAGS = -I$(top_srcdir)/src
test4_optioniterator_LDFLAGS  = 
test4_optioniterator_LDADD    =
test4_optioniterator_SOURCES  = $(LIBSINRMAPS_SOURCES) \
												cmd/test4-optioniterator.cpp
nodist_EXTRA_test4_optioniterator_SOURCES = dummy.cxx


# GUI-BASED

vis0_placement_CPPFLAGS = -I$(top_srcdir)/src
vis0_placement_LDFLAGS  = 
vis0_placement_LDADD    = 
vis0_placement_SOURCES  = $(LIBSINRMAPS_SOURCES) \
															cmd/vis0-placement.cu
nodist_EXTRA_vis0_placement_SOURCES = dummy.cxx


# Crowncom Simulations

crowncom0_placement_CPPFLAGS = -I$(top_srcdir)/src
crowncom0_placement_LDFLAGS  = 
crowncom0_placement_LDADD    = 
crowncom0_placement_SOURCES  = $(LIBSINRMAPS_SOURCES) \
															cmd/crowncom0-placement.cu
nodist_EXTRA_crowncom0_placement_SOURCES = dummy.cxx


# WiFiUS Simulations

wifius0_modesvbands_CPPFLAGS = -I$(top_srcdir)/src
wifius0_modesvbands_LDFLAGS  = 
wifius0_modesvbands_LDADD    = 
wifius0_modesvbands_SOURCES  = $(LIBSINRMAPS_SOURCES) \
															cmd/wifius0-modesvbands.cu
nodist_EXTRA_wifius0_modesvbands_SOURCES = dummy.cxx


# Enable silent rules for nvcc
# See: http://www.gnu.org/software/automake/manual/html_node/Automake-silent_002drules-Option.html
nvcc_verbose		= $(nvcc_verbose_@AM_V@)
nvcc_verbose_		= $(nvcc_verbose_@AM_DEFAULT_V@)
nvcc_verbose_0	= @echo "  NVCC     $@";
.cu.o:
	$(nvcc_verbose)$(NVCC) $(NVCCFLAGS) -I$(top_srcdir)/src -c -o $@ $<

CLEANFILES = 
MAINTAINERCLEANFILES = Makefile.in
