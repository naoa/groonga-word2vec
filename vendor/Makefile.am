AM_CFLAGS =					\
	$(GROONGA_CFLAGS)

AM_LDFLAGS =					\
	-avoid-version				\
	-module					\
	-no-undefined				\
        -lm

LIBS =						\
	$(GROONGA_LIBS)

bin_PROGRAMS =
if NOINST_WORD2VEC
else
bin_PROGRAMS += word2vec/word2vec
endif
