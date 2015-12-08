AM_CFLAGS =					\
	$(GROONGA_CFLAGS)

AM_LDFLAGS =					\
	-avoid-version				\
	-module					\
	-no-undefined

LIBS =						\
	$(GROONGA_LIBS)

glove_cooccur_plugins_LTLIBRARIES =
glove_cooccur_plugins_LTLIBRARIES += glove_cooccur.la
glove_plugins_LTLIBRARIES =
glove_plugins_LTLIBRARIES += glove.la
glove_shuffle_plugins_LTLIBRARIES =
glove_shuffle_plugins_LTLIBRARIES += glove_shuffle.la
glove_vocab_count_plugins_LTLIBRARIES =
glove_vocab_count_plugins_LTLIBRARIES += glove_vocab_count.la

glove_cooccur_la_SOURCES = \
glove_cooccur.c
glove_la_SOURCES = \
glove.c
glove_shuffle_la_SOURCES = \
glove_shuffle.c
glove_vocab_count_la_SOURCES = \
glove_vocab_count.c

glove_cooccur_la_CFLAGS = $(AM_CFLAGS) $(AM_CFLAGS)
glove_cooccur_la_LIBADD = $(LIBS)
glove_cooccur_la_LDFLAGS = $(AM_LDFLAGS)
glove_la_CFLAGS = $(AM_CFLAGS) $(AM_CFLAGS)
glove_la_LIBADD = $(LIBS)
glove_la_LDFLAGS = $(AM_LDFLAGS)
glove_shuffle_la_CFLAGS = $(AM_CFLAGS) $(AM_CFLAGS)
glove_shuffle_la_LIBADD = $(LIBS)
glove_shuffle_la_LDFLAGS = $(AM_LDFLAGS)
glove_vocab_count_la_CFLAGS = $(AM_CFLAGS) $(AM_CFLAGS)
glove_vocab_count_la_LIBADD = $(LIBS)
glove_vocab_count_la_LDFLAGS = $(AM_LDFLAGS)
