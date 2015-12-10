AM_CFLAGS =					\
	$(GROONGA_CFLAGS)

AM_LDFLAGS =					\
	-avoid-version				\
	-module					\
	-no-undefined

LIBS =						\
	$(GROONGA_LIBS)

bin_PROGRAMS = word2vec/word2vec

word2vec_plugins_LTLIBRARIES =
word2vec_plugins_LTLIBRARIES += word2vec.la

word2vec_la_SOURCES = \
word2vec.cpp

word2vec_la_CPPFLAGS = $(AM_CFLAGS) $(AM_CPPFLAGS) $(MECAB_CPPFLAGS)
word2vec_la_LIBADD = $(LIBS) $(MECAB_LIBS) $(RE2_LIBS)
word2vec_la_LDFLAGS = $(AM_LDFLAGS) $(MECAB_LDFLAGS)
