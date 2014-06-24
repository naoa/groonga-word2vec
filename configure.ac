AC_PREREQ(2.59)
AC_INIT([groonga-word2vec], 0.0.1, [naoya@createfield.com])

AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_SRCDIR([word2vec/word2vec.cpp])
AM_CONFIG_HEADER([config.h])

AM_INIT_AUTOMAKE(foreign)
m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])])

AC_PROG_CXX

AC_PROG_LIBTOOL

if test "$GCC" = "yes"; then
  CFLAGS="$CFLAGS -Wall -Wextra"
  CFLAGS="$CFLAGS -Wmissing-declarations -Wmissing-prototypes"
fi

AC_ARG_ENABLE(debug,
  [AS_HELP_STRING([--enable-debug],
                  [use debug flags (default=no)])],
  [sample_debug="$enableval"],
  [sample_debug="no"])
if test "x$sample_debug" != "xno"; then
  if test "$CLANG" = "yes"; then
    CFLAGS="$CFLAGS -O0 -g"
  elif test "$GCC" = "yes"; then
    CFLAGS="$CFLAGS -O0 -g3"
  fi
fi

# MeCab
# NOTE: MUST be checked last
AC_ARG_WITH(mecab,
  [AS_HELP_STRING([--with-mecab],
    [use MeCab for morphological analysis. [default=yes]])],
  [with_mecab="$withval"],
  [with_mecab="yes"])
AC_MSG_CHECKING([whether enable MeCab])
AC_MSG_RESULT($with_mecab)
if test "x$with_mecab" = "xyes"; then
  # mecab-config
  AC_ARG_WITH(mecab-config,
    [AS_HELP_STRING([--with-mecab-config=PATH],
      [set mecab-config location. [default=auto-detect]])],
    [if test "$cross_compiling" = "yes"; then
       MECAB_CONFIG="$withval"
     else
       AC_CHECK_FILE("$withval", MECAB_CONFIG="$withval", MECAB_CONFIG=no)
     fi],
    [AC_PATH_PROG(MECAB_CONFIG, mecab-config, no)])
  if test "x$MECAB_CONFIG" = "xno"; then
    with_mecab="no"
  else
    MECAB_CPPFLAGS="-I`$MECAB_CONFIG --inc-dir`"
    MECAB_LDFLAGS="-L`$MECAB_CONFIG --libs-only-L`"
    _SAVE_LIBS="$LIBS"
    _SAVE_LDFLAGS="$LDFLAGS"
    LDFLAGS="$LDFLAGS $MECAB_LDFLAGS"
    AC_SEARCH_LIBS(mecab_new,
                   mecab,
                   [MECAB_LIBS="-lmecab $PTHREAD_LIBS"],
                   [AC_MSG_ERROR("No libmecab found")],
                   $PTHREAD_LIBS)
    LDFLAGS="$_SAVE_LDFLAGS"
    LIBS="$_SAVE_LIBS"
    _SAVE_CPPFLAGS="$CPPFLAGS"
    CPPFLAGS="$CPPFLAGS $MECAB_CPPFLAGS"
    AC_CHECK_HEADER(mecab.h, , [AC_MSG_ERROR("No mecab.h found")])
    AC_CHECK_TYPE([mecab_dictionary_info_t],
                  [AC_DEFINE([HAVE_MECAB_DICTIONARY_INFO_T],
                             [1],
                             [Define to 1 if MeCab has the type `mecab_dictionary_info_t'.])],
                  [],
                  [[#include <mecab.h>]])
    CPPFLAGS="$_SAVE_CPPFLAGS"
    AC_SUBST(MECAB_CPPFLAGS)
    AC_SUBST(MECAB_LDFLAGS)
    AC_SUBST(MECAB_LIBS)
  fi
fi
if test "x$with_mecab" = "xyes"; then
  AC_DEFINE(GRN_WITH_MECAB, [1], [use MeCab])
fi
AM_CONDITIONAL(WITH_MECAB, test "x$with_mecab" = "xyes")

# RE2

with_re2=no
AC_CHECK_LIB(re2, main,
     [with_re2=yes],
     [AC_MSG_ERROR([RE2 library failed.])])
if test "x$with_re2" = "xyes"; then
  RE2_LIBS="-lre2"
  AC_SUBST(RE2_LIBS)
  AC_DEFINE(GRN_WITH_RE2, [1], [use RE2])
fi
AM_CONDITIONAL(WITH_RE2, test "x$with_re2" = "xyes")


GROONGA_REQUIRED_VERSION=4.0.2
PKG_CHECK_MODULES([GROONGA], [groonga >= ${GROONGA_REQUIRED_VERSION}])

_PKG_CONFIG(GROONGA_PLUGINS_DIR, [variable=pluginsdir],    [groonga])
_PKG_CONFIG(GROONGA,             [variable=groonga],       [groonga])
_PKG_CONFIG(GROONGA_HTTPD,       [variable=groonga_httpd], [groonga])

GROONGA_PLUGINS_DIR="${pkg_cv_GROONGA_PLUGINS_DIR}"
GROONGA="${pkg_cv_GROONGA}"
GROONGA_HTTPD="${pkg_cv_GROONGA_HTTPD}"

AC_SUBST(GROONGA_PLUGINS_DIR)
AC_SUBST(GROONGA)
AC_SUBST(GROONGA_HTTPD)

word2vec_pluginsdir="\${GROONGA_PLUGINS_DIR}/word2vec"
AC_SUBST(word2vec_pluginsdir)

AC_CONFIG_FILES([
  Makefile
  word2vec/Makefile
  test/Makefile
])

AC_OUTPUT

echo "$PACKAGE_NAME $PACKAGE_VERSION configuration:"
echo "-----------------------"
echo "  Compiler:              ${CC}"
echo "  CFLAGS:                ${CFLAGS}"
echo "  CXXFLAGS:              ${CXXFLAGS}"
echo "  Libraries:             ${LIBS}"
echo
echo "groonga-word2vec"
echo "  CFLAGS:                ${GROONGA_CFLAGS}"
echo "  Libraries:             ${GROONGA_LIBS}"
echo "  install directory:     ${word2vec_pluginsdir}"
echo
echo "  MeCab: $with_mecab"
if test "x$with_mecab" = "xyes"; then
  echo "  CPPFLAGS:              ${MECAB_CPPFLAGS}"
  echo "  LDFLAGS:               ${MECAB_LDFLAGS}"
  echo "  LIBS:                  ${MECAB_LIBS}"
fi
echo
echo "  RE2: $with_re2"
if test "x$with_re2" = "xyes"; then
  echo "  LIBS:                  ${RE2_LIBS}"
fi
