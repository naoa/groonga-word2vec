/* 
  Copyright(C) 2014 Naoya Murakami <naoya@createfield.com>

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Library General Public
  License as published by the Free Software Foundation; version 2
  of the License.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
  Library General Public License for more details.

  You should have received a copy of the GNU Library General Public
  License along with this library; if not, write to the Free
  Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston,
  MA 02110-1301, USA

  This file includes the original word2vec's code. https://code.google.com/p/word2vec/
  The following is the header of the file:

    Copyright 2013 Google Inc. All Rights Reserved.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#include <groonga/plugin.h>
#include <groonga/tokenizer.h>

#include <iostream>
#include <string>

using namespace std;

#include <mecab.h>
#include <re2/re2.h>

#ifdef __GNUC__
# define GNUC_UNUSED __attribute__((__unused__))
#else
# define GNUC_UNUSED
#endif

static mecab_t *sole_mecab = NULL;
static grn_plugin_mutex *sole_mecab_mutex = NULL;
static grn_encoding sole_mecab_encoding = GRN_ENC_NONE;

#define GRN_EXPANDER_NONE 0
#define GRN_EXPANDER_EXPANDED 1
#define GRN_EXPANDER_TSV 2

#define MAX_SYNONYM_BYTES 4096

static grn_hash *synonyms = NULL;

/* distance.c analogy.c */

const long long max_size = 2000; // max length of strings
const long long N = 40; // number of closest words that will be shown
const long long max_w = 50; // max length of vocabulary entries

long long words, size = 0;
float *M = NULL;
char *load_vocab = NULL;

static char *
right_trim(char *s, char del) {
  int i;
  int count = 0;
  if ( s == NULL ) {
    return NULL;
  }
  i = strlen(s);
  while ( --i >= 0 && s[i] == del ) count++;
  if (i == 0) {
    s[i] = '\0';
  } else {
    s[i+1] = '\0';
  }
  return s;
}

static grn_encoding
translate_mecab_charset_to_grn_encoding(const char *charset)
{
  if (strcasecmp(charset, "euc-jp") == 0) {
    return GRN_ENC_EUC_JP;
  } else if (strcasecmp(charset, "utf-8") == 0 ||
             strcasecmp(charset, "utf8") == 0) {
    return GRN_ENC_UTF8;
  } else if (strcasecmp(charset, "shift_jis") == 0 ||
             strcasecmp(charset, "shift-jis") == 0 ||
             strcasecmp(charset, "sjis") == 0) {
    return GRN_ENC_SJIS;
  }
  return GRN_ENC_NONE;
}

static grn_encoding
get_mecab_encoding(mecab_t *mecab)
{
  grn_encoding encoding = GRN_ENC_NONE;
  const mecab_dictionary_info_t *dictionary_info;
  dictionary_info = mecab_dictionary_info(mecab);
  if (dictionary_info) {
    const char *charset = dictionary_info->charset;
    encoding = translate_mecab_charset_to_grn_encoding(charset);
  }
  return encoding;
}

static void
check_mecab_dictionary_encoding(GNUC_UNUSED grn_ctx *ctx)
{
#ifdef HAVE_MECAB_DICTIONARY_DEBUG_T
  mecab_t *mecab;

  mecab = mecab_new2("-Owakati");
  if (mecab) {
    grn_encoding encoding;
    int have_same_encoding_dictionary = 0;

    encoding = GRN_CTX_GET_ENCODING(ctx);
    have_same_encoding_dictionary = encoding == get_mecab_encoding(mecab);
    mecab_destroy(mecab);

    if (!have_same_encoding_dictionary) {
      GRN_PLUGIN_ERROR(ctx, GRN_TOKENIZER_ERROR,
                       "[plugin][word2vec][train] "
                       "MeCab has no dictionary that uses the context encoding"
                       ": <%s>",
                       grn_encoding_to_string(encoding));
    }
  } else {
    GRN_PLUGIN_ERROR(ctx, GRN_TOKENIZER_ERROR,
                     "[plugin][word2vec][train] "
                     "mecab_new2 failed in check_mecab_dictionary_encoding: %s",
                     mecab_strerror(NULL));
  }
#endif
}

static grn_rc
mecab_init(grn_ctx *ctx)
{
  sole_mecab = NULL;
  sole_mecab_mutex = grn_plugin_mutex_open(ctx);
  if (!sole_mecab_mutex) {
    GRN_PLUGIN_ERROR(ctx, GRN_NO_MEMORY_AVAILABLE,
                     "[plugin][word2vec][train] grn_plugin_mutex_open() failed");
    return ctx->rc;
  }
  check_mecab_dictionary_encoding(ctx);
  return ctx->rc;
}

static grn_rc
mecab_fin(grn_ctx *ctx)
{
  if (sole_mecab) {
    mecab_destroy(sole_mecab);
    sole_mecab = NULL;
  }
  if (sole_mecab_mutex) {
    grn_plugin_mutex_close(ctx, sole_mecab_mutex);
    sole_mecab_mutex = NULL;
  }
  return GRN_SUCCESS;
}

static const char *
sparse(grn_ctx *ctx, const char *string, char *mecab_option)
{
  const char *parsed;

  if (!sole_mecab) {
    grn_plugin_mutex_lock(ctx, sole_mecab_mutex);
    if (!sole_mecab) {
      sole_mecab = mecab_new2(mecab_option);
      if (!sole_mecab) {
        GRN_PLUGIN_ERROR(ctx, GRN_TOKENIZER_ERROR,
                         "[plugin][word2vec][train] "
                         "mecab_new2() failed on mecab_init(): %s",
                         mecab_strerror(NULL));
      } else {
        sole_mecab_encoding = get_mecab_encoding(sole_mecab);
      }
    }
    grn_plugin_mutex_unlock(ctx, sole_mecab_mutex);
  }
  parsed = mecab_sparse_tostr(sole_mecab, string);
  if (!parsed) {
    GRN_PLUGIN_ERROR(ctx, GRN_TOKENIZER_ERROR,
                     "[plugin][word2vec][train] "
                     "split mecab_sparse_tostr() failed len=%d err=%s",
                     strlen(string),
                     mecab_strerror(sole_mecab));
    return NULL;
  }
  return parsed;
}

static const char *
normalize(grn_ctx *ctx, grn_obj *buf, char* normalizer_name, int normalizer_len, grn_obj *outbuf)
{
  grn_obj *normalizer;
  const char *normalized;
  grn_obj *grn_string;

  unsigned int normalized_length_in_bytes;
  unsigned int normalized_n_characters;

  normalizer = grn_ctx_get(ctx,
                           normalizer_name,
                           normalizer_len);

  grn_string = grn_string_open(ctx,
                               GRN_TEXT_VALUE(buf), GRN_TEXT_LEN(buf),
                               normalizer, 0);
  grn_obj_unlink(ctx, normalizer);

  grn_string_get_normalized(ctx, grn_string,
                            &normalized,
                            &normalized_length_in_bytes,
                            &normalized_n_characters);

  GRN_TEXT_PUTS(ctx, outbuf, normalized);
  const char *ret_normalized;
  ret_normalized = GRN_TEXT_VALUE(outbuf);

  grn_obj_unlink(ctx, grn_string);

  return ret_normalized;
}

static void
word2vec_unload(grn_ctx *ctx)
{
  if(M != NULL){
    GRN_PLUGIN_FREE(ctx, M);
    M = NULL;
  }
  if(load_vocab != NULL){
    GRN_PLUGIN_FREE(ctx, load_vocab);
    load_vocab = NULL;
  }
}

static void
get_model_file_path(grn_ctx *ctx, char *file_name)
{
   grn_obj *db;
   db = grn_ctx_db(ctx);
   const char *path;
   path = grn_obj_path(ctx, db);
   strcpy(file_name, path);
   strcat(file_name, "_w2v.bin");
}

static void
get_train_file_path(grn_ctx *ctx, char *file_name)
{
   grn_obj *db;
   db = grn_ctx_db(ctx);
   const char *path;
   path = grn_obj_path(ctx, db);
   strcpy(file_name, path);
   strcat(file_name, "_w2v.txt");
}

static grn_bool
word2vec_load(grn_ctx *ctx, grn_user_data *user_data)
{
  FILE *f;
  char file_name[max_size];
  float len;
  long long a, b;
  char ch;

  grn_obj *var;
  var = grn_plugin_proc_get_var(ctx, user_data, "file_path", -1);

  if(GRN_TEXT_LEN(var) == 0) {
    get_model_file_path(ctx, file_name);
  } else {
    strcpy(file_name, GRN_TEXT_VALUE(var));
  }

  f = fopen(file_name, "rb");
  if (f == NULL) {
    GRN_PLUGIN_LOG(ctx, GRN_LOG_WARNING,
                   "[plugin][word2vec][load] "
                   "Input file not found : %s", 
                   file_name);
    return GRN_FALSE;
  }

  word2vec_unload(ctx);

  fscanf(f, "%lld", &words);
  fscanf(f, "%lld", &size);
  load_vocab = (char *)GRN_PLUGIN_MALLOC(ctx, (long long)words * max_w * sizeof(char));
  M = (float *)GRN_PLUGIN_MALLOC(ctx, (long long)words * (long long)size * sizeof(float));
  if (M == NULL) {
    GRN_PLUGIN_LOG(ctx, GRN_LOG_WARNING,
                   "[plugin][word2vec][load] "
                   "Cannot allocate memory: %lld MB %lld %lld", 
                   (long long)words * size * sizeof(float) / 1048576, words, size);
    return GRN_FALSE;
  }
  for (b = 0; b < words; b++) {
    fscanf(f, "%s%c", &load_vocab[b * max_w], &ch);
    for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1, f);
    len = 0;
    for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
    len = sqrt(len);
    for (a = 0; a < size; a++) M[a + b * size] /= len;
  }
  fclose(f);

  return GRN_TRUE;
}

static grn_obj *
command_word2vec_load(grn_ctx *ctx, GNUC_UNUSED int nargs, GNUC_UNUSED grn_obj **args,
                      grn_user_data *user_data)
{
  if(word2vec_load(ctx, user_data) == GRN_TRUE) {
    grn_ctx_output_bool(ctx, GRN_TRUE);
  } else {
    grn_ctx_output_bool(ctx, GRN_FALSE);
  }
  return NULL;
}

static grn_obj *
command_word2vec_unload(grn_ctx *ctx, GNUC_UNUSED int nargs, GNUC_UNUSED grn_obj **args,
                        GNUC_UNUSED grn_user_data *user_data)
{
  word2vec_unload(ctx);
  grn_ctx_output_bool(ctx, GRN_TRUE);
  return NULL;
}

static grn_obj *
command_word2vec_distance(grn_ctx *ctx, GNUC_UNUSED int nargs, GNUC_UNUSED grn_obj **args,
                          grn_user_data *user_data)
{
  char st1[max_size];
  char bestw[N][max_size];
  char st[100][max_size];
  float dist, len, bestd[N], vec[max_size];
  long long a, b, c, d, cn, bi[100];

  for (a = 0; a < N; a++) bestd[a] = 0;
  for (a = 0; a < N; a++) bestw[a][0] = 0;

  if(M == NULL || load_vocab == NULL){
    if(word2vec_load(ctx, user_data) == GRN_FALSE) {
      grn_ctx_output_bool(ctx, GRN_FALSE);
      return NULL;
    }
  }

  grn_obj *var;
  int offset = 0;
  int limit = 10;
  float threshold = -1;
  char *normalizer_name = (char *)"NormalizerAuto";
  int normalizer_len = 14;
  char *term_filter = NULL;
  char *output_filter = NULL;
  char *mecab_option = NULL;
  int expander_mode = 0;

  long long st_pos = 0;

  var = grn_plugin_proc_get_var(ctx, user_data, "offset", -1);
  if(GRN_TEXT_LEN(var) != 0) {
    offset = atoi(GRN_TEXT_VALUE(var));
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "limit", -1);
  if(GRN_TEXT_LEN(var) != 0) {
    limit = atoi(GRN_TEXT_VALUE(var));
    if (limit < 0) {
      limit = N;
    }
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "threshold", -1);
  if(GRN_TEXT_LEN(var) != 0) {
    threshold = atof(GRN_TEXT_VALUE(var));
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "normalizer", -1);
  if(GRN_TEXT_LEN(var) != 0) {
    normalizer_name = GRN_TEXT_VALUE(var);
    normalizer_len = GRN_TEXT_LEN(var);
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "term_filter", -1);
  if(GRN_TEXT_LEN(var) != 0) {
    term_filter = GRN_TEXT_VALUE(var);
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "output_filter", -1);
  if(GRN_TEXT_LEN(var) != 0) {
    output_filter = GRN_TEXT_VALUE(var);
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "mecab_option", -1);
  if(GRN_TEXT_LEN(var) != 0) {
    mecab_option = GRN_TEXT_VALUE(var);
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "expander_mode", -1);
  if(GRN_TEXT_LEN(var) != 0) {
    expander_mode = atoi(GRN_TEXT_VALUE(var));
  }

  var = grn_plugin_proc_get_var(ctx, user_data, "term", -1);
  a = GRN_TEXT_LEN(var);

  if(a == 0) {
    GRN_PLUGIN_LOG(ctx, GRN_LOG_NOTICE,
                   "[plugin][word2vec][distance] empty term");
    grn_ctx_output_bool(ctx, GRN_FALSE);
    return NULL;
  } else {
    const char *term;
    grn_obj buf;
    GRN_TEXT_INIT(&buf, 0);
    GRN_BULK_REWIND(&buf);

    if(normalizer_name != ""){
      term = normalize(ctx, var, normalizer_name, normalizer_len, &buf);
    } else {
      term = GRN_TEXT_VALUE(var);
    }
    right_trim((char *)term, '\n');
    right_trim((char *)term, ' ');
    if(mecab_option != NULL && strlen(term) > 0){
      term = sparse(ctx, term, mecab_option);
    }
    right_trim((char *)term, '\n');
    right_trim((char *)term, ' ');

    strcpy(st1, term);
    st1[strlen(st1) + 1] = 0;
    grn_obj_unlink(ctx, &buf);
  }

  GRN_PLUGIN_LOG(ctx, GRN_LOG_DEBUG,
                 "[plugin][word2vec][distance] st1 = %s",st1);

  cn = 0;
  b = 0;
  c = 0;
  while (1) {
    st[cn][b] = st1[c];
    b++;
    c++;
    st[cn][b] = 0;
    if (st1[c] == 0) break;
    if (st1[c] == ' ') {
      cn++;
      b = 0;
      c++;
    }
  }
  cn++;
  for (a = 0; a < cn; a++) {
    for (b = 0; b < words; b++) if (!strcmp(&load_vocab[b * max_w], st[a])) break;
    if (b == words) b = -1;
    bi[a] = b;
    st_pos = bi[a];
    if (b == -1) {
      if (expander_mode == GRN_EXPANDER_NONE) {
        grn_ctx_output_array_open(ctx, "RESULT", 2);
        grn_ctx_output_array_open(ctx, "INPUT_WORD", 2);
        grn_ctx_output_cstr(ctx, st1);
        grn_ctx_output_float(ctx, st_pos);
        grn_ctx_output_array_close(ctx);
        grn_ctx_output_array_open(ctx, "SIMILAR_WORDS", 2);
        grn_ctx_output_array_open(ctx, "SIMILAR_WORD", 1);
        grn_ctx_output_cstr(ctx, "Output of dictionary word!");
        grn_ctx_output_array_close(ctx);
        grn_ctx_output_array_close(ctx);
        grn_ctx_output_array_close(ctx);
      } else {
        grn_ctx_output_cstr(ctx, st1);
      }
      return NULL;
    }
  }
  for (a = 0; a < size; a++) vec[a] = 0;
  for (b = 0; b < cn; b++) {
    if (bi[b] == -1) continue;
    for (a = 0; a < size; a++) vec[a] += M[a + bi[b] * size];
  }
  len = 0;
  for (a = 0; a < size; a++) len += vec[a] * vec[a];
  len = sqrt(len);
  for (a = 0; a < size; a++) vec[a] /= len;
  for (a = 0; a < N; a++) bestd[a] = 0;
  for (a = 0; a < N; a++) bestw[a][0] = 0;
  for (c = 0; c < words; c++) {
    a = 0;
    for (b = 0; b < cn; b++) if (bi[b] == c) a = 1;
    if (a == 1) continue;
    dist = 0;
    for (a = 0; a < size; a++) dist += vec[a] * M[a + c * size];
    for (a = 0; a < N; a++) {
      if (dist > bestd[a]) {
        for (d = N - 1; d > a; d--) {
          bestd[d] = bestd[d - 1];
          strcpy(bestw[d], bestw[d - 1]);
        }
        bestd[a] = dist;
        strcpy(bestw[a], &load_vocab[c * max_w]);
        break;
      }
    }
  }

  if (threshold > 0 || term_filter != NULL) {
    for (a = 0; a < N; a++) {
      if (threshold > 0 && bestd[a] < threshold) {
        bestd[a] = 0;
      }
      if (term_filter != NULL) {
        string s = bestw[a];
        string t = term_filter;
        if ( RE2::FullMatch(s, t) ) {
          bestd[a] = 0;
        }
      }
    }
  }

  unsigned int max;
  if (offset + limit > N) {
    max = N; 
  } else {
    max = offset + limit;
  }

  if (expander_mode == GRN_EXPANDER_EXPANDED) {
    grn_obj buf;
    GRN_TEXT_INIT(&buf, 0);
    GRN_BULK_REWIND(&buf);
    GRN_TEXT_PUTS(ctx, &buf, "((");
    GRN_TEXT_PUTS(ctx, &buf, st1);
    for (a = offset; a < max; a++) {
      if (output_filter != NULL) {
        string s = bestw[a];
        re2::RE2::GlobalReplace(&s, output_filter, "");
        strcpy(bestw[a],s.c_str());
      }
      if (strlen(bestw[a]) > 0 && bestd[a] != 0) {
        if ( a < max - 1) {
          GRN_TEXT_PUTS(ctx, &buf, ") OR (");
          GRN_TEXT_PUTS(ctx, &buf, bestw[a]);
        } else {
          GRN_TEXT_PUTS(ctx, &buf, bestw[a]);
        }
      } else {
        if (max < N) {
          max++;
        }
      }
    }
    GRN_TEXT_PUTS(ctx, &buf, "))");
    grn_ctx_output_obj(ctx, &buf, NULL);
    grn_obj_unlink(ctx, &buf);
  }
  else if (expander_mode == GRN_EXPANDER_TSV) {
    grn_obj buf;
    GRN_TEXT_INIT(&buf, 0);
    GRN_BULK_REWIND(&buf);
    GRN_TEXT_PUTS(ctx, &buf, st1);
    for (a = offset; a < max; a++) {
      if (output_filter != NULL) {
        string s = bestw[a];
        re2::RE2::GlobalReplace(&s, output_filter, "");
        strcpy(bestw[a],s.c_str());
      }
      if (strlen(bestw[a]) > 0 && bestd[a] != 0) {
        if ( a < max - 1){
          GRN_TEXT_PUTC(ctx, &buf, '\t');
          GRN_TEXT_PUTS(ctx, &buf, bestw[a]);
        } else {
          GRN_TEXT_PUTS(ctx, &buf, bestw[a]);
        }
      } else {
        if (max < N) {
          max++;
        }
      }
    }
    grn_ctx_output_obj(ctx, &buf, NULL);
    grn_obj_unlink(ctx, &buf);
  }
  else {
    grn_ctx_output_array_open(ctx, "RESULT", 2);
    grn_ctx_output_array_open(ctx, "INPUT_WORD", 2);
    grn_ctx_output_cstr(ctx, st1);
    grn_ctx_output_float(ctx, st_pos);
    grn_ctx_output_array_close(ctx);
    grn_ctx_output_array_open(ctx, "SIMILAR_WORDS", N);
    for (a = offset; a < max; a++) {
      if (output_filter != NULL) {
        string s = bestw[a];
        re2::RE2::GlobalReplace(&s, output_filter, "");
        strcpy(bestw[a],s.c_str());
      }
      if (strlen(bestw[a]) > 0 && bestd[a] != 0) {
        grn_ctx_output_array_open(ctx, "SIMILAR_WORD", 2);
        grn_ctx_output_cstr(ctx, bestw[a]);
        grn_ctx_output_float(ctx, bestd[a]);
        grn_ctx_output_array_close(ctx);
      } else {
        if (max < N) {
          max++;
        }
      }
    }
    grn_ctx_output_array_close(ctx);
    grn_ctx_output_array_close(ctx);
  }

  return NULL;
}

static grn_obj *
command_word2vec_analogy(grn_ctx *ctx, GNUC_UNUSED int nargs, GNUC_UNUSED grn_obj **args,
                         grn_user_data *user_data)
{
  char st1[max_size];
  char bestw[N][max_size];
  char st[100][max_size];
  float dist, len, bestd[N], vec[max_size];
  long long a, b, c, d, cn, bi[100];

  for (a = 0; a < N; a++) bestd[a] = 0;
  for (a = 0; a < N; a++) bestw[a][0] = 0;

  if(M == NULL || load_vocab == NULL){
    if(word2vec_load(ctx, user_data) == GRN_FALSE) {
      grn_ctx_output_bool(ctx, GRN_FALSE);
      return NULL;
    }
  }

  grn_obj *var;
  int offset = 0;
  int limit = 10;
  float threshold = -1;
  char *normalizer_name = (char *)"NormalizerAuto";
  int normalizer_len = 14;
  char *term_filter = NULL;
  char *output_filter = NULL;
  char *mecab_option = NULL;
  int expander_mode = 0;

  long long st_pos = 0;

  var = grn_plugin_proc_get_var(ctx, user_data, "offset", -1);
  if(GRN_TEXT_LEN(var) != 0) {
    offset = atoi(GRN_TEXT_VALUE(var));
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "limit", -1);
  if(GRN_TEXT_LEN(var) != 0) {
    limit = atoi(GRN_TEXT_VALUE(var));
    if (limit < 0) {
      limit = N;
    }
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "threshold", -1);
  if(GRN_TEXT_LEN(var) != 0) {
    threshold = atof(GRN_TEXT_VALUE(var));
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "normalizer", -1);
  if(GRN_TEXT_LEN(var) != 0) {
    normalizer_name = GRN_TEXT_VALUE(var);
    normalizer_len = GRN_TEXT_LEN(var);
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "term_filter", -1);
  if(GRN_TEXT_LEN(var) != 0) {
    term_filter = GRN_TEXT_VALUE(var);
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "output_filter", -1);
  if(GRN_TEXT_LEN(var) != 0) {
    output_filter = GRN_TEXT_VALUE(var);
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "mecab_option", -1);
  if(GRN_TEXT_LEN(var) != 0) {
    mecab_option = GRN_TEXT_VALUE(var);
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "expander_mode", -1);
  if(GRN_TEXT_LEN(var) != 0) {
    expander_mode = atoi(GRN_TEXT_VALUE(var));
  }

  var = grn_plugin_proc_get_var(ctx, user_data, "positive_term1", -1);
  a = GRN_TEXT_LEN(var);

  if(a == 0) {
    GRN_PLUGIN_LOG(ctx, GRN_LOG_NOTICE,
                   "[plugin][word2vec][analogy] empty positive_term1");
    grn_ctx_output_bool(ctx, GRN_FALSE);
    return NULL;
  } else {
    const char *term;

    grn_obj buf;
    GRN_TEXT_INIT(&buf, 0);
    GRN_BULK_REWIND(&buf);

    if(normalizer_name != ""){
      term = normalize(ctx, var, normalizer_name, normalizer_len, &buf);
    } else {
      term = GRN_TEXT_VALUE(var);
    }

    right_trim((char *)term, '\n');
    right_trim((char *)term, ' ');
    if(mecab_option != NULL && strlen(term) > 0){
      term = sparse(ctx, term, mecab_option);
    }
    right_trim((char *)term, '\n');
    right_trim((char *)term, ' ');

    strcpy(st1, term);
    strcat(st1, " ");

    grn_obj_unlink(ctx, &buf);

  }

  var = grn_plugin_proc_get_var(ctx, user_data, "negative_term", -1);
  a = GRN_TEXT_LEN(var);

  if(a == 0) {
    GRN_PLUGIN_LOG(ctx, GRN_LOG_NOTICE,
                   "[plugin][word2vec][analogy] empty negative_term");
    grn_ctx_output_bool(ctx, GRN_FALSE);
    return NULL;
  } else {
    const char *term;

    grn_obj buf;
    GRN_TEXT_INIT(&buf, 0);
    GRN_BULK_REWIND(&buf);

    if(normalizer_name != ""){
      term = normalize(ctx, var, normalizer_name, normalizer_len, &buf);
    } else {
      term = GRN_TEXT_VALUE(var);
    }

    right_trim((char *)term, '\n');
    right_trim((char *)term, ' ');
    if(mecab_option != NULL && strlen(term) > 0){
      term = sparse(ctx, term, mecab_option);
    }
    right_trim((char *)term, '\n');
    right_trim((char *)term, ' ');

    strcat(st1, term);
    strcat(st1, " ");

    grn_obj_unlink(ctx, &buf);
  }

  var = grn_plugin_proc_get_var(ctx, user_data, "positive_term2", -1);
  a = GRN_TEXT_LEN(var);

  if(a == 0) {
    GRN_PLUGIN_LOG(ctx, GRN_LOG_NOTICE,
                   "[plugin][word2vec][analogy] empty positive_term2");
    grn_ctx_output_bool(ctx, GRN_FALSE);
    return NULL;
  } else {
    const char *term;

    grn_obj buf;
    GRN_TEXT_INIT(&buf, 0);
    GRN_BULK_REWIND(&buf);

    if(normalizer_name != ""){
      term = normalize(ctx, var, normalizer_name, normalizer_len, &buf);
    } else {
      term = GRN_TEXT_VALUE(var);
    }

    right_trim((char *)term, '\n');
    right_trim((char *)term, ' ');
    if(mecab_option != NULL && strlen(term) > 0){
      term = sparse(ctx, term, mecab_option);
    }
    right_trim((char *)term, '\n');
    right_trim((char *)term, ' ');

    strcat(st1, term);

    grn_obj_unlink(ctx, &buf);
  }
  st1[strlen(st1) + 1] = 0;

  printf("%s\n",st1);
  GRN_PLUGIN_LOG(ctx, GRN_LOG_DEBUG,
                 "[plugin][word2vec][analogy] st1 = %s",st1);

  cn = 0;
  b = 0;
  c = 0;
  while (1) {
    st[cn][b] = st1[c];
    b++;
    c++;
    st[cn][b] = 0;
    if (st1[c] == 0) break;
    if (st1[c] == ' ') {
      cn++;
      b = 0;
      c++;
    }
  }
  cn++;
  if (cn < 3) {
    grn_ctx_output_array_open(ctx, "ANALOGY_WORDS", 1);
    grn_ctx_output_cstr(ctx, "three words are needed at the input to perform the calculation");
    grn_ctx_output_array_close(ctx);
    return NULL;
  }
  for (a = 0; a < cn; a++) {
    for (b = 0; b < words; b++) if (!strcmp(&load_vocab[b * max_w], st[a])) break;
    if (b == words) b = 0;
    bi[a] = b;
    st_pos = bi[a];
    if (b == 0) {
      if (expander_mode == GRN_EXPANDER_NONE) {
        grn_ctx_output_array_open(ctx, "RESULT", 2);
        grn_ctx_output_array_open(ctx, "INPUT_WORD", 2);
        grn_ctx_output_cstr(ctx, st1);
        grn_ctx_output_float(ctx, st_pos);
        grn_ctx_output_array_close(ctx);
        grn_ctx_output_array_open(ctx, "ANALOGY_WORDS", 2);
        grn_ctx_output_array_open(ctx, "ANALOGY_WORD", 1);
        grn_ctx_output_cstr(ctx, "Output of dictionary word!");
        grn_ctx_output_array_close(ctx);
        grn_ctx_output_array_close(ctx);
        grn_ctx_output_array_close(ctx);
      } else {
        grn_ctx_output_cstr(ctx, st1);
      }
      return NULL;
    }
  }
  if (b == 0) return NULL;
  for (a = 0; a < size; a++) vec[a] = M[a + bi[1] * size] - M[a + bi[0] * size] + M[a + bi[2] * size];
  len = 0;
  for (a = 0; a < size; a++) len += vec[a] * vec[a];
  len = sqrt(len);
  for (a = 0; a < size; a++) vec[a] /= len;
  for (a = 0; a < N; a++) bestd[a] = 0;
  for (a = 0; a < N; a++) bestw[a][0] = 0;
  for (c = 0; c < words; c++) {
    if (c == bi[0]) continue;
    if (c == bi[1]) continue;
    if (c == bi[2]) continue;
    a = 0;
    for (b = 0; b < cn; b++) if (bi[b] == c) a = 1;
    if (a == 1) continue;
    dist = 0;
    for (a = 0; a < size; a++) dist += vec[a] * M[a + c * size];
    for (a = 0; a < N; a++) {
      if (dist > bestd[a]) {
      for (d = N - 1; d > a; d--) {
        bestd[d] = bestd[d - 1];
        strcpy(bestw[d], bestw[d - 1]);
      }
      bestd[a] = dist;
      strcpy(bestw[a], &load_vocab[c * max_w]);
      break;
      }
    }
  }

  if (threshold > 0 || term_filter != NULL) {
    for (a = 0; a < N; a++) {
      if (threshold > 0 && bestd[a] < threshold) {
        bestd[a] = 0;
      }
      if (term_filter != NULL) {
        string s = bestw[a];
        string t = term_filter;
        if ( RE2::FullMatch(s, t) ) {
          bestd[a] = 0;
        }
      }
    }
  }

  unsigned int max;
  if (offset + limit > N) {
    max = N;
  } else {
    max = offset + limit;
  }

  if (expander_mode == GRN_EXPANDER_EXPANDED) {
    grn_obj buf;
    GRN_TEXT_INIT(&buf, 0);
    GRN_BULK_REWIND(&buf);
    GRN_TEXT_PUTS(ctx, &buf, "((");
    GRN_TEXT_PUTS(ctx, &buf, st1);
    for (a = offset; a < max; a++) {
      if (output_filter != NULL) {
        string s = bestw[a];
        re2::RE2::GlobalReplace(&s, output_filter, "");
        strcpy(bestw[a],s.c_str());
      }
      if (strlen(bestw[a]) > 0 && bestd[a] != 0) {
        if ( a < max - 1) {
          GRN_TEXT_PUTS(ctx, &buf, ") OR (");
          GRN_TEXT_PUTS(ctx, &buf, bestw[a]);
        } else {
          GRN_TEXT_PUTS(ctx, &buf, bestw[a]);
        }
      } else {
        if (max < N) {
          max++;
        }
      }
    }
    GRN_TEXT_PUTS(ctx, &buf, "))");
    grn_ctx_output_obj(ctx, &buf, NULL);
    grn_obj_unlink(ctx, &buf);
  }
  else if (expander_mode == GRN_EXPANDER_TSV) {
    grn_obj buf;
    GRN_TEXT_INIT(&buf, 0);
    GRN_BULK_REWIND(&buf);
    GRN_TEXT_PUTS(ctx, &buf, st1);
    for (a = offset; a < max; a++) {
      if (output_filter != NULL) {
        string s = bestw[a];
        re2::RE2::GlobalReplace(&s, output_filter, "");
        strcpy(bestw[a],s.c_str());
      }
      if (strlen(bestw[a]) > 0 && bestd[a] != 0) {
        if ( a < max - 1){
          GRN_TEXT_PUTC(ctx, &buf, '\t');
          GRN_TEXT_PUTS(ctx, &buf, bestw[a]);
        } else {
          GRN_TEXT_PUTS(ctx, &buf, bestw[a]);
        }
      } else {
        if (max < N) {
          max++;
        }
      }
    }
    grn_ctx_output_obj(ctx, &buf, NULL);
    grn_obj_unlink(ctx, &buf);
  }
  else {
    grn_ctx_output_array_open(ctx, "RESULT", 2);
    grn_ctx_output_array_open(ctx, "INPUT_WORD", 2);
    grn_ctx_output_cstr(ctx, st1);
    grn_ctx_output_float(ctx, st_pos);
    grn_ctx_output_array_close(ctx);
    grn_ctx_output_array_open(ctx, "ANALOGY_WORDS", N);
    for (a = offset; a < max; a++) {
      if (output_filter != NULL) {
        string s = bestw[a];
        re2::RE2::GlobalReplace(&s, output_filter, "");
        strcpy(bestw[a],s.c_str());
      }
      if (strlen(bestw[a]) > 0 && bestd[a] != 0) {
        grn_ctx_output_array_open(ctx, "ANALOGY_WORD", 2);
        grn_ctx_output_cstr(ctx, bestw[a]);
        grn_ctx_output_float(ctx, bestd[a]);
        grn_ctx_output_array_close(ctx);
      } else {
        if (max < N) {
          max++;
        }
      }
    }
    grn_ctx_output_array_close(ctx);
    grn_ctx_output_array_close(ctx);
  }

  return NULL;
}

/* word2vec.c */

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

const int vocab_hash_size = 30000000; // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real; // Precision of float numbers

struct vocab_word {
  long long cn;
  int *point;
  char *word, *code, codelen;
};

char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
struct vocab_word *vocab;
int binary = 1, cbow = 0, debug_mode = 2, window = 5, min_count = 5, num_threads = 4, min_reduce = 1;
int *vocab_hash;
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 200;
long long train_words = 0, word_count_actual = 0, file_size = 0, classes = 0;
real alpha = 0.025, starting_alpha, sample = 1e-3;
real *syn0, *syn1, *syn1neg, *expTable;
clock_t start;

int hs = 1, negative = 0;
const int table_size = 1e8;
int *table;

static void InitUnigramTable() {
  int a, i;
  long long train_words_pow = 0;
  real d1, power = 0.75;
  table = (int *)malloc(table_size * sizeof(int));
  if (table == NULL) {
    fprintf(stderr, "cannot allocate memory for the table\n");
    exit(1);
  }
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  d1 = pow(vocab[i].cn, power) / (real)train_words_pow;
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (real)table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / (real)train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
static void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--; // Truncate too long words
  }
  word[a] = 0;
}

// Returns hash value of a word
static int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
static int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
static int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

// Adds a word to the vocabulary
static int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;
  vocab_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash] = vocab_size - 1;
  return vocab_size - 1;
}

// Used later for sorting by word counts
static int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

static void DestroyVocab() {
  int a;

  for (a = 0; a < vocab_size; a++) {
    if (vocab[a].word != NULL) {
      free(vocab[a].word);
    }
    if (vocab[a].code != NULL) {
      free(vocab[a].code);
    }
    if (vocab[a].point != NULL) {
      free(vocab[a].point);
    }
  }
  free(vocab[vocab_size].word);
  free(vocab);
}

// Sorts the vocabulary by frequency using word counts
static void SortVocab() {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  size = vocab_size;
  train_words = 0;
  for (a = 1; a < size; a++) { // Skip </s>
    // Words occuring less than min_count times will be discarded from the vocab
    if (vocab[a].cn < min_count) {
      vocab_size--;
      free(vocab[a].word);
      vocab[a].word = NULL;
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash=GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
    }
  }
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
  // Allocate memory for the binary tree construction
  for (a = 0; a < vocab_size; a++) {
    vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}

// Reduces the vocabulary by removing infrequent tokens
static void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
    vocab[b].cn = vocab[a].cn;
    vocab[b].word = vocab[a].word;
    b++;
  } else free(vocab[a].word);
  vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
static void CreateBinaryTree() {
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH];
  long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
  for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
  pos1 = vocab_size - 1;
  pos2 = vocab_size;
  // Following algorithm constructs the Huffman tree by adding one node at a time
  for (a = 0; a < vocab_size - 1; a++) {
    // First, find two smallest nodes 'min1, min2'
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1--;
      } else {
        min1i = pos2;
        pos2++;
      }
    } else {
      min1i = pos2;
      pos2++;
    }
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        pos1--;
      } else {
        min2i = pos2;
        pos2++;
      }
    } else {
      min2i = pos2;
      pos2++;
    }
    count[vocab_size + a] = count[min1i] + count[min2i];
    parent_node[min1i] = vocab_size + a;
    parent_node[min2i] = vocab_size + a;
    binary[min2i] = 1;
  }
  // Now assign binary code to each vocabulary word
  for (a = 0; a < vocab_size; a++) {
    b = a;
    i = 0;
    while (1) {
      code[i] = binary[b];
      point[i] = b;
      i++;
      b = parent_node[b];
      if (b == vocab_size * 2 - 2) break;
    }
    vocab[a].codelen = i;
    vocab[a].point[0] = vocab_size - 2;
    for (b = 0; b < i; b++) {
      vocab[a].code[i - b - 1] = code[b];
      vocab[a].point[i - b] = point[b] - vocab_size;
    }
  }
  free(count);
  free(binary);
  free(parent_node);
}

static void LearnVocabFromTrainFile() {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  vocab_size = 0;
  AddWordToVocab((char *)"</s>");
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    train_words++;
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }
    i = SearchVocab(word);
    if (i == -1) {
      a = AddWordToVocab(word);
      vocab[a].cn = 1;
    } else vocab[i].cn++;
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  file_size = ftell(fin);
  fclose(fin);
}

static void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

static void ReadVocab() {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }

  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_size = 0;
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    a = AddWordToVocab(word);
    fscanf(fin, "%lld%c", &vocab[a].cn, &c);
    i++;
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);
}

static void InitNet() {
  long long a, b;
  a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
  if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
  if (hs) {
    a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (b = 0; b < layer1_size; b++) for (a = 0; a < vocab_size; a++)
     syn1[a * layer1_size + b] = 0;
  }
  if (negative>0) {
    a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (b = 0; b < layer1_size; b++) for (a = 0; a < vocab_size; a++)
     syn1neg[a * layer1_size + b] = 0;
  }
  for (b = 0; b < layer1_size; b++) for (a = 0; a < vocab_size; a++)
   syn0[a * layer1_size + b] = (rand() / (real)RAND_MAX - 0.5) / layer1_size;
  CreateBinaryTree();
}

static void DestroyNet() {
  if (syn0 != NULL) {
    free(syn0);
  }
  if (syn1 != NULL) {
    free(syn1);
  }
  if (syn1neg != NULL) {
    free(syn1neg);
  }
}

static void *TrainModelThread(void *id) {
  long long a, b, d, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long l1, l2, c, target, label;
  unsigned long long next_random = (long long)id;
  real f, g;
  clock_t now;
  real *neu1 = (real *)calloc(layer1_size, sizeof(real));
  real *neu1e = (real *)calloc(layer1_size, sizeof(real));
  FILE *fi = fopen(train_file, "rb");
  if (fi == NULL) {
    fprintf(stderr, "no such file or directory: %s", train_file);
    exit(1);
  }
  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
  while (1) {
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now=clock();
        printf("%cAlpha: %f Progress: %.2f%% Words/thread/sec: %.2fk ", 13, alpha,
         word_count_actual / (real)(train_words + 1) * 100,
         word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
      alpha = starting_alpha * (1 - word_count_actual / (real)(train_words + 1));
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }
    if (sentence_length == 0) {
      while (1) {
        word = ReadWordIndex(fi);
        if (feof(fi)) break;
        if (word == -1) continue;
        word_count++;
        if (word == 0) break;
        // The subsampling randomly discards frequent words while keeping the ranking same
        if (sample > 0) {
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
          next_random = next_random * (unsigned long long)25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        }
        sen[sentence_length] = word;
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }
    if (feof(fi)) break;
    if (word_count > train_words / num_threads) break;
    word = sen[sentence_position];
    if (word == -1) continue;
    for (c = 0; c < layer1_size; c++) neu1[c] = 0;
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
    next_random = next_random * (unsigned long long)25214903917 + 11;
    b = next_random % window;
    if (cbow) { //train the cbow architecture
      // in -> hidden
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        for (c = 0; c < layer1_size; c++) neu1[c] += syn0[c + last_word * layer1_size];
      }
      if (hs) for (d = 0; d < vocab[word].codelen; d++) {
        f = 0;
        l2 = vocab[word].point[d] * layer1_size;
        // Propagate hidden -> output
        for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1[c + l2];
        if (f <= -MAX_EXP) continue;
        else if (f >= MAX_EXP) continue;
        else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
        // 'g' is the gradient multiplied by the learning rate
        g = (1 - vocab[word].code[d] - f) * alpha;
        // Propagate errors output -> hidden
        for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
        // Learn weights hidden -> output
        for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * neu1[c];
      }
      // NEGATIVE SAMPLING
      if (negative > 0) for (d = 0; d < negative + 1; d++) {
        if (d == 0) {
          target = word;
          label = 1;
        } else {
          next_random = next_random * (unsigned long long)25214903917 + 11;
          target = table[(next_random >> 16) % table_size];
          if (target == 0) target = next_random % (vocab_size - 1) + 1;
          if (target == word) continue;
          label = 0;
        }
        l2 = target * layer1_size;
        f = 0;
        for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2];
        if (f > MAX_EXP) g = (label - 1) * alpha;
        else if (f < -MAX_EXP) g = (label - 0) * alpha;
        else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
        for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
        for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];
      }
      // hidden -> in
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        for (c = 0; c < layer1_size; c++) syn0[c + last_word * layer1_size] += neu1e[c];
      }
    } else { //train skip-gram
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        l1 = last_word * layer1_size;
        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
        // HIERARCHICAL SOFTMAX
        if (hs) for (d = 0; d < vocab[word].codelen; d++) {
          f = 0;
          l2 = vocab[word].point[d] * layer1_size;
          // Propagate hidden -> output
          for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1[c + l2];
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          // 'g' is the gradient multiplied by the learning rate
          g = (1 - vocab[word].code[d] - f) * alpha;
          // Propagate errors output -> hidden
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
          // Learn weights hidden -> output
          for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * syn0[c + l1];
        }
        // NEGATIVE SAMPLING
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            target = word;
            label = 1;
          } else {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            label = 0;
          }
          l2 = target * layer1_size;
          f = 0;
          for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
          for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
        }
        // Learn weights input -> hidden
        for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
      }
    }
    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  fclose(fi);
  free(neu1);
  free(neu1e);
  pthread_exit(NULL);
}

static void TrainModel() {
  long a, b, c, d;
  FILE *fo;
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  if (pt == NULL) {
    fprintf(stderr, "cannot allocate memory for threads\n");
    exit(1);
  }
  printf("Starting training using file %s\n", train_file);
  starting_alpha = alpha;
  if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
  if (save_vocab_file[0] != 0) SaveVocab();
  if (output_file[0] == 0) return;
  InitNet();
  if (negative > 0) InitUnigramTable();
  start = clock();
  for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
  for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
  fo = fopen(output_file, "wb");
  if (fo == NULL) {
    fprintf(stderr, "Cannot open %s: permission denied\n", output_file);
    exit(1);
  }
  if (classes == 0) {
    // Save the word vectors
    fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
    for (a = 0; a < vocab_size; a++) {
      if (vocab[a].word != NULL) {
        fprintf(fo, "%s ", vocab[a].word);
      }
      if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
      else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
      fprintf(fo, "\n");
    }
  } else {
    // Run K-means on the word vectors
    int clcn = classes, iter = 10, closeid;
    int *centcn = (int *)malloc(classes * sizeof(int));
    if (centcn == NULL) {
      fprintf(stderr, "cannot allocate memory for centcn\n");
      exit(1);
    }
    int *cl = (int *)calloc(vocab_size, sizeof(int));
    real closev, x;
    real *cent = (real *)calloc(classes * layer1_size, sizeof(real));
    for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;
    for (a = 0; a < iter; a++) {
      for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;
      for (b = 0; b < clcn; b++) centcn[b] = 1;
      for (c = 0; c < vocab_size; c++) {
        for (d = 0; d < layer1_size; d++) {
          cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
          centcn[cl[c]]++;
        }
      }
      for (b = 0; b < clcn; b++) {
        closev = 0;
        for (c = 0; c < layer1_size; c++) {
          cent[layer1_size * b + c] /= centcn[b];
          closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
        }
        closev = sqrt(closev);
        for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;
      }
      for (c = 0; c < vocab_size; c++) {
        closev = -10;
        closeid = 0;
        for (d = 0; d < clcn; d++) {
          x = 0;
          for (b = 0; b < layer1_size; b++) x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];
          if (x > closev) {
            closev = x;
            closeid = d;
          }
        }
        cl[c] = closeid;
      }
    }
    // Save the K-means classes
    for (a = 0; a < vocab_size; a++) fprintf(fo, "%s %d\n", vocab[a].word, cl[a]);
    free(centcn);
    free(cent);
    free(cl);
  }
  fclose(fo);
  free(table);
  free(pt);
  DestroyVocab();
}

static GNUC_UNUSED int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

#define MAX_LINE_LENGTH 65535

static grn_bool
file_to_train_file(grn_ctx *ctx, char *train_file,
                   char *input_file,
                   char *normalizer_name, int normalizer_len,
                   char *input_filter, char *mecab_option)
{
  FILE *fp;
  if ((fp = fopen(input_file, "r")) == NULL) {
    printf("file open error\n");
    return GRN_FALSE;
  }
  char data[MAX_LINE_LENGTH];
  FILE *fo = fopen(train_file, "wb");
  grn_obj buf, buf2;
  GRN_TEXT_INIT(&buf, 0);
  GRN_TEXT_INIT(&buf2, 0);

  while ( fgets(data, MAX_LINE_LENGTH, fp) != NULL ) {

    GRN_BULK_REWIND(&buf);
    GRN_TEXT_PUTS(ctx, &buf, data);

    const char *column_value = NULL;
    grn_obj *grn_string;

    if (normalizer_name != ""){

      grn_obj *normalizer;
      const char *normalized;
      unsigned int normalized_length_in_bytes;
      unsigned int normalized_n_characters;

      normalizer = grn_ctx_get(ctx,
                       normalizer_name,
                       normalizer_len);

      grn_string = grn_string_open(ctx,
                           GRN_TEXT_VALUE(&buf), GRN_TEXT_LEN(&buf),
                           normalizer, 0);
      grn_obj_unlink(ctx, normalizer);

      grn_string_get_normalized(ctx, grn_string,
                        &normalized,
                        &normalized_length_in_bytes,
                        &normalized_n_characters);
      column_value = normalized;
    } else {
      column_value = GRN_TEXT_VALUE(&buf);
    }

    if (input_filter != NULL) {
      string s = column_value;
      re2::RE2::GlobalReplace(&s, input_filter, " ");
      re2::RE2::GlobalReplace(&s, "[ ]+", " ");
      column_value = s.c_str();
    }

    GRN_BULK_REWIND(&buf2);
    GRN_TEXT_PUTS(ctx, &buf2, column_value);
    column_value = GRN_TEXT_VALUE(&buf2);

    right_trim((char *)column_value, '\n');
    right_trim((char *)column_value, ' ');
    if(mecab_option != NULL && strlen(column_value) > 0){
      column_value = sparse(ctx, column_value, mecab_option);
    }
    right_trim((char *)column_value, '\n');
    right_trim((char *)column_value, ' ');


    if(strlen(column_value) > 0){
      fprintf(fo, "%s\n", column_value);
    }
    grn_obj_unlink(ctx, grn_string);

  }
  grn_obj_unlink(ctx, &buf);
  grn_obj_unlink(ctx, &buf2);

  fclose(fo);
  fclose(fp);
  return GRN_TRUE;

}

static grn_bool
column_to_train_file(grn_ctx *ctx, char *train_file,
                     char *table_name, int table_len,
                     char *column_name, int column_len,
                     char *normalizer_name, int normalizer_len,
                     char *input_filter, char *mecab_option)
{
  grn_obj *table = grn_ctx_get(ctx, table_name, table_len);
  if (table) {
    FILE *fo = fopen(train_file, "wb");

    grn_obj *column;
    column = grn_obj_column(ctx, table, column_name, column_len);

    grn_table_cursor *cur;
    if ((cur = grn_table_cursor_open(ctx, table, NULL, 0, NULL, 0, 0, -1,
                                     GRN_CURSOR_BY_ID))) {
      grn_id id;
      grn_obj buf, buf2;
      GRN_TEXT_INIT(&buf, 0);
      GRN_TEXT_INIT(&buf2, 0);

      while ((id = grn_table_cursor_next(ctx, cur)) != GRN_ID_NIL) {
        GRN_BULK_REWIND(&buf);
        grn_obj_get_value(ctx, column, id, &buf);

        const char *column_value = NULL;
        grn_obj *grn_string;

        if (normalizer_name != ""){
          /* Don't use normalize function because its need buffer copy. */

          grn_obj *normalizer;
          const char *normalized;
          unsigned int normalized_length_in_bytes;
          unsigned int normalized_n_characters;

          normalizer = grn_ctx_get(ctx,
                                   normalizer_name,
                                   normalizer_len);

          grn_string = grn_string_open(ctx,
                                       GRN_TEXT_VALUE(&buf), GRN_TEXT_LEN(&buf),
                                       normalizer, 0);
          grn_obj_unlink(ctx, normalizer);

          grn_string_get_normalized(ctx, grn_string,
                                    &normalized,
                                    &normalized_length_in_bytes,
                                    &normalized_n_characters);
          column_value = normalized;
        } else {
          column_value = GRN_TEXT_VALUE(&buf);
        }


        if (input_filter != NULL) {
          string s = column_value;
          re2::RE2::GlobalReplace(&s, input_filter, " ");
          re2::RE2::GlobalReplace(&s, "[ ]+", " ");
          column_value = s.c_str();
        }

        GRN_BULK_REWIND(&buf2);
        GRN_TEXT_PUTS(ctx, &buf2, column_value);
        column_value = GRN_TEXT_VALUE(&buf2);
        
        right_trim((char *)column_value, '\n');
        right_trim((char *)column_value, ' ');
        if(mecab_option != NULL && strlen(column_value) > 0){
          column_value = sparse(ctx, column_value, mecab_option);
        }
        right_trim((char *)column_value, '\n');
        right_trim((char *)column_value, ' ');


        if(strlen(column_value) > 0){
          fprintf(fo, "%s\n", column_value);
        }

        grn_obj_unlink(ctx, grn_string);
      }
      grn_obj_unlink(ctx, &buf);
      grn_obj_unlink(ctx, &buf2);
      grn_table_cursor_close(ctx, cur);
    }
    grn_obj_unlink(ctx, table);
    table = NULL;
    fclose(fo);
    return GRN_TRUE;
  } else {
    return GRN_FALSE;
  }
}


static grn_obj *
command_word2vec_train(grn_ctx *ctx, GNUC_UNUSED int nargs, GNUC_UNUSED grn_obj **args,
                       grn_user_data *user_data)
{
  grn_obj *var;

  char *table_name = NULL;
  unsigned int table_len = 0;
  char *column_name = NULL;
  unsigned int column_len = 0;
  char *normalizer_name = (char *)"NormalizerAuto";
  unsigned int normalizer_len = 14;
  char *input_filter = NULL;
  char *mecab_option = (char *)"-Owakati";
  char *input_file = NULL;

  var = grn_plugin_proc_get_var(ctx, user_data, "table", -1);
  if(GRN_TEXT_LEN(var) != 0) {
    table_name = GRN_TEXT_VALUE(var);
    table_len = GRN_TEXT_LEN(var);
  }

  var = grn_plugin_proc_get_var(ctx, user_data, "column", -1);
  if(GRN_TEXT_LEN(var) != 0) {
    column_name = GRN_TEXT_VALUE(var);
    column_len = GRN_TEXT_LEN(var);
  }

  var = grn_plugin_proc_get_var(ctx, user_data, "normalizer", -1);
  if(GRN_TEXT_LEN(var) != 0) {
    normalizer_name = GRN_TEXT_VALUE(var);
    normalizer_len = GRN_TEXT_LEN(var);
  }

  var = grn_plugin_proc_get_var(ctx, user_data, "input_filter", -1);
  if(GRN_TEXT_LEN(var) != 0) {
    input_filter = GRN_TEXT_VALUE(var);
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "mecab_option", -1);
  if(GRN_TEXT_LEN(var) != 0) {
    mecab_option = GRN_TEXT_VALUE(var);
  }

  var = grn_plugin_proc_get_var(ctx, user_data, "input_file", -1);
  if(GRN_TEXT_LEN(var) != 0) {
    input_file = GRN_TEXT_VALUE(var);
  }

  train_file[0] = 0;
  output_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;

  var = grn_plugin_proc_get_var(ctx, user_data, "size", -1);
  if(GRN_TEXT_LEN(var) != 0) {
    layer1_size = atoi(GRN_TEXT_VALUE(var));
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "train_file", -1);
  if(GRN_TEXT_LEN(var) != 0) {
    strcpy(train_file, GRN_TEXT_VALUE(var));
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "save_vocab_file", -1);
  if(GRN_TEXT_LEN(var) != 0) {
    strcpy(save_vocab_file, GRN_TEXT_VALUE(var));
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "read_vocab_file", -1);
  if(GRN_TEXT_LEN(var) != 0) {
    strcpy(read_vocab_file, GRN_TEXT_VALUE(var));
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "debug", -1);
  if(GRN_TEXT_LEN(var) != 0) {
    debug_mode = atoi(GRN_TEXT_VALUE(var));
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "binary", -1);
  if(GRN_TEXT_LEN(var) != 0) {
    binary = atoi(GRN_TEXT_VALUE(var));
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "cbow", -1);
  if(GRN_TEXT_LEN(var) != 0) {
    cbow = atoi(GRN_TEXT_VALUE(var));
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "alpha", -1);
  if(GRN_TEXT_LEN(var) != 0) {
    alpha = atof(GRN_TEXT_VALUE(var));
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "output_file", -1);
  if(GRN_TEXT_LEN(var) != 0) {
    strcpy(output_file, GRN_TEXT_VALUE(var));
  } else {
    get_model_file_path(ctx, output_file);
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "window", -1);
  if(GRN_TEXT_LEN(var) != 0) {
    window = atoi(GRN_TEXT_VALUE(var));
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "sample", -1);
  if(GRN_TEXT_LEN(var) != 0) {
    sample = atof(GRN_TEXT_VALUE(var));
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "hs", -1);
  if(GRN_TEXT_LEN(var) != 0) {
    hs = atoi(GRN_TEXT_VALUE(var));
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "negative", -1);
  if(GRN_TEXT_LEN(var) != 0) {
    negative = atoi(GRN_TEXT_VALUE(var));
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "threads", -1);
  if(GRN_TEXT_LEN(var) != 0) {
    num_threads = atoi(GRN_TEXT_VALUE(var));
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "min-count", -1);
  if(GRN_TEXT_LEN(var) != 0) {
    min_count = atoi(GRN_TEXT_VALUE(var));
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "classes", -1);
  if(GRN_TEXT_LEN(var) != 0) {
    classes = atoi(GRN_TEXT_VALUE(var));
  }

  if (train_file[0] == 0) {
    get_train_file_path(ctx, train_file);
  }

  if(table_name != NULL && column_name != NULL) {
    if(column_to_train_file(ctx, train_file,
                            table_name, table_len,
                            column_name, column_len,
                            normalizer_name, normalizer_len,
                            input_filter, mecab_option) == GRN_TRUE)
     {
        printf("Dump column to train file %s\n", train_file);
     } else {
        printf("Dump column to train file %s failed.\n", train_file);
        return NULL;
     }
  }
  else if(input_file != NULL) {
    if(file_to_train_file(ctx, train_file,
                          input_file,
                          normalizer_name, normalizer_len,
                          input_filter, mecab_option) == GRN_TRUE)
     {
        printf("input file %s to train file %s\n", input_file, train_file);
     } else {
        printf("input file %s to train file %s failed.\n", input_file, train_file);
        return NULL;
     }
  }

  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));

  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  if (expTable == NULL) {
    fprintf(stderr, "out of memory\n");
    exit(1);
  }
  int i;
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1); // Precompute f(x) = x / (x + 1)
  }
  TrainModel();
  DestroyNet();
  free(vocab_hash);
  free(expTable);

  return NULL;
}

static grn_obj *
func_query_expander_word2ec(grn_ctx *ctx, int nargs, grn_obj **args,
                            grn_user_data *user_data)
{
  grn_rc rc = GRN_END_OF_DATA;
  grn_id id;
  grn_obj *term, *expanded_term;
  void *value;
  grn_obj *rc_object;
  grn_obj *var;

  term = args[0];
  expanded_term = args[1];

  rc = GRN_SUCCESS;

  grn_obj buf;
  GRN_TEXT_INIT(&buf, 0);
  GRN_BULK_REWIND(&buf);
  GRN_TEXT_PUTS(ctx, &buf, "word2vec_distance ");
  GRN_TEXT_PUTS(ctx, &buf, GRN_TEXT_VALUE(term));
  GRN_TEXT_PUTS(ctx, &buf, " --expander_mode 1");
  GRN_TEXT_PUTS(ctx, &buf, " --limit 3");
  GRN_TEXT_PUTS(ctx, &buf, " --threshold 0.75");

  grn_ctx_send(ctx, GRN_TEXT_VALUE(&buf) , GRN_TEXT_LEN(&buf), GRN_CTX_QUIET);

  char *result = NULL;
  unsigned int result_size = 0;
  int recv_flags;
  grn_ctx_recv(ctx, &result, &result_size, &recv_flags);

  result[result_size] = '\0';

  result++;
  right_trim(result, '"');
  
  if (result_size > 0) {
    const char *query = result;
    GRN_TEXT_PUTS(ctx, expanded_term, query);
    rc = GRN_SUCCESS;
  }

  rc_object = grn_plugin_proc_alloc(ctx, user_data, GRN_DB_INT32, 0);
  if (rc_object) {
    GRN_INT32_SET(ctx, rc_object, rc);
  }

  grn_obj_unlink(ctx, &buf);

  return rc_object;
}

grn_rc
GRN_PLUGIN_INIT(GNUC_UNUSED grn_ctx *ctx)
{
  mecab_init(ctx);
  return GRN_SUCCESS;
}

grn_rc
GRN_PLUGIN_REGISTER(grn_ctx *ctx)
{
  grn_expr_var vars[22];

  grn_plugin_expr_var_init(ctx, &vars[0], "file_path", -1);
  grn_plugin_command_create(ctx, "word2vec_load", -1, command_word2vec_load, 1, vars);
  grn_plugin_command_create(ctx, "word2vec_unload", -1, command_word2vec_unload, 0, vars);

  grn_plugin_expr_var_init(ctx, &vars[0], "term", -1);
  grn_plugin_expr_var_init(ctx, &vars[1], "offset", -1);
  grn_plugin_expr_var_init(ctx, &vars[2], "limit", -1);
  grn_plugin_expr_var_init(ctx, &vars[3], "threshold", -1);
  grn_plugin_expr_var_init(ctx, &vars[4], "normalizer", -1);
  grn_plugin_expr_var_init(ctx, &vars[5], "term_filter", -1);
  grn_plugin_expr_var_init(ctx, &vars[6], "output_filter", -1);
  grn_plugin_expr_var_init(ctx, &vars[7], "mecab_option", -1);
  grn_plugin_expr_var_init(ctx, &vars[8], "file_path", -1);
  grn_plugin_expr_var_init(ctx, &vars[9], "expander_mode", -1);
  grn_plugin_command_create(ctx, "word2vec_distance", -1, command_word2vec_distance, 10, vars);

  grn_plugin_expr_var_init(ctx, &vars[0], "positive_term1", -1);
  grn_plugin_expr_var_init(ctx, &vars[1], "negative_term", -1);
  grn_plugin_expr_var_init(ctx, &vars[2], "positive_term2", -1);
  grn_plugin_expr_var_init(ctx, &vars[3], "offset", -1);
  grn_plugin_expr_var_init(ctx, &vars[4], "limit", -1);
  grn_plugin_expr_var_init(ctx, &vars[5], "threshold", -1);
  grn_plugin_expr_var_init(ctx, &vars[6], "normalizer", -1);
  grn_plugin_expr_var_init(ctx, &vars[7], "term_filter", -1);
  grn_plugin_expr_var_init(ctx, &vars[8], "output_filter", -1);
  grn_plugin_expr_var_init(ctx, &vars[9], "mecab_option", -1);
  grn_plugin_expr_var_init(ctx, &vars[10], "file_path", -1);
  grn_plugin_expr_var_init(ctx, &vars[11], "expander_mode", -1);
  grn_plugin_command_create(ctx, "word2vec_analogy", -1, command_word2vec_analogy, 12, vars);

  grn_plugin_expr_var_init(ctx, &vars[0], "table", -1);
  grn_plugin_expr_var_init(ctx, &vars[1], "column", -1);
  grn_plugin_expr_var_init(ctx, &vars[2], "train_file", -1);
  grn_plugin_expr_var_init(ctx, &vars[3], "output_file", -1);
  grn_plugin_expr_var_init(ctx, &vars[4], "normalizer", -1);
  grn_plugin_expr_var_init(ctx, &vars[5], "input_filter", -1);
  grn_plugin_expr_var_init(ctx, &vars[6], "mecab_option", -1);
  grn_plugin_expr_var_init(ctx, &vars[7], "input_file", -1);
  grn_plugin_expr_var_init(ctx, &vars[8], "save_vocab_file", -1);
  grn_plugin_expr_var_init(ctx, &vars[9], "read_vocab_file", -1);
  grn_plugin_expr_var_init(ctx, &vars[10], "threads", -1);
  grn_plugin_expr_var_init(ctx, &vars[11], "size", -1);
  grn_plugin_expr_var_init(ctx, &vars[12], "debug", -1);
  grn_plugin_expr_var_init(ctx, &vars[13], "binary", -1);
  grn_plugin_expr_var_init(ctx, &vars[14], "cbow", -1);
  grn_plugin_expr_var_init(ctx, &vars[15], "alpha", -1);
  grn_plugin_expr_var_init(ctx, &vars[16], "window", -1);
  grn_plugin_expr_var_init(ctx, &vars[17], "sample", -1);
  grn_plugin_expr_var_init(ctx, &vars[18], "hs", -1);
  grn_plugin_expr_var_init(ctx, &vars[19], "negative", -1);
  grn_plugin_expr_var_init(ctx, &vars[20], "min-count", -1);
  grn_plugin_expr_var_init(ctx, &vars[21], "classes", -1);
  grn_plugin_command_create(ctx, "word2vec_train", -1, command_word2vec_train, 22, vars);

  grn_proc_create(ctx, "QueryExpanderWord2vec", strlen("QueryExpanderWord2vec"),
                  GRN_PROC_FUNCTION,
                  func_query_expander_word2ec, NULL, NULL,
                  0, NULL);

  return ctx->rc;
}

grn_rc
GRN_PLUGIN_FIN(GNUC_UNUSED grn_ctx *ctx)
{
  word2vec_unload(ctx);
  mecab_fin(ctx);

  return GRN_SUCCESS;
}
