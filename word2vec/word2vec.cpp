/*
  Copyright(C) 2014-2015 Naoya Murakami <naoya@createfield.com>

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
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#include <groonga/plugin.h>

#include <iostream>
#include <string>

#include "Eigen/Dense"

#define print(var)  \
  std::cout<<#var"= "<<std::endl<<(var)<<std::endl

using namespace std;
using namespace Eigen;

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

#define CONST_STR_LEN(x) x, x ? sizeof(x) - 1 : 0

#define DEFAULT_SORTBY          "-_score"
#define DEFAULT_OUTPUT_COLUMNS  "_id,_score"

#define DEFAULT_N_SORT 40
#define INSERTION_SORT_THRESHOLD 200

#define DOC_ID_PREFIX "doc_id:"
#define DOC_ID_PREFIX_LEN 7

#define MAX_COLUMNS 20
#define MAX_TERMS 100

const long long max_size = 2000; // max length of strings
const long long max_length_of_vocab_word = 255; // max length of vocabulary entries

#define MAX_MODEL 20

long long n_words[MAX_MODEL], dim_size[MAX_MODEL] = {0};
float *M[MAX_MODEL] = {NULL};
static grn_hash *model_idxes = NULL;
static grn_pat *vocab[MAX_MODEL]  = {NULL};

typedef struct {
  double score;
  int n_subrecs;
  int subrecs[1];
} grn_rset_recinfo;

typedef struct {
  char *input_filter;
  char *mecab_option;
  char *normalizer_name;
  unsigned int normalizer_len;
  grn_bool is_sentence_vectors;
  grn_bool is_phrase[MAX_COLUMNS];
  grn_bool is_mecab[MAX_COLUMNS];
  grn_bool is_remove_symbol[MAX_COLUMNS];
  grn_bool is_remove_alpha[MAX_COLUMNS];
  int weights[MAX_COLUMNS];
  string label[MAX_COLUMNS];
} train_option;


static void
output_header(grn_ctx *ctx, int nhits)
{
  grn_ctx_output_array_open(ctx, "RESULTSET", 2);
  grn_ctx_output_array_open(ctx, "NHITS", 1);
  grn_ctx_output_int32(ctx, nhits);
  grn_ctx_output_array_close(ctx);
  grn_ctx_output_array_open(ctx, "COLUMNS", 2);
  grn_ctx_output_array_open(ctx, "COLUMN", 2);
  grn_ctx_output_cstr(ctx, "_key");
  grn_ctx_output_cstr(ctx, "ShortText");
  grn_ctx_output_array_close(ctx);
  grn_ctx_output_array_open(ctx, "COLUMN", 2);
  grn_ctx_output_cstr(ctx, "_value");
  grn_ctx_output_cstr(ctx, "Float");
  grn_ctx_output_array_close(ctx);
  grn_ctx_output_array_close(ctx);
}

static void
calc_edit_distance(grn_ctx *ctx, grn_obj *res, const char *term)
{
  grn_obj *var;
  grn_obj *expr;
  GRN_EXPR_CREATE_FOR_QUERY(ctx, res, expr, var);
  if (expr) {
    grn_table_cursor *tc;
    grn_obj gterm;
    GRN_TEXT_INIT(&gterm, 0);
    GRN_TEXT_SET(ctx, &gterm, term, strlen(term));
    grn_obj *score = grn_obj_column(ctx, res,
                                    GRN_COLUMN_NAME_SCORE,
                                    GRN_COLUMN_NAME_SCORE_LEN);
    grn_obj *key = grn_obj_column(ctx, res,
                                  GRN_COLUMN_NAME_KEY,
                                  GRN_COLUMN_NAME_KEY_LEN);
    grn_expr_append_obj(ctx, expr,
                        score,
                        GRN_OP_GET_VALUE, 1);
    grn_expr_append_obj(ctx, expr,
                        grn_ctx_get(ctx, CONST_STR_LEN("edit_distance")),
                        GRN_OP_PUSH, 1);
    grn_expr_append_obj(ctx, expr,
                        key,
                        GRN_OP_GET_VALUE, 1);
    grn_expr_append_const(ctx, expr, &gterm, GRN_OP_PUSH, 1);
    grn_expr_append_op(ctx, expr, GRN_OP_CALL, 2);
    grn_expr_append_op(ctx, expr, GRN_OP_PLUS_ASSIGN, 2);

    if ((tc = grn_table_cursor_open(ctx, res, NULL, 0, NULL, 0, 0, -1, GRN_CURSOR_BY_ID))) {
      grn_id id;
      while ((id = grn_table_cursor_next(ctx, tc)) != GRN_ID_NIL) {
        GRN_RECORD_SET(ctx, var, id);
        grn_expr_exec(ctx, expr, 0);
      }
      grn_table_cursor_close(ctx, tc);

    }
    grn_obj_unlink(ctx, score);
    grn_obj_unlink(ctx, key);
    grn_obj_unlink(ctx, expr);
    grn_obj_unlink(ctx, &gterm);
  } else {
    GRN_PLUGIN_LOG(ctx, GRN_LOG_ERROR,
       "[word2vec_distance] error on building expr. for calcurating edit distance");
  }
}

static void
add_record(grn_ctx *ctx, grn_obj *res, void *word, int word_length, double score)
{
  void *value;
  if (grn_hash_add(ctx, (grn_hash *)res, word, word_length, &value, NULL)) {
    grn_rset_recinfo *ri;
    ri = (grn_rset_recinfo *)value;
    ri->score = score;
  }
}

static void
add_record_value(grn_ctx *ctx, grn_obj *res, void *word, int word_length, double score)
{
  grn_id id;
  id = grn_table_add(ctx, res, word, word_length, NULL);
  if (id) {
    grn_obj value;
    GRN_FLOAT_INIT(&value, 0);
    GRN_FLOAT_SET(ctx, &value, score);
    grn_obj_set_value(ctx, res, id, &value, GRN_OBJ_SET);
    grn_obj_unlink(ctx, &value);
  }
}

static void
add_doc_id(grn_ctx *ctx, grn_obj *res, grn_id doc_id, double score)
{
  void *value;
  if (grn_hash_add(ctx, (grn_hash *)res, &doc_id, sizeof(grn_id), &value, NULL)) {
    grn_rset_recinfo *ri;
    ri = (grn_rset_recinfo *)value;
    ri->score = score;
  }
}

static void
output_for_expander(grn_ctx *ctx, grn_obj *res, int offset, int limit, const char *sortby_val, unsigned int sortby_len, grn_obj *outbuf)
{
  grn_obj *sorted;
  if ((sorted = grn_table_create(ctx, NULL, 0, NULL, GRN_OBJ_TABLE_NO_KEY, NULL, res))) {
    uint32_t nkeys;
    grn_table_sort_key *keys;
    const char *oc_val;
    unsigned int oc_len;
    int offset = 0;
    int limit = 1;
    grn_obj buf;
    GRN_TEXT_INIT(&buf, 0);
    if (!sortby_val || !sortby_len) {
      sortby_val = DEFAULT_SORTBY;
      sortby_len = sizeof(DEFAULT_SORTBY) - 1;
    }
    if (!oc_val || !oc_len) {
      oc_val = DEFAULT_OUTPUT_COLUMNS;
      oc_len = sizeof(DEFAULT_OUTPUT_COLUMNS) - 1;
    }

    if ((keys = grn_table_sort_key_from_str(ctx, sortby_val, sortby_len, res, &nkeys))) {
      grn_table_sort(ctx, res, offset, limit, sorted, keys, nkeys);
      grn_table_cursor *tc;
      if ((tc = grn_table_cursor_open(ctx, sorted, NULL, 0, NULL, 0, offset, limit, GRN_CURSOR_BY_ID))) {
        grn_id id;
        grn_obj *column = grn_obj_column(ctx, sorted,
                                         GRN_COLUMN_NAME_KEY,
                                         GRN_COLUMN_NAME_KEY_LEN);
        while ((id = grn_table_cursor_next(ctx, tc))) {
          GRN_BULK_REWIND(&buf);
          grn_obj_get_value(ctx, column, id, &buf);
          GRN_TEXT_PUTS(ctx, outbuf, ") OR (");
          GRN_TEXT_PUT(ctx, outbuf, GRN_TEXT_VALUE(&buf), GRN_TEXT_LEN(&buf));
        }
        grn_obj_unlink(ctx, column);
        grn_table_cursor_close(ctx, tc);
      }
      grn_table_sort_key_close(ctx, keys, nkeys);
    }
    grn_obj_unlink(ctx, sorted);
    grn_obj_unlink(ctx, &buf);
  } else {
    GRN_PLUGIN_LOG(ctx, GRN_LOG_ERROR, "[word2vec_distance] cannot create temporary sort table.");
  }
}

static void
output(grn_ctx *ctx, grn_obj *res, int offset, int limit, const char *oc_val, unsigned int oc_len, const char *sortby_val, unsigned int sortby_len)
{
  grn_obj *sorted;
  if ((sorted = grn_table_create(ctx, NULL, 0, NULL, GRN_OBJ_TABLE_NO_KEY, NULL, res))) {
    uint32_t nkeys;
    grn_obj_format format;
    grn_table_sort_key *keys;
    if (!sortby_val || !sortby_len) {
      sortby_val = DEFAULT_SORTBY;
      sortby_len = sizeof(DEFAULT_SORTBY) - 1;
    }
    if (!oc_val || !oc_len) {
      oc_val = DEFAULT_OUTPUT_COLUMNS;
      oc_len = sizeof(DEFAULT_OUTPUT_COLUMNS) - 1;
    }

    if ((keys = grn_table_sort_key_from_str(ctx, sortby_val, sortby_len, res, &nkeys))) {
      grn_table_sort(ctx, res, offset, limit, sorted, keys, nkeys);

      GRN_OBJ_FORMAT_INIT(&format, grn_table_size(ctx, res), 0, limit, offset);
      format.flags = GRN_OBJ_FORMAT_WITH_COLUMN_NAMES;
      grn_obj_columns(ctx, sorted, oc_val, oc_len, &format.columns);
      grn_ctx_output_obj(ctx, sorted, &format);
      GRN_OBJ_FORMAT_FIN(ctx, &format);
      grn_table_sort_key_close(ctx, keys, nkeys);
    }
    grn_obj_unlink(ctx, sorted);
  } else {
    GRN_PLUGIN_LOG(ctx, GRN_LOG_ERROR, "[word2vec] cannot create temporary sort table.");
  }
}

static char *
right_trim(char *s, char del) {
  int i;
  int count = 0;
  if ( s == NULL ) {
    return NULL;
  }
  i = strlen(s);
  while ( --i >= 0 && s[i] == del ) count++;
  if (count > 0) {
    if (i == 0) {
      s[i] = '\0';
    } else if (i > 0) {
      s[i+1] = '\0';
    }
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
get_reference_value(grn_ctx *ctx, grn_obj *column_value, grn_obj *buf)
{
  grn_obj *temp_table;
  grn_id temp_id;
  grn_obj temp_key;
  char key_name[GRN_TABLE_MAX_KEY_SIZE];
  int key_len;
  const char *column_value_p;
  temp_table = grn_ctx_at(ctx, column_value->header.domain);
  temp_id = GRN_RECORD_VALUE(column_value);
  GRN_OBJ_INIT(&temp_key, GRN_BULK, 0, temp_table->header.domain);
  key_len = grn_table_get_key(ctx, temp_table, temp_id, key_name, GRN_TABLE_MAX_KEY_SIZE);
  GRN_BULK_REWIND(buf);
  GRN_TEXT_SET(ctx, buf, key_name, key_len);
  GRN_TEXT_PUTC(ctx, buf, '\0');
  column_value_p = GRN_TEXT_VALUE(buf);

  grn_obj_unlink(ctx, temp_table);
  GRN_OBJ_FIN(ctx, &temp_key);
  return column_value_p;
}

static const char *
get_reference_vector_value(grn_ctx *ctx, grn_obj *column_values, int i, grn_obj *buf, grn_obj *record)
{
  grn_id id;
  const char *column_value_p;
  id = grn_uvector_get_element(ctx, column_values, i, NULL);
  GRN_RECORD_SET(ctx, record, id);
  column_value_p = get_reference_value(ctx, record, buf);
  return column_value_p;
}

static const char *
normalize(grn_ctx *ctx, grn_obj *buf,
          char* normalizer_name, int normalizer_len,
          grn_obj *outbuf)
{
  grn_obj *normalizer;
  const char *normalized;
  grn_obj *grn_string = NULL;

  unsigned int normalized_length_in_bytes;
  unsigned int normalized_n_characters;
  const char *ret_normalized;

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

  GRN_BULK_REWIND(outbuf);
  if (normalized_length_in_bytes > 0) {
    GRN_TEXT_SET(ctx, outbuf, normalized, normalized_length_in_bytes);
  }
  GRN_TEXT_PUTC(ctx, outbuf, '\0');
  ret_normalized = GRN_TEXT_VALUE(outbuf);

  if (grn_string) {
    grn_obj_unlink(ctx, grn_string);
    grn_string = NULL;
  }

  return ret_normalized;
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

static int
get_model_idx(grn_ctx *ctx, const char *file_name)
{
  int model_idx = 0;
  if (!model_idxes) {
    model_idxes = grn_hash_create(ctx, NULL,
                                    GRN_TABLE_MAX_KEY_SIZE,
                                    0,
                                    GRN_OBJ_TABLE_HASH_KEY|GRN_OBJ_KEY_VAR_SIZE);
    if (!model_idxes) {
      return 0;
    }
  }
  model_idx = grn_hash_get(ctx, model_idxes,
                             file_name, strlen(file_name),
                             NULL);
  if (model_idx == GRN_ID_NIL) {
    model_idx = grn_hash_add(ctx, model_idxes,
                               file_name, strlen(file_name),
                               NULL, NULL);
    if (model_idx > MAX_MODEL || model_idx == GRN_ID_NIL) {
      GRN_PLUGIN_LOG(ctx, GRN_LOG_ERROR,
                     "[word2vec_load] "
                     "Couldn't get model index : %s",
                     file_name);
      return 0;
    }
  }
  if (model_idx) {
    model_idx--;
  }
  return model_idx;
}

static void
word2vec_unload(grn_ctx *ctx, int i)
{
  if (vocab[i] != NULL) {
    grn_pat_close(ctx, vocab[i]);
    vocab[i] = NULL;
  }
  if (M[i] != NULL){
    GRN_PLUGIN_FREE(ctx, M[i]);
    M[i] = NULL;
  }
  n_words[i] = 0;
  dim_size[i] = 0;
}

static grn_bool
word2vec_load(grn_ctx *ctx, const char *file_name, int model_idx, int binary)
{
  FILE *f;
  float len;
  long long a, b;
  grn_obj buf;

  f = fopen(file_name, "rb");
  if (f == NULL) {
    GRN_PLUGIN_LOG(ctx, GRN_LOG_ERROR,
                   "[word2vec_load] "
                   "Input file not found : %s",
                   file_name);
    return GRN_FALSE;
  }

  word2vec_unload(ctx, model_idx);

  fscanf(f, "%lld", &n_words[model_idx]);
  fscanf(f, "%lld", &dim_size[model_idx]);
  M[model_idx] = (float *)GRN_PLUGIN_MALLOC(ctx, (long long)n_words[model_idx] * (long long)dim_size[model_idx] * sizeof(float));

  vocab[model_idx] = grn_pat_create(ctx, NULL,
                                    GRN_TABLE_MAX_KEY_SIZE,
                                    0,
                                    GRN_OBJ_TABLE_PAT_KEY|GRN_OBJ_KEY_VAR_SIZE);

  if (vocab[model_idx] == NULL) {
    GRN_PLUGIN_LOG(ctx, GRN_LOG_ERROR,
                   "[word2vec_load] "
                   "Cannot create vocab table");
    return GRN_FALSE;
  }

  GRN_TEXT_INIT(&buf, 0);
  for (b = 0; b < n_words[model_idx]; b++) {
    a = 0;
    GRN_BULK_REWIND(&buf);
    if (binary == 0) {
      char format[20];
      char word[max_length_of_vocab_word];
      float value;
      sprintf(format,"%%%ds",max_length_of_vocab_word);
      fscanf(f, format, word);
      if (!grn_pat_add(ctx, vocab[model_idx], word, strlen(word), NULL, NULL)) {
        GRN_PLUGIN_LOG(ctx, GRN_LOG_ERROR, "[word2vec_load] Faild load vocab");
        grn_obj_unlink(ctx, &buf);
        return GRN_FALSE;
      }
      for (a = 0; a < dim_size[model_idx]; a++) {
        fscanf(f, "%f", &M[model_idx][a + b * dim_size[model_idx]]);
        len += M[model_idx][a + b * dim_size[model_idx]] * M[model_idx][a + b * dim_size[model_idx]];
      }
    } else {
      while (1) {
        char c = fgetc(f);
        if (c != '\n' && c!= ' ') GRN_TEXT_PUTC(ctx, &buf, c);
        if (feof(f) || (c == ' ')) break;
        if ((a < max_length_of_vocab_word) && (c != '\n')) a++;
      }

      if (!grn_pat_add(ctx, vocab[model_idx], GRN_TEXT_VALUE(&buf), GRN_TEXT_LEN(&buf), NULL, NULL)) {
        GRN_PLUGIN_LOG(ctx, GRN_LOG_ERROR, "[word2vec_load] Faild load vocab");
        grn_obj_unlink(ctx, &buf);
        return GRN_FALSE;
      }
      for (a = 0; a < dim_size[model_idx]; a++) fread(&M[model_idx][a + b * dim_size[model_idx]], sizeof(float), 1, f);
    }
    len = 0;
    for (a = 0; a < dim_size[model_idx]; a++) len += M[model_idx][a + b * dim_size[model_idx]] * M[model_idx][a + b * dim_size[model_idx]];
    len = sqrt(len);
    for (a = 0; a < dim_size[model_idx]; a++) M[model_idx][a + b * dim_size[model_idx]] /= len;
  }
  grn_obj_unlink(ctx, &buf);
  fclose(f);

  return GRN_TRUE;
}

static grn_obj *
command_word2vec_load(grn_ctx *ctx, GNUC_UNUSED int nargs, GNUC_UNUSED grn_obj **args,
                      grn_user_data *user_data)
{
  char file_name[max_size];
  grn_obj *var;
  int binary = 1;
  var = grn_plugin_proc_get_var(ctx, user_data, "file_path", -1);
  if (GRN_TEXT_LEN(var) == 0) {
    get_model_file_path(ctx, file_name);
  } else {
    strcpy(file_name, GRN_TEXT_VALUE(var));
    file_name[GRN_TEXT_LEN(var)] = '\0';
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "binary", -1);
  if (GRN_TEXT_LEN(var) != 0) {
    binary = atoi(GRN_TEXT_VALUE(var));
  }

  if (word2vec_load(ctx, file_name, get_model_idx(ctx, file_name), binary) == GRN_TRUE) {
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
  for (int i = 0; i < MAX_MODEL; i++) {
    word2vec_unload(ctx, i);
  }
  grn_ctx_output_bool(ctx, GRN_TRUE);
  return NULL;
}

static grn_obj *
command_word2vec_distance(grn_ctx *ctx, GNUC_UNUSED int nargs, GNUC_UNUSED grn_obj **args,
                          grn_user_data *user_data)
{
  const char *input;
  long long N = DEFAULT_N_SORT;
  char input_term[MAX_TERMS][max_length_of_vocab_word];
  long long found_row_idx[MAX_TERMS];
  char op[MAX_TERMS] = {'+'};
  float dist, len;
  float *vec;
  char **bestw;
  float *bestd;
  long long *besti;
  long long a, b;
  int input_n_words = 0;
  grn_obj *var;
  int offset = 0;
  int limit = 10;
  float threshold = -1;
  char *normalizer_name = (char *)"NormalizerAuto";
  int normalizer_len = 14;
  char *stop_filter = NULL;
  char *prefix_filter = NULL;
  char *output_filter = NULL;
  char *mecab_option = NULL;
  int expander_mode = 0;
  grn_bool is_phrase = GRN_FALSE;
  grn_bool is_sentence_vectors = GRN_FALSE;
  unsigned int edit_distance = 0;
  char *table_name = NULL;
  unsigned int table_len = 0;
  char *column_names = NULL;
  unsigned int column_names_len = 0;
  char *sortby = NULL;
  unsigned int sortby_len = 0;
  char file_name[max_size];
  int model_idx = 0;
  grn_obj *table = NULL;
  grn_obj *res = NULL;
  grn_pat_cursor *pc;
  int binary = 1;
  int pca = 0;
  int pca_centered = 1;
  int total_count = 0;

  var = grn_plugin_proc_get_var(ctx, user_data, "file_path", -1);
  if (GRN_TEXT_LEN(var) == 0) {
    get_model_file_path(ctx, file_name);
  } else {
    strcpy(file_name, GRN_TEXT_VALUE(var));
    file_name[GRN_TEXT_LEN(var)] = '\0';
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "binary", -1);
  if (GRN_TEXT_LEN(var) != 0) {
    binary = atoi(GRN_TEXT_VALUE(var));
  }

  model_idx = get_model_idx(ctx, file_name);
  if (M[model_idx] == NULL || vocab[model_idx] == NULL) {
    if (word2vec_load(ctx, file_name, model_idx, binary) == GRN_FALSE) {
      grn_ctx_output_bool(ctx, GRN_FALSE);
      return NULL;
    }
  }

  var = grn_plugin_proc_get_var(ctx, user_data, "offset", -1);
  if (GRN_TEXT_LEN(var) != 0) {
    offset = atoi(GRN_TEXT_VALUE(var));
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "limit", -1);
  if (GRN_TEXT_LEN(var) != 0) {
    limit = atoi(GRN_TEXT_VALUE(var));
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "n_sort", -1);
  if (GRN_TEXT_LEN(var) != 0) {
    N = atoi(GRN_TEXT_VALUE(var));
    if (N < 0) {
      N = DEFAULT_N_SORT;
    }
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "threshold", -1);
  if (GRN_TEXT_LEN(var) != 0) {
    threshold = atof(GRN_TEXT_VALUE(var));
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "normalizer", -1);
  if (GRN_TEXT_LEN(var) != 0) {
    if (GRN_TEXT_LEN(var) == 4 && memcmp(GRN_TEXT_VALUE(var), "NONE", 4) == 0) {
      normalizer_len = 0;
    } else {
      normalizer_name = GRN_TEXT_VALUE(var);
      normalizer_len = GRN_TEXT_LEN(var);
    }
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "prefix_filter", -1);
  if (GRN_TEXT_LEN(var) != 0) {
    prefix_filter = GRN_TEXT_VALUE(var);
    prefix_filter[GRN_TEXT_LEN(var)] = '\0';
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "stop_filter", -1);
  if (GRN_TEXT_LEN(var) != 0) {
    stop_filter = GRN_TEXT_VALUE(var);
    stop_filter[GRN_TEXT_LEN(var)] = '\0';
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "output_filter", -1);
  if (GRN_TEXT_LEN(var) != 0) {
    output_filter = GRN_TEXT_VALUE(var);
    output_filter[GRN_TEXT_LEN(var)] = '\0';
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "mecab_option", -1);
  if (GRN_TEXT_LEN(var) != 0) {
    if (GRN_TEXT_LEN(var) == 4 && memcmp(GRN_TEXT_VALUE(var), "NONE", 4) == 0) {
      mecab_option = NULL;
    } else {
      mecab_option = GRN_TEXT_VALUE(var);
    }
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "expander_mode", -1);
  if (GRN_TEXT_LEN(var) != 0) {
    expander_mode = atoi(GRN_TEXT_VALUE(var));
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "is_phrase", -1);
  if (GRN_TEXT_LEN(var) != 0) {
    is_phrase = atoi(GRN_TEXT_VALUE(var));
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "edit_distance", -1);
  if (GRN_TEXT_LEN(var) != 0) {
    edit_distance = atoi(GRN_TEXT_VALUE(var));
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "pca", -1);
  if (GRN_TEXT_LEN(var) != 0) {
    pca = atoi(GRN_TEXT_VALUE(var));
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "pca_centered", -1);
  if (GRN_TEXT_LEN(var) != 0) {
    pca_centered = atoi(GRN_TEXT_VALUE(var));
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "sentence_vectors", -1);
  if (GRN_TEXT_LEN(var) != 0) {
    is_sentence_vectors = atoi(GRN_TEXT_VALUE(var));
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "table", -1);
  if (GRN_TEXT_LEN(var) != 0) {
    table_name = GRN_TEXT_VALUE(var);
    table_len = GRN_TEXT_LEN(var);
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "column", -1);
  if (GRN_TEXT_LEN(var) != 0) {
    column_names = GRN_TEXT_VALUE(var);
    column_names_len = GRN_TEXT_LEN(var);
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "sortby", -1);
  if (GRN_TEXT_LEN(var) != 0) {
    sortby = GRN_TEXT_VALUE(var);
    sortby_len = GRN_TEXT_LEN(var);
  }

  var = grn_plugin_proc_get_var(ctx, user_data, "term", -1);

  if (GRN_TEXT_LEN(var) == 0) {
    GRN_PLUGIN_LOG(ctx, GRN_LOG_NOTICE,
                   "[plugin][word2vec][distance] empty term");
    grn_ctx_output_bool(ctx, GRN_FALSE);
    return NULL;
  } else {
    grn_obj buf;
    int array_len;
    char *result_array[MAX_TERMS];
    int op_row = 1;

    GRN_TEXT_INIT(&buf, 0);
    GRN_BULK_REWIND(&buf);

    if (normalizer_len){
      input = normalize(ctx, var, normalizer_name, normalizer_len, &buf);
    } else {
      GRN_TEXT_PUTC(ctx, var, '\0');
      input = GRN_TEXT_VALUE(var);
    }
    right_trim((char *)input, '\n');
    right_trim((char *)input, ' ');
    if (mecab_option != NULL && strlen(input) > 0){
      input = sparse(ctx, input, mecab_option);
      right_trim((char *)input, '\n');
      right_trim((char *)input, ' ');
    }
    if (is_phrase) {
      string s = input;
      re2::RE2::GlobalReplace(&s, " ", "_");
      strcpy((char *)input, s.c_str());
    }
    {
       const char *s, *e, *l;
       s = input;
       e = input;
       l = input + strlen(input) + 1;
       for (e = input; e < l ; e++) {
         if (e[0] == ' ' || e[0] == '\0') {
           memcpy(input_term[input_n_words], s, e - s);
           input_term[input_n_words][e - s] = '\0';
           input_n_words++;
           s = e + 1;
         } else if (e > input && e < l - 1 && e[-1] == ' ' && e[0] == '+' && e[1] == ' ') {
            op[op_row] = '+';
            op_row++;
            e++;
            s = e + 1;
         } else if (e > input && e < l - 1 && e[-1] == ' ' && e[0] == '-' && e[1] == ' ') {
            op[op_row] = '-';
            op_row++;
            e++;
            s = e + 1;
         }
       }
    }
    grn_obj_unlink(ctx, &buf);
  }

  for (a = 0; a < input_n_words; a++) {
    found_row_idx[a] = grn_pat_get(ctx, vocab[model_idx], input_term[a], strlen(input_term[a]), NULL);

    found_row_idx[a]--;
    if (found_row_idx[a] == -1) {
      if (expander_mode == GRN_EXPANDER_NONE) {
        output_header(ctx, 0);
        grn_ctx_output_array_open(ctx, "HIT", 2);
        grn_ctx_output_cstr(ctx, "Output of dictionary word!");
        grn_ctx_output_float(ctx, 0);
        grn_ctx_output_array_close(ctx);
        grn_ctx_output_array_close(ctx);
      } else {
        grn_ctx_output_cstr(ctx, input);
      }
      return NULL;
    }
  }

  if ((is_sentence_vectors && table_len)) {
    table = grn_ctx_get(ctx, table_name, table_len);
    if (!table) {
      GRN_PLUGIN_LOG(ctx, GRN_LOG_ERROR,
                     "[word2vec_distance] couldn't open table %.*s", table_len, table_name);
      return NULL;
    }
    if (table) {
      res = grn_table_create(ctx, NULL, 0, NULL,
                             GRN_TABLE_HASH_KEY|GRN_OBJ_WITH_SUBREC,
                             table, NULL);
      if (!res) {
        GRN_PLUGIN_LOG(ctx, GRN_LOG_ERROR,
                       "[word2vec_distance] couldn't create temp table");
        grn_obj_unlink(ctx, table);
        return NULL;
      }
    }
  } else if (!pca || pca && N >= INSERTION_SORT_THRESHOLD) {
    res = grn_table_create(ctx, NULL, 0, NULL,
                           GRN_TABLE_HASH_KEY|GRN_OBJ_WITH_SUBREC,
                           grn_ctx_at(ctx, GRN_DB_SHORT_TEXT), grn_ctx_at(ctx, GRN_DB_FLOAT));
    if (!res) {
      GRN_PLUGIN_LOG(ctx, GRN_LOG_ERROR,
                     "[word2vec_distance] couldn't create temp table");
      return NULL;
    }
  }

  vec = (float *)GRN_PLUGIN_MALLOC(ctx, dim_size[model_idx] * sizeof(float));

  if (input_n_words == 1) {
    for (a = 0; a < dim_size[model_idx]; a++) vec[a] = 0;
    for (a = 0; a < dim_size[model_idx]; a++) vec[a] += M[model_idx][a + found_row_idx[b] * dim_size[model_idx]];
  } else {
    for (a = 0; a < dim_size[model_idx]; a++) vec[a] = 0;
    for (a = 0; a < dim_size[model_idx]; a++) {
      for (b = 0; b < input_n_words; b++) {
        if (op[b] == '-') {
          vec[a] -= M[model_idx][a + found_row_idx[b] * dim_size[model_idx]];
        } else {
          vec[a] += M[model_idx][a + found_row_idx[b] * dim_size[model_idx]];
        }
      }
    }
  }

  len = 0;
  for (a = 0; a < dim_size[model_idx]; a++) len += vec[a] * vec[a];
  len = sqrt(len);
  for (a = 0; a < dim_size[model_idx]; a++) vec[a] /= len;

  bestw = (char **)GRN_PLUGIN_MALLOC(ctx, N * sizeof(char *));
  for (a = 0; a < N; a++) {
    bestw[a] = (char *)GRN_PLUGIN_MALLOC(ctx, max_length_of_vocab_word * sizeof(char));
  }
  bestd = (float *)GRN_PLUGIN_MALLOC(ctx, N * sizeof(float));
  besti = (long long *)GRN_PLUGIN_MALLOC(ctx, N * sizeof(long long));
  for (a = 0; a < N; a++) {
    bestd[a] = -1;
    besti[a] = 0;
    bestw[a][0] = 0;
  }

  if (is_sentence_vectors) {
    pc = grn_pat_cursor_open(ctx, vocab[model_idx], DOC_ID_PREFIX, DOC_ID_PREFIX_LEN, NULL, 0, 0, -1, GRN_CURSOR_PREFIX);
  } else if (prefix_filter != NULL) {
    pc = grn_pat_cursor_open(ctx, vocab[model_idx], prefix_filter, strlen(prefix_filter), NULL, 0, 0, -1, GRN_CURSOR_PREFIX);
  } else {
    pc = grn_pat_cursor_open(ctx, vocab[model_idx], NULL, 0, NULL, 0, 0, -1, GRN_CURSOR_BY_ID);
  }
  if (pc) {
    long long word_idx;
    while ((word_idx = grn_pat_cursor_next(ctx, pc)) != GRN_ID_NIL) {
      char key_name[GRN_TABLE_MAX_KEY_SIZE];
      int key_len;
      /* convert grn_id to idx of array */
      word_idx--;
      a = 0;
      /* skip same word */
      for (b = 0; b < input_n_words; b++) if (found_row_idx[b] == word_idx) a = 1;
      if (a == 1) continue;

      /* get word */
      key_len = grn_pat_get_key(ctx, vocab[model_idx], word_idx + 1, key_name, GRN_TABLE_MAX_KEY_SIZE);
      key_name[key_len] = '\0';

      /* filter by regexp */
      if (stop_filter != NULL) {
        string s = key_name;
        string t = stop_filter;
        if ( RE2::FullMatch(s, t) ) {
          continue;
        }
      }

      /* calc distance */
      dist = 0;
      for (a = 0; a < dim_size[model_idx]; a++) dist += vec[a] * M[model_idx][a + word_idx * dim_size[model_idx]];

      /* skip if distance is under threshold */
      if (threshold > 0 && dist < threshold) {
        continue;
      }

      /* Insertion sort to bestw[N] */
      if (N < INSERTION_SORT_THRESHOLD) {
        for (a = 0; a < N; a++) {
          if (dist > bestd[a]) {
            for (long long d = N - 1; d > a; d--) {
              bestd[d] = bestd[d - 1];
              besti[d] = besti[d - 1];
              strcpy(bestw[d], bestw[d - 1]);
            }
            bestd[a] = dist;
            besti[a] = word_idx;
            strcpy(bestw[a], key_name);
            break;
          }
        }
      } else {
        if (key_len > 0) {
          if (is_sentence_vectors && table_len) {
            char *doc_id_p;
            grn_id doc_id = 0;
            doc_id_p = key_name;
            doc_id_p += DOC_ID_PREFIX_LEN;
            doc_id = atoi(doc_id_p);
            if (doc_id) {
              add_record_value(ctx, res, &doc_id, sizeof(grn_id), dist);
            }
          } else {
            add_record_value(ctx, res, key_name, strlen(key_name), dist);
          }
        }
      }
    }
    grn_pat_cursor_close(ctx, pc);
  }

  if (N < INSERTION_SORT_THRESHOLD) {
    for (a = 0; a < N; a++) {
      if (strlen(bestw[a]) > 0) {
        if (output_filter != NULL || is_phrase) {
          string s = bestw[a];
          if (is_phrase) {
            re2::RE2::GlobalReplace(&s, "_", " ");
          }
          if (output_filter != NULL) {
            re2::RE2::GlobalReplace(&s, output_filter, "");
          }
          strcpy(bestw[a], s.c_str());
        }

        total_count++;
        if (is_sentence_vectors && table_len) {
          char *doc_id_p;
          grn_id doc_id = 0;
          doc_id_p = bestw[a];
          doc_id_p += DOC_ID_PREFIX_LEN;
          doc_id = atoi(doc_id_p);
          if (doc_id) {
            add_record_value(ctx, res, &doc_id, sizeof(grn_id), bestd[a]);
          }
        } else if (!pca) {
          add_record_value(ctx, res, bestw[a], strlen(bestw[a]), bestd[a]);
        }
      }
    }
  } else {
    grn_obj *sorted;
    if ((sorted = grn_table_create(ctx, NULL, 0, NULL, GRN_OBJ_TABLE_NO_KEY, NULL, res))) {
      uint32_t nkeys;
      grn_table_sort_key *keys;
      grn_table_cursor *cur;
      grn_id id;
      if ((keys = grn_table_sort_key_from_str(ctx, "-_value", strlen("-_value"), res, &nkeys))) {
        grn_table_sort(ctx, res, 0, -1, sorted, keys, nkeys);
        grn_table_sort_key_close(ctx, keys, nkeys);
        if (cur = grn_table_cursor_open(ctx, sorted, NULL, 0, NULL, 0, 0, N,
                                        GRN_CURSOR_BY_ID)) {
          grn_obj buf;
          grn_obj value_buf;
          GRN_TEXT_INIT(&buf, 0);
          GRN_FLOAT_INIT(&value_buf, 0);
          grn_obj *key = grn_obj_column(ctx, sorted,
                                           GRN_COLUMN_NAME_KEY,
                                           GRN_COLUMN_NAME_KEY_LEN);
          grn_obj *value = grn_obj_column(ctx, sorted,
                                           GRN_COLUMN_NAME_VALUE,
                                           GRN_COLUMN_NAME_VALUE_LEN);

          total_count = 0;
          while ((id = grn_table_cursor_next(ctx, cur)) != GRN_ID_NIL) {
            GRN_BULK_REWIND(&buf);
            GRN_BULK_REWIND(&value_buf);
            grn_obj_get_value(ctx, key, id, &buf);

            memcpy(bestw[total_count], GRN_TEXT_VALUE(&buf), GRN_TEXT_LEN(&buf));
            bestw[total_count][GRN_TEXT_LEN(&buf)] = '\0';
            if (output_filter != NULL || is_phrase) {
              string s = bestw[total_count];
              if (is_phrase) {
                re2::RE2::GlobalReplace(&s, "_", " ");
              }
              if (output_filter != NULL) {
                re2::RE2::GlobalReplace(&s, output_filter, "");
              }
              strcpy(bestw[total_count], s.c_str());
            }
            grn_obj_get_value(ctx, value, id, &value_buf);
            bestd[total_count] = GRN_FLOAT_VALUE(&value_buf);
            id = grn_table_get(ctx, res, GRN_TEXT_VALUE(&buf), GRN_TEXT_LEN(&buf));
            besti[total_count] = id - 1;
            total_count++;
          }
          grn_obj_unlink(ctx, key);
          grn_obj_unlink(ctx, value);
          grn_obj_unlink(ctx, &value_buf);
          grn_obj_unlink(ctx, &buf);
          grn_table_cursor_close(ctx, cur);
        }
      }
      grn_obj_unlink(ctx, sorted);
    } else {
      GRN_PLUGIN_LOG(ctx, GRN_LOG_ERROR, "[word2vec] cannot create temporary sort table.");
    }
  }

  if (expander_mode == GRN_EXPANDER_EXPANDED) {
    grn_obj buf;
    GRN_TEXT_INIT(&buf, 0);
    GRN_BULK_REWIND(&buf);
    GRN_TEXT_PUTS(ctx, &buf, "((");
    GRN_TEXT_PUTS(ctx, &buf, input_term[0]);
    output_for_expander(ctx, res, offset, limit, "-_value", strlen("-_value"), &buf);
    GRN_TEXT_PUTS(ctx, &buf, "))");
    grn_ctx_output_obj(ctx, &buf, NULL);
    grn_obj_unlink(ctx, &buf);
  } else if (pca) {
    /* Map to matrix of Eigen */
    MatrixXf X(total_count + 1, dim_size[model_idx]);
    for (b = 0; b < dim_size[model_idx]; b++) {
      X(0, b) = M[model_idx][b + found_row_idx[0] * dim_size[model_idx]];
    }
    for (a = 0; a < total_count; a++) {
      if (strlen(bestw[a]) > 0) {
        for (b = 0; b < dim_size[model_idx]; b++) {
          X(a+1, b) = M[model_idx][b + besti[a] * dim_size[model_idx]];
        }
      }
    }
    /* PCA reduce dimension to pca */
    if (pca_centered == 1) {
      X = X.rowwise() - X.colwise().mean();
    }
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(X, Eigen::ComputeThinU | Eigen::ComputeThinV);
    MatrixXf proj = svd.matrixU() * svd.singularValues().asDiagonal();
    MatrixXf Y = proj.leftCols(pca);

    /* output */
    /* header */
    grn_ctx_output_array_open(ctx, "RESULTSET", 2 + total_count + 1);
    grn_ctx_output_array_open(ctx, "NHITS", 1);
    grn_ctx_output_int32(ctx, total_count + 1);
    grn_ctx_output_array_close(ctx);

    /* columns */
    grn_ctx_output_array_open(ctx, "COLUMNS", 2 + pca);
    grn_ctx_output_array_open(ctx, "COLUMN", 2);
    grn_ctx_output_cstr(ctx, "_key");
    grn_ctx_output_cstr(ctx, "ShortText");
    grn_ctx_output_array_close(ctx);
    grn_ctx_output_array_open(ctx, "COLUMN", 2);
    grn_ctx_output_cstr(ctx, "_value");
    grn_ctx_output_cstr(ctx, "Float");
    grn_ctx_output_array_close(ctx);
    for (a = 0; a < pca; a++) {
      grn_ctx_output_array_open(ctx, "COLUMN", 2);
      grn_ctx_output_cstr(ctx, "_pca_value");
      grn_ctx_output_cstr(ctx, "Float");
      grn_ctx_output_array_close(ctx);
    }
    grn_ctx_output_array_close(ctx);

    /* input term */
    grn_ctx_output_array_open(ctx, "HIT", 2 + pca);
    if (output_filter != NULL || is_phrase) {
      string s = input_term[0];
      if (is_phrase) {
        re2::RE2::GlobalReplace(&s, "_", " ");
      }
      if (output_filter != NULL) {
        re2::RE2::GlobalReplace(&s, output_filter, "");
      }
      strcpy(input_term[0], s.c_str());
    }

    grn_ctx_output_cstr(ctx, input_term[0]);
    grn_ctx_output_float(ctx, 1);
    for (b = 0; b < pca; b++) {
      grn_ctx_output_float(ctx, Y(0,b));
    }
    grn_ctx_output_array_close(ctx);

    /* neighbor terms */
    {
      unsigned int max;
      if (offset + limit > N) {
        max = N;
      } else {
        max = offset + limit;
      }
      for (a = offset; a < max; a++) {
        if (strlen(bestw[a]) > 0) {
          grn_ctx_output_array_open(ctx, "HIT", 2 + pca);
          grn_ctx_output_cstr(ctx, bestw[a]);
          grn_ctx_output_float(ctx, bestd[a]);
          for (b = 0; b < pca; b++) {
            grn_ctx_output_float(ctx, Y(a+1,b));
          }
          grn_ctx_output_array_close(ctx);
        }
      }
    }
    grn_ctx_output_array_close(ctx);
  } else {
    if (is_sentence_vectors && table_len) {
      output(ctx, res, offset, limit, column_names, column_names_len, sortby, sortby_len);
    } else if (edit_distance) {
      calc_edit_distance(ctx, res, input_term[0]);
      output(ctx, res, offset, limit, "_key,_score", strlen("_key,_score"), "_score", strlen("_score"));
    } else {
      output(ctx, res, offset, limit, "_key,_value", strlen("_key,_value"), "-_value", strlen("-_value"));
    }
  }

  for (a = 0; a < N; a++) {
    GRN_PLUGIN_FREE(ctx, bestw[a]);
    bestw[a] = NULL;
  }
  GRN_PLUGIN_FREE(ctx, bestw);
  bestw = NULL;
  GRN_PLUGIN_FREE(ctx, bestd);
  bestd = NULL;
  GRN_PLUGIN_FREE(ctx, besti);
  besti = NULL;

  GRN_PLUGIN_FREE(ctx, vec);
  vec = NULL;

  if (res) {
    grn_obj_close(ctx, res);
    res = NULL;
  }
  if (table) {
    grn_obj_close(ctx, table);
    table = NULL;
  }

  return NULL;
}

static grn_bool
is_record(grn_ctx *ctx, grn_obj *obj)
{
  grn_obj *domain = grn_ctx_at(ctx, obj->header.domain);
  if (domain) {
    grn_id type = domain->header.type;
    grn_obj_unlink(ctx, domain);
    switch (type) {
    case GRN_TABLE_HASH_KEY :
    case GRN_TABLE_PAT_KEY :
    case GRN_TABLE_NO_KEY :
      return GRN_TRUE;
    }
  }
  return GRN_FALSE;
}

static grn_bool
filter_and_add_vector_element(grn_ctx *ctx,
                              grn_obj *vbuf,
                              int i,
                              const char **column_value_p,
                              grn_obj *get_buf,
                              train_option option)
{
  if (option.normalizer_len) {
    *column_value_p = normalize(ctx, get_buf,
                                option.normalizer_name,
                                option.normalizer_len,
                                get_buf);
  }

  if (option.mecab_option != NULL && option.is_mecab[i] &&
    strlen(*column_value_p) > 0){
    *column_value_p = sparse(ctx, *column_value_p, option.mecab_option);
    right_trim((char *)*column_value_p, '\n');
    right_trim((char *)*column_value_p, ' ');
  }

  if (option.input_filter != NULL || option.is_phrase[i] ||
      option.is_remove_symbol[i] || option.is_remove_alpha[i] || option.label[i].length() > 0) {
    string s = *column_value_p;
    if (option.input_filter != NULL) {
      re2::RE2::GlobalReplace(&s, option.input_filter, " ");
      re2::RE2::GlobalReplace(&s, "[ ]+", " ");
    }
    if (option.is_remove_symbol[i]) {
      re2::RE2::GlobalReplace(&s, "(<[^>]*>)|([0-9,.;:&^/\\-−#'\"()\\[\\]、。【】「」~・])", " ");
      re2::RE2::GlobalReplace(&s, "[ ]+", " ");
    }
    if (option.is_remove_alpha[i]) {
      re2::RE2::GlobalReplace(&s, "([a-zA-Z]+)", " ");
      re2::RE2::GlobalReplace(&s, "[ ]+", " ");
    }
    if (option.is_phrase[i]) {
      re2::RE2::GlobalReplace(&s, " ", "_");
    }
    if (option.label[i].length() > 0) {
      s = option.label[i] + s;
    }
    GRN_BULK_REWIND(get_buf);
    GRN_TEXT_SET(ctx, get_buf, s.c_str(), s.length());
    GRN_TEXT_PUTC(ctx, get_buf, '\0');
    *column_value_p = GRN_TEXT_VALUE(get_buf);
  }
  right_trim((char *)*column_value_p, '\n');
  right_trim((char *)*column_value_p, ' ');

  if (strlen(*column_value_p) == 0){
    return GRN_FALSE;
  } else {
    for (int w = 0; w < option.weights[i]; w++) {
      grn_vector_add_element(ctx, vbuf,
                             *column_value_p,
                             strlen(*column_value_p),
                             0, GRN_DB_TEXT);
    }
    return GRN_TRUE;
  }
}

static grn_bool
column_to_train_file(grn_ctx *ctx, char *train_file,
                     char *table_name, int table_len,
                     const char *column_names,
                     char *filter,
                     train_option option)
{
  grn_obj *table = grn_ctx_get(ctx, table_name, table_len);
  if (table) {
    FILE *fo = fopen(train_file, "wb");
    char column_name_array[MAX_COLUMNS][max_size];
    int i, t, array_len = 0;
    grn_obj *columns[MAX_COLUMNS];
    grn_table_cursor *cur;
    grn_obj *result = NULL;

    /* parse column option */
    {
       const char *s, *e, *l;
       s = column_names;
       e = column_names;
       l = column_names + strlen(column_names) + 1;
       for (e = column_names; e < l ; e++) {
         if (e[0] == ',' || e[0] == '\0') {
           memcpy(column_name_array[array_len], s, e - s);
           column_name_array[array_len][e - s] = '\0';
           array_len++;
           s = e + 1;
         }
       }
    }
    for (i = 0; i < array_len; i++) {
      if (column_name_array[i][strlen(column_name_array[i]) - 1] == ']') {
        string s = column_name_array[i];
        re2::RE2::FullMatch(s, ".*\\[(.+)\\]", &option.label[i]);
        re2::RE2::GlobalReplace(&s, "\\[.+\\]$", "");
        strcpy(column_name_array[i], s.c_str());
      } else {
        option.label[i] = "";
      }
      if (column_name_array[i][strlen(column_name_array[i]) - 2] == '*' &&
          column_name_array[i][strlen(column_name_array[i]) - 1] >= '2' &&
          column_name_array[i][strlen(column_name_array[i]) - 1] <= '9') {
        option.weights[i] = (int)(column_name_array[i][strlen(column_name_array[i]) - 1]) - (int)('0');
        right_trim(column_name_array[i], column_name_array[i][strlen(column_name_array[i]) - 1]);
        right_trim(column_name_array[i], column_name_array[i][strlen(column_name_array[i]) - 1]);
      } else {
        option.weights[i] = 1;
      }
      if (column_name_array[i][strlen(column_name_array[i]) - 1] == '$') {
        option.is_remove_symbol[i] = GRN_TRUE;
        right_trim(column_name_array[i], '$');
      } else {
        option.is_remove_symbol[i] = GRN_FALSE;
      }
      if (column_name_array[i][strlen(column_name_array[i]) - 1] == '@') {
        option.is_remove_alpha[i] = GRN_TRUE;
        right_trim(column_name_array[i], '@');
      } else {
        option.is_remove_alpha[i] = GRN_FALSE;
      }
      if (column_name_array[i][strlen(column_name_array[i]) - 1] == '_') {
        option.is_phrase[i] = GRN_TRUE;
        right_trim(column_name_array[i], '_');
      } else {
        option.is_phrase[i] = GRN_FALSE;
      }
      if (column_name_array[i][strlen(column_name_array[i]) - 1] == '/') {
        option.is_mecab[i] = GRN_TRUE;
        right_trim(column_name_array[i], '/');
      } else {
        option.is_mecab[i] = GRN_FALSE;
      }
      columns[i] = grn_obj_column(ctx, table, column_name_array[i], strlen(column_name_array[i]));
    }

    /* select by script */
    if (filter) {
      grn_obj *v, *cond;

      GRN_EXPR_CREATE_FOR_QUERY(ctx, table, cond, v);
      grn_expr_parse(ctx, cond,
                     filter,
                     strlen(filter),
                     NULL,
                     GRN_OP_MATCH,
                     GRN_OP_AND,
                     GRN_EXPR_SYNTAX_SCRIPT);

      result = grn_table_create(ctx, NULL, 0, NULL,
                                GRN_TABLE_HASH_KEY|GRN_OBJ_WITH_SUBREC,
                                table, NULL);
      if (result) {
        grn_table_select(ctx, table, cond, result, GRN_OP_OR);
      }
      if (cond) {
        grn_obj_unlink(ctx, cond);
      }
    }
    if (result) {
      cur = grn_table_cursor_open(ctx, result, NULL, 0, NULL, 0, 0, -1,
                                  GRN_CURSOR_BY_ID);
    } else {
      cur = grn_table_cursor_open(ctx, table, NULL, 0, NULL, 0, 0, -1,
                                  GRN_CURSOR_BY_ID);
    }

    /* output and filter column */
    if (cur) {
      grn_id id;
      grn_obj column_value, get_buf;
      grn_obj vbuf;
      GRN_TEXT_INIT(&column_value, 0);
      GRN_TEXT_INIT(&get_buf, 0);
      GRN_TEXT_INIT(&vbuf, GRN_OBJ_VECTOR);

      while ((id = grn_table_cursor_next(ctx, cur)) != GRN_ID_NIL) {
        if (result) {
          grn_table_get_key(ctx, result, id, &id, sizeof(unsigned int));
        }
        GRN_BULK_REWIND(&vbuf);
        for (i = 0; i < array_len; i++) {
          const char *column_value_p = NULL;
          GRN_BULK_REWIND(&column_value);
          grn_obj_get_value(ctx, columns[i], id, &column_value);

          /* reference vector */
          if ((&(column_value))->header.type == GRN_UVECTOR) {
            grn_obj record;
            int n;
            GRN_RECORD_INIT(&record, 0, ((&column_value))->header.domain);
            GRN_BULK_REWIND(&get_buf);
            n = grn_vector_size(ctx, &column_value);
            for (int s = 0; s < n; s++) {
              column_value_p = get_reference_vector_value(ctx, &column_value, s, &get_buf, &record);
              filter_and_add_vector_element(ctx, &vbuf, i,
                                            &column_value_p,
                                            &get_buf,
                                            option);
           }
            GRN_OBJ_FIN(ctx, &record);
          } else {
            /* vector column */
            if ((&(column_value))->header.type == GRN_VECTOR) {
              grn_obj record;
              int n;
              GRN_RECORD_INIT(&record, 0, ((&column_value))->header.domain);
              GRN_BULK_REWIND(&get_buf);
              n = grn_vector_size(ctx, &column_value);
              for (int s = 0; s < n; s++) {
                unsigned int length;
                length = grn_vector_get_element(ctx, &column_value, s, &column_value_p, NULL, NULL);

                GRN_BULK_REWIND(&get_buf);
                GRN_TEXT_SET(ctx, &get_buf, column_value_p, length);
                GRN_TEXT_PUTC(ctx, &get_buf, '\0');
                column_value_p = GRN_TEXT_VALUE(&get_buf);
                filter_and_add_vector_element(ctx, &vbuf, i,
                                              &column_value_p,
                                              &get_buf,
                                              option);
              }
              GRN_OBJ_FIN(ctx, &record);
            } else {
              /* reference column */
              if (is_record(ctx, &column_value)) {
                column_value_p = get_reference_value(ctx, &column_value, &column_value);
              }
              GRN_BULK_REWIND(&get_buf);
              GRN_TEXT_SET(ctx, &get_buf, GRN_TEXT_VALUE(&column_value), GRN_TEXT_LEN(&column_value));
              GRN_TEXT_PUTC(ctx, &get_buf, '\0');
              column_value_p = GRN_TEXT_VALUE(&get_buf);
              filter_and_add_vector_element(ctx, &vbuf, i,
                                            &column_value_p,
                                            &get_buf,
                                            option);
            }
          }
        }
        if (option.is_sentence_vectors) {
          fprintf(fo, "%.*s%d ", DOC_ID_PREFIX_LEN, DOC_ID_PREFIX, id);
        }
        for (t = 0; t < grn_vector_size(ctx, &vbuf); t++) {
          const char *value;
          unsigned int length;
          length = grn_vector_get_element(ctx, &vbuf, t, &value, NULL,  NULL);
          if (t > 0) {
            fprintf(fo, " ");
          }
          fprintf(fo, "%.*s", length, value);
        }
        fprintf(fo, "\n");
      }
      for (i = 0; i < array_len; i++) {
        if (columns[i]) {
          grn_obj_unlink(ctx, columns[i]);
        }
      }
      grn_obj_unlink(ctx, &column_value);
      grn_obj_unlink(ctx, &get_buf);
      grn_obj_unlink(ctx, &vbuf);
      grn_table_cursor_close(ctx, cur);
      if (result) {
        grn_obj_unlink(ctx, result);
        result = NULL;
      }
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
command_dump_to_train_file(grn_ctx *ctx, GNUC_UNUSED int nargs, GNUC_UNUSED grn_obj **args,
                           grn_user_data *user_data)
{
  grn_obj *var;
  char *table_name = NULL;
  unsigned int table_len = 0;
  char *column_names = NULL;
  char *filter = NULL;
  char *normalizer_name = (char *)"NormalizerAuto";
  unsigned int normalizer_len = 14;
  char *input_filter = NULL;
  char *mecab_option = (char *)"-Owakati";
  grn_bool is_sentence_vectors = GRN_FALSE;
  char train_file[max_size];

  var = grn_plugin_proc_get_var(ctx, user_data, "table", -1);
  if (GRN_TEXT_LEN(var) != 0) {
    table_name = GRN_TEXT_VALUE(var);
    table_len = GRN_TEXT_LEN(var);
  }

  var = grn_plugin_proc_get_var(ctx, user_data, "column", -1);
  if (GRN_TEXT_LEN(var) != 0) {
    column_names = GRN_TEXT_VALUE(var);
    column_names[GRN_TEXT_LEN(var)] = '\0';
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "filter", -1);
  if (GRN_TEXT_LEN(var) != 0) {
    filter = GRN_TEXT_VALUE(var);
    filter[GRN_TEXT_LEN(var)] = '\0';
  }

  var = grn_plugin_proc_get_var(ctx, user_data, "normalizer", -1);
  if (GRN_TEXT_LEN(var) != 0) {
    if (GRN_TEXT_LEN(var) == 4 && memcmp(GRN_TEXT_VALUE(var), "NONE", 4) == 0) {
      normalizer_len = 0;
    } else {
      normalizer_name = GRN_TEXT_VALUE(var);
      normalizer_len = GRN_TEXT_LEN(var);
    }
  }

  var = grn_plugin_proc_get_var(ctx, user_data, "input_filter", -1);
  if (GRN_TEXT_LEN(var) != 0) {
    input_filter = GRN_TEXT_VALUE(var);
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "mecab_option", -1);
  if (GRN_TEXT_LEN(var) != 0) {
    if (GRN_TEXT_LEN(var) == 4 && memcmp(GRN_TEXT_VALUE(var), "NONE", 4) == 0) {
      mecab_option = NULL;
    } else {
      mecab_option = GRN_TEXT_VALUE(var);
    }
  }

  train_file[0] = 0;

  var = grn_plugin_proc_get_var(ctx, user_data, "train_file", -1);
  if (GRN_TEXT_LEN(var) != 0) {
    strcpy(train_file, GRN_TEXT_VALUE(var));
    train_file[GRN_TEXT_LEN(var)] = '\0';
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "sentence_vectors", -1);
  if (GRN_TEXT_LEN(var) != 0) {
    is_sentence_vectors = atoi(GRN_TEXT_VALUE(var));
  }

  if (train_file[0] == 0) {
    get_train_file_path(ctx, train_file);
  }

  if (table_name != NULL && column_names != NULL) {
    train_option option;
    option.input_filter = input_filter;
    option.mecab_option = mecab_option;
    option.normalizer_name = normalizer_name;
    option.normalizer_len = normalizer_len;
    option.is_sentence_vectors = is_sentence_vectors;

    if(column_to_train_file(ctx, train_file,
                            table_name, table_len,
                            column_names, filter,
                            option) == GRN_TRUE)
     {
        GRN_PLUGIN_LOG(ctx, GRN_LOG_NOTICE,
                       "[dump_to_train_file] Dump column to train file %s", train_file);
     } else {
        GRN_PLUGIN_LOG(ctx, GRN_LOG_ERROR,
                       "[dump_to_train_file] Dump column to train file %s failed.", train_file);
        return NULL;
     }
    grn_ctx_output_bool(ctx, GRN_TRUE);
  } else {
    GRN_PLUGIN_LOG(ctx, GRN_LOG_ERROR,
                   "[dump_to_train_file] Missing table name or column name.");
    grn_ctx_output_bool(ctx, GRN_FALSE);
  }

  return NULL;
}

static grn_obj *
command_word2vec_train(grn_ctx *ctx, GNUC_UNUSED int nargs, GNUC_UNUSED grn_obj **args,
                       grn_user_data *user_data)
{
  grn_obj *var;
  char train_file[max_size], output_file[max_size];
  grn_bool is_output_file = GRN_FALSE;
  grn_obj cmd;
  GRN_TEXT_INIT(&cmd, 0);
  GRN_BULK_REWIND(&cmd);
  GRN_TEXT_PUTS(ctx, &cmd, "word2vec");

  train_file[0] = 0;
  output_file[0] = 0;

  var = grn_plugin_proc_get_var(ctx, user_data, "size", -1);
  if (GRN_TEXT_LEN(var) != 0) {
    GRN_TEXT_PUTS(ctx, &cmd, " -size ");
    GRN_TEXT_PUTS(ctx, &cmd, GRN_TEXT_VALUE(var));
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "train_file", -1);
  if (GRN_TEXT_LEN(var) != 0) {
    GRN_TEXT_PUTS(ctx, &cmd, " -train ");
    GRN_TEXT_PUTS(ctx, &cmd, GRN_TEXT_VALUE(var));
  } else {
    get_train_file_path(ctx, train_file);
    GRN_TEXT_PUTS(ctx, &cmd, " -train ");
    GRN_TEXT_PUTS(ctx, &cmd, train_file);
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "save_vocab_file", -1);
  if (GRN_TEXT_LEN(var) != 0) {
    GRN_TEXT_PUTS(ctx, &cmd, " -save-vocab ");
    GRN_TEXT_PUTS(ctx, &cmd, GRN_TEXT_VALUE(var));
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "read_vocab_file", -1);
  if (GRN_TEXT_LEN(var) != 0) {
    GRN_TEXT_PUTS(ctx, &cmd, " -read-vocab ");
    GRN_TEXT_PUTS(ctx, &cmd, GRN_TEXT_VALUE(var));
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "debug", -1);
  if (GRN_TEXT_LEN(var) != 0) {
    GRN_TEXT_PUTS(ctx, &cmd, " -debug ");
    GRN_TEXT_PUTS(ctx, &cmd, GRN_TEXT_VALUE(var));
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "binary", -1);
  if (GRN_TEXT_LEN(var) != 0) {
    GRN_TEXT_PUTS(ctx, &cmd, " -binary ");
    GRN_TEXT_PUTS(ctx, &cmd, GRN_TEXT_VALUE(var));
  } else {
    GRN_TEXT_PUTS(ctx, &cmd, " -binary 1");
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "cbow", -1);
  if (GRN_TEXT_LEN(var) != 0) {
    GRN_TEXT_PUTS(ctx, &cmd, " -cbow ");
    GRN_TEXT_PUTS(ctx, &cmd, GRN_TEXT_VALUE(var));
  } else {
    GRN_TEXT_PUTS(ctx, &cmd, " -cbow 0");
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "alpha", -1);
  if (GRN_TEXT_LEN(var) != 0) {
    GRN_TEXT_PUTS(ctx, &cmd, " -alpha ");
    GRN_TEXT_PUTS(ctx, &cmd, GRN_TEXT_VALUE(var));
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "output_file", -1);
  if (GRN_TEXT_LEN(var) != 0) {
    GRN_TEXT_PUTS(ctx, &cmd, " -output ");
    GRN_TEXT_PUTS(ctx, &cmd, GRN_TEXT_VALUE(var));
  } else {
    get_model_file_path(ctx, output_file);
    GRN_TEXT_PUTS(ctx, &cmd, " -output ");
    GRN_TEXT_PUTS(ctx, &cmd, output_file);
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "window", -1);
  if (GRN_TEXT_LEN(var) != 0) {
    GRN_TEXT_PUTS(ctx, &cmd, " -window ");
    GRN_TEXT_PUTS(ctx, &cmd, GRN_TEXT_VALUE(var));
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "sample", -1);
  if (GRN_TEXT_LEN(var) != 0) {
    GRN_TEXT_PUTS(ctx, &cmd, " -sample ");
    GRN_TEXT_PUTS(ctx, &cmd, GRN_TEXT_VALUE(var));
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "hs", -1);
  if (GRN_TEXT_LEN(var) != 0) {
    GRN_TEXT_PUTS(ctx, &cmd, " -hs ");
    GRN_TEXT_PUTS(ctx, &cmd, GRN_TEXT_VALUE(var));
  } else {
    GRN_TEXT_PUTS(ctx, &cmd, " -hs 1");
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "negative", -1);
  if (GRN_TEXT_LEN(var) != 0) {
    GRN_TEXT_PUTS(ctx, &cmd, " -negative ");
    GRN_TEXT_PUTS(ctx, &cmd, GRN_TEXT_VALUE(var));
  } else {
    GRN_TEXT_PUTS(ctx, &cmd, " -negative 0");
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "threads", -1);
  if (GRN_TEXT_LEN(var) != 0) {
    GRN_TEXT_PUTS(ctx, &cmd, " -threads ");
    GRN_TEXT_PUTS(ctx, &cmd, GRN_TEXT_VALUE(var));
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "iter", -1);
  if (GRN_TEXT_LEN(var) != 0) {
    GRN_TEXT_PUTS(ctx, &cmd, " -iter ");
    GRN_TEXT_PUTS(ctx, &cmd, GRN_TEXT_VALUE(var));
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "classes", -1);
  if (GRN_TEXT_LEN(var) != 0) {
    GRN_TEXT_PUTS(ctx, &cmd, " -classes ");
    GRN_TEXT_PUTS(ctx, &cmd, GRN_TEXT_VALUE(var));
  }
  var = grn_plugin_proc_get_var(ctx, user_data, "sentence_vectors", -1);
  if (GRN_TEXT_LEN(var) != 0) {
    GRN_TEXT_PUTS(ctx, &cmd, " -sentence-vectors ");
    GRN_TEXT_PUTS(ctx, &cmd, GRN_TEXT_VALUE(var));
    GRN_TEXT_PUTS(ctx, &cmd, " -min-count 1");
  } else {
    var = grn_plugin_proc_get_var(ctx, user_data, "min_count", -1);
    if (GRN_TEXT_LEN(var) != 0) {
      GRN_TEXT_PUTS(ctx, &cmd, " -min-count ");
      GRN_TEXT_PUTS(ctx, &cmd, GRN_TEXT_VALUE(var));
    }
  }

  {
    char buff[1024];
    GRN_LOG(ctx, GRN_LOG_NOTICE, "[word2vec_train] %.*s", GRN_TEXT_LEN(&cmd), GRN_TEXT_VALUE(&cmd));
    FILE *fp = popen(GRN_TEXT_VALUE(&cmd), "r");
    while (fgets(buff, sizeof(buff), fp)) {
      GRN_LOG(ctx, GRN_LOG_NOTICE, "[word2vec_train] %s", right_trim(buff, '\n'));
    }
    pclose(fp);
  }
  grn_obj_unlink(ctx, &cmd);
  grn_ctx_output_bool(ctx, GRN_TRUE);

  return NULL;
}

static grn_obj *
func_query_expander_word2vec(grn_ctx *ctx, int nargs, grn_obj **args,
                             grn_user_data *user_data)
{
  grn_rc rc = GRN_END_OF_DATA;
  grn_id id;
  grn_obj *term, *expanded_term;
  void *value;
  grn_obj *rc_object;
  grn_obj *var;
  const char *env;

  term = args[0];
  expanded_term = args[1];

  rc = GRN_SUCCESS;

  grn_obj buf;
  GRN_TEXT_INIT(&buf, 0);
  GRN_BULK_REWIND(&buf);
  GRN_TEXT_PUTS(ctx, &buf, "word2vec_distance ");
  GRN_TEXT_PUTS(ctx, &buf, GRN_TEXT_VALUE(term));
  GRN_TEXT_PUTS(ctx, &buf, " --expander_mode 1");
  GRN_TEXT_PUTS(ctx, &buf, " --normalizer \"NONE\"");
  GRN_TEXT_PUTS(ctx, &buf, " --limit ");
  env = getenv("GRN_WORD2VEC_EXPANDER_LIMIT");
  if (env) {
    GRN_TEXT_PUTS(ctx, &buf, env);
  } else {
    GRN_TEXT_PUTS(ctx, &buf, "3");
  }
  GRN_TEXT_PUTS(ctx, &buf, " --threshold ");
  env = getenv("GRN_WORD2VEC_EXPANDER_THRESHOLD");
  if (env) {
    GRN_TEXT_PUTS(ctx, &buf, env);
  } else {
    GRN_TEXT_PUTS(ctx, &buf, "0.75");
  }
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
  grn_expr_var vars[21];

  grn_plugin_expr_var_init(ctx, &vars[0], "file_path", -1);
  grn_plugin_expr_var_init(ctx, &vars[1], "binary", -1);
  grn_plugin_command_create(ctx, "word2vec_load", -1, command_word2vec_load, 2, vars);
  grn_plugin_command_create(ctx, "word2vec_unload", -1, command_word2vec_unload, 0, vars);

  grn_plugin_expr_var_init(ctx, &vars[0], "term", -1);
  grn_plugin_expr_var_init(ctx, &vars[1], "offset", -1);
  grn_plugin_expr_var_init(ctx, &vars[2], "limit", -1);
  grn_plugin_expr_var_init(ctx, &vars[3], "n_sort", -1);
  grn_plugin_expr_var_init(ctx, &vars[4], "threshold", -1);
  grn_plugin_expr_var_init(ctx, &vars[5], "normalizer", -1);
  grn_plugin_expr_var_init(ctx, &vars[6], "prefix_filter", -1);
  grn_plugin_expr_var_init(ctx, &vars[7], "stop_filter", -1);
  grn_plugin_expr_var_init(ctx, &vars[8], "output_filter", -1);
  grn_plugin_expr_var_init(ctx, &vars[9], "mecab_option", -1);
  grn_plugin_expr_var_init(ctx, &vars[10], "file_path", -1);
  grn_plugin_expr_var_init(ctx, &vars[11], "binary", -1);
  grn_plugin_expr_var_init(ctx, &vars[12], "expander_mode", -1);
  grn_plugin_expr_var_init(ctx, &vars[13], "is_phrase", -1);
  grn_plugin_expr_var_init(ctx, &vars[14], "edit_distance", -1);
  grn_plugin_expr_var_init(ctx, &vars[15], "pca", -1);
  grn_plugin_expr_var_init(ctx, &vars[16], "pca_centered", -1);
  grn_plugin_expr_var_init(ctx, &vars[17], "sentence_vectors", -1);
  grn_plugin_expr_var_init(ctx, &vars[18], "table", -1);
  grn_plugin_expr_var_init(ctx, &vars[19], "column", -1);
  grn_plugin_expr_var_init(ctx, &vars[20], "sortby", -1);
  grn_plugin_command_create(ctx, "word2vec_distance", -1, command_word2vec_distance, 21, vars);

  grn_plugin_expr_var_init(ctx, &vars[0], "table", -1);
  grn_plugin_expr_var_init(ctx, &vars[1], "column", -1);
  grn_plugin_expr_var_init(ctx, &vars[2], "filter", -1);
  grn_plugin_expr_var_init(ctx, &vars[3], "train_file", -1);
  grn_plugin_expr_var_init(ctx, &vars[4], "normalizer", -1);
  grn_plugin_expr_var_init(ctx, &vars[5], "input_filter", -1);
  grn_plugin_expr_var_init(ctx, &vars[6], "mecab_option", -1);
  grn_plugin_expr_var_init(ctx, &vars[7], "sentence_vectors", -1);
  grn_plugin_command_create(ctx, "dump_to_train_file", -1, command_dump_to_train_file, 8, vars);

  grn_plugin_expr_var_init(ctx, &vars[0], "train_file", -1);
  grn_plugin_expr_var_init(ctx, &vars[1], "output_file", -1);
  grn_plugin_expr_var_init(ctx, &vars[2], "save_vocab_file", -1);
  grn_plugin_expr_var_init(ctx, &vars[3], "read_vocab_file", -1);
  grn_plugin_expr_var_init(ctx, &vars[4], "threads", -1);
  grn_plugin_expr_var_init(ctx, &vars[5], "size", -1);
  grn_plugin_expr_var_init(ctx, &vars[6], "debug", -1);
  grn_plugin_expr_var_init(ctx, &vars[7], "binary", -1);
  grn_plugin_expr_var_init(ctx, &vars[8], "cbow", -1);
  grn_plugin_expr_var_init(ctx, &vars[9], "alpha", -1);
  grn_plugin_expr_var_init(ctx, &vars[10], "window", -1);
  grn_plugin_expr_var_init(ctx, &vars[11], "sample", -1);
  grn_plugin_expr_var_init(ctx, &vars[12], "hs", -1);
  grn_plugin_expr_var_init(ctx, &vars[13], "negative", -1);
  grn_plugin_expr_var_init(ctx, &vars[14], "iter", -1);
  grn_plugin_expr_var_init(ctx, &vars[15], "min_count", -1);
  grn_plugin_expr_var_init(ctx, &vars[16], "classes", -1);
  grn_plugin_expr_var_init(ctx, &vars[17], "is_output_file", -1);
  grn_plugin_expr_var_init(ctx, &vars[18], "sentence_vectors", -1);
  grn_plugin_command_create(ctx, "word2vec_train", -1, command_word2vec_train, 19, vars);

  grn_proc_create(ctx, "QueryExpanderWord2vec", strlen("QueryExpanderWord2vec"),
                  GRN_PROC_FUNCTION,
                  func_query_expander_word2vec, NULL, NULL,
                  0, NULL);

  return ctx->rc;
}

grn_rc
GRN_PLUGIN_FIN(GNUC_UNUSED grn_ctx *ctx)
{
  for (int i = 0; i < MAX_MODEL; i++) {
    word2vec_unload(ctx, i);
  }
  if (model_idxes) {
    grn_hash_close(ctx, model_idxes);
    model_idxes = NULL;
  }
  mecab_fin(ctx);

  return GRN_SUCCESS;
}
