//  Tool to extract unigram counts
//
//  GloVe: Global Vectors for Word Representation
//  Copyright (c) 2014 The Board of Trustees of
//  The Leland Stanford Junior University. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
//
//  For more information, bug reports, fixes, contact:
//    Jeffrey Pennington (jpennin@stanford.edu)
//    GlobalVectors@googlegroups.com
//    http://nlp.stanford.edu/projects/glove/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <groonga/plugin.h>

#ifdef __GNUC__
# define GNUC_UNUSED __attribute__((__unused__))
#else
# define GNUC_UNUSED
#endif

#define MAX_STRING_LENGTH 1000
#define TSIZE	1048576
#define SEED	1159241
#define HASHFN  bitwisehash

typedef struct vocabulary {
    char *word;
    long long count;
} VOCAB;

typedef struct hashrec {
    char *word;
    long long count;
    struct hashrec *next;
} HASHREC;

int verbose = 2; // 0, 1, or 2
long long min_count = 1; // min occurrences for inclusion in vocab
long long max_vocab = 0; // max_vocab = 0 for no limit
char input_file[MAX_STRING_LENGTH];
char output_file[MAX_STRING_LENGTH];

/* Efficient string comparison */
static int scmp( char *s1, char *s2 ) {
    while(*s1 != '\0' && *s1 == *s2) {s1++; s2++;}
    return(*s1 - *s2);
}


/* Vocab frequency comparison; break ties alphabetically */
static int CompareVocabTie(const void *a, const void *b) {
    long long c;
    if( (c = ((VOCAB *) b)->count - ((VOCAB *) a)->count) != 0) return ( c > 0 ? 1 : -1 );
    else return (scmp(((VOCAB *) a)->word,((VOCAB *) b)->word));
    
}

/* Vocab frequency comparison; no tie-breaker */
static int CompareVocab(const void *a, const void *b) {
    long long c;
    if( (c = ((VOCAB *) b)->count - ((VOCAB *) a)->count) != 0) return ( c > 0 ? 1 : -1 );
    else return 0;
}

/* Move-to-front hashing and hash function from Hugh Williams, http://www.seg.rmit.edu.au/code/zwh-ipl/ */

/* Simple bitwise hash function */
static unsigned int bitwisehash(char *word, int tsize, unsigned int seed) {
    char c;
    unsigned int h;
    h = seed;
    for(; (c =* word) != '\0'; word++) h ^= ((h << 5) + c + (h >> 2));
    return((unsigned int)((h&0x7fffffff) % tsize));
}

/* Create hash table, initialise pointers to NULL */
static HASHREC ** inithashtable() {
    int	i;
    HASHREC **ht;
    ht = (HASHREC **) malloc( sizeof(HASHREC *) * TSIZE );
    for(i = 0; i < TSIZE; i++) ht[i] = (HASHREC *) NULL;
    return(ht);
}

/* Search hash table for given string, insert if not found */
static void hashinsert(HASHREC **ht, char *w) {
    HASHREC	*htmp, *hprv;
    unsigned int hval = HASHFN(w, TSIZE, SEED);
    
    for(hprv = NULL, htmp = ht[hval]; htmp != NULL && scmp(htmp->word, w) != 0; hprv = htmp, htmp = htmp->next);
    if(htmp == NULL) {
        htmp = (HASHREC *) malloc( sizeof(HASHREC) );
        htmp->word = (char *) malloc( strlen(w) + 1 );
        strcpy(htmp->word, w);
        htmp->count = 1;
        htmp->next = NULL;
        if( hprv==NULL )
            ht[hval] = htmp;
        else
            hprv->next = htmp;
    }
    else {
        /* new records are not moved to front */
        htmp->count++;
        if(hprv != NULL) {
            /* move to front on access */
            hprv->next = htmp->next;
            htmp->next = ht[hval];
            ht[hval] = htmp;
        }
    }
    return;
}

static int get_counts() {
    long long i = 0, j = 0, vocab_size = 12500;
    char format[20];
    char str[MAX_STRING_LENGTH + 1];
    HASHREC **vocab_hash = inithashtable();
    HASHREC *htmp;
    VOCAB *vocab;
    FILE *fid, *fout;
    fid = fopen(input_file, "r");
    fout = fopen(output_file, "w");
    
    fprintf(stderr, "BUILDING VOCABULARY\n");
    if(verbose > 1) fprintf(stderr, "Processed %lld tokens.", i);
    sprintf(format,"%%%ds",MAX_STRING_LENGTH);
    while(fscanf(fid, format, str) != EOF) { // Insert all tokens into hashtable
        hashinsert(vocab_hash, str);
        if(((++i)%100000) == 0) if(verbose > 1) fprintf(stderr,"\033[11G%lld tokens.", i);
    }
    if(verbose > 1) fprintf(stderr, "\033[0GProcessed %lld tokens.\n", i);
    vocab = malloc(sizeof(VOCAB) * vocab_size);
    for(i = 0; i < TSIZE; i++) { // Migrate vocab to array
        htmp = vocab_hash[i];
        while (htmp != NULL) {
            vocab[j].word = htmp->word;
            vocab[j].count = htmp->count;
            j++;
            if(j>=vocab_size) {
                vocab_size += 2500;
                vocab = (VOCAB *)realloc(vocab, sizeof(VOCAB) * vocab_size);
            }
            htmp = htmp->next;
        }
    }
    if(verbose > 1) fprintf(stderr, "Counted %lld unique words.\n", j);
    if(max_vocab > 0 && max_vocab < j)
        // If the vocabulary exceeds limit, first sort full vocab by frequency without alphabetical tie-breaks.
        // This results in pseudo-random ordering for words with same frequency, so that when truncated, the words span whole alphabet
        qsort(vocab, j, sizeof(VOCAB), CompareVocab);
    else max_vocab = j;
    qsort(vocab, max_vocab, sizeof(VOCAB), CompareVocabTie); //After (possibly) truncating, sort (possibly again), breaking ties alphabetically
    
    for(i = 0; i < max_vocab; i++) {
        if(vocab[i].count < min_count) { // If a minimum frequency cutoff exists, truncate vocabulary
            if(verbose > 0) fprintf(stderr, "Truncating vocabulary at min count %lld.\n",min_count);
            break;
        }
        fprintf(fout, "%s %lld\n",vocab[i].word,vocab[i].count);
    }
    
    if(i == max_vocab && max_vocab < j) if(verbose > 0) fprintf(stderr, "Truncating vocabulary at size %lld.\n", max_vocab);
    fprintf(stderr, "Using vocabulary of size %lld.\n\n", i);
    fclose(fid);
    fclose(fout);
    return 0;
}

static GNUC_UNUSED int find_arg(char *str, int argc, char **argv) {
    int i;
    for (i = 1; i < argc; i++) {
        if(!scmp(str, argv[i])) {
            if (i == argc - 1) {
                printf("No argument given for %s\n", str);
                exit(1);
            }
            return i;
        }
    }
    return -1;
}

static void
get_input_file_path(grn_ctx *ctx, char *file_name)
{
   grn_obj *db;
   db = grn_ctx_db(ctx);
   const char *path;
   path = grn_obj_path(ctx, db);
   strcpy(file_name, path);
   strcat(file_name, "_w2v.txt");
}

static void
get_output_file_path(grn_ctx *ctx, char *file_name)
{
   grn_obj *db;
   db = grn_ctx_db(ctx);
   const char *path;
   path = grn_obj_path(ctx, db);
   strcpy(file_name, path);
   strcat(file_name, "_w2v.vocab");
}

static grn_obj *
command_glove_vocab_count(grn_ctx *ctx, GNUC_UNUSED int nargs, GNUC_UNUSED grn_obj **args,
                          grn_user_data *user_data)
{
    grn_obj *var;

    var = grn_plugin_proc_get_var(ctx, user_data, "input_file", -1);
    if (GRN_TEXT_LEN(var) != 0) {
        strcpy(input_file, GRN_TEXT_VALUE(var));
        input_file[GRN_TEXT_LEN(var)] = '\0';
    } else {
      get_input_file_path(ctx, input_file);
    }
    var = grn_plugin_proc_get_var(ctx, user_data, "output_file", -1);
    if (GRN_TEXT_LEN(var) != 0) {
        strcpy(output_file, GRN_TEXT_VALUE(var));
        output_file[GRN_TEXT_LEN(var)] = '\0';
    }
    if (output_file[0] == 0) {
      get_output_file_path(ctx, output_file);
    }
    var = grn_plugin_proc_get_var(ctx, user_data, "verbose", -1);
    if (GRN_TEXT_LEN(var) != 0) verbose = atoi(GRN_TEXT_VALUE(var));
    var = grn_plugin_proc_get_var(ctx, user_data, "max_vocab", -1);
    if (GRN_TEXT_LEN(var) != 0) max_vocab = atoll(GRN_TEXT_VALUE(var));
    var = grn_plugin_proc_get_var(ctx, user_data, "min_count", -1);
    if (GRN_TEXT_LEN(var) != 0) min_count = atoll(GRN_TEXT_VALUE(var));

    grn_ctx_output_bool(ctx, !get_counts());
    return NULL;
}


grn_rc
GRN_PLUGIN_INIT(GNUC_UNUSED grn_ctx *ctx)
{
  return GRN_SUCCESS;
}

grn_rc
GRN_PLUGIN_REGISTER(grn_ctx *ctx)
{
  grn_expr_var vars[5];
  grn_plugin_expr_var_init(ctx, &vars[0], "input_file", -1);
  grn_plugin_expr_var_init(ctx, &vars[1], "output_file", -1);
  grn_plugin_expr_var_init(ctx, &vars[2], "verbose", -1);
  grn_plugin_expr_var_init(ctx, &vars[3], "max_vocab", -1);
  grn_plugin_expr_var_init(ctx, &vars[4], "min_count", -1);
  grn_plugin_command_create(ctx, "glove_vocab_count", -1, command_glove_vocab_count, 5, vars);
  return ctx->rc;
}

grn_rc
GRN_PLUGIN_FIN(GNUC_UNUSED grn_ctx *ctx)
{
  return GRN_SUCCESS;
}
