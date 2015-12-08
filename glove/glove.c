//  GloVe: Global Vectors for Word Representation
//
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
#include <math.h>
#include <pthread.h>

#include <groonga/plugin.h>

#ifdef __GNUC__
# define GNUC_UNUSED __attribute__((__unused__))
#else
# define GNUC_UNUSED
#endif

#define _FILE_OFFSET_BITS 64
#define MAX_STRING_LENGTH 1000

typedef double real;

typedef struct cooccur_rec {
    int word1;
    int word2;
    real val;
} CREC;

int verbose = 2; // 0, 1, or 2
int use_unk_vec = 1; // 0 or 1
int num_threads = 8; // pthreads
int num_iter = 25; // Number of full passes through cooccurrence matrix
int vector_size = 50; // Word vector size
int save_gradsq = 0; // By default don't save squared gradient values
int use_binary = 1; // 0: save as text files; 1: save as binary; 2: both. For binary, save both word and context word vectors.
int model = 2; // For text file output only. 0: concatenate word and context vectors (and biases) i.e. save everything; 1: Just save word vectors (no bias); 2: Save (word + context word) vectors (no biases)
real eta = 0.05; // Initial learning rate
real alpha = 0.75, x_max = 100.0; // Weighting function parameters, not extremely sensitive to corpus, though may need adjustment for very small or very large corpora
real *W, *gradsq, *cost;
long long num_lines, *lines_per_thread, vocab_size;
char *vocab_file, *input_file, *save_W_file, *save_gradsq_file;

/* Efficient string comparison */
static int scmp( char *s1, char *s2 ) {
    while(*s1 != '\0' && *s1 == *s2) {s1++; s2++;}
    return(*s1 - *s2);
}

static void initialize_parameters() {
	long long a, b;
	vector_size++; // Temporarily increment to allocate space for bias
    
	/* Allocate space for word vectors and context word vectors, and correspodning gradsq */
	a = posix_memalign((void **)&W, 128, 2 * vocab_size * (vector_size + 1) * sizeof(real)); // Might perform better than malloc
    if (W == NULL) {
        fprintf(stderr, "Error allocating memory for W\n");
        exit(1);
    }
    a = posix_memalign((void **)&gradsq, 128, 2 * vocab_size * (vector_size + 1) * sizeof(real)); // Might perform better than malloc
	if (gradsq == NULL) {
        fprintf(stderr, "Error allocating memory for gradsq\n");
        exit(1);
    }
	for (b = 0; b < vector_size; b++) for (a = 0; a < 2 * vocab_size; a++) W[a * vector_size + b] = (rand() / (real)RAND_MAX - 0.5) / vector_size;
	for (b = 0; b < vector_size; b++) for (a = 0; a < 2 * vocab_size; a++) gradsq[a * vector_size + b] = 1.0; // So initial value of eta is equal to initial learning rate
	vector_size--;
}

/* Train the GloVe model */
static void *glove_thread(void *vid) {
    long long a, b ,l1, l2;
    long long id = (long long) vid;
    CREC cr;
    real diff, fdiff, temp1, temp2;
    FILE *fin;
    fin = fopen(input_file, "rb");
    fseeko(fin, (num_lines / num_threads * id) * (sizeof(CREC)), SEEK_SET); //Threads spaced roughly equally throughout file
    cost[id] = 0;
    
    for(a = 0; a < lines_per_thread[id]; a++) {
        fread(&cr, sizeof(CREC), 1, fin);
        if(feof(fin)) break;
        
        /* Get location of words in W & gradsq */
        l1 = (cr.word1 - 1LL) * (vector_size + 1); // cr word indices start at 1
        l2 = ((cr.word2 - 1LL) + vocab_size) * (vector_size + 1); // shift by vocab_size to get separate vectors for context words
        
        /* Calculate cost, save diff for gradients */
        diff = 0;
        for(b = 0; b < vector_size; b++) diff += W[b + l1] * W[b + l2]; // dot product of word and context word vector
        diff += W[vector_size + l1] + W[vector_size + l2] - log(cr.val); // add separate bias for each word
        fdiff = (cr.val > x_max) ? diff : pow(cr.val / x_max, alpha) * diff; // multiply weighting function (f) with diff
        cost[id] += 0.5 * fdiff * diff; // weighted squared error
        
        /* Adaptive gradient updates */
        fdiff *= eta; // for ease in calculating gradient
        for(b = 0; b < vector_size; b++) {
            // learning rate times gradient for word vectors
            temp1 = fdiff * W[b + l2];
            temp2 = fdiff * W[b + l1];
            // adaptive updates
            W[b + l1] -= temp1 / sqrt(gradsq[b + l1]);
            W[b + l2] -= temp2 / sqrt(gradsq[b + l2]);
            gradsq[b + l1] += temp1 * temp1;
            gradsq[b + l2] += temp2 * temp2;
        }
        // updates for bias terms
        W[vector_size + l1] -= fdiff / sqrt(gradsq[vector_size + l1]);
        W[vector_size + l2] -= fdiff / sqrt(gradsq[vector_size + l2]);
        fdiff *= fdiff;
        gradsq[vector_size + l1] += fdiff;
        gradsq[vector_size + l2] += fdiff;
        
    }
    
    fclose(fin);
    pthread_exit(NULL);
}

/* Save params to file */
static int save_params() {
    long long a, b;
    char format[20];
    char output_file[MAX_STRING_LENGTH], output_file_gsq[MAX_STRING_LENGTH];
    char *word = malloc(sizeof(char) * MAX_STRING_LENGTH);
    FILE *fid, *fout, *fgs = NULL;
    
    if(use_binary > 0) { // Save parameters in binary file
        sprintf(output_file,"%s.bin",save_W_file);
        fout = fopen(output_file,"wb");
        if(fout == NULL) {fprintf(stderr, "Unable to open file %s.\n",save_W_file); return 1;}
        for(a = 0; a < 2 * (long long)vocab_size * (vector_size + 1); a++) fwrite(&W[a], sizeof(real), 1,fout);
        fclose(fout);
        if(save_gradsq > 0) {
            sprintf(output_file_gsq,"%s.bin",save_gradsq_file);
            fgs = fopen(output_file_gsq,"wb");
            if(fgs == NULL) {fprintf(stderr, "Unable to open file %s.\n",save_gradsq_file); return 1;}
            for(a = 0; a < 2 * (long long)vocab_size * (vector_size + 1); a++) fwrite(&gradsq[a], sizeof(real), 1,fgs);
            fclose(fgs);
        }
    }
    if(use_binary != 1) { // Save parameters in text file
        sprintf(output_file,"%s.txt",save_W_file);
        if(save_gradsq > 0) {
            sprintf(output_file_gsq,"%s.txt",save_gradsq_file);
            fgs = fopen(output_file_gsq,"wb");
            if(fgs == NULL) {fprintf(stderr, "Unable to open file %s.\n",save_gradsq_file); return 1;}
        }
        fout = fopen(output_file,"wb");
        if(fout == NULL) {fprintf(stderr, "Unable to open file %s.\n",save_W_file); return 1;}
        fid = fopen(vocab_file, "r");
        sprintf(format,"%%%ds",MAX_STRING_LENGTH);
        if(fid == NULL) {fprintf(stderr, "Unable to open file %s.\n",vocab_file); return 1;}
        for(a = 0; a < vocab_size; a++) {
            if(fscanf(fid,format,word) == 0) return 1;
            // input vocab cannot contain special <unk> keyword
            if(strcmp(word, "<unk>") == 0) return 1;
            fprintf(fout, "%s",word);
            if(model == 0) { // Save all parameters (including bias)
                for(b = 0; b < (vector_size + 1); b++) fprintf(fout," %lf", W[a * (vector_size + 1) + b]);
                for(b = 0; b < (vector_size + 1); b++) fprintf(fout," %lf", W[(vocab_size + a) * (vector_size + 1) + b]);
            }
            if(model == 1) // Save only "word" vectors (without bias)
                for(b = 0; b < vector_size; b++) fprintf(fout," %lf", W[a * (vector_size + 1) + b]);
            if(model == 2) // Save "word + context word" vectors (without bias)
                for(b = 0; b < vector_size; b++) fprintf(fout," %lf", W[a * (vector_size + 1) + b] + W[(vocab_size + a) * (vector_size + 1) + b]);
            fprintf(fout,"\n");
            if(save_gradsq > 0) { // Save gradsq
                fprintf(fgs, "%s",word);
                for(b = 0; b < (vector_size + 1); b++) fprintf(fgs," %lf", gradsq[a * (vector_size + 1) + b]);
                for(b = 0; b < (vector_size + 1); b++) fprintf(fgs," %lf", gradsq[(vocab_size + a) * (vector_size + 1) + b]);
                fprintf(fgs,"\n");
            }
            if(fscanf(fid,format,word) == 0) return 1; // Eat irrelevant frequency entry
        }

        if (use_unk_vec) {
            real* unk_vec = (real*)calloc((vector_size + 1), sizeof(real));
            real* unk_context = (real*)calloc((vector_size + 1), sizeof(real));
            word = "<unk>";

            int num_rare_words = vocab_size < 100 ? vocab_size : 100;

            for(a = vocab_size - num_rare_words; a < vocab_size; a++) {
                for(b = 0; b < (vector_size + 1); b++) {
                    unk_vec[b] += W[a * (vector_size + 1) + b] / num_rare_words;
                    unk_context[b] += W[(vocab_size + a) * (vector_size + 1) + b] / num_rare_words;
                }
            }

            fprintf(fout, "%s",word);
            if(model == 0) { // Save all parameters (including bias)
                for(b = 0; b < (vector_size + 1); b++) fprintf(fout," %lf", unk_vec[b]);
                for(b = 0; b < (vector_size + 1); b++) fprintf(fout," %lf", unk_context[b]);
            }
            if(model == 1) // Save only "word" vectors (without bias)
                for(b = 0; b < vector_size; b++) fprintf(fout," %lf", unk_vec[b]);
            if(model == 2) // Save "word + context word" vectors (without bias)
                for(b = 0; b < vector_size; b++) fprintf(fout," %lf", unk_vec[b] + unk_context[b]);
            fprintf(fout,"\n");

            free(unk_vec);
            free(unk_context);
        }

        fclose(fid);
        fclose(fout);
        if(save_gradsq > 0) fclose(fgs);
    }
    return 0;
}

/* Train model */
static int train_glove() {
    long long a, file_size;
    int b;
    FILE *fin;
    real total_cost = 0;
    fprintf(stderr, "TRAINING MODEL\n");
    
    fin = fopen(input_file, "rb");
    if(fin == NULL) {fprintf(stderr,"Unable to open cooccurrence file %s.\n",input_file); return 1;}
    fseeko(fin, 0, SEEK_END);
    file_size = ftello(fin);
    num_lines = file_size/(sizeof(CREC)); // Assuming the file isn't corrupt and consists only of CREC's
    fclose(fin);
    fprintf(stderr,"Read %lld lines.\n", num_lines);
    if(verbose > 1) fprintf(stderr,"Initializing parameters...");
    initialize_parameters();
    if(verbose > 1) fprintf(stderr,"done.\n");
    if(verbose > 0) fprintf(stderr,"vector size: %d\n", vector_size);
    if(verbose > 0) fprintf(stderr,"vocab size: %lld\n", vocab_size);
    if(verbose > 0) fprintf(stderr,"x_max: %lf\n", x_max);
    if(verbose > 0) fprintf(stderr,"alpha: %lf\n", alpha);
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    lines_per_thread = (long long *) malloc(num_threads * sizeof(long long));
    
    // Lock-free asynchronous SGD
    for(b = 0; b < num_iter; b++) {
        total_cost = 0;
        for (a = 0; a < num_threads - 1; a++) lines_per_thread[a] = num_lines / num_threads;
        lines_per_thread[a] = num_lines / num_threads + num_lines % num_threads;
        for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, glove_thread, (void *)a);
        for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
        for (a = 0; a < num_threads; a++) total_cost += cost[a];
        fprintf(stderr,"iter: %03d, cost: %lf\n", b+1, total_cost/num_lines);
    }
    return save_params();
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

static grn_obj *
command_glove_train(grn_ctx *ctx, GNUC_UNUSED int nargs, GNUC_UNUSED grn_obj **args,
                    grn_user_data *user_data)
{
    grn_obj *var;

    int i;
    FILE *fid;
    vocab_file = malloc(sizeof(char) * MAX_STRING_LENGTH);
    input_file = malloc(sizeof(char) * MAX_STRING_LENGTH);
    save_W_file = malloc(sizeof(char) * MAX_STRING_LENGTH);
    save_gradsq_file = malloc(sizeof(char) * MAX_STRING_LENGTH);
    
    var = grn_plugin_proc_get_var(ctx, user_data, "verbose", -1);
    if (GRN_TEXT_LEN(var) != 0) verbose = atoi(GRN_TEXT_VALUE(var));
    var = grn_plugin_proc_get_var(ctx, user_data, "vector_size", -1);
    if (GRN_TEXT_LEN(var) != 0) vector_size = atoi(GRN_TEXT_VALUE(var));
    var = grn_plugin_proc_get_var(ctx, user_data, "iter", -1);
    if (GRN_TEXT_LEN(var) != 0) num_iter = atoi(GRN_TEXT_VALUE(var));
    var = grn_plugin_proc_get_var(ctx, user_data, "threads", -1);
    if (GRN_TEXT_LEN(var) != 0) num_threads = atoi(GRN_TEXT_VALUE(var));

    cost = malloc(sizeof(real) * num_threads);
    var = grn_plugin_proc_get_var(ctx, user_data, "alpha", -1);
    if (GRN_TEXT_LEN(var) != 0) alpha = atof(GRN_TEXT_VALUE(var));
    var = grn_plugin_proc_get_var(ctx, user_data, "x_max", -1);
    if (GRN_TEXT_LEN(var) != 0) x_max = atof(GRN_TEXT_VALUE(var));
    var = grn_plugin_proc_get_var(ctx, user_data, "eta", -1);
    if (GRN_TEXT_LEN(var) != 0) eta = atof(GRN_TEXT_VALUE(var));
    var = grn_plugin_proc_get_var(ctx, user_data, "binary", -1);
    if (GRN_TEXT_LEN(var) != 0) use_binary = atoi(GRN_TEXT_VALUE(var));
    var = grn_plugin_proc_get_var(ctx, user_data, "model", -1);
    if (GRN_TEXT_LEN(var) != 0) model = atoi(GRN_TEXT_VALUE(var));
    if(model != 0 && model != 1) model = 2;
    var = grn_plugin_proc_get_var(ctx, user_data, "save_gradsq", -1);
    if (GRN_TEXT_LEN(var) != 0) save_gradsq = atoi(GRN_TEXT_VALUE(var));
    var = grn_plugin_proc_get_var(ctx, user_data, "vocab_file", -1);
    if (GRN_TEXT_LEN(var) != 0) {
        strcpy(vocab_file, GRN_TEXT_VALUE(var));
        vocab_file[GRN_TEXT_LEN(var)] = '\0';
    } else {
        strcpy(vocab_file, (char *)"vocab.txt");
    }
    var = grn_plugin_proc_get_var(ctx, user_data, "save_file", -1);
    if (GRN_TEXT_LEN(var) != 0) {
        strcpy(save_W_file, GRN_TEXT_VALUE(var));
        save_W_file[GRN_TEXT_LEN(var)] = '\0';
    } else {
        strcpy(save_W_file, (char *)"vectors");
    }
    var = grn_plugin_proc_get_var(ctx, user_data, "gradsq_file", -1);
    if (GRN_TEXT_LEN(var) != 0) {
        strcpy(save_gradsq_file, GRN_TEXT_VALUE(var));
        save_gradsq_file[GRN_TEXT_LEN(var)] = '\0';
        save_gradsq = 1;
    } else if (save_gradsq > 0) {
        strcpy(save_gradsq_file, (char *)"gradsq");
    }
    var = grn_plugin_proc_get_var(ctx, user_data, "input_file", -1);
    if (GRN_TEXT_LEN(var) != 0) {
        strcpy(input_file, GRN_TEXT_VALUE(var));
        input_file[GRN_TEXT_LEN(var)] = '\0';
    } else {
        strcpy(input_file, (char *)"cooccurrence.shuf.bin");
    }
    vocab_size = 0;
    fid = fopen(vocab_file, "r");
    if(fid == NULL) {fprintf(stderr, "Unable to open vocab file %s.\n",vocab_file); return NULL;}
    while ((i = getc(fid)) != EOF) if (i == '\n') vocab_size++; // Count number of entries in vocab_file
    fclose(fid);
    grn_ctx_output_bool(ctx, train_glove());
    
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
  grn_expr_var vars[14];
  grn_plugin_expr_var_init(ctx, &vars[0], "verbose", -1);
  grn_plugin_expr_var_init(ctx, &vars[1], "vector_size", -1);
  grn_plugin_expr_var_init(ctx, &vars[2], "iter", -1);
  grn_plugin_expr_var_init(ctx, &vars[3], "threads", -1);
  grn_plugin_expr_var_init(ctx, &vars[4], "alpha", -1);
  grn_plugin_expr_var_init(ctx, &vars[5], "x_max", -1);
  grn_plugin_expr_var_init(ctx, &vars[6], "eta", -1);
  grn_plugin_expr_var_init(ctx, &vars[7], "binary", -1);
  grn_plugin_expr_var_init(ctx, &vars[8], "model", -1);
  grn_plugin_expr_var_init(ctx, &vars[9], "save_gradsq", -1);
  grn_plugin_expr_var_init(ctx, &vars[10], "vocab_file", -1);
  grn_plugin_expr_var_init(ctx, &vars[11], "save_file", -1);
  grn_plugin_expr_var_init(ctx, &vars[12], "gradsq_file", -1);
  grn_plugin_expr_var_init(ctx, &vars[13], "input_file", -1);
  grn_plugin_command_create(ctx, "glove_train", -1, command_glove_train, 14, vars);
  return ctx->rc;
}

grn_rc
GRN_PLUGIN_FIN(GNUC_UNUSED grn_ctx *ctx)
{
  return GRN_SUCCESS;
}
