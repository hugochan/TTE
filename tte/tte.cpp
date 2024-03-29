/*
 This is the tool ....
 Created on Sep, 2016

 @author: hugo
 */

// Format of the training file:
//
// The training file contains serveral lines, each line represents a DIRECTED edge in the network.
// More specifically, each line has the following format "<u> <v> <w>", meaning an edge from <u> to <v> with weight as <w>.
// <u> <v> and <w> are seperated by ' ' or '\t' (blank or tab)
// For UNDIRECTED edge, the user should use two DIRECTED edges to represent it.

#include <iostream>
#include <vector>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <random>
#include <pthread.h>
#include <gsl/gsl_rng.h>

using namespace std;

#define MAX_STRING 200
#define SIGMOID_BOUND 12
#define NEG_SAMPLING_POWER 0.75
#define WW_TYPE 0 // word-word
#define WD_TYPE 1 // word-doc
#define DT_TYPE 2 // doc-topic
#define TW_TYPE 3 // topic-word
#define WORD_TYPE 0 // word
#define DOC_TYPE 1 // doc
#define TOPIC_TYPE 2 // topic
#define min(a,b) (a<b?a:b)

const int hash_table_size = 30000000;
const int neg_table_size = 1e8;
const int sigmoid_table_size = 100000;

typedef double real;   // Precision of float numbers

struct ClassVertex {
    double degree[2]; // if word vertex, word-word network out-degree and word-doc network out-degree
    char *name;
};

char wwnet_file[MAX_STRING], wdnet_file[MAX_STRING], word_embedding_file[MAX_STRING], doc_embedding_file[MAX_STRING], topic_embedding_file[MAX_STRING], doc_topic_dist_file[MAX_STRING], topic_word_dist_file[MAX_STRING],
    readable_doc_topic_dist_file[MAX_STRING], readable_topic_word_dist_file[MAX_STRING];
struct ClassVertex *word_vertex, *doc_vertex;
int is_binary = 0, n_topics = 0, num_threads = 1, dim = 100, num_negative = 5, n_top = 10, join_threads = 0;
int *word_hash_table, *doc_hash_table, *ww_neg_table, *wd_neg_table, *wt_neg_table, *td_neg_table;
int max_num_word_vertices = 1000, max_num_doc_vertices = 1000, num_word_vertices = 0, num_doc_vertices = 0, num_topic_vertices = 0;
long long total_samples = 1, current_sample_count = 0, num_ww_edges = 0, num_wd_edges = 0;
real init_rho = 0.025, rho, epsilon = 1e-6, update_negtable_ratio = 1e-5;
real *word_emb_vertex, *doc_emb_vertex, *topic_emb_vertex, *sigmoid_table;
real *word_emb_context, *doc_emb_context, *topic_emb_context;
real **doc_topic_dist, **topic_word_dist;
int *ww_edge_source_id, *ww_edge_target_id, *wd_edge_source_id, *wd_edge_target_id;
double *ww_edge_weight, *wd_edge_weight;

// Parameters for edge sampling
long long *ww_alias, *wd_alias;
double *ww_prob, *wd_prob;

const gsl_rng_type * gsl_T;
gsl_rng * gsl_r;

// Generate uniform random numbers
random_device                  rand_dev;
mt19937                        generator(rand_dev());


/* global mutex variable */
pthread_mutex_t my_mutex = PTHREAD_MUTEX_INITIALIZER;



/* Build a hash table, mapping each vertex name to a unique vertex id */
unsigned int Hash(char *key)
{
    unsigned int seed = 131;
    unsigned int hash = 0;
    while (*key)
    {
        hash = hash * seed + (*key++);
    }
    return hash % hash_table_size;
}

int *InitHashTable()
{
    int *vertex_hash_table = (int *)malloc(hash_table_size * sizeof(int));
    for (int k = 0; k != hash_table_size; k++) vertex_hash_table[k] = -1;
    return vertex_hash_table;
}

void InsertHashTable(int *vertex_hash_table, char *key, int value)
{
    int addr = Hash(key);
    while (vertex_hash_table[addr] != -1) addr = (addr + 1) % hash_table_size;
    vertex_hash_table[addr] = value;
}

int SearchHashTable(struct ClassVertex *vertex, int *vertex_hash_table, char *key)
{
    int addr = Hash(key);
    while (1)
    {
        if (vertex_hash_table[addr] == -1) return -1;
        if (!strcmp(key, vertex[vertex_hash_table[addr]].name)) return vertex_hash_table[addr];
        addr = (addr + 1) % hash_table_size;
    }
    return -1;
}

/* Fastly compute sigmoid function */
void InitSigmoidTable()
{
    real x;
    sigmoid_table = (real *)malloc((sigmoid_table_size + 1) * sizeof(real));
    for (int k = 0; k != sigmoid_table_size; k++)
    {
        x = 2.0 * SIGMOID_BOUND * k / sigmoid_table_size - SIGMOID_BOUND;
        sigmoid_table[k] = 1 / (1 + exp(-x));
    }
}

real FastSigmoid(real x)
{
    if (x > SIGMOID_BOUND) return 1;
    else if (x < -SIGMOID_BOUND) return 0;
    int k = (x + SIGMOID_BOUND) * sigmoid_table_size / SIGMOID_BOUND / 2;
    return sigmoid_table[k];
}

/* Compute dot product.*/
real dot(real *vec_u, real *vec_v, int size)
{
    real x = 0;
    for (int c = 0; c != size; c++) x += vec_u[c] * vec_v[c];
    return x;
}

/* Add a vertex to the vertex set */
struct ClassVertex *AddVertex(int *vertex_hash_table, struct ClassVertex *vertex, int *num_vertices, char *name, int *max_num_vertices)
{
    int length = (int)strlen(name) + 1;
    if (length > MAX_STRING) length = MAX_STRING;
    if (*num_vertices >= *max_num_vertices)
    {
        (*max_num_vertices) += 1000;
        struct ClassVertex *tmp = (struct ClassVertex *)realloc(vertex, *max_num_vertices * sizeof(struct ClassVertex));
        if (tmp == NULL)
        {
            printf("Error: memory reallocation failed!\n");
            exit(1);
        }
        vertex = tmp;

    }
    vertex[*num_vertices].degree[0] = 0; // ww
    vertex[*num_vertices].degree[1] = 0; // wd
    vertex[*num_vertices].name = (char *)calloc(length, sizeof(char));
    memcpy(vertex[*num_vertices].name, name, length - 1);
    vertex[*num_vertices].name[length - 1] = '\0';
    (*num_vertices)++;

    InsertHashTable(vertex_hash_table, name, *num_vertices - 1);
    return vertex;
}

/* Read network from the training file */
void ReadData(char *network_file, int type)
{
    int *source_num_vertices, *target_num_vertices, *source_max_num_vertices, *target_max_num_vertices;
    long long *num_edges = nullptr;
    struct ClassVertex *source_vertex, *target_vertex;
    int *source_hash_table, *target_hash_table;
    int *edge_source_id, *edge_target_id;
    double *edge_weight;

    if (type == WW_TYPE) // word-word network
    {
        source_num_vertices = &num_word_vertices;
        target_num_vertices = &num_word_vertices;
        source_max_num_vertices = &max_num_word_vertices;
        target_max_num_vertices = &max_num_word_vertices;
        num_edges = &num_ww_edges;
        source_vertex = word_vertex;
        target_vertex = word_vertex;
        source_hash_table = word_hash_table;
        target_hash_table = word_hash_table;
    }
    else if (type == WD_TYPE)// word-doc network
    {
        source_num_vertices = &num_word_vertices;
        target_num_vertices = &num_doc_vertices;
        source_max_num_vertices = &max_num_word_vertices;
        target_max_num_vertices = &max_num_doc_vertices;
        num_edges = &num_wd_edges;
        source_vertex = word_vertex;
        target_vertex = doc_vertex;
        source_hash_table = word_hash_table;
        target_hash_table = doc_hash_table;
    }
    else
    {
        printf("ERROR: unknown type %d", type);
        exit(1);
    }

    FILE *fin;
    char name_v1[MAX_STRING], name_v2[MAX_STRING], str[2 * MAX_STRING + 10000];
    int vid;
    double weight;

    fin = fopen(network_file, "rb");
    if (fin == NULL)
    {
        printf("ERROR: network file not found!\n");
        exit(1);
    }

    while (fgets(str, sizeof(str), fin)) (*num_edges)++;
    fclose(fin);

    edge_source_id = (int *)malloc(*num_edges*sizeof(int));
    edge_target_id = (int *)malloc(*num_edges*sizeof(int));
    edge_weight = (double *)malloc(*num_edges*sizeof(double));

    if (edge_source_id == NULL || edge_target_id == NULL || edge_weight == NULL)
    {
        printf("Error: memory allocation failed!\n");
        exit(1);
    }

    fin = fopen(network_file, "rb");

    for (int k = 0; k != *num_edges; k++)
    {
        fscanf(fin, "%s %s %lf", name_v1, name_v2, &weight);

        if (k % 10000 == 0)
        {
            printf("Reading edges: %.3lf%%%c", k / (double)(*num_edges + 1) * 100, 13);
            fflush(stdout);
        }

        // source vertex
        vid = SearchHashTable(source_vertex, source_hash_table, name_v1);

        if (vid == -1)
        {
            source_vertex = AddVertex(source_hash_table, source_vertex, source_num_vertices, name_v1, source_max_num_vertices);
            if (type == WW_TYPE) // word-word
                target_vertex = source_vertex;
            vid = *source_num_vertices - 1;
        }

        if (type == WW_TYPE) // word-word
            source_vertex[vid].degree[0] += weight;
        else if (type == WD_TYPE) // word-doc
            source_vertex[vid].degree[1] += weight;

        edge_source_id[k] = vid;

        // target vertex
        vid = SearchHashTable(target_vertex, target_hash_table, name_v2);
        if (vid == -1)
        {
            target_vertex = AddVertex(target_hash_table, target_vertex, target_num_vertices, name_v2, target_max_num_vertices);
            if (type == WW_TYPE) // word-word
                source_vertex = target_vertex;
            vid = *target_num_vertices - 1;
        }
        edge_target_id[k] = vid;

        edge_weight[k] = weight;
    }
    fclose(fin);

    if (type == WW_TYPE) // word-word network
    {
        word_vertex = source_vertex;

        ww_edge_source_id = edge_source_id;
        ww_edge_target_id = edge_target_id;
        ww_edge_weight = edge_weight;
    }
    else if (type == WD_TYPE) // word-doc network
    {
        word_vertex = source_vertex;
        doc_vertex = target_vertex;

        wd_edge_source_id = edge_source_id;
        wd_edge_target_id = edge_target_id;
        wd_edge_weight = edge_weight;
    }
}

/* The alias sampling algorithm, which is used to sample an edge in O(1) time. */
void InitAliasTable(long long num_edges, double *edge_weight, int type)
{
    long long *alias = (long long *)calloc(num_edges, sizeof(long long));
    double *prob = (double *)calloc(num_edges, sizeof(double));
    if (alias == NULL || prob == NULL)
    {
        printf("Error: memory allocation failed!\n");
        exit(1);
    }

    double *norm_prob = (double*)calloc(num_edges, sizeof(double));
    long long *large_block = (long long*)calloc(num_edges, sizeof(long long));
    long long *small_block = (long long*)calloc(num_edges, sizeof(long long));
    if (norm_prob == NULL || large_block == NULL || small_block == NULL)
    {
        printf("Error: memory allocation failed!\n");
        exit(1);
    }

    double sum = 0;
    long long cur_small_block, cur_large_block;
    long long num_small_block = 0, num_large_block = 0;

    for (long long k = 0; k != num_edges; k++) sum += edge_weight[k];
    for (long long k = 0; k != num_edges; k++) norm_prob[k] = edge_weight[k] * num_edges / sum;

    for (long long k = num_edges - 1; k >= 0; k--)
    {
        if (norm_prob[k]<1)
            small_block[num_small_block++] = k;
        else
            large_block[num_large_block++] = k;
    }

    while (num_small_block && num_large_block)
    {
        cur_small_block = small_block[--num_small_block];
        cur_large_block = large_block[--num_large_block];
        prob[cur_small_block] = norm_prob[cur_small_block];
        alias[cur_small_block] = cur_large_block;
        norm_prob[cur_large_block] = norm_prob[cur_large_block] + norm_prob[cur_small_block] - 1;
        if (norm_prob[cur_large_block] < 1)
            small_block[num_small_block++] = cur_large_block;
        else
            large_block[num_large_block++] = cur_large_block;
    }

    while (num_large_block) prob[large_block[--num_large_block]] = 1;
    while (num_small_block) prob[small_block[--num_small_block]] = 1;

    if (type == WW_TYPE) // word-word network
    {
        ww_alias = alias;
        ww_prob = prob;
    }
    else if (type == WD_TYPE)// word-doc network
    {
        wd_alias = alias;
        wd_prob = prob;
    }
    else
    {
        printf("ERROR: unknown type %d", type);
        free(norm_prob);
        free(small_block);
        free(large_block);
        exit(1);
    }
    free(norm_prob);
    free(small_block);
    free(large_block);
}

long long SampleAnEdge(double rand_value1, double rand_value2, int type)
{
    long long k = 0;

    if (type == WW_TYPE) // word-word network
    {
        k = num_ww_edges * rand_value1;
        return rand_value2 < ww_prob[k] ? k : ww_alias[k];

    }
    else if (type == WD_TYPE) // word-doc network
    {
        k = num_wd_edges * rand_value1;
        return rand_value2 < wd_prob[k] ? k : wd_alias[k];
    }
    else
    {
        printf("ERROR: unknown output type %d", type);
        exit(1);
    }
}

/* Initialize the vertex embedding and the context embedding */
real *InitVector(int num_vertices)
{
    long long a, b;
    real *emb_vertex = nullptr;
    // real *emb_context;

    posix_memalign((void **)&emb_vertex, 128, (long long)num_vertices * dim * sizeof(real));
    if (emb_vertex == NULL) { printf("Error: memory allocation failed\n"); exit(1); }

    for (b = 0; b < dim; b++) for (a = 0; a < num_vertices; a++)
        emb_vertex[a * dim + b] = (rand() / (real)RAND_MAX - 0.5) / dim;

    // a = posix_memalign((void **)&emb_context, 128, (long long)num_vertices * dim * sizeof(real));
    // if (emb_context == NULL) { printf("Error: memory allocation failed\n"); exit(1); }
    // for (b = 0; b < dim; b++) for (a = 0; a < num_vertices; a++)
    //  emb_context[a * dim + b] = 0;
    return emb_vertex;
}

real **InitCondDist(int source_num_vertices, int target_num_vertices)
{
    real **cond_dist = (real **)calloc(target_num_vertices, sizeof(real *));
    if (cond_dist == NULL)
    {
        printf("Error: memory allocation failed!\n");
        exit(1);
    }
    for (int i = 0; i != target_num_vertices; i++)
    {
        cond_dist[i] = (real *)calloc(source_num_vertices, sizeof(real));
        if (cond_dist[i] == NULL)
        {
            printf("Error: memory allocation failed!\n");
            exit(1);
        }
    }
    return cond_dist;
}

void FreeCondDist(real **cond_dist, int size)
{
    for (int i = 0; i != size; i++)
        free(cond_dist[i]);
    free(cond_dist);
}

void FreeVertex(struct ClassVertex *vertex, int num_vertices)
{
    for (int i = 0; i != num_vertices; i++)
        free(vertex[i].name);
    free(vertex);
}

/* Sample negative vertex samples according to vertex degrees */
void InitNegTable(int type)
{
    int num_vertices = 0;
    struct ClassVertex *vertex;
    if (type == WW_TYPE || type == WD_TYPE) // word-word or word-doc network
    {
        num_vertices = num_word_vertices;
        vertex = word_vertex;
    }
    else
    {
        printf("ERROR: unknown output type %d", type);
        exit(1);
    }

    double sum = 0, cur_sum = 0, por = 0;
    int vid = 0;
    int *neg_table = (int *)malloc(neg_table_size * sizeof(int));
    for (int k = 0; k != num_vertices; k++) sum += pow(vertex[k].degree[type], NEG_SAMPLING_POWER);
    for (int k = 0; k != neg_table_size; k++)
    {
        if ((double)(k + 1) / neg_table_size > por)
        {
            cur_sum += pow(vertex[vid].degree[type], NEG_SAMPLING_POWER);
            por = cur_sum / sum;
            vid++;
        }
        neg_table[k] = vid - 1;
    }

    if (type == WW_TYPE) // word-word network
        ww_neg_table = neg_table;
    else if (type == WD_TYPE) // word-doc network
        wd_neg_table = neg_table;
}


/* Sample negative vertex samples according to vertex "popularity".
 We cannot afford to compute "degrees" (i.e., sum of weights) for
 negative sampling in word-topic and topic-doc networks. Thus, we
 adopt a cheap way to achieve the same goal.
 */
void InitNegTable4Topic(int *neg_table, real *vec_u, real *vec_v, int num_vec_u, int num_vec_v)
{
    int i, j;
    // Compute mean of vec_v
    real x;
    real *mean_vec_v = (real *)calloc(dim, sizeof(real));
    for (i = 0; i < dim; i++)
    {
        x = 0;
        for (j = 0; j < num_vec_v; j++)
            x += vec_v[j * dim + i];
        mean_vec_v[i] = x / num_vec_v;
    }

    // Assign proportions
    double sum = 0, cur_sum = 0, por = 0;
    int vid = 0;
    double *weight_vec = (double *)calloc(num_vec_u, sizeof(double));

    for (i = 0; i != num_vec_u; i++)
    {
        x = dot(&vec_u[i * dim], mean_vec_v, dim);
        //weight_vec[i] = pow(exp(x), NEG_SAMPLING_POWER);
        weight_vec[i] = FastSigmoid(x);
        sum += weight_vec[i];
    }

    for (i = 0; i != neg_table_size; i++)
    {
        if ((double)(i + 1) / neg_table_size > por)
        {
            cur_sum += weight_vec[vid];
            por = cur_sum / sum;
            vid++;
        }
        neg_table[i] = vid - 1;
    }
    free(mean_vec_v);
    free(weight_vec);
}

/* Compute distance between two embeddings*/
real dist_emb(real *vec_u, real *vec_v, int num_vertices)
{
    real x = 0;
    for (int c = 0; c != num_vertices * dim; c++) x += pow((vec_u[c] - vec_v[c]), 2);
    return x;
}

/* Fastly generate a random integer */
int Rand(unsigned long long &seed)
{
    seed = seed * 25214903917 + 11;
    return (seed >> 16) % neg_table_size;
}

/* Update embeddings in word-word and word-doc networks.
 vec_u and vec_v are source and target embeddings, respectively.
 */
void Update(real *vec_u, real *vec_v, real *vec_error, int label)
{
    real x, g;
    x = dot(vec_u, vec_v, dim);
    g = (label - FastSigmoid(x)) * rho;
    for (int c = 0; c != dim; c++) vec_error[c] += g * vec_u[c]; // vec_error is used to update target embeddings
    for (int c = 0; c != dim; c++) vec_u[c] += g * vec_v[c];
}

/* Update embeddings in word-topic and topic-doc networks.
 vec_u and vec_v are source and target embeddings, respectively.
 */
void Update2(real *vec_u, real *vec_v, real *vec_error, int label, real part_grad)
{
    real x, g;
    x = dot(vec_u, vec_v, dim);
    g = part_grad * (label - FastSigmoid(x)) * rho;
    for (int c = 0; c != dim; c++) vec_error[c] += g * vec_u[c]; // vec_error is used to update target embeddings
    for (int c = 0; c != dim; c++) vec_u[c] += g * vec_v[c];
}

/* Compute p(u/v) * [log p(u/v) +1]
 which is part of the gradient for word-topic and topic-doc networks
 */
real CalcPartGrad(long long target_vertex, long long *sample_list, real *source_emb_vertex, real *target_emb_vertex)
{
    long long u, v = target_vertex, lu, lv;
    int d;
    real x, log_pr = 0;

    lv = v * dim;
    // compute log p(u/v) using the negative sampling equation
    for (d = 0; d != num_negative + 1; d++)
    {
        u = sample_list[d];
        lu = u * dim;

        if (d == 0) // positive sample
        {
            x = dot(&source_emb_vertex[lu], &target_emb_vertex[lv], dim);
            log_pr += logl((double)FastSigmoid(x));
        }
        else // negative sample
        {
            x = dot(&source_emb_vertex[lu], &target_emb_vertex[lv], dim);
            log_pr += logl((double)FastSigmoid(-x));
        }
    }
    x = exp(log_pr) * (log_pr + 1);
    return isnan(x)? 0 : x;
}


/* Compute conditional distribution using the original softmax function*/
void CalcCondDist(real **cond_dist, real *source_emb_vertex, real *target_emb_vertex, int source_num_vertices, int target_num_vertices, int start_idx, int end_idx)
{
    int i, j;
    long long lu, lv;
    real sum;

    for (j = start_idx; j != end_idx; j++)
    {
        lv = j * dim;
        sum = 0;
        for (i = 0; i != source_num_vertices; i++)
        {
            lu = i * dim;
            cond_dist[j][i] = exp(dot(&source_emb_vertex[lu], &target_emb_vertex[lv], dim));
            sum += cond_dist[j][i];
        }
        // normalization
        for (i = 0; i != source_num_vertices; i++)
            cond_dist[j][i] /= sum;
    }
}

void *MonitorThread(void *id)
{
    real word_diff, doc_diff, topic_diff;
    real *pre_word_emb = InitVector(num_word_vertices); // word
    real *pre_doc_emb = InitVector(num_doc_vertices); // doc
    real *pre_topic_emb = InitVector(n_topics); // topic

    while (current_sample_count < total_samples)
    {
        if (current_sample_count % (long long)(update_negtable_ratio * total_samples) == 0) // update topic realted neg_table
        {
            InitNegTable4Topic(wt_neg_table, word_emb_vertex, topic_emb_vertex, num_word_vertices, n_topics); // word-topic
            InitNegTable4Topic(td_neg_table, topic_emb_vertex, doc_emb_vertex, n_topics, num_doc_vertices); // topic-doc
            printf("\nupdate topic related neg_table");
        }

        // check convergence
        if (current_sample_count > total_samples - num_threads)
        {
            printf("\ncopy pre emb");
            memcpy(pre_word_emb, word_emb_vertex, num_word_vertices * dim * sizeof(real));
            memcpy(pre_doc_emb, doc_emb_vertex, num_doc_vertices * dim * sizeof(real));
            memcpy(pre_topic_emb, topic_emb_vertex, n_topics * dim * sizeof(real));
        }
    }

    word_diff = dist_emb(word_emb_vertex, pre_word_emb, num_word_vertices);
    doc_diff = dist_emb(doc_emb_vertex, pre_doc_emb, num_doc_vertices);
    topic_diff = dist_emb(topic_emb_vertex, pre_topic_emb, n_topics);
    if (word_diff <= epsilon && doc_diff <= epsilon && topic_diff <= epsilon) printf("\nconverged");
    else printf("\nnot converged");
    printf("\nword_diff: %f, doc_diff: %f, topic_diff: %f\n", word_diff, doc_diff, topic_diff);

    free(pre_word_emb);
    free(pre_doc_emb);
    free(pre_topic_emb);
    pthread_exit(NULL);
}


void *TrainLINEThread(void *id)
{
    int label;
    long long u, v, lu, lv, source;
    long long last_count = 0, count, curedge;
    unsigned long long seed = (long long)id;
    real *vec_error = (real *)calloc(dim, sizeof(real));
    long long *sample_list = (long long *)calloc(num_negative+1, sizeof(long long));
    real part_grad;
    long long batch_size = total_samples / num_threads;
    if ((long long)id == 0) batch_size = total_samples - batch_size * (num_threads - 1);

    for (count = 0; count < batch_size; count++)
    {
        // 1) sample an edge from Eww and draw num_negative negative edges
        curedge = SampleAnEdge(gsl_rng_uniform(gsl_r), gsl_rng_uniform(gsl_r), WW_TYPE);
        u = ww_edge_source_id[curedge];
        v = ww_edge_target_id[curedge];

        lv = v * dim;
        for (int c = 0; c != dim; c++) vec_error[c] = 0;

        // NEGATIVE SAMPLING
        for (int d = 0; d != num_negative + 1; d++)
        {
            if (d == 0) // positive sample
            {
                source = u;
                label = 1;
            }
            else // negative samples
            {
                source = v;
                while (source == v) // source id should be distinguished from target id
                    source = ww_neg_table[Rand(seed)];
                label = 0;
            }
            lu = source * dim;
//            pthread_mutex_lock(&my_mutex);
            Update(&word_emb_vertex[lu], &word_emb_vertex[lv], vec_error, label);
//            pthread_mutex_unlock(&my_mutex);
        }
        // update target embedding
//        pthread_mutex_lock(&my_mutex);
        for (int c = 0; c != dim; c++) word_emb_vertex[c + lv] += vec_error[c];
//        pthread_mutex_unlock(&my_mutex);


        // 2) sample an edge from Ewd and draw num_negative negative edges
        curedge = SampleAnEdge(gsl_rng_uniform(gsl_r), gsl_rng_uniform(gsl_r), WD_TYPE);
        u = wd_edge_source_id[curedge];
        v = wd_edge_target_id[curedge];

        lv = v * dim;
        for (int c = 0; c != dim; c++) vec_error[c] = 0;

        // NEGATIVE SAMPLING
        for (int d = 0; d != num_negative + 1; d++)
        {
            if (d == 0) // positive sample
            {
                source = u;
                label = 1;
            }
            else // negative samples
            {
                source = wd_neg_table[Rand(seed)];
                label = 0;
            }
            lu = source * dim;
//            pthread_mutex_lock(&my_mutex);
            Update(&word_emb_vertex[lu], &doc_emb_vertex[lv], vec_error, label);
//            pthread_mutex_unlock(&my_mutex);
        }
        // update target embedding
//        pthread_mutex_lock(&my_mutex);
        for (int c = 0; c != dim; c++) doc_emb_vertex[c + lv] += vec_error[c];
//        pthread_mutex_unlock(&my_mutex);


        // 3) sample a pair of word-topic and draw num_negative "negative" pairs
        uniform_int_distribution<int>  topic_distr(0, n_topics-1);
        uniform_int_distribution<int>  word_distr(0, num_word_vertices-1);

        u = word_distr(generator);
        v = topic_distr(generator);

        lv = v * dim;
        for (int c = 0; c != dim; c++) vec_error[c] = 0;
        for (int d = 0; d != num_negative + 1; d++) sample_list[d] = -1;

        // NEGATIVE SAMPLING
        for (int d = 0; d != num_negative + 1; d++)
        {
            if (d == 0) // positive sample
            {
                sample_list[d] = u;
            }
            else // negative samples
            {
                sample_list[d] = wt_neg_table[Rand(seed)]; // samples a negative edge
            }
        }

        part_grad = CalcPartGrad(v, sample_list, word_emb_vertex, topic_emb_vertex);

        for (int d = 0; d != num_negative + 1; d++)
        {
            source = sample_list[d];
            label = (d == 0)? 1:0;
            lu = source * dim;
//            pthread_mutex_lock(&my_mutex);
            Update2(&word_emb_vertex[lu], &topic_emb_vertex[lv], vec_error, label, part_grad);
//            pthread_mutex_unlock(&my_mutex);
        }
        // update target embedding
//        pthread_mutex_lock(&my_mutex);
        for (int c = 0; c != dim; c++) topic_emb_vertex[c + lv] += vec_error[c];
//        pthread_mutex_unlock(&my_mutex);


        // 4) sample a pair of topic-doc and draw num_negative "negative" pairs
        uniform_int_distribution<int>  doc_distr(0, num_doc_vertices-1);

        u = topic_distr(generator);
        v = doc_distr(generator);

        lv = v * dim;
        for (int c = 0; c != dim; c++) vec_error[c] = 0;
        for (int d = 0; d != num_negative + 1; d++) sample_list[d] = -1;

        // NEGATIVE SAMPLING
        for (int d = 0; d != num_negative + 1; d++)
        {
            if (d == 0) // positive sample
            {
                sample_list[d] = u;
            }
            else // negative samples
            {
                sample_list[d] = td_neg_table[Rand(seed)]; // samples a negative edge
            }
        }

        part_grad = CalcPartGrad(v, sample_list, topic_emb_vertex, doc_emb_vertex);

        for (int d = 0; d != num_negative + 1; d++)
        {
            source = sample_list[d];
            label = (d == 0)? 1:0;
            lu = source * dim;
//            pthread_mutex_lock(&my_mutex);
            Update2(&topic_emb_vertex[lu], &doc_emb_vertex[lv], vec_error, label, part_grad);
//            pthread_mutex_unlock(&my_mutex);
        }
        // update target embedding
//        pthread_mutex_lock(&my_mutex);
        for (int c = 0; c != dim; c++) doc_emb_vertex[c + lv] += vec_error[c];
//        pthread_mutex_unlock(&my_mutex);

        pthread_mutex_lock(&my_mutex);
        current_sample_count += 1;
        pthread_mutex_unlock(&my_mutex);

        if (count - last_count == 10000)
        {
            last_count = count;
            printf("%cRho: %f  Progress: %.3lf%%", 13, rho, (real)current_sample_count / (real)total_samples * 100);
            fflush(stdout);
            rho = init_rho * (1 - current_sample_count / (real)total_samples);
            if (rho < init_rho * 0.0001) rho = init_rho * 0.0001;
        }
    }

    free(vec_error);
    free(sample_list);
    printf("%cRho: %f  Progress: %.3lf%%", 13, rho, (real)current_sample_count / (real)total_samples * 100);
    fflush(stdout);
    join_threads += 1;
    pthread_exit(NULL);
}

void OutputVector(char *out_file, real *emb_vertex, int num_vertices, int type)
{
    FILE *fo = fopen(out_file, "wb");
    if (fo == NULL)
    {
        printf("ERROR: failed to create a file!\n");
        exit(1);
    }
    fprintf(fo, "# number of nodes: %d, number of dimensions: %d\n", num_vertices, dim);
    for (int a = 0; a < num_vertices; a++)
    {
        if (type == TOPIC_TYPE)
            fprintf(fo, "# topic %d)\n", a);
        else if (type == WORD_TYPE)
        {
            //printf("%d)\n", a);
            //fflush(stdout);
            //fprintf(fo, "%d)\n", a);
            fprintf(fo, "# %s\n", word_vertex[a].name);
        }
        else if (type == DOC_TYPE)
        {
            //printf("%d)\n", a);
            //fflush(stdout);
            //fprintf(fo, "%d)\n", a);
            fprintf(fo, "# %s\n", doc_vertex[a].name);
        }
        else
        {
            printf("Error: unknown type %d", type);
            fclose(fo);
            exit(1);
        }
        if (is_binary) for (int b = 0; b < dim; b++) fwrite(&emb_vertex[a * dim + b], sizeof(real), 1, fo);
        else for (int b = 0; b < dim; b++) {

            //printf("%f ", emb_vertex[a * dim + b]);
            //fflush(stdout);
            fprintf(fo, "%lf ", emb_vertex[a * dim + b]);
        }
        //printf("\n");
        fprintf(fo, "\n");
    }
    fclose(fo);
}

/* Output doc-topic and topic-word distributions*/
void OutputCondDist(char *out_file, real **cond_dist, int source_num_vertices, int target_num_vertices, int type)
{
    FILE *fo = fopen(out_file, "wb");
    if (fo == NULL)
    {
        printf("ERROR: failed to create a file!\n");
        exit(1);
    }
    fprintf(fo, "# number of nodes: %d, number of dimensions: %d\n", target_num_vertices, source_num_vertices);
    for (int a = 0; a < target_num_vertices; a++)
    {
        if (type == DT_TYPE)
        {
            //fprintf(fo, "%d)\n", a);
            //printf("%d)\n", a);
            //fflush(stdout);
            fprintf(fo, "# %s\n", doc_vertex[a].name);
        }
        else if (type == TW_TYPE)
        {
            fprintf(fo, "# topic %d)\n", a);
            //printf("topic %d)\n", a);
        }
        else
        {
            printf("Error: unknown type %d", type);
            fclose(fo);
            exit(1);
        }

        if (is_binary)
            for (int b = 0; b < source_num_vertices; b++) fwrite(&cond_dist[a][b], sizeof(real), 1, fo);
        else
            for (int b = 0; b < source_num_vertices; b++) fprintf(fo, "%lf ", cond_dist[a][b]);
        //printf("\n");
        fprintf(fo, "\n");
    }
    fclose(fo);
}


/* compare function */
int sortByValue(const pair<string, real> &lhs, const pair<string, real> &rhs)
{
    return lhs.second < rhs.second ? 1 : lhs.second == lhs.second ? 0 : -1;
}


/* Output readable doc-topic and topic-word distributions */
void OutputReadableCondDist(char *out_file, real **cond_dist, int size_vec, int n_select, int n_top, int type)
{
    FILE *fo = fopen(out_file, "wb");
    if (fo == NULL)
    {
        printf("ERROR: failed to create a file!\n");
        exit(1);
    }
    fprintf(fo, "# number of nodes: %d, number of dimensions: %d\n", n_select, n_top);
    for (int a = 0; a < n_select; a++)
    {
        vector<pair<string, real>> vec;

        if (type == DT_TYPE)
        {
            //fprintf(fo, "%d)\n", a);
            //printf("%d)\n", a);
            //fflush(stdout);
            fprintf(fo, "# %s\n", doc_vertex[a].name);

            // copy data to vector
            for (int b = 0; b < size_vec; b++)
            {
                pair<string, real> item;
                item.first = string("topic ") + to_string(b) + string(")");
                item.second = cond_dist[a][b];
                vec.push_back(item);
            }
        }
        else if (type == TW_TYPE)
        {
            fprintf(fo, "# topic %d)\n", a);
            //printf("topic %d)\n", a);

            // copy data to vector
            for (int b = 0; b < size_vec; b++)
            {
                pair<string, real> item;
                item.first = word_vertex[b].name;
                item.second = cond_dist[a][b];
                vec.push_back(item);
            }
        }
        else
        {
            printf("Error: unknown type %d", type);
            fclose(fo);
            exit(1);
        }

        sort(vec.rbegin(), vec.rend(), sortByValue);

        if (is_binary)
        {
            for (int b = 0; b < n_top; b++)
                fwrite(&cond_dist[a][b], sizeof(real), 1, fo);
        }
        else
            for (int b = 0; b < n_top; b++)
            {
                fprintf(fo, " %lf*%s +", vec[b].second, vec[b].first.c_str());
            }
        //printf("\n");
        fprintf(fo, " ...\n");
    }
    fclose(fo);
}


void TrainLINE() {
    long a;
    pthread_t *pt = (pthread_t *)malloc((num_threads + 1) * sizeof(pthread_t));

    printf("--------------------------------\n");
    printf("Samples: %lldM\n", total_samples / 1000000);
    printf("Negative: %d\n", num_negative);
    printf("Dimension: %d\n", dim);
    printf("Initial rho: %lf\n", init_rho);
    printf("--------------------------------\n");


    /* Default output settings */

    if (strlen(word_embedding_file) == 0)
        strcpy(word_embedding_file, "word_vec.txt");

    if (strlen(doc_embedding_file) == 0)
        strcpy(doc_embedding_file, "doc_vec.txt");

    if (strlen(topic_embedding_file) == 0)
        strcpy(topic_embedding_file, "topic_vec.txt");

    if (strlen(doc_topic_dist_file) == 0)
        strcpy(doc_topic_dist_file, "dt_dist.txt");

    if (strlen(topic_word_dist_file) == 0)
        strcpy(topic_word_dist_file, "tw_dist.txt");

    if (strlen(readable_doc_topic_dist_file) == 0)
        strcpy(readable_doc_topic_dist_file, "readable_dt_dist.txt");

    if (strlen(readable_topic_word_dist_file) == 0)
        strcpy(readable_topic_word_dist_file, "readable_tw_dist.txt");

    /* Init hash tables */
    word_hash_table = InitHashTable();
    doc_hash_table = InitHashTable();


    /* Read word-word and word-doc networks*/
    ReadData(wwnet_file, 0);
    ReadData(wdnet_file, 1);
    printf("Number of words: %d          \n", num_word_vertices);
    printf("Number of documents: %d          \n", num_doc_vertices);
    printf("Number of topics: %d          \n", n_topics);


    /* Init alias tables */
    InitAliasTable(num_ww_edges, ww_edge_weight, 0); // word-word network
    InitAliasTable(num_wd_edges, wd_edge_weight, 1); // word-doc network

    /* Init text embeddings */
    word_emb_vertex = InitVector(num_word_vertices); // word
    doc_emb_vertex = InitVector(num_doc_vertices); // doc
    topic_emb_vertex = InitVector(n_topics); // topic

    /* Init negative sampling and sigmoid tables */
    InitNegTable(WW_TYPE); // word-word network
    InitNegTable(WD_TYPE); // word-doc network
    wt_neg_table = (int *)malloc(neg_table_size * sizeof(int)); // word-topic network
    td_neg_table = (int *)malloc(neg_table_size * sizeof(int)); // topic-doc network

    InitSigmoidTable();

    gsl_rng_env_setup();
    gsl_T = gsl_rng_rand48;
    gsl_r = gsl_rng_alloc(gsl_T);
    gsl_rng_set(gsl_r, 314159265);


    clock_t start = clock();
    printf("--------------------------------\n");
    pthread_create(&pt[num_threads], NULL, MonitorThread, (void *)(long)num_threads); // monitor thread
    for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainLINEThread, (void *)a);
    for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
    pthread_join(pt[num_threads], NULL);

    printf("\n");
    clock_t finish = clock();
    printf("Total time: %lf\n", (double)(finish - start) / CLOCKS_PER_SEC);

    /* Output text embeddings */
    OutputVector(word_embedding_file, word_emb_vertex, num_word_vertices, WORD_TYPE);
    OutputVector(doc_embedding_file, doc_emb_vertex, num_doc_vertices, DOC_TYPE);
    OutputVector(topic_embedding_file, topic_emb_vertex, n_topics, TOPIC_TYPE);

    /* Init doc-topic and topic-word distributions */
    doc_topic_dist = InitCondDist(n_topics, num_doc_vertices);
    topic_word_dist = InitCondDist(num_word_vertices, n_topics);

    /* Compute doc-topic and topic-word distributions */
    CalcCondDist(doc_topic_dist, topic_emb_vertex, doc_emb_vertex, n_topics, num_doc_vertices, 0, num_doc_vertices);
    CalcCondDist(topic_word_dist, word_emb_vertex, topic_emb_vertex, num_word_vertices, n_topics, 0, n_topics);

    /* Output doc-topic and topic-word distributions */
    OutputCondDist(doc_topic_dist_file, doc_topic_dist, n_topics, num_doc_vertices, DT_TYPE); // doc-topic
    OutputCondDist(topic_word_dist_file, topic_word_dist, num_word_vertices, n_topics, DT_TYPE); // topic-word

    /* Output readable doc-topic and topic-word distributions */
    OutputReadableCondDist(readable_doc_topic_dist_file, doc_topic_dist, n_topics, num_doc_vertices, min(n_top, n_topics), DT_TYPE);
    OutputReadableCondDist(readable_topic_word_dist_file, topic_word_dist, num_word_vertices, n_topics, min(n_top, num_word_vertices), TW_TYPE);


    // free memory
    free(word_emb_vertex);
    free(doc_emb_vertex);
    free(topic_emb_vertex);
    free(word_hash_table);
    free(doc_hash_table);
    free(ww_edge_source_id);
    free(ww_edge_target_id);
    free(ww_edge_weight);
    free(wd_edge_source_id);
    free(wd_edge_target_id);
    free(wd_edge_weight);
    free(ww_alias);
    free(ww_prob);
    free(wd_alias);
    free(wd_prob);
    free(ww_neg_table);
    free(wd_neg_table);
    free(wt_neg_table);
    free(td_neg_table);
    free(sigmoid_table);
    FreeCondDist(doc_topic_dist, num_doc_vertices);
    FreeCondDist(topic_word_dist, num_topic_vertices);
    free(pt);
}

int ArgPos(char *str, int argc, char **argv) {
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

int main(int argc, char **argv) {
    int i;
    if (argc == 1) {
        printf("TTE: Large Information Network Embedding\n\n");
        printf("Options:\n");
        printf("Parameters for training:\n");
        printf("\t-train <file>\n");
        printf("\t\tUse network data from <file> to train the model\n");
        printf("\t-output <file>\n");
        printf("\t\tUse <file> to save the learnt embeddings\n");
        printf("\t-binary <int>\n");
        printf("\t\tSave the learnt embeddings in binary moded; default is 0 (off)\n");
        printf("\t-size <int>\n");
        printf("\t\tSet dimension of vertex embeddings; default is 100\n");
        printf("\t-negative <int>\n");
        printf("\t\tNumber of negative examples; default is 5\n");
        printf("\t-samples <int>\n");
        printf("\t\tSet the number of training samples as <int>Million; default is 1\n");
        printf("\t-threads <int>\n");
        printf("\t\tUse <int> threads (default 1)\n");
        printf("\t-rho <float>\n");
        printf("\t\tSet the starting learning rate; default is 0.025\n");
        printf("\nExamples:\n");
        printf("./line -ww word_word_net.txt -wd word_doc_net.txt -out_word word_vec.txt -out_doc doc_vec.txt -out_topic topic_vec.txt -out_dt_dist dt_dist.txt -out_tw_dist tw_dist.txt -out_rdt_dist readable_dt_dist.txt -out_rtw_dist readable_tw_dist.txt -n_top 5 -binary 1 -n_topics 20 -size 200 -negative 5 -samples 100 -rho 0.025 -epsilon 3.0 -threads 20\n\n");
        return 0;
    }
    if ((i = ArgPos((char *)"-ww", argc, argv)) > 0) strcpy(wwnet_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-wd", argc, argv)) > 0) strcpy(wdnet_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-out_word", argc, argv)) > 0) strcpy(word_embedding_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-out_doc", argc, argv)) > 0) strcpy(doc_embedding_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-out_topic", argc, argv)) > 0) strcpy(topic_embedding_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-out_dt_dist", argc, argv)) > 0) strcpy(doc_topic_dist_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-out_tw_dist", argc, argv)) > 0) strcpy(topic_word_dist_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-out_rdt_dist", argc, argv)) > 0) strcpy(readable_doc_topic_dist_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-out_rtw_dist", argc, argv)) > 0) strcpy(readable_topic_word_dist_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-n_top", argc, argv)) > 0) n_top = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) is_binary = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-n_topics", argc, argv)) > 0) n_topics = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-size", argc, argv)) > 0) dim = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) num_negative = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-samples", argc, argv)) > 0) total_samples = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-rho", argc, argv)) > 0) init_rho = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-epsilon", argc, argv)) > 0) epsilon = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
    total_samples *= 1000000;
    rho = init_rho;
    word_vertex = (struct ClassVertex *)calloc(max_num_word_vertices, sizeof(struct ClassVertex));
    doc_vertex = (struct ClassVertex *)calloc(max_num_doc_vertices, sizeof(struct ClassVertex));
    TrainLINE();
    FreeVertex(word_vertex, num_word_vertices);
    FreeVertex(doc_vertex, num_doc_vertices);
    return 0;
}
