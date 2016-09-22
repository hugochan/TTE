/*
This is the tool ....

Contact Author: Jian Tang, Microsoft Research, jiatang@microsoft.com, tangjianpku@gmail.com
Publication: Jian Tang, Meng Qu, Mingzhe Wang, Ming Zhang, Jun Yan, Qiaozhu Mei. "LINE: Large-scale Information Network Embedding". In WWW 2015.
*/

// Format of the training file:
//
// The training file contains serveral lines, each line represents a DIRECTED edge in the network.
// More specifically, each line has the following format "<u> <v> <w>", meaning an edge from <u> to <v> with weight as <w>.
// <u> <v> and <w> are seperated by ' ' or '\t' (blank or tab)
// For UNDIRECTED edge, the user should use two DIRECTED edges to represent it.

#include <iostream>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <gsl/gsl_rng.h>


#define MAX_STRING 100
#define SIGMOID_BOUND 6
#define NEG_SAMPLING_POWER 0.75
#define WW_TYPE 0
#define WD_TYPE 1
#define WORD_TYPE 0
#define DOC_TYPE 1
#define TOPIC_TYPE 2

const int hash_table_size = 30000000;
const int neg_table_size = 1e8;
const int sigmoid_table_size = 1000;


typedef float real;                    // Precision of float numbers

struct ClassVertex {
	// double ww_degree; // if word vertex, word-word net out-degree; else, reserved
	// double wd_degree; // if word vertex, word-doc net out-degree; else, reserved
	double degree[2]; // if word vertex, word-word network out-degree and word-doc network out-degree
	char *name;
};


// char network_file[MAX_STRING], embedding_file[MAX_STRING];
char wwnet_file[MAX_STRING], wdnet_file[MAX_STRING], word_embedding_file[MAX_STRING], doc_embedding_file[MAX_STRING], topic_embedding_file[MAX_STRING];
struct ClassVertex *word_vertex, *doc_vertex;
int is_binary = 0, n_topics = 0, num_threads = 1, order = 2, dim = 100, num_negative = 5;
int *word_hash_table, *doc_hash_table, *ww_neg_table, *wd_neg_table;
int max_num_vertices = 1, num_word_vertices = 0, num_doc_vertices = 0, num_topic_vertices = 0;
long long total_samples = 1, current_sample_count = 0, num_ww_edges = 0, num_wd_edges = 0;
real init_rho = 0.025, rho;
real *word_emb_vertex, *word_emb_context, *doc_emb_vertex, *doc_emb_context, *topic_emb_vertex, *topic_emb_context, *sigmoid_table;

int *ww_edge_source_id, *ww_edge_target_id, *wd_edge_source_id, *wd_edge_target_id;
double *ww_edge_weight, *wd_edge_weight;

// Parameters for edge sampling
long long *ww_alias, *wd_alias;
double *ww_prob, *wd_prob;

const gsl_rng_type * gsl_T;
gsl_rng * gsl_r;


// Generate uniform random numbers
std::random_device                  rand_dev;
std::mt19937                        generator(rand_dev());


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

/* Add a vertex to the vertex set */
struct ClassVertex *AddVertex(int *vertex_hash_table, struct ClassVertex *vertex, int *num_vertices, char *name)
{
	int length = strlen(name) + 1;
	if (length > MAX_STRING) length = MAX_STRING;
	vertex[*num_vertices].name = (char *)calloc(length, sizeof(char));
	strcpy(vertex[*num_vertices].name, name);
	vertex[*num_vertices].degree[0] = 0; // ww
	vertex[*num_vertices].degree[1] = 0; // wd
	(*num_vertices)++;
	if (*num_vertices + 2 >= max_num_vertices)
	{
		max_num_vertices += 1000;
		vertex = (struct ClassVertex *)realloc(vertex, max_num_vertices * sizeof(struct ClassVertex));

	}
	InsertHashTable(vertex_hash_table, name, *num_vertices - 1);
	return vertex;
}

/* Read network from the training file */
void ReadData(char *network_file, int type)
{
	int *source_num_vertices, *target_num_vertices;
	long long *num_edges;
	struct ClassVertex *source_vertex, *target_vertex;
	int *source_hash_table, *target_hash_table;
	int *edge_source_id, *edge_target_id;
	double *edge_weight;

	if (type == WW_TYPE) // word-word network
	{
		source_num_vertices = &num_word_vertices;
		target_num_vertices = &num_word_vertices;
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
		num_edges = &num_wd_edges;
		source_vertex = word_vertex;
		target_vertex = doc_vertex;
		source_hash_table = word_hash_table;
		target_hash_table = doc_hash_table;
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
	printf("Number of edges: %lld          \n\n", *num_edges);

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
			source_vertex = AddVertex(source_hash_table, source_vertex, source_num_vertices, name_v1);
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
			target_vertex = AddVertex(target_hash_table, target_vertex, target_num_vertices, name_v2);
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
	long long *alias = (long long *)malloc(num_edges*sizeof(long long));
	double *prob = (double *)malloc(num_edges*sizeof(double));
	if (alias == NULL || prob == NULL)
	{
		printf("Error: memory allocation failed!\n");
		exit(1);
	}

	double *norm_prob = (double*)malloc(num_edges*sizeof(double));
	long long *large_block = (long long*)malloc(num_edges*sizeof(long long));
	long long *small_block = (long long*)malloc(num_edges*sizeof(long long));
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

	free(norm_prob);
	free(small_block);
	free(large_block);
}

long long SampleAnEdge(double rand_value1, double rand_value2, int type)
{
	int num_edges = 0;
	double *prob = NULL;
	long long *alias = NULL;

	if (type == WW_TYPE) // word-word network
	{
		num_edges = num_ww_edges;
		prob = ww_prob;
		alias = ww_alias;
	}
	else if (type == WD_TYPE) // word-doc network
	{
		num_edges = num_wd_edges;
		prob = wd_prob;
		alias = wd_alias;
	}
	long long k = (long long)num_edges * rand_value1;
	return rand_value2 < prob[k] ? k : alias[k];
}

/* Initialize the vertex embedding and the context embedding */
void InitVector(int type, int num_vertices)
{
	long long a, b;
	real *emb_vertex, *emb_context;

	a = posix_memalign((void **)&emb_vertex, 128, (long long)num_vertices * dim * sizeof(real));
	if (emb_vertex == NULL) { printf("Error: memory allocation failed\n"); exit(1); }
	for (b = 0; b < dim; b++) for (a = 0; a < num_vertices; a++)
		emb_vertex[a * dim + b] = (rand() / (real)RAND_MAX - 0.5) / dim;

	a = posix_memalign((void **)&emb_context, 128, (long long)num_vertices * dim * sizeof(real));
	if (emb_context == NULL) { printf("Error: memory allocation failed\n"); exit(1); }
	for (b = 0; b < dim; b++) for (a = 0; a < num_vertices; a++)
		emb_context[a * dim + b] = 0;

	if (type == WORD_TYPE) // word
	{
		word_emb_vertex = emb_vertex;
		word_emb_context = emb_context;
	}
	else if (type == DOC_TYPE) // doc
	{
		doc_emb_vertex = emb_vertex;
		doc_emb_context = emb_context;
	}
	else if (type == TOPIC_TYPE) // topic
	{
		topic_emb_vertex = emb_vertex;
		topic_emb_context = emb_context;
	}
	else
	{
		printf("ERROR: unknown output type %d", type);
		exit(1);
	}
}

/* Sample negative vertex samples according to vertex degrees */
void InitNegTable(int type)
{
	int num_vertices = 0;
	struct ClassVertex *vertex;
	if (type == WW_TYPE) // word-word network
	{
		num_vertices = num_word_vertices;
		vertex = word_vertex;
	}
	else if (type == WD_TYPE) // word-doc network
	{
		num_vertices = num_word_vertices;
		vertex = word_vertex;
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

/* Fastly compute sigmoid function */
void InitSigmoidTable()
{
	real x;
	sigmoid_table = (real *)malloc((sigmoid_table_size + 1) * sizeof(real));
	for (int k = 0; k != sigmoid_table_size; k++)
	{
		x = 2 * SIGMOID_BOUND * k / sigmoid_table_size - SIGMOID_BOUND;
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

/* Compute -p(u/v) * [log p(u/v) +1]
which is part of the gradient for word-topic and topic-doc networks
*/
real calc_part_grad(long long target_vertex, long long *sample_list, real *source_emb_vertex, real *target_emb_vertex)
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
	return -exp(log_pr) * (log_pr + 1);
}

void *TrainLINEThread(void *id)
{
	long long u, v, lu, lv, source, label;
	long long count = 0, last_count = 0, curedge;
	unsigned long long seed = (long long)id;
	real *vec_error = (real *)calloc(dim, sizeof(real));
	// real *emb_vertex, *emb_context;
	long long *sample_list = (long long *)calloc(num_negative+1, sizeof(long long));
	real part_grad;

	while (1)
	{
		//judge for exit
		if (count > total_samples / num_threads + 2) break;

		if (count - last_count > 10000)
		{
			current_sample_count += count - last_count;
			last_count = count;
			printf("%cRho: %f  Progress: %.3lf%%", 13, rho, (real)current_sample_count / (real)(total_samples + 1) * 100);
			fflush(stdout);
			rho = init_rho * (1 - current_sample_count / (real)(total_samples + 1));
			if (rho < init_rho * 0.0001) rho = init_rho * 0.0001;
		}


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
			Update(&word_emb_vertex[lu], &word_emb_vertex[lv], vec_error, label);
		}

		// update target embedding
		for (int c = 0; c != dim; c++) word_emb_vertex[c + lv] += vec_error[c];



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
			Update(&word_emb_vertex[lu], &doc_emb_vertex[lv], vec_error, label);
		}

		// update target embedding
		for (int c = 0; c != dim; c++) doc_emb_vertex[c + lv] += vec_error[c];


		// 3) sample a pair of word-topic and draw num_negative "negative" pairs
		std::uniform_int_distribution<int>  topic_distr(0, n_topics-1);
		std::uniform_int_distribution<int>  word_distr(0, num_word_vertices-1);

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
				sample_list[d] = word_distr(generator); // uniformly samples a negative edge
			}
		}

		part_grad = calc_part_grad(v, sample_list, word_emb_vertex, topic_emb_vertex);

		for (int d = 0; d != num_negative + 1; d++)
		{
			source = sample_list[d];
			label = (d == 0)? 1:0;
			lu = source * dim;

			Update2(&word_emb_vertex[lu], &topic_emb_vertex[lv], vec_error, label, part_grad);
		}

		// update target embedding
		for (int c = 0; c != dim; c++) topic_emb_vertex[c + lv] += vec_error[c];





		// 4) sample a pair of topic-doc and draw num_negative "negative" pairs
		std::uniform_int_distribution<int>  doc_distr(0, num_doc_vertices-1);

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
				sample_list[d] = topic_distr(generator); // uniformly samples a negative edge
			}
		}

		part_grad = calc_part_grad(v, sample_list, topic_emb_vertex, doc_emb_vertex);

		for (int d = 0; d != num_negative + 1; d++)
		{
			source = sample_list[d];
			label = (d == 0)? 1:0;
			lu = source * dim;

			Update2(&topic_emb_vertex[lu], &doc_emb_vertex[lv], vec_error, label, part_grad);
		}

		// update target embedding
		for (int c = 0; c != dim; c++) doc_emb_vertex[c + lv] += vec_error[c];


		count++;
	}
	free(vec_error);
	free(sample_list);
	pthread_exit(NULL);
}

void OutputVector(int type)
{
	char *out_file = NULL;
	int num_vertices;
	struct ClassVertex *vertex = NULL;
	real *emb_vertex = NULL;

	if (type == WORD_TYPE) // word
	{
		out_file = word_embedding_file;
		num_vertices = num_word_vertices;
		vertex = word_vertex;
		emb_vertex = word_emb_vertex;
	}
	else if (type == DOC_TYPE) // doc
	{
		out_file = doc_embedding_file;
		num_vertices = num_doc_vertices;
		vertex = doc_vertex;
		emb_vertex = doc_emb_vertex;
	}
	else if (type == TOPIC_TYPE) // topic
	{
		out_file = topic_embedding_file;
		num_vertices = n_topics;
		emb_vertex = topic_emb_vertex;
	}
	else
	{
		printf("ERROR: unknown output type %d", type);
		exit(1);
	}

	FILE *fo = fopen(out_file, "wb");
	fprintf(fo, "#%d %d-D\n", num_vertices, dim);
	for (int a = 0; a < num_vertices; a++)
	{
		if (type == TOPIC_TYPE)
			fprintf(fo, "topic %d) ", a);
		else
			fprintf(fo, "%s ", vertex[a].name);
		if (is_binary) for (int b = 0; b < dim; b++) fwrite(&emb_vertex[a * dim + b], sizeof(real), 1, fo);
		else for (int b = 0; b < dim; b++) fprintf(fo, "%lf ", emb_vertex[a * dim + b]);
		fprintf(fo, "\n");
	}
	fclose(fo);
}

void TrainLINE() {
	long a;
	pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));

	if (order != 1 && order != 2)
	{
		printf("Error: order should be eighther 1 or 2!\n");
		exit(1);
	}
	printf("--------------------------------\n");
	printf("Order: %d\n", order);
	printf("Samples: %lldM\n", total_samples / 1000000);
	printf("Negative: %d\n", num_negative);
	printf("Dimension: %d\n", dim);
	printf("Initial rho: %lf\n", init_rho);
	printf("--------------------------------\n");

	word_hash_table = InitHashTable();
	doc_hash_table = InitHashTable();


	/* Read word-word and word-doc networks*/
	ReadData(wwnet_file, 0);
	ReadData(wdnet_file, 1);
	printf("Number of words: %d          \n", num_word_vertices);
	printf("Number of documents: %d          \n", num_doc_vertices);

	InitAliasTable(num_ww_edges, ww_edge_weight, 0); // word-word network
	InitAliasTable(num_wd_edges, wd_edge_weight, 1); // word-doc network
	InitVector(WORD_TYPE, num_word_vertices); // word
	InitVector(DOC_TYPE, num_doc_vertices); // doc
	InitVector(TOPIC_TYPE, n_topics); // topic

	InitNegTable(WW_TYPE); // word-word network
	InitNegTable(WD_TYPE); // word-doc network
	InitSigmoidTable();

	gsl_rng_env_setup();
	gsl_T = gsl_rng_rand48;
	gsl_r = gsl_rng_alloc(gsl_T);
	gsl_rng_set(gsl_r, 314159265);

	num_threads = 1;
	clock_t start = clock();
	printf("--------------------------------\n");
	for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainLINEThread, (void *)a);
	for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
	printf("\n");
	clock_t finish = clock();
	printf("Total time: %lf\n", (double)(finish - start) / CLOCKS_PER_SEC);

	OutputVector(WORD_TYPE); // word
	OutputVector(DOC_TYPE); // doc
	OutputVector(TOPIC_TYPE); // topic
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
		printf("LINE: Large Information Network Embedding\n\n");
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
		printf("\t-order <int>\n");
		printf("\t\tThe type of the model; 1 for first order, 2 for second order; default is 2\n");
		printf("\t-negative <int>\n");
		printf("\t\tNumber of negative examples; default is 5\n");
		printf("\t-samples <int>\n");
		printf("\t\tSet the number of training samples as <int>Million; default is 1\n");
		printf("\t-threads <int>\n");
		printf("\t\tUse <int> threads (default 1)\n");
		printf("\t-rho <float>\n");
		printf("\t\tSet the starting learning rate; default is 0.025\n");
		printf("\nExamples:\n");
		printf("./line -ww word_word_net.txt -wd word_doc_net.txt -out_word word_vec.txt -out_doc doc_vec.txt -out_topic topic_vec.txt -binary 1 -n_topics 20 -size 200 -order 2 -negative 5 -samples 100 -rho 0.025 -threads 20\n\n");
		return 0;
	}
	if ((i = ArgPos((char *)"-ww", argc, argv)) > 0) strcpy(wwnet_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-wd", argc, argv)) > 0) strcpy(wdnet_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-out_word", argc, argv)) > 0) strcpy(word_embedding_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-out_doc", argc, argv)) > 0) strcpy(doc_embedding_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-out_topic", argc, argv)) > 0) strcpy(topic_embedding_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) is_binary = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-n_topics", argc, argv)) > 0) n_topics = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-size", argc, argv)) > 0) dim = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-order", argc, argv)) > 0) order = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) num_negative = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-samples", argc, argv)) > 0) total_samples = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-rho", argc, argv)) > 0) init_rho = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
	total_samples *= 1000000;
	rho = init_rho;
	word_vertex = (struct ClassVertex *)calloc(max_num_vertices, sizeof(struct ClassVertex));
	doc_vertex = (struct ClassVertex *)calloc(max_num_vertices, sizeof(struct ClassVertex));
	TrainLINE();
	return 0;
}
