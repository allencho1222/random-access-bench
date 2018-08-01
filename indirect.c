/*
* indirect.c
*
*  Created on: Sep 7, 2017
*      Author: gswart
*
*  Modified on: Sep 22, 2017
*      Author: skchavan
*        Added segmented gather
*/

#include <stdlib.h>
#include <sys/time.h>
#include <stdio.h>

enum {
                rows = 1U << 26,
                array = 1U << 26,
                groups = 1U << 10,
        segment_bits = 12,
        segments = array / (1U << segment_bits)
};

struct Row {
                unsigned int measure;
                unsigned int group;
};

struct Row A[array];

unsigned int in[rows];
struct Row out[rows];
unsigned long long agg1[groups];
unsigned long long agg2[groups];

struct Row out2[rows];
struct Row * B[segments];

static unsigned int seed;

static unsigned long diff(const struct timeval * new, const struct timeval * old) {
                return (new->tv_sec - old->tv_sec)*1000000 + (new->tv_usec - old->tv_usec);
}

int main() {

                // Random fill indirection array A
                unsigned i;
                for (i = 0; i < array; i++) {
                                A[i].measure = rand_r(&seed) % array;
                                A[i].group = rand_r(&seed) % groups;
                }

                // Fill segmented array B
                for (i = 0; i < segments; i++) {
                B[i] = & (A[i * (1U << segment_bits)]);
        }

                // Random fill input
                for (i = 0; i < rows; i++)
                                in[i] = rand_r(&seed) % array;

                // Zero aggregates
                for (i = 0; i < groups; i++) {
                                agg1[i] = 0;
                                agg2[i] = 0;
                }

                struct timeval t0, t1, t2, t3, t4, t5;

                gettimeofday(&t0, 0);

                // Gather rows
                for (i = 0; i < rows; i++) {
                                out[i] = A[in[i]];
                }

                gettimeofday(&t1, 0);

                // Group rows
                for (i = 0; i < rows; i++) {
                                agg1[out[i].group] += out[i].measure;
                }

                gettimeofday(&t2, 0);

                // Indirect Gather rows
                for (i = 0; i < rows; i++) {
                                out[i] = A[A[in[i]].measure];
                }

                gettimeofday(&t3, 0);

                // Fused gather group
                for (i = 0; i < rows; i++) {
                                agg2[A[in[i]].group] += A[in[i]].measure;
                }

                gettimeofday(&t4, 0);

                // Segmented gather
                for (i = 0; i < rows; i++) {
                int segment_number = (in[i] >> segment_bits);
                int segment_offset = (in[i] & ((1U << segment_bits) - 1));
                                out2[i] = B[segment_number][segment_offset];
                }

                gettimeofday(&t5, 0);

                printf("Gather: %lu, Group: %lu, Ind Gather: %lu, Fused Gather Group: %lu\n"
               "Segmented Gather: %lu\n",
                                                diff(&t1, &t0), diff(&t2, &t1), diff(&t3, &t2), diff(&t4, &t3),
                        diff(&t5, &t4));

                for (i = 0; i < groups; i++) {
                                if (agg1[i] != agg2[i]) printf("Agg doesn't match: %d\n", i);
                }

                return 0;
}

