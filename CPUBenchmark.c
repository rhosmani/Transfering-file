#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <immintrin.h>

#define ITERATION1 1000000
#define ITERATION2 1000000
#define EXPERIMENT_FREQUENCY 3

struct stat st = {0};
struct InputParameters
{
	char precisionType[3];  // "SP\0" Or "DP\0"
	int threadCount;		// 1 2 4 8
};

void *computeArithmeticOperations(void *param)
{
	struct InputParameters *inp = (struct InputParameters *) param;
	long long int total_iterations = (long long int) ITERATION1 / (10 * (*inp).threadCount); 

	__m256 set1 = _mm256_set_ps(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0);
	__m256 set2 = _mm256_set_ps(1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0);
	__m256 result;

	__m256d set1d = _mm256_set_pd(2.21, 26.4, 214.4, 46.4);
	__m256d set2d = _mm256_set_pd(65.1, 55.4, 15.1, 25.5);
	__m256d resultd;

	if (strcmp((*inp).precisionType, "SP") == 0)
	{
		for (int j = 0; j < ITERATION2; j++)
		{
			for (long long int i = 0; i < total_iterations; i++)
			{
				result = _mm256_add_ps(set1, set2);
			}
		}

	}
	else if (strcmp((*inp).precisionType, "DP") == 0)
	{
		for (int j = 0; j < ITERATION2; j++)
		{
			for (long long int i = 0; i < total_iterations; i++)
			{
				resultd = _mm256_add_pd(set1d, set2d);
			}
		}
	}
	pthread_exit(NULL);
	return NULL;
}


int main(int argc, char *argv[]) {

	struct InputParameters *inp = malloc(sizeof(struct InputParameters));
	double th_Gops, efficiency;

	if (argc > 1)
	{
		strcpy((*inp).precisionType, argv[1]);
		(*inp).threadCount = atoi(argv[2]);
	}

	else
	{
		printf("Error: Insufficient input paraments. Specify Precision Type followed by Thread Count.\n Example: CPUBenchmark SP 1\n");
		exit(1);
	}

	printf("\nCommencing CPU Benchmark...\n");

	printf("\nPrecisionType: %s \t Thread #: %d \n", (*inp).precisionType, (*inp).threadCount );

	double total_time_taken[EXPERIMENT_FREQUENCY];
	double throughput[EXPERIMENT_FREQUENCY];


	for (int i = 0; i < EXPERIMENT_FREQUENCY; i++)
	{
		printf("\nExperiment Number: %d\n", i + 1);
		struct timeval process_start_time, process_end_time;
		pthread_t threadIdList[(*inp).threadCount];
		gettimeofday(&process_start_time, NULL);

		for (int i = 0; i < (*inp).threadCount ; ++i)
			pthread_create(&threadIdList[i], NULL, computeArithmeticOperations, (void *) inp);

		for (int i = 0; i < (*inp).threadCount ; i++)
			pthread_join(threadIdList[i], NULL);

		gettimeofday(&process_end_time, NULL);


		total_time_taken[i] = (float) (process_end_time.tv_usec - process_start_time.tv_usec) / 1000000 + (float) (process_end_time.tv_sec - process_start_time.tv_sec);

		printf("Total time : %f\n", total_time_taken[i]);
		printf("Operations per Second : %lld\n", (long long int)ITERATION1 * ITERATION2);
		throughput[i] = (double) 1000 / total_time_taken[i];			//----- Converting ops to Gops = total iterations/ (total time * 10^ 9)
		printf("CPUBench : %f Gops\n", throughput[i]);
	}

	double avg_throughput = 0;
	for (int i = 0; i < EXPERIMENT_FREQUENCY; i++)
	{
		avg_throughput += throughput[i];
	}
	avg_throughput /= EXPERIMENT_FREQUENCY;

	if (strcmp((*inp).precisionType, "SP") == 0)
	{
		th_Gops = 147.2;
	}
	else if (strcmp((*inp).precisionType, "DP") == 0)
	{
		th_Gops = 73.6;
	}

	efficiency = (avg_throughput / th_Gops) * 100;

	printf("PrecisionType \t\t ThreadCount \t\t AverageProcessorSpeed\n");
	printf("%s\t\t %d\t\t\t %.2f\t\n", (*inp).precisionType, (*inp).threadCount, avg_throughput);

	FILE *outputFilePointer;
	if (stat("./output", &st) == -1) {
		mkdir("./output", 0700);
	}
	outputFilePointer = fopen("./output/output.txt" , "a");
	char outputSTR[1024];
	sprintf(outputSTR, "%s \t %d \t %f \t %f \t %f\n", (*inp).precisionType, (*inp).threadCount, avg_throughput, th_Gops, efficiency);
	fwrite(outputSTR, 1, strlen(outputSTR), outputFilePointer);
	fclose(outputFilePointer);

	return 0;
}

