#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

//Variable declaration
#define TOTAL_ITERATIONS 1000000000
#define TEST_SIZE 1000
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

	long long int total_iterations = (long long int) TOTAL_ITERATIONS / (25 * (*inp).threadCount);
	if (strcmp((*inp).precisionType, "SP") == 0)
	{
		int total = 61;
		for (int j = 0; j < TEST_SIZE; j++)
		{
			for (long long int i = 0; i < total_iterations; i++)
			{
				total += (int) 1452 + 1432 - 4295 + 4134
				         + 4734 - 6843 + 6531 + 8774 + 520
				         + 1085 + 1546 + 8561 + 4510 - 17413
				         + 6241 + 452 - 913 + 7923 + 1521
				         + 8536 - 210 + 955 + 1630 - 6734;
			}
		}
	}
	else if (strcmp((*inp).precisionType, "DP") == 0)
	{
		double total = 40;
		for (int j = 0; j < TEST_SIZE; j++)
		{
			for (long long int i = 0; i < total_iterations; i++)
			{
				total += (double) 14452.23 + 1432.821 - 14295.1298 + 4134.894
				         + 34734.33 - 26843.65 + 165312.555 + 1521.8774 + 520.83
				         + 21085.74 + 1546.153 + 8561.9415 + 4510.4325 - 17413.672;
				total += (double) 6241.65841 + 452.85 - 913.74 + 7923.5684 + 1521.23;
				total += (double) 6.2 - 210.654 + 955.782 + 1630.9874 - 6734.564;
			}
		}
	}
	pthread_exit(NULL);
	return NULL;
}


int main(int argc, char *argv[]) {

	struct InputParameters *inp = malloc(sizeof(struct InputParameters));
	double theoGOps, efficiency;
	
	if (argc > 1)
	{
		strcpy((*inp).precisionType, argv[1]);
		(*inp).threadCount = atoi(argv[2]);
	}

	else
	{
		printf("Error: Insufficient input paraments. Specify Precision Type followed by Thread Count.\n Example: CPUBenchmark SP 2\n");
		exit(1);
	}
	
	printf("\nCommencing CPU Benchmark...\n");

	double total_time_taken[EXPERIMENT_FREQUENCY];
	double processor_speed[EXPERIMENT_FREQUENCY];


	for (int i = 0; i < EXPERIMENT_FREQUENCY; i++)
	{
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
		printf("ops : %lld\n", (long long int)TOTAL_ITERATIONS * TEST_SIZE);
		processor_speed[i] = (double)TEST_SIZE / total_time_taken[i];
		printf("processor_speed : %f\n", processor_speed[i]);
	}

	double avg_processor_speed = 0;
	for (int i = 0; i < EXPERIMENT_FREQUENCY; i++)
	{
		avg_processor_speed += processor_speed[i];
	}
	avg_processor_speed /= EXPERIMENT_FREQUENCY;

	if (strcmp((*inp).precisionType, "SP") == 0)
	{
		theoGOps = 147.20;
	}
	else if (strcmp((*inp).precisionType, "DP") == 0)
	{
		theoGOps = 294.4;
	}

	efficiency = avg_processor_speed * 100 / theoGOps;

	printf("PrecisionType \t\t ThreadCount \t\t AverageProcessorSpeed\n");
	printf("%s\t\t %d\t\t\t %.2f\t\n", (*inp).precisionType, (*inp).threadCount, avg_processor_speed);

	FILE *outputFilePointer;
	if (stat("./output", &st) == -1) {
   		 mkdir("./output", 0700);
	}
	outputFilePointer = fopen("./output/output.txt" , "a");
	char outputSTR[1024];
	sprintf(outputSTR, "%s \t %d \t %f \t %f \t %f\n", (*inp).precisionType, (*inp).threadCount, avg_processor_speed, theoGOps, efficiency);
	fwrite(outputSTR, 1, strlen(outputSTR), outputFilePointer);
	fclose(outputFilePointer);

	return 0;
}

