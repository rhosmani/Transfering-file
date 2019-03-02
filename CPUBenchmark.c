#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#define ITERATION1 1000000
#define ITERATION2 1000000
#define EXPERIMENT_FREQUENCY 3

struct stat st = {0};
struct InputParameters
{
	char precisionType[3];  // "SP\0" Or "DP\0"
	int threadCount;		// 1 2 4 8
};

float a = 10.4, b = 15.6, c = 25.4, d = 65.4, e = 4643.45, f = 44.45, g = 45.4, h = 946.1, j = 464.4, k = 43.14;
double aa = 464.446, bb = 789.444, cc = 7823.44, dd = 0.22564, ee = 7464.46, ff = 7464.4, gg = 13.111, hh = 48.4156, jj = 464.164, kk = 5464.44;


void *computeArithmeticOperations(void *param)
{
	struct InputParameters *inp = (struct InputParameters *) param;
	long long int total_iterations = (long long int) ITERATION1 / (10 * (*inp).threadCount);

	if (strcmp((*inp).precisionType, "SP") == 0)
	{
		for (int j = 0; j < ITERATION2; j++)
		{
			float total = 1.111;
			for (long long int i = 0; i < total_iterations; i++)
			{
				total = total + a + b - c * d + e - f * g - h + j * k  ;
			}
		}
	}
	else if (strcmp((*inp).precisionType, "DP") == 0)
	{
		for (int j = 0; j < ITERATION2; j++)
		{
			double total = 1.2131;
			for (long long int i = 0; i < total_iterations; i++)
			{
				total = total + aa + bb - cc * dd + ee - ff * gg - hh + jj * kk  ;
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
		printf("Error: Insufficient input paraments. Specify Precision Type followed by Thread Count.\n Example: CPUBenchmark SP 2\n");
		exit(1);
	}

	printf("\nCommencing CPU Benchmark...\n");

	printf("\nPrecisionType: %s \t Thread #: %d \n", (*inp).precisionType, (*inp).threadCount );

	double total_time_taken[EXPERIMENT_FREQUENCY];
	double processor_speed[EXPERIMENT_FREQUENCY];


	for (int i = 0; i < EXPERIMENT_FREQUENCY; i++)
	{
		printf("\nExperiment Number: %d\n",i+1);
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
		processor_speed[i] = (double) 1000 / total_time_taken[i];			//----- Converting ops to Gops = total iterations/ (total time * 10^ 9) 
		printf("Processor_speed : %f Gops\n", processor_speed[i]);
	}

	double avg_processor_speed = 0;
	for (int i = 0; i < EXPERIMENT_FREQUENCY; i++)
	{
		avg_processor_speed += processor_speed[i];
	}
	avg_processor_speed /= EXPERIMENT_FREQUENCY;

	if (strcmp((*inp).precisionType, "SP") == 0)
	{
		th_Gops = 294.4; 
	}
	else if (strcmp((*inp).precisionType, "DP") == 0)
	{
		th_Gops = 147.2;
	}

	efficiency = (avg_processor_speed / th_Gops) * 100;

	printf("PrecisionType \t\t ThreadCount \t\t AverageProcessorSpeed\n");
	printf("%s\t\t %d\t\t\t %.2f\t\n", (*inp).precisionType, (*inp).threadCount, avg_processor_speed);

	FILE *outputFilePointer;
	if (stat("./output", &st) == -1) {
		mkdir("./output", 0700);
	}
	outputFilePointer = fopen("./output/output.txt" , "a");
	char outputSTR[1024];
	sprintf(outputSTR, "%s \t %d \t %f \t %f \t %f\n", (*inp).precisionType, (*inp).threadCount, avg_processor_speed, th_Gops, efficiency);
	fwrite(outputSTR, 1, strlen(outputSTR), outputFilePointer);
	fclose(outputFilePointer);

	return 0;
}

