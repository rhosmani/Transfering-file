#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#define WORK_LOAD 1000000000
#define TEST_SIZE 100
#define EXPERIMENT_FREQUENCY 3

struct stat st = {0};
struct InputParameters
{
	char benchmarkType[4];  // "RWR\0" Or "RWS\0"
	int threadCount;		// 1 2 4 8
	long int block_size;
	char *data;
};

void *read_write_sequential(void *arg)
{
	struct InputParameters *inp = (struct InputParameters*) arg;
	char *outputDataBlock;
	long long int totalIterations;
	
	
	if((*inp).block_size == 1) //For calculating latency
	{
		totalIterations = (long long int) 1000000 / (long long int) ((*inp).block_size * (*inp).threadCount);	
		outputDataBlock = malloc((long long int) 1000000  * sizeof(char));
	}
	else
	{
		totalIterations = (long long int) WORK_LOAD / (long long int) ((*inp).block_size * (*inp).threadCount);	
		outputDataBlock = malloc((long long int) WORK_LOAD  * sizeof(char));
	}
	
	for(int i = 0; i < TEST_SIZE; i++)
	{

		if((*inp).block_size == 1)                     //Memory Latency
			outputDataBlock = malloc((long long int) 1000000  * sizeof(char));
		else
			outputDataBlock = malloc((long long int) WORK_LOAD  * sizeof(char));
		
		for(long long int j = 0; j < totalIterations; j++)
		{
			memcpy(outputDataBlock + (j * (*inp).block_size), (*inp).data + (j * (*inp).block_size) , (*inp).block_size);
		}
		free(outputDataBlock);	
	}
	pthread_exit(NULL);
	return NULL;
}

void *read_write_random(void *arg)
{
	struct InputParameters *inp = (struct InputParameters*) arg;
	char *outputDataBlock;
	outputDataBlock = malloc((long long int) WORK_LOAD  * sizeof(char));
	long long int totalIterations;
	
	if((*inp).block_size == 1)
		totalIterations = (long long int) 1000000 / (long long int) ((*inp).block_size * (*inp).threadCount);	
	else
		totalIterations = (long long int) WORK_LOAD / (long long int) ((*inp).block_size * (*inp).threadCount);	
	
	for(int i = 0; i < TEST_SIZE; i++)
	{
	outputDataBlock = malloc((long long int) WORK_LOAD  * sizeof(char));
		for(long long int j = 0; j < totalIterations; j++)
		{
			long long int l = rand() % totalIterations;
			memcpy(outputDataBlock + (l * (*inp).block_size), (*inp).data + (l * (*inp).block_size) , (*inp).block_size);
		}
	free(outputDataBlock);	
	}
	pthread_exit(NULL);
	return NULL;
}


int main(int argc, char *argv[]){

	struct InputParameters *inp = malloc(sizeof(struct InputParameters));
	double throughput[EXPERIMENT_FREQUENCY],latency[EXPERIMENT_FREQUENCY];
	float total_time_taken;
	double avg_throughput = 0;
	double avg_latency = 0;
	
	if (argc > 1)
	{
		strcpy((*inp).benchmarkType, argv[1]);
		(*inp).block_size = atoi(argv[2]);
		(*inp).threadCount = atoi(argv[3]);
	}

	else
	{
		printf("Error: Insufficient input paraments. Specify Benchmark Type, block_size followed by Thread Count.\n 
			myRAMBench <benchmarkType> <block_size> <thread_Count>\n
			Example: myRAMBench RWS 10 2\n");
		exit(1);
	}

	printf("\nBenchmark Type: %s \t Block Size: %ld \t Threads: %d\n",(*inp).benchmarkType,(*inp).block_size,(*inp).threadCount);


	for(int exp_no = 0; exp_no < EXPERIMENT_FREQUENCY; exp_no++)
	{
		printf("Experiment number: %d\n", exp_no+1);
		(*inp).data = malloc((long int)WORK_LOAD * sizeof(char));
		memset((*inp).data , '\0', (long int) WORK_LOAD);   //Copying '\0' in data for WORK_LOAD times
		
		struct timeval process_start_time, process_end_time;

		if( strcmp((*inp).benchmarkType , "RWS") == 0)				//----------------RWS
		{	
			pthread_t threadslist[(*inp).threadCount];
			gettimeofday(&process_start_time, NULL);
			for(int i = 0; i < (*inp).threadCount; i++)
				pthread_create(&threadslist[i], NULL, read_write_sequential, (void *) inp);
			for(int i = 0; i < (*inp).threadCount; i++)
				pthread_join(threadslist[i], NULL);
			gettimeofday(&process_end_time, NULL);
		
			total_time_taken = (float) (process_end_time.tv_usec - process_start_time.tv_usec) / 1000000 + (float) (process_end_time.tv_sec - process_start_time.tv_sec);
		}
		else if(strcmp((*inp).benchmarkType , "RWR") == 0)														//-----------------RWR
		{
			pthread_t threadslist[(*inp).threadCount];
			gettimeofday(&process_start_time, NULL);
			for(int i = 0; i < (*inp).threadCount; i++)
				pthread_create(&threadslist[i], NULL, read_write_random, (void *) inp);
			for(int i = 0; i < (*inp).threadCount; i++)
				pthread_join(threadslist[i], NULL);
			gettimeofday(&process_end_time, NULL);

			total_time_taken = (float) (process_end_time.tv_usec - process_start_time.tv_usec) / 1000000 + (float) (process_end_time.tv_sec - process_start_time.tv_sec);	// in seconds
		}

		if((*inp).block_size == 1)
		{
			latency[exp_no] = (double) (total_time_taken * 1000000) / (TEST_SIZE * 1000000);			
			printf("\nTotal time taken : %f\tLatency : %f\n",total_time_taken, latency[exp_no]);
			avg_latency += latency[exp_no];
		}
		else
		{
			throughput[exp_no] = (double) WORK_LOAD / ( total_time_taken * WORK_LOAD);
			throughput[exp_no] *= TEST_SIZE; //-----100GB data processed
			printf("\nTotal time taken : %f\tThroughput: %f\n",total_time_taken, throughput[exp_no]);
			avg_throughput += throughput[exp_no];
		}
		free((*inp).data);
	}

	double MyRAMBench_throughput;
	double MyRAMBench_latency;
	double theoretical_throughput;
	double theoretical_latency;
	double MyRAMBenchEfficiency;
	if((*inp).block_size == 1)
	{
		avg_latency /= EXPERIMENT_FREQUENCY;
		MyRAMBench_latency = avg_latency;
		theoretical_latency = 0.015;		
		MyRAMBenchEfficiency = ((theoretical_latency - MyRAMBench_latency)/theoretical_latency) * 100;
		printf("\nBenchmark Type: %s \t Threads: %d \t Block Size: %ld \t Experiment RAM Benchmark (Latency): %f \t Theoretical Latency: %f \t RAM Benchmark Efficiency: %f\n", (*inp).benchmarkType, (*inp).threadCount, (*inp).block_size, MyRAMBench_latency, theoretical_latency, MyRAMBenchEfficiency);
	}
	else
	{
		avg_throughput /= EXPERIMENT_FREQUENCY;
		MyRAMBench_throughput = avg_throughput;
		theoretical_throughput = (((2299.998 * 2 * 64 * 2)/ 8) / 1000);
		MyRAMBenchEfficiency = (MyRAMBench_throughput / theoretical_throughput) * 100;
		printf("\nBenchmark Type: %s \t Threads: %d \t Block Size: %ld \t Experiment RAM Benchmark (Throughput): %f \t Theoretical Throughput: %f \t RAM Benchmark Efficiency: %f\n", (*inp).benchmarkType, (*inp).threadCount, (*inp).block_size, MyRAMBench_throughput, theoretical_throughput, MyRAMBenchEfficiency);
	
	}
	

	FILE *outputFilePointer;
	if (stat("./output", &st) == -1) {
		mkdir("./output", 0700);
	}
	outputFilePointer = fopen("./output/myRAMBench_output.txt" , "a");
	char outputSTR[1024]
;	
	if((*inp).block_size == 1)
		sprintf(outputSTR, "%s %d %ld %f %f %f\n", (*inp).benchmarkType, (*inp).threadCount, (*inp).block_size, MyRAMBench_latency, theoretical_latency, MyRAMBenchEfficiency);
	else
		sprintf(outputSTR, "%s %d %ld %f %f %f\n", (*inp).benchmarkType, (*inp).threadCount, (*inp).block_size, MyRAMBench_throughput, theoretical_throughput, MyRAMBenchEfficiency);
	fwrite(outputSTR, 1, strlen(outputSTR), outputFilePointer);
	fclose(outputFilePointer);
	return 0;
}