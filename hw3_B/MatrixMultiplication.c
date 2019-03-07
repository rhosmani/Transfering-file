#include <stdio.h>
#include<stdlib.h>
#include<pthread.h>
#include <string.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#define MAX_SP 22360
#define MAX_DP 15808
#define EXPERIMENT_FREQUENCY 3

const static int SPMatrixSize = 16384;
const static int DPMatrixSize = 16384;

#define MATRIX_SIZE 16384

// Initializing matrices
float *matrixSP1[MATRIX_SIZE];
float *matrixSP2[MATRIX_SIZE];
float *resultSPMatrix[MATRIX_SIZE];

double *matrixDP1[MATRIX_SIZE];
double *matrixDP2[MATRIX_SIZE];
double *resultDPMatrix[MATRIX_SIZE];

double randomNumberGen(int n) {
   return ((n / 9.0) * (2.0)) - 1.0;
}
struct stat st = {0};
static int step = 0;

struct InputParameters
{
   char precisionType[3];  // "SP\0" Or "DP\0"
   int threadCount;     // 1 2 4 8
};


void *matrixMultiplication(void *arg)      // each thread
{
   struct InputParameters *p = (struct InputParameters *) arg;
   int iter = step++;
   double r;
   int head_val = ((iter * SPMatrixSize) / ((*p).threadCount));
   int tail_val = ((iter + 1) * SPMatrixSize) / ((*p).threadCount);
   
   if (strcmp((*p).precisionType, "SP") == 0)
   {
      for (int i = head_val; i < tail_val; ++i)
         for (int j = 0; j < SPMatrixSize; ++j)
            for (int k = 0; k < SPMatrixSize; ++k )
               resultSPMatrix[i][j] += matrixSP1[i][k] * matrixSP2[k][j];

      r = (float) resultSPMatrix[head_val][tail_val - 1];
   }
   else if (strcmp((*p).precisionType, "DP") == 0)
   {
      for (int i = head_val; i < tail_val; ++i)
         for (int j = 0; j < DPMatrixSize; ++j)
            for (int k = 0; k < DPMatrixSize; ++k)
               resultDPMatrix[i][j] += matrixDP1[i][k] * matrixDP2[k][j];
      r = resultDPMatrix[head_val][tail_val - 1];
   }
   pthread_exit(NULL);
   return NULL;
}

int main(int argc, char *argv[])
{

   struct InputParameters *inp = malloc(sizeof(struct InputParameters));
   double th_Gops, efficiency;

   if (argc > 1)
   {
      strcpy((*inp).precisionType, argv[1]);
      (*inp).threadCount = atoi(argv[2]);
   }

   else
   {
      printf("Error: Insufficient input paraments. Specify Precision Type followed by Thread Count.\n Example: MatrixBenchmark SP 1\n");
      exit(1);
   }


   if (strcmp((*inp).precisionType, "SP") == 0)
   {

      //Creating SP Matrix
      for (int i = 0; i < SPMatrixSize; ++i) {
         matrixSP1[i] = (float *)malloc (SPMatrixSize * sizeof(float));
         matrixSP2[i] = (float *)malloc (SPMatrixSize * sizeof(float));
      }

      for (int i = 0; i < SPMatrixSize; ++i) 
            resultSPMatrix[i] = (float *)malloc(SPMatrixSize * sizeof(float));

      //Intitalizing SP Matrix
      for (int i = 0; i < SPMatrixSize; ++i)
      {
         for (int j = 0; j < SPMatrixSize; ++j)
         {
            matrixSP1[i][j] = (float)randomNumberGen(rand() % 10);
            matrixSP2[i][j] = (float)randomNumberGen(rand() % 10);
         }
      }

   }
   else if (strcmp((*inp).precisionType, "DP") == 0)
   {
      //Creating DP Matrix
      for (int i = 0; i < DPMatrixSize; ++i) {
         matrixDP1[i] = (double *)malloc (DPMatrixSize * sizeof(double));
         matrixDP2[i] = (double *)malloc (DPMatrixSize * sizeof(double));
      }

      for (int i = 0; i < DPMatrixSize; ++i) 
            resultDPMatrix[i] = (double *)malloc(DPMatrixSize * sizeof(double));


      //Intitalizing DP Matrix
      for (int i = 0; i < DPMatrixSize; ++i)
      {
         for (int j = 0; j < DPMatrixSize; ++j)
         {
            matrixDP1[i][j] = (double)randomNumberGen(rand() % 10);
            matrixDP2[i][j] = (double)randomNumberGen(rand() % 10);
         }
      }
   }

   printf("\nCommencing Matrix Multiplication Benchmark...\n");
   printf("\nPrecisionType: %s \t Thread #: %d \n", (*inp).precisionType, (*inp).threadCount );

   double total_time_taken[EXPERIMENT_FREQUENCY];
   long double throughput[EXPERIMENT_FREQUENCY];
   int n = SPMatrixSize;

   for (int i = 0; i < EXPERIMENT_FREQUENCY; i++)
   {
      step = 0;
      printf("\nExperiment Number: %d\n", i + 1);
      struct timeval process_start_time, process_end_time;
      pthread_t threadIdList[(*inp).threadCount];
      gettimeofday(&process_start_time, NULL);

      for (int i = 0; i < (*inp).threadCount ; ++i)
         pthread_create(&threadIdList[i], NULL, matrixMultiplication, (void *) inp);

      for (int i = 0; i < (*inp).threadCount ; i++)
         pthread_join(threadIdList[i], NULL);

      gettimeofday(&process_end_time, NULL);
      total_time_taken[i] = (float) (process_end_time.tv_usec - process_start_time.tv_usec) / 1000000 + (float) (process_end_time.tv_sec - process_start_time.tv_sec);

      printf("Total time : %f\n", total_time_taken[i]);
      long double ops = (long double)(2 * n * n * n)/total_time_taken[i];//Matrix Bench 2n^3
      printf("Operations per Second : %Lf\n", ops);  
      throughput[i] = (double) ops / 1000000000;        
      printf("MatrixBench : %Lf Gops\n", throughput[i]);

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