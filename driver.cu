#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// Includes CUDA
#include <cuda_runtime.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check

#define DIMENSION 512
#define INPUT_RAW inputData

// #define MIN_EPSILON_ERROR 5e-3f

// Auto-Verification Code
bool testResult = true;

// Perform Median Filter on data
__global__ void medianFilterKernel(float *inputData, float *outputData, int width, int height, int filterSize)
{

  const unsigned int windowSize = filterSize * filterSize;
  float window[DIMENSION];

  int iterator;

  // calculate normalized texture coordinates
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  //const unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x; 

  int radiusX = filterSize / 2;
  int radiusY = filterSize / 2;

  // if( (x >= (width - radiusX)) || (y >= height - radiusY) || (x == 0) || (y == 0)) return;
  if( (x >= (width - radiusX)) || (y >= height - radiusY) || (x == 0) || (y == 0)) {
    outputData[y * width + x] = inputData[y * width + x];
    return;
  }

  // --- Fill array private to the threads
  iterator = 0;
  for (int row = x - radiusX; row <= x + radiusX; row++) {
    for (int column = y - radiusY; column <= y + radiusY; column++) {
      if (iterator < windowSize && iterator >= 0) {
      window[iterator] = inputData[column * width + row];
      iterator++;
      }
    }
  }

  // --- Sort private array to find the median using Bubble Sort
  for (int i = 0; i <= (windowSize/2); ++i) {
    // --- Find the position of the minimum element
    int minVal = i;
    for (int l = i + 1; l < windowSize; ++l) if (window[l] < window[minVal]) minVal = l;

    // --- Put found minimum element in its place
    float temp = window[i];
    window[i] = window[minVal];
    window[minVal] = temp;
  }

  // --- Pick the middle one
  outputData[y * width + x] = window[windowSize/2]; 
}

////////////////////////////////////////////////////////////////////////////////
// Declaration, forward
void runTest(int argc, char **argv);

int main(int argc, char **argv) {

  runTest(argc, argv);
  cudaDeviceReset();

  exit(testResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

void runTest(int argc, char **argv) {
  int devID = findCudaDevice(argc, (const char **) argv);

  // Take input, if given
  int windowSize;
  const char *imageFilename = NULL;
  const char *outputFilename = NULL;
  if (argc == 4) {
    // Take Window size
    sscanf(argv[1], "%d", &windowSize);

    // Take Input file name
    imageFilename = argv[2];

    // Take Output file name
    outputFilename = argv[3];
  } else if (argc == 1){
    windowSize = 3;
    imageFilename = "lena.pgm";
    outputFilename = "lena_out.pgm";
  } else {
    printf("Usage: ./driver windowSize inputFile.pgm outputFile.pgm");
  }

  // load image from disk
  unsigned int width, height;
  char *imagePath = sdkFindFilePath(imageFilename, argv[0]);

  if (imagePath == NULL) {
    printf("Unable to source image file: %s\n", imageFilename);
    exit(EXIT_FAILURE);
  }

  float *inputData = NULL;
  sdkLoadPGM(imagePath, &inputData, &width, &height);

  unsigned int size = width * height * sizeof(float);
  printf("Loaded '%s', %d x %d pixels\n", imageFilename, width, height);

  // Copy input data to device
  float *hData = NULL;
  checkCudaErrors(cudaMalloc((void **) &hData, size));

  // Allocate device memory for result
  float *dData = NULL;
  checkCudaErrors(cudaMalloc((void **) &dData, size));

  checkCudaErrors(cudaMemcpy(hData, INPUT_RAW, size, cudaMemcpyHostToDevice));

  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

  // Warmup
  medianFilterKernel<<<dimGrid, dimBlock, 0>>>(hData, dData, width, height, windowSize);

  checkCudaErrors(cudaDeviceSynchronize());
  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);

  // Execute the kernel
  medianFilterKernel<<<dimGrid, dimBlock, 0>>>(hData, dData, width, height, windowSize);

  // Check if kernel execution generated an error
  getLastCudaError("Kernel execution failed");

  checkCudaErrors(cudaDeviceSynchronize());
  sdkStopTimer(&timer);
  printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
  printf("%.2f Mpixels/sec\n",
         (width *height / (sdkGetTimerValue(&timer) / 1000.0f)) / 1e6);
  sdkDeleteTimer(&timer);

  // Allocate mem for the result on host side
  float *hOutputData = (float *) malloc(size);
  
  // copy result from device to host
  checkCudaErrors(cudaMemcpy(hOutputData, dData, size, cudaMemcpyDeviceToHost));
  
  // Write result to file
  sdkSavePGM(outputFilename, hOutputData, width, height);
  printf("Wrote '%s'\n", outputFilename);
  
  // Write regression file if necessary
  if (checkCmdLineFlag(argc, (const char **) argv, "regression"))
  {
    // Write file for regression test
    sdkWriteFile<float>("./data/regression.dat", hOutputData, width*height, 0.0f, false);
  } else {
    // We need to reload the data from disk,
    // because it is inverted upon output
    sdkLoadPGM(outputFilename, &hOutputData, &width, &height);

    printf("Comparing files\n");
    printf("\toutput:    <%s>\n", outputFilename);
    
  }
  
  system("bin/standard 3 lena.pgm > lena_out.pgm");
  system("bin/diff lena.pgm lena_out.pgm");

  checkCudaErrors(cudaFree(dData));
  free(imagePath);
  free(inputData);
}