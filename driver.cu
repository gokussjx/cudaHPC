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

// Exit true
bool testResult = true;

// Median Filter KERNEL
__global__ void medianFilterKernel(float *inputData, float *outputData, int width, int height, int filterSize)
{

  // Specify window size
  const unsigned int windowSize = filterSize * filterSize;
  float window[DIMENSION];

  // calculate normalized texture coordinates
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  int radiusX = filterSize / 2;
  int radiusY = filterSize / 2;

  // if( (x >= (width - radiusX)) || (y >= height - radiusY) || (x == 0) || (y == 0)) return;
  if( (x >= (width - radiusX)) || (y >= height - radiusY) || (x == 0) || (y == 0)) {
    outputData[y * width + x] = inputData[y * width + x];
    return;
  }

  // --- Fill array private to the threads
  int iterator = 0;
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

  // --- Pick the middle one. It has the median!
  outputData[y * width + x] = window[windowSize/2]; 
}

// Forward Declaration
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

  // Load image data to variable, store width and height
  float *inputData = NULL;
  sdkLoadPGM(imagePath, &inputData, &width, &height);

  // Allocate size
  unsigned int size = width * height * sizeof(float);
  printf("Loaded '%s', %d x %d pixels\n", imageFilename, width, height);

  // Copy input data to device
  float *hData = NULL;
  checkCudaErrors(cudaMalloc((void **) &hData, size));

  // Allocate device memory for result
  float *dData = NULL;
  checkCudaErrors(cudaMalloc((void **) &dData, size));

  // Copy image data from host to device
  checkCudaErrors(cudaMemcpy(hData, INPUT_RAW, size, cudaMemcpyHostToDevice));

  // Timing analysis loops
  short windowSizeHolder[] = {3, 7, 11, 15};
  short blockSizeHolder[] = {8, 16};

  // Loop blockSize: 8 or 16
  for(short blockSizeIndex = 0; blockSizeIndex < 2; blockSizeIndex++) {
    // Loop windowSize: 3, 7, 11, 15
    for(short windowSizeIndex = 0; windowSizeIndex < 4; windowSizeIndex++) {
      // Iterate each, 10 times
      for(short loop = 0; loop < 10; loop++) {

        // Specify block and grid dimensions
        dim3 dimBlock(blockSizeHolder[blockSizeIndex], blockSizeHolder[blockSizeIndex], 1);
        dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

        // Warmup
        medianFilterKernel<<<dimGrid, dimBlock, 0>>>(hData, dData, width, height, windowSizeHolder[windowSizeIndex]);

        // Synchronize, and start timer
        checkCudaErrors(cudaDeviceSynchronize());
        StopWatchInterface *timer = NULL;
        sdkCreateTimer(&timer);
        sdkStartTimer(&timer);
        // Execute the kernel
        medianFilterKernel<<<dimGrid, dimBlock, 0>>>(hData, dData, width, height, windowSizeHolder[windowSizeIndex]);

        // Check if kernel execution generated an error
        getLastCudaError("Kernel execution failed");

        // Synchronize, and stop timer
        checkCudaErrors(cudaDeviceSynchronize());
        sdkStopTimer(&timer);
        printf("Processing time: %fms\n", sdkGetTimerValue(&timer));
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
      }

      // Ask Golden function to generate its output, compare and provide match info
      char goldenFunction[2048];
      sprintf(goldenFunction, "bin/standard %d lena.pgm > lena_out_gold.pgm", windowSizeHolder[windowSizeIndex]);
      system(goldenFunction);
      system("bin/diff lena_out_gold.pgm lena_out.pgm");      
    }
  }

  // Free data
  checkCudaErrors(cudaFree(dData));
  free(imagePath);
  free(inputData);
}