#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <unistd.h>

#define DIMENSION 512

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// Includes CUDA
#include <cuda_runtime.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check

#define INPUT(I,x,y) input##I[((y)*(512*3))+(x)*3]

// #define MIN_EPSILON_ERROR 5e-3f

////////////////////////////////////////////////////////////////////////////////
// Define the files that are to be save and the reference images for validation
// const char *imageFilename = "lena.pgm";
//const char *refFilename   = "ref_rotated.pgm";

// Declare texture reference for 2D float texture
// texture<float, 2, cudaReadModeElementType> tex;

// Auto-Verification Code
bool testResult = true;

////////////////////////////////////////////////////////////////////////////////
//! Perform Median Filter on data
//! @param outputData  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void medianFilterKernel(float *inputData, float *outputData, int width, int height, int filterSize)
{

 const unsigned short windowSize = filterSize * filterSize;
  // unsigned short window[windowSize];
 float *window = new float[windowSize];

 int iterator;

  // calculate normalized texture coordinates
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  //const unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x; 

  if( (x >= (width - 1)) || (y >= height - 1) || (x == 0) || (y == 0)) return;

  // --- Fill array private to the threads
  iterator = 0;
  for (int row = x - 1; row <= x + 1; row++) {
    for (int column = y - 1; column <= y + 1; column++) {
      window[iterator] = inputData[column * width + row];
      iterator++;
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

  free(window);
}

////////////////////////////////////////////////////////////////////////////////
// Declaration, forward
float load_buffer[DIMENSION*DIMENSION];
float* load(int fd){
    int ct0a=4; struct stat _fstat;
    if (fstat(fd, &_fstat) == -1) { perror("fstat()"); exit(1); }
    unsigned char *p=(unsigned char*)mmap(NULL, _fstat.st_size, PROT_READ|PROT_WRITE, MAP_PRIVATE, fd, 0);
    do{ p=(unsigned char*)memchr(p, 0x0a, 64)+1; } while(--ct0a);
    for(int i=0;i<DIMENSION*DIMENSION;++i) // abandon green/blue channels
      load_buffer[i]=(float)p[i*3]/255.0f; // this is what sdkLoadPGM acutally does according to hexdump
    return load_buffer;
}
void runTest(int argc, char **argv);

int main(int argc, char **argv) {
#if 0 // This does not work. Why? Only supports P5 format?
  float *inputData = NULL; unsigned width, height;
  sdkLoadPGM(sdkFindFilePath("lena.pgm", argv[0]), &inputData, &width, &height);
  sdkSavePGM("lena_out.pgm", inputData, width, height);
  exit(0);
#endif

  //printf("%s starting...\n", sampleName);

  // Process command-line arguments
  // if (argc == 1) {
    // if (checkCmdLineFlag(argc, (const char **) argv, "input")) {
    //   getCmdLineArgumentString(argc, (const char **) argv, "input", (char **) &imageFilename);

    //   // if (checkCmdLineFlag(argc, (const char **) argv, "reference")) {
    //   //     getCmdLineArgumentString(argc,
    //   //                              (const char **) argv,
    //   //                              "reference",
    //   //                              (char **) &refFilename);
    // } else {
    //   printf("-input flag should be used");
    //   exit(EXIT_FAILURE);
    // }
  // }

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

  printf("ARGC: %d\n", argc);
  printf("ARGV[0]: %s\n", argv[0]);
  printf("ARGV[1]: %s\n", argv[1]);
  printf("IMAGEPATH: %s\n", imagePath);

  if (imagePath == NULL) {
    printf("Unable to source image file: %s\n", imageFilename);
    exit(EXIT_FAILURE);
  }

#if 1
#define INPUT_RAW inputData
  float *inputData = NULL;
  sdkLoadPGM(imagePath, &inputData, &width, &height);
#else
#define INPUT_RAW input1
  float *input1=load(open("lena.ppm", O_RDONLY));
  width = height = DIMENSION;
#endif
  // {
  //   FILE * pFile;
  //   pFile = fopen ("inputDataFile.txt", "wb");
  //   printf("input[0]: %f\n", INPUT_RAW[0]);
  //   fwrite(INPUT_RAW, 128, 1, pFile);
  //   fclose(pFile);
  // }


  unsigned int size = width * height * sizeof(float);
  printf("Loaded '%s', %d x %d pixels\n", imageFilename, width, height);

  //Load reference image from image (output)
  // float *hDataRef = (float *) malloc(size);
  // char *refPath = sdkFindFilePath(refFilename, argv[0]);

  // if (refPath == NULL)
  // {
  //     printf("Unable to find reference image file: %s\n", refFilename);
  //     exit(EXIT_FAILURE);
  // }

  // sdkLoadPGM(refPath, &hDataRef, &width, &height);

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
  // transformKernel<<<dimGrid, dimBlock, 0>>>(dData, width, height, angle);
  // medianFilterKernel<<<dimGrid, dimBlock, 0>>>(hData, dData, width, height, 3);

  checkCudaErrors(cudaDeviceSynchronize());
  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);

  // Execute the kernel
  //transformKernel<<<dimGrid, dimBlock, 0>>>(dData, width, height, angle);
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
    // printf("\treference: <%s>\n", refPath);

    // testResult = compareData(hOutputData, hDataRef, width*height, MAX_EPSILON_ERROR, 0.15f);
  }

  checkCudaErrors(cudaFree(dData));
  // checkCudaErrors(cudaFreeArray(cuArray));
  free(imagePath);
  // free(refPath);
  free(inputData);
  // free(imageFilename);
  // free(outputFilename);
}