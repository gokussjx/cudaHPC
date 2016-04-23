#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

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

#define MIN_EPSILON_ERROR 5e-3f

////////////////////////////////////////////////////////////////////////////////
// Define the files that are to be save and the reference images for validation
const char *imageFilename = "lena.ppm";
//const char *refFilename   = "ref_rotated.pgm";

// Declare texture reference for 2D float texture
texture<float, 2, cudaReadModeElementType> tex;

// Auto-Verification Code
bool testResult = true;

////////////////////////////////////////////////////////////////////////////////
// Kernels
////////////////////////////////////////////////////////////////////////////////
//! Write to a cuArray (texture data source) using surface writes
//! @param gIData input data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void surfaceWriteKernel(float *gIData, int width, int height) {
    // calculate surface coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // read from global memory and write to cuarray (via surface reference)
    surf2Dwrite(gIData[y * width + x],
      outputSurface, x * 4, y, cudaBoundaryModeTrap);
}

int main(int argc, char **argv) {

  printf("%s starting...\n", sampleName);

  // Process command-line arguments
  if (argc == 1) {
    if (checkCmdLineFlag(argc, (const char **) argv, "input")) {
        getCmdLineArgumentString(argc,
                                 (const char **) argv,
                                 "input",
                                 (char **) &imageFilename);

        if (checkCmdLineFlag(argc, (const char **) argv, "reference")) {
            getCmdLineArgumentString(argc,
                                     (const char **) argv,
                                     "reference",
                                     (char **) &refFilename);
      } else {
            printf("-input flag should be used");
            exit(EXIT_FAILURE);
      }
    } else {
        printf("-reference flag should be used with -input flag");
        exit(EXIT_FAILURE);
    }
  }

  runTest(argc, argv);
  cudaDeviceReset();

  exit(testResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

void runTest(int argc, char **argv) {
    int devID = findCudaDevice(argc, (const char **) argv);

    // load image from disk
    float *hData = NULL;
    unsigned int width, height;
    char *imagePath = sdkFindFilePath(imageFilename, argv[0]);

    if (imagePath == NULL) {
        printf("Unable to source image file: %s\n", imageFilename);
        exit(EXIT_FAILURE);
    }

    sdkLoadPGM(imagePath, &hData, &width, &height);

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

    // Allocate device memory for result
    float *dData = NULL;
    checkCudaErrors(cudaMalloc((void **) &dData, size));

    // Allocate array and copy image data
    cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray *cuArray;
    checkCudaErrors(cudaMallocArray(&cuArray,
                                    &channelDesc,
                                    width,
                                    height));
    checkCudaErrors(cudaMemcpyToArray(cuArray,
                                      0,
                                      0,
                                      hData,
                                      size,
                                      cudaMemcpyHostToDevice));

    // Set texture parameters
    tex.addressMode[0] = cudaAddressModeWrap;
    tex.addressMode[1] = cudaAddressModeWrap;
    tex.filterMode = cudaFilterModeLinear;
    tex.normalized = true;    // access with normalized texture coordinates

    // Bind the array to the texture
    checkCudaErrors(cudaBindTextureToArray(tex, cuArray, channelDesc));

    dim3 dimBlock(8, 8, 1);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

    // Warmup
    transformKernel<<<dimGrid, dimBlock, 0>>>(dData, width, height, angle);

    checkCudaErrors(cudaDeviceSynchronize());
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    // Execute the kernel
    transformKernel<<<dimGrid, dimBlock, 0>>>(dData, width, height, angle);

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
    checkCudaErrors(cudaMemcpy(hOutputData,
                               dData,
                               size,
                               cudaMemcpyDeviceToHost));

    // Write result to file
    char outputFilename[1024];
    strcpy(outputFilename, imagePath);
    strcpy(outputFilename + strlen(imagePath) - 4, "_out.pgm");
    sdkSavePGM(outputFilename, hOutputData, width, height);
    printf("Wrote '%s'\n", outputFilename);

    // Write regression file if necessary
    if (checkCmdLineFlag(argc, (const char **) argv, "regression"))
    {
        // Write file for regression test
        sdkWriteFile<float>("./data/regression.dat",
                            hOutputData,
                            width*height,
                            0.0f,
                            false);
    }
    else
    {
        // We need to reload the data from disk,
        // because it is inverted upon output
        sdkLoadPGM(outputFilename, &hOutputData, &width, &height);

        printf("Comparing files\n");
        printf("\toutput:    <%s>\n", outputFilename);
        printf("\treference: <%s>\n", refPath);

        testResult = compareData(hOutputData,
                                 hDataRef,
                                 width*height,
                                 MAX_EPSILON_ERROR,
                                 0.15f);
    }

    checkCudaErrors(cudaFree(dData));
    checkCudaErrors(cudaFreeArray(cuArray));
    free(imagePath);
    free(refPath);
}