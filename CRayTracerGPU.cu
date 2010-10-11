////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#include "CRayTracerGPU.h"



// Global variable representing CUDA texture for screen pixels
texture<float, 2, cudaReadModeElementType> g_screenTexture;
// Array representing CUDA texture pixels
cudaArray* g_screenArray;

__global__ void cuda_kernel_texture_2d(unsigned char* surface, int width, int height, size_t pitch)
{
    
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;
	
    // get a pointer to the pixel at (x,y)
    float* pixel = (float*)(surface + y*pitch) + 4*x;

	pixel[0] = 1; // red
	pixel[1] = 0; // green
	pixel[2] = 0; // blue
	pixel[3] = 1; // alpha
}


extern "C" 
void runKernels(void* surface, int width, int height, size_t pitch)
{
    cudaError_t error = cudaSuccess;

    dim3 Db = dim3( 16, 16 ); // block dimensions are fixed to be 256 threads
    dim3 Dg = dim3( (width+Db.x-1)/Db.x, (height+Db.y-1)/Db.y );
    
    cuda_kernel_texture_2d<<<Dg,Db>>>( (unsigned char*)surface, width, height, pitch);

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("cuda_kernel_texture_2d() failed to launch error = %d\n", error);
    }
}



E_COMPUTING_TYPE CRayTracerGPU::getType()
{
	return ECT_CUDA;
}

CColor CRayTracerGPU::traceRay( CRay& ray, int depthLevel )
{
	return CColor();
}