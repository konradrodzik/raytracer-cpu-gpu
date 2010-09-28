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
    float* pixel;

	if (x >= width || y >= height) return;
	
    // get a pointer to the pixel at (x,y)
    pixel = (float*)(surface + y*pitch) + 4*x;

	pixel[0] = 1;	// red
	pixel[1] = 1;	// green
	pixel[2] = 0;	// blue
	pixel[3] = 255; // alpha
}

CRayTracerGPU::CRayTracerGPU( int width, int height )
: CRayTracer(width, height)
{

}

E_COMPUTING_TYPE CRayTracerGPU::getType()
{
	return ECT_CUDA;
}

void CRayTracerGPU::calculateScene()
{
	//cudaArray *cuArray;
	//cudaGraphicsSubResourceGetMappedArray( &cuArray, m_cudatextureWrapper.cudaResource, 0, 0);

		


	int N=800*600;
	int block_size = 100;
	int n_blocks = N/block_size + (N%block_size == 0 ? 0:1);
	

	cuda_kernel_texture_2d <<< n_blocks,block_size >>> ((unsigned char*)m_cudatextureWrapper.cudaLinearMemory, m_cudatextureWrapper.width, m_cudatextureWrapper.height, m_cudatextureWrapper.pitch);


	// then we want to copy cudaLinearMemory to the D3D texture, via its mapped form : cudaArray
	//cudaMemcpyToArray(cuArray, 0, 0, m_cudatextureWrapper.cudaLinearMemory, m_cudatextureWrapper.pitch * m_cudatextureWrapper.height, cudaMemcpyDeviceToDevice);
}

CColor CRayTracerGPU::traceRay( CRay& ray, int depthLevel )
{
	return CColor();
}