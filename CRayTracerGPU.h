////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#ifndef __H_CRayTracerGPU_H__
#define __H_CRayTracerGPU_H__


#include <cuda.h>
#include <builtin_types.h>
#include <cuda_runtime_api.h>
#include <cuda_d3d9_interop.h>
//#include <cutil_inline_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "CVector3.h"
#include "CRay.h"
#include "CPlane.h"
#include "CTexture.h"
#include "CMaterial.h"
#include "CCamera.h"
#include "CBasePrimitive.h"
#include "CSpherePrimitive.h"
#include "CPlanePrimitive.h"
#include "CLight.h"
#include "CScene.h"
#include "CRayTracer.h"

#include <d3d9.h>




using namespace std;



// Global structure and object of CUDA texture resource
struct CUDA_tex
{
	CUDA_tex() : pTexture(NULL), cudaResource(NULL), cudaLinearMemory(NULL), pitch(0), width(0), height(0) {}
	IDirect3DTexture9		* pTexture;
	cudaGraphicsResource	*cudaResource;
	void					*cudaLinearMemory;
	size_t					pitch;
	int width;
	int height;	
};


class CRayTracerGPU : public CRayTracer
{
public:
	// Initialize constructor
	CRayTracerGPU(int width, int height);

	// Get raytracer type
	virtual E_COMPUTING_TYPE getType();

	// Calculate scene
	virtual void calculateScene();

	HRESULT registerCUDA(IDirect3DDevice9* device, IDirect3DTexture9* texture)
	{
		cudaD3D9SetDirect3DDevice(device);


		// Assign texture from framework to CUDA wrapper
		m_cudatextureWrapper.pTexture = texture;
		m_cudatextureWrapper.width = m_width;
		m_cudatextureWrapper.height = m_height;

		// Register CUDA resource and assign it to our D3D9 texture
		cudaGraphicsD3D9RegisterResource(&m_cudatextureWrapper.cudaResource, m_cudatextureWrapper.pTexture, cudaGraphicsRegisterFlagsNone);
		cudaMallocPitch(&m_cudatextureWrapper.cudaLinearMemory, &m_cudatextureWrapper.pitch, m_cudatextureWrapper.width * sizeof(float) * 4, m_cudatextureWrapper.height);
		cudaMemset(m_cudatextureWrapper.cudaLinearMemory, 1, m_cudatextureWrapper.pitch * m_cudatextureWrapper.height);

		return S_OK;
	}

private:
	// Trace single ray
	virtual CColor traceRay(CRay& ray, int depthLevel);

private:
	CUDA_tex m_cudatextureWrapper;	// Wrapper for texture
};

#endif;