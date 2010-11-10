////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#ifndef __H_CRayTracerGPU_H__
#define __H_CRayTracerGPU_H__


#include <cuda.h>
#include <builtin_types.h>
#include <cuda_runtime_api.h>
#include <cuda_d3d9_interop.h>


#include "MathFunctions.h"
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
//#include <cutil_inline.h>




using namespace std;



struct CUDA_CurrentScene
{
public:
	int width;
	int height;
	CCamera camera;
	CSpherePrimitive* sphereArray;
	CPlanePrimitive* planeArray;
	int m_sphereCount;
	int m_planeCount;

	CUDA_CurrentScene() : width(0), height(0), sphereArray(NULL), planeArray(NULL), m_sphereCount(0), m_planeCount(0)
	{

	}
};



extern CScene* g_currentScene;


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



extern "C" 
{
	void runKernels(void* surface, int width, int height, size_t pitch, CUDA_CurrentScene* scene, CSpherePrimitive* spheres, CPlanePrimitive* planes);
}


class CRayTracerGPU : public CRayTracer
{
public:
	// Initialize constructor
	CRayTracerGPU(int width, int height) : CRayTracer(width, height)
	{
		m_deviceScene = NULL;
	}

	// Get raytracer type
	virtual E_COMPUTING_TYPE getType();

	// Calculate scene
	virtual void calculateScene()
	{
		cudaArray *cuArray;
		cudaError_t error;
		cudaStream_t 	stream = 0;
		const int nbResources = 1;
		cudaGraphicsResource * ppResources[nbResources] = 
		{
			m_cudatextureWrapper.cudaResource,
		};
		error = cudaGraphicsMapResources(nbResources, ppResources, stream);
		error = cudaGraphicsSubResourceGetMappedArray( &cuArray, m_cudatextureWrapper.cudaResource, 0, 0);



		// Clear output color
		CColor outputColor = CColor(0.0f, 0.0f, 0.0f);
		// Get current camera
		CCamera* camera = m_currentScene->getCamera();
		camera->initialize();



		runKernels(m_cudatextureWrapper.cudaLinearMemory, m_cudatextureWrapper.width, m_cudatextureWrapper.height, m_cudatextureWrapper.pitch, 
				m_deviceScene, m_spherePrimitives, m_planePrimitives);



		// then we want to copy cudaLinearMemory to the D3D texture, via its mapped form : cudaArray
		error = cudaMemcpyToArray(cuArray, 0, 0, m_cudatextureWrapper.cudaLinearMemory, m_cudatextureWrapper.pitch * m_cudatextureWrapper.height, cudaMemcpyDeviceToDevice);
		cudaGraphicsUnmapResources(	nbResources, ppResources, stream);
	}



	HRESULT registerCUDA(IDirect3DDevice9* device, IDirect3DTexture9* texture)
	{
		cudaError_t error;
		error = cudaD3D9SetDirect3DDevice(device);

		// Assign texture from framework to CUDA wrapper
		m_cudatextureWrapper.pTexture = texture;
		m_cudatextureWrapper.width = m_width;
		m_cudatextureWrapper.height = m_height;

		// Register CUDA resource and assign it to our D3D9 texture
		error = cudaGraphicsD3D9RegisterResource(&m_cudatextureWrapper.cudaResource, m_cudatextureWrapper.pTexture, cudaGraphicsRegisterFlagsNone);
		//cutilCheckMsg("cudaGraphicsD3D9RegisterResource (g_texture_2d) failed");
		error = cudaMallocPitch(&m_cudatextureWrapper.cudaLinearMemory, &m_cudatextureWrapper.pitch, m_cudatextureWrapper.width * sizeof(float) * 4, m_cudatextureWrapper.height);
		//cutilCheckMsg("cudaMallocPitch (g_texture_2d) failed");
		error = cudaMemset(m_cudatextureWrapper.cudaLinearMemory, 1, m_cudatextureWrapper.pitch * m_cudatextureWrapper.height);

		CCamera* cam = m_currentScene->getCamera();
		cam->initialize();
		m_hostScene.width = m_width;
		m_hostScene.height = m_height;
		m_hostScene.camera = *cam;

		// Spheres
		m_hostScene.m_sphereCount = m_currentScene->getSphereCount();
		m_hostScene.sphereArray = new CSpherePrimitive[m_hostScene.m_sphereCount];
		m_currentScene->fillSphereArray(m_hostScene.sphereArray);
		cudaMalloc((void**)&m_spherePrimitives, sizeof( CSpherePrimitive ) * m_hostScene.m_sphereCount);
		cudaMemcpy(m_spherePrimitives, m_hostScene.sphereArray, sizeof( CSpherePrimitive ) * m_hostScene.m_sphereCount, cudaMemcpyHostToDevice);


		// Planes
		m_hostScene.m_planeCount = m_currentScene->getPlaneCount();
		m_hostScene.planeArray = new CPlanePrimitive[m_hostScene.m_planeCount];
		m_currentScene->fillPlaneArray(m_hostScene.planeArray);
		cudaMalloc((void**)&m_planePrimitives, sizeof( CPlanePrimitive ) * m_hostScene.m_planeCount);
		cudaMemcpy(m_planePrimitives, m_hostScene.planeArray, sizeof( CPlanePrimitive ) * m_hostScene.m_planeCount, cudaMemcpyHostToDevice);





		CUDA_CurrentScene tmp;
		tmp.width = 666;
		tmp.height = 666;


		cudaMalloc((void**)&m_deviceScene, sizeof( CUDA_CurrentScene ));
		cudaMemcpy(m_deviceScene, &m_hostScene, sizeof( CUDA_CurrentScene ), cudaMemcpyHostToDevice);


		

		cudaMemcpy(&tmp, m_deviceScene, sizeof(CUDA_CurrentScene), cudaMemcpyDeviceToHost);

		int raytraceDepth = RAYTRACE_DEPTH;
		cudaMemcpyToSymbol("g_RAYTRACE_DEPTH", &raytraceDepth, sizeof(int), 0, cudaMemcpyHostToDevice);


		return S_OK;
	}

private:
	// Trace single ray
	virtual CColor traceRay(CRay& ray, int depthLevel);

private:
	CUDA_tex m_cudatextureWrapper;	// Wrapper for texture


	CUDA_CurrentScene m_hostScene;
	CUDA_CurrentScene *m_deviceScene;
	CSpherePrimitive *m_spherePrimitives;
	CPlanePrimitive *m_planePrimitives;
};

#endif;