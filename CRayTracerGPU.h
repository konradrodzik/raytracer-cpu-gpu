////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#ifndef __H_CRayTracerGPU_H__
#define __H_CRayTracerGPU_H__


#include <cuda.h>
#include <builtin_types.h>
#include <cuda_runtime_api.h>
#include <cuda_d3d9_interop.h>
#include <cutil.h>

#include "MathFunctions.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "CVector3.h"
#include "CRay.h"
#include "CPlane.h"
#include "CAABBox.h"
#include "CTexture.h"
#include "CMaterial.h"
#include "CCamera.h"
#include "CBasePrimitive.h"
#include "CSpherePrimitive.h"
#include "CPlanePrimitive.h"
#include "CBoxPrimitive.h"
#include "CLight.h"
#include "CScene.h"
#include "CRayTracer.h"
#include "CRTProfiler.h"

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
	CBoxPrimitive* boxArray;
	int m_sphereCount;
	int m_planeCount;
	int m_boxCount;

	CUDA_CurrentScene()
	: width(0)
	, height(0)
	, sphereArray(NULL)
	, planeArray(NULL)
	, boxArray(NULL)
	, m_sphereCount(0)
	, m_planeCount(0)
	, m_boxCount(0)
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
	void runKernels(void* surface, int width, int height, size_t pitch, CUDA_CurrentScene* scene, CSpherePrimitive* spheres, CPlanePrimitive* planes, CBoxPrimitive* boxes);
}



class CRayTracerGPU : public CRayTracer
{
public:
	// Initialize constructor
	CRayTracerGPU(int width, int height) : CRayTracer(width, height)
	{
		m_deviceScene = NULL;

		// Create cutil timer
		CUT_SAFE_CALL(cutCreateTimer(&m_wholeTimer));
	}

	// CUDA raytracer destructor
	~CRayTracerGPU()
	{
		CUT_SAFE_CALL(cutDeleteTimer(m_wholeTimer));
		unregisterCUDA();
	}

	// Get raytracer type
	virtual E_COMPUTING_TYPE getType();

	void updateCudaCamera(CCamera* camera)
	{
		cudaMemcpy(&m_deviceScene->camera, camera, sizeof( CCamera ), cudaMemcpyHostToDevice);
	}

	// Calculate scene
	virtual void calculateScene()
	{
		// Create current profile
		m_currentProfile = new SProfiledScene();
		m_currentProfile->m_scene = m_currentScene;
		CUT_SAFE_CALL(cutResetTimer(m_wholeTimer));
		CUT_SAFE_CALL(cutStartTimer(m_wholeTimer));

		runKernels(m_cudatextureWrapper.cudaLinearMemory, m_cudatextureWrapper.width, m_cudatextureWrapper.height, m_cudatextureWrapper.pitch, 
				m_deviceScene, m_spherePrimitives, m_planePrimitives, m_boxPrimitives);
		
		CUT_SAFE_CALL(cutStopTimer(m_wholeTimer));
		m_currentProfile->m_frameTime += cutGetTimerValue(m_wholeTimer);
		m_profiler->addSceneProfile(m_currentProfile);

		// then we want to copy cudaLinearMemory to the D3D texture, via its mapped form : cudaArray
		cudaMemcpyToArray(m_cuArray, 0, 0, m_cudatextureWrapper.cudaLinearMemory, m_cudatextureWrapper.pitch * m_cudatextureWrapper.height, cudaMemcpyDeviceToDevice);
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

		// Boxes
		m_hostScene.m_boxCount = m_currentScene->getBoxCount();
		m_hostScene.boxArray = new CBoxPrimitive[m_hostScene.m_boxCount];
		m_currentScene->fillBoxArray(m_hostScene.boxArray);
		cudaMalloc((void**)&m_boxPrimitives, sizeof( CBoxPrimitive ) * m_hostScene.m_boxCount);
		cudaMemcpy(m_boxPrimitives, m_hostScene.boxArray, sizeof( CBoxPrimitive ) * m_hostScene.m_boxCount, cudaMemcpyHostToDevice);


		cudaMalloc((void**)&m_deviceScene, sizeof( CUDA_CurrentScene ));
		cudaMemcpy(m_deviceScene, &m_hostScene, sizeof( CUDA_CurrentScene ), cudaMemcpyHostToDevice);

		// Map cuda resources...
		error = cudaGraphicsMapResources(1, &m_cudatextureWrapper.cudaResource, 0);
		error = cudaGraphicsSubResourceGetMappedArray( &m_cuArray, m_cudatextureWrapper.cudaResource, 0, 0);

		return S_OK;
	}

	void unregisterCUDA()
	{
		// DEVICE scene
		if(m_deviceScene)
		{
			cudaFree(m_deviceScene);
			m_deviceScene = NULL;
		}

		// Unmap cuda resources
		cudaGraphicsUnmapResources(	1, &m_cudatextureWrapper.cudaResource, 0);

		// DEVICE primitives
		if(m_spherePrimitives)
		{
			cudaFree(m_spherePrimitives);
			m_spherePrimitives = NULL;
		}

		if(m_planePrimitives)
		{
			cudaFree(m_planePrimitives);
			m_planePrimitives = NULL;
		}

		if(m_boxPrimitives)
		{
			cudaFree(m_boxPrimitives);
			m_boxPrimitives = NULL;
		}
		


		// HOST primitives
		if(m_hostScene.sphereArray)
		{
			delete[] m_hostScene.sphereArray;
			m_hostScene.sphereArray = NULL;
		}
		if(m_hostScene.planeArray)
		{
			delete[] m_hostScene.planeArray;
			m_hostScene.planeArray = NULL;
		}
		if(m_hostScene.boxArray)
		{
			delete[] m_hostScene.boxArray;
			m_hostScene.boxArray = NULL;
		}



		// unregister the CUDA resources
		cudaGraphicsUnregisterResource(m_cudatextureWrapper.cudaResource);
		cudaFree(m_cudatextureWrapper.cudaLinearMemory);

		// direct 3d texture release
		if(m_cudatextureWrapper.pTexture) 
		{
			m_cudatextureWrapper.pTexture->Release();
			m_cudatextureWrapper.pTexture = NULL;
		}
		
		cudaThreadExit();
	}

private:
	// Trace single ray
	virtual CColor traceRay(CRay& ray, int depthLevel);

private:
	CUDA_tex m_cudatextureWrapper;	// Wrapper for texture
	cudaArray *m_cuArray;


	CUDA_CurrentScene m_hostScene;
	CUDA_CurrentScene *m_deviceScene;
	CSpherePrimitive *m_spherePrimitives;
	CPlanePrimitive *m_planePrimitives;
	CBoxPrimitive *m_boxPrimitives;


	unsigned int m_wholeTimer;			// Main timer for whole raytraced scene
};

#endif;