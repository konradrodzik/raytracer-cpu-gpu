////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#include "CRayTracerGPU.h"
#include <vector_types.h>
#include <vector_functions.h>
#include <math_functions.h>
#include <cutil_math.h>



// Global variable representing CUDA texture for screen pixels
texture<float, 2, cudaReadModeElementType> g_screenTexture;
// Array representing CUDA texture pixels
cudaArray* g_screenArray;


__constant__ int g_RAYTRACE_DEPTH;



struct CUDA_Ray
{	
	__device__ CUDA_Ray(){};
	__device__ CUDA_Ray(const float3 &o,const float3 &d)
	{
		m_origin = o;
		m_direction = d;
		m_direction = normalize(m_direction);
		m_invDirection = make_float3(1.0/m_direction.x,1.0/m_direction.y,1.0/m_direction.z);
	}
	float3 m_origin;
	float3 m_direction;
	float3 m_invDirection;
};



CScene* g_currentScene;





// SPHERE INTERSECTION

__device__ int SphereIntersect(CSpherePrimitive* sphere, CUDA_Ray& ray, float& distance )
{
	float3 v = ray.m_origin - make_float3(sphere->m_center.m_x, sphere->m_center.m_y, sphere->m_center.m_z);
	int intResult = 0;
	float b = -dot(v, ray.m_direction);
	float det = (b * b) - dot(v, v) + sphere->m_sqrRadius;

	if(det > 0)
	{
		det = sqrtf(det);

		float point1 = b - det;
		float point2 = b + det;

		if(point2 > 0)
		{
			if(point1 < 0)
			{
				if(point2 < distance)
				{
					distance = point2;
					intResult = -1;
				}
			}
			else
			{
				if(point1 < distance)
				{
					distance = point1;
					intResult = 1;
				}
			}
		}
	}

	return intResult;
}

// END OF SPHERE INTERSECTION

// PLANE INTERSECTION

__device__ int PlaneIntersect(CPlanePrimitive* plane, CUDA_Ray& ray, float& distance)
{
	CPlane planeObj = plane->getPlane();
	float3 planeNormal = make_float3(planeObj.k[0], planeObj.k[1], planeObj.k[2]);
	float d = dot(planeNormal, ray.m_direction);

	if(d != 0)
	{
		float dist = -(dot(planeNormal, ray.m_origin) + planeObj.k[3]) / d;
		if(dist > 0 && dist < distance)
		{
			distance = dist;
			return 1;
		}
	}

	return 0;
}

// END OF PLANE INTERSECTION













// TODO: USE SHARED MEMORY FOR PRIMITIVES!!!
__device__ float3 traceRay(CUDA_Ray& ray, int depthLevel, CUDA_CurrentScene* scene, CSpherePrimitive* spheres, CPlanePrimitive* planes)
{
	if (depthLevel > RAYTRACE_DEPTH) 
		return make_float3(0.0f, 0.0f, 0.0f);
		
	float distance = 10000000.0f;
	float3 outColor = make_float3(0.0f, 0.0f, 0.0f);
	//CBasePrimitive* hitPrimitive = NULL;
	CSpherePrimitive* spherePrimitive = NULL;
	CPlanePrimitive* planePrimitive = NULL;
	int result = 0;
	float3 primitiveIntersection;
	bool isLight = false;


	int index;
	
	// Spheres intersection
	for(index = 0; index < scene->m_sphereCount; ++index)
	{
		if(int res = SphereIntersect(&spheres[index], ray, distance))	// spheres intersection
		{
			spherePrimitive = &spheres[index];
			result = res;
		}
	}
	
	// Planes intersection
	for(index = 0; index < scene->m_planeCount; ++index)
	{
		if(int res = PlaneIntersect(&planes[index], ray, distance)) // planes intersection
		{
			planePrimitive = &planes[index];
			spherePrimitive = NULL;
			result = res;
		}
	}

	// no hit, terminate ray
	if (!spherePrimitive && !planePrimitive) return make_float3(0.0f, 0.0f, 0.0f);

	// handle intersection
	if (spherePrimitive && spherePrimitive->m_isLight)
	{
		// Hit light, return light color
		//return make_float3( 1.0f, 1.0f, 1.0f );
		CVector3 col = spherePrimitive->getMaterial()->getColor();
		return make_float3(col.m_x, col.m_y, col.m_z);
	}

	// determine color at point of intersection
	// get intersection point and normal of intersection
	primitiveIntersection = ray.m_origin + ray.m_direction * distance;
	float3 N;
	if(spherePrimitive) 
		N = spherePrimitive->getNormal(primitiveIntersection);
	else if(planePrimitive)
		N = planePrimitive->getNormal(primitiveIntersection);
	
	
	
	
	// trace spherical lights
	for ( int l = 0; l < scene->m_sphereCount; l++ )
	{
		CSpherePrimitive* p = &spheres[l];
		if (p->m_isLight)
		{
			CSpherePrimitive* light = p;
			
			// SHADOWS BEGIN
			float shadow = 1.0f;
			CUDA_Ray shadowRay;
			if (light->m_type == EPT_SPHERE)	// POINT LIGHTS
			{
				
				float3 L = light->getCenterEx() - primitiveIntersection;
				float tdist = length( L );
				L *= (1.0f / tdist);
				shadowRay = CUDA_Ray( primitiveIntersection + L * RAYTRACE_EPSILON, L );
				
				// Spheres
				for ( int s = 0; s < scene->m_sphereCount; ++s )
				{
					CSpherePrimitive* pr = &spheres[s];
					if ((pr != light) && (SphereIntersect( pr, shadowRay, tdist )))
					{
						shadow = 0;
						break;
					}
				}
				
				// Planes
				for ( int s = 0; s < scene->m_planeCount; ++s )
				{
					CPlanePrimitive* pr = &planes[s];
					if ((PlaneIntersect( pr, shadowRay, tdist )))
					{
						shadow = 0;
						break;
					}
				}
			}
		
		
			if(shadow > 0.0f)
			{
				// BEGIN DIFFUSE LAMBERT SHADING
				float3 L = shadowRay.m_direction;
				if (spherePrimitive && spherePrimitive->getMaterial()->getDiffuse() > 0)
				{
					float dotProduct = dot(shadowRay.m_direction, N);
					if(dotProduct > 0) {
						outColor = outColor + dotProduct * spherePrimitive->getMaterial()->getDiffuse() * shadow * spherePrimitive->getColor(primitiveIntersection) * light->getMaterial()->getColorEx();
					}
				}
				else if(planePrimitive && planePrimitive->getMaterial()->getDiffuse() > 0)
				{
					float dotProduct = dot(shadowRay.m_direction, N);
					if(dotProduct > 0) {
						outColor = outColor + dotProduct * planePrimitive->getMaterial()->getDiffuse() * shadow * planePrimitive->getColor(primitiveIntersection) * light->getMaterial()->getColorEx();
					}
				}
				// END DIFFUSE LAMBERT SHADING
				
				// BEGIN SPECULAR
				if (spherePrimitive && spherePrimitive->getMaterial()->getSpecular() > 0)
				{
					float3 V = ray.m_direction;
					float3 R = L - 2.0f * dot( L, N ) * N;
					float dotf = dot( V, R );
					if (dotf > 0)
					{
						float spec = powf( dotf, 20 ) * spherePrimitive->getMaterial()->getSpecular() * shadow;
						outColor += spec * light->getMaterial()->getColorEx();
					}
				}
				else if (planePrimitive && planePrimitive->getMaterial()->getSpecular() > 0)
				{
					float3 V = ray.m_direction;
					float3 R = L - 2.0f * dot( L, N ) * N;
					float dotf = dot( V, R );
					if (dotf > 0)
					{
						float spec = powf( dotf, 20 ) * planePrimitive->getMaterial()->getSpecular() * shadow;
						outColor += spec * light->getMaterial()->getColorEx();
					}
				}
				// END SPECULAR
			}
		}
	}
	
	

	return outColor;
}

__device__ void calcRayDir(CUDA_Ray& ray, float x, float y, CCamera* camera)
{
	float tmpX = ((2.0f * x + 1.0f  - (float)camera->m_screenWidth) * 0.5f);
	float tmpY = ((2.0f * y + 1.0f - (float)camera->m_screenHeight) * 0.5f);
	
	float3 dx = make_float3(camera->m_dx.m_x*tmpX, camera->m_dx.m_y*tmpX, camera->m_dx.m_z*tmpX);
	float3 dy = make_float3(camera->m_dy.m_x*tmpY, camera->m_dy.m_y*tmpY, camera->m_dy.m_z*tmpY);

	float3 dir = make_float3(camera->m_direction.m_x, camera->m_direction.m_y, camera->m_direction.m_z) + dx + dy;
	ray.m_direction = normalize(dir);
}


__global__ void rayTraceKernel(unsigned char* surface, int width, int height, size_t pitch, CUDA_CurrentScene* scene, CSpherePrimitive* spheres, CPlanePrimitive* planes)
{
    
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;
	
	// Calculate new ray
	CUDA_Ray ray;
	float3 outColor = make_float3(0.0f, 0.0f, 0.0f);
	CVector3 cameraPos = scene->camera.m_position;
	ray.m_origin = make_float3(cameraPos.m_x, cameraPos.m_y, cameraPos.m_z);
	
	for(float samplingY = y; samplingY < y + 1.0f; samplingY += 0.5f)
	for(float samplingX = x; samplingX < x + 1.0f; samplingX += 0.5f)
	{
		calcRayDir(ray, samplingX, samplingY, &scene->camera);
		float3 tmpColor = traceRay(ray, 1, scene, spheres, planes);
		outColor += 0.25f * tmpColor;
	}
	
	//calcRayDir(ray, x, y, &scene->camera);
	//outColor = traceRay(ray, 1, scene, spheres, planes);
	
    // get a pointer to the pixel at (x,y)
    float* pixel = (float*)(surface + y*pitch) + 4*x;
	pixel[0] = outColor.x; // red
	pixel[1] = outColor.y; // green
	pixel[2] = outColor.z; // blue
	pixel[3] = 1; // alpha
	
}


extern "C" 
void runKernels(void* surface, int width, int height, size_t pitch, CUDA_CurrentScene* scene, CSpherePrimitive* spheres, CPlanePrimitive* planes)
{
    cudaError_t error = cudaSuccess;

    dim3 Db = dim3( 16, 16 ); // block dimensions are fixed to be 256 threads
    dim3 Dg = dim3( (width+Db.x-1)/Db.x, (height+Db.y-1)/Db.y );
    
    rayTraceKernel<<<Dg,Db>>>( (unsigned char*)surface, width, height, pitch, scene, spheres, planes);

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