////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#include "CRayTracerGPU.h"
#include <vector_types.h>
#include <vector_functions.h>
#include <math_functions.h>
#include <cutil_math.h>



// Global variable representing CUDA texture for screen pixels
//texture<float, 2, cudaReadModeElementType> g_screenTexture;
// Array representing CUDA texture pixels
//cudaArray* g_screenArray;

//__constant__ int g_RAYTRACE_DEPTH;

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
	
	__device__ void init()
	{
		m_direction = normalize(m_direction);
		m_invDirection = make_float3(1.0/m_direction.x,1.0/m_direction.y,1.0/m_direction.z);
	}
	
	float3 m_origin;
	float3 m_direction;
	float3 m_invDirection;
	float3 m_tmpColor;
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

// BOX INTERSECTION

__device__ int BoxIntersect(CBoxPrimitive* box, CUDA_Ray& ray, float& distance)
{
	// ray position y-rotation
	/*CAABBox boxObj = box->getBox();
	float m_sinAngleY = box->getSinusAngle();
	float m_cosAngleY = box->getCosinusAngle();

	float3 origin = ray.m_origin;
	float3 direction = ray.m_direction;
	float ray_x = (origin.x - boxObj.m_position.m_x) * m_cosAngleY + (origin.z - boxObj.m_position.m_z) * m_sinAngleY;
	float ray_y = origin.y;
	float ray_z = (origin.z - boxObj.m_position.m_z) * m_cosAngleY - (origin.x - boxObj.m_position.m_x) * m_sinAngleY;

	// ray direction y-rotation
	float dir_x = direction.x * m_cosAngleY + direction.z * m_sinAngleY;
	float dir_y = direction.y;
	float dir_z = direction.z * m_cosAngleY - direction.x * m_sinAngleY;

	// ray direction sign
	int dir_sx = dir_x < 0 ? 1 : 0; 
	int dir_sy = dir_y < 0 ? 1 : 0; 
	int dir_sz = dir_z < 0 ? 1 : 0;

	CVector3 aabb[2];
	aabb[0].m_x = boxObj.m_position.m_x; 
	aabb[0].m_y = boxObj.m_position.m_y; 
	aabb[0].m_z = boxObj.m_position.m_z;

	aabb[1].m_x = boxObj.m_size.m_x;
	aabb[1].m_y = boxObj.m_size.m_y;
	aabb[1].m_z = boxObj.m_size.m_z;

	float tmin   = (aabb[    dir_sx].m_x - ray_x) / dir_x;
	float tymax  = (aabb[1 - dir_sy].m_y - ray_y) / dir_y;
	if (tmin > tymax) return 0;

	float tmax   = (aabb[1 - dir_sx].m_x - ray_x) / dir_x;
	float tymin  = (aabb[    dir_sy].m_y - ray_y) / dir_y;
	if (tymin > tmax) return 0;
	if (tymin > tmin) { tmin = tymin; }

	float tzmax  = (aabb[1 - dir_sz].m_z - ray_z) / dir_z;
	if (tmin > tzmax) return 0;
	if (tymax < tmax) tmax = tymax;

	float tzmin  = (aabb[    dir_sz].m_z - ray_z) / dir_z;
	if (tzmin > tmax) return PRIM_MISS;
	float tmp;
	if (tzmin > tmin) { tmp = tzmin; }
	else tmp = tmin;

	if (tmp > 0.1f && tmp < distance)
	{
		distance = tmp;
		return 1;
	}

	return 0;*/
	
	
	
	
	CAABBox boxObj = box->getBox();
	float tmin, tmax, tymin, tymax, tzmin, tzmax;
	float3 invdir;
	float3 direction = ray.m_direction;
	if(direction.x != 0.0f)
		invdir.x = 1.0f / direction.x;
	if(direction.y != 0.0f)
		invdir.y = 1.0f / direction.y;
	if(direction.z != 0.0f)
		invdir.z = 1.0f / direction.z;

	int sign[3];
	sign[0] = invdir.x < 0;
	sign[1] = invdir.y < 0;
	sign[2] = invdir.z < 0;
	float3 bounds[2];
	bounds[0] = boxObj.getPos();
	bounds[1] = boxObj.getPos() + boxObj.getS();

	tmin = (bounds[sign[0]].x - ray.m_origin.x) * invdir.x;
	tmax = (bounds[1-sign[0]].x - ray.m_origin.x) * invdir.x;
	tymin = (bounds[sign[1]].y - ray.m_origin.y) * invdir.y;
	tymax = (bounds[1-sign[1]].y - ray.m_origin.y) * invdir.y;

	if ( tmin > tymax || tymin > tmax)
		return 0;
	if (tymin > tmin)
		tmin = tymin;
	if (tymax < tmax)
		tmax = tymax;

	tzmin = (bounds[sign[2]].z - ray.m_origin.z) * invdir.z;
	tzmax = (bounds[1-sign[2]].z - ray.m_origin.z) * invdir.z;
	if ( (tmin > tzmax) || (tzmin > tmax) )
		return 0;
	if (tzmin > tmin)
		tmin = tzmin;
	if (tzmax < tmax)
		tmax = tzmax;

	if(tmin > 0.1f && tmin < distance)
	{
		distance = tmin;
		return 1;
	}

	return 0;
	
	
	
	
	
	
}

// END OF BOX INTERSECTION













// TODO: USE SHARED MEMORY FOR PRIMITIVES!!!
__device__ float3 traceRay(CUDA_Ray& ray, int depthLevel, CUDA_Ray* outReflectionRay, CUDA_Ray* outRefractionRay, CUDA_CurrentScene* scene, CSpherePrimitive* spheres, CPlanePrimitive* planes, CBoxPrimitive* boxes)
{
	if (depthLevel > RAYTRACE_DEPTH) 
		return make_float3(0.0f, 0.0f, 0.0f);
		
	float distance = 10000000.0f;
	float3 outColor = make_float3(0.0f, 0.0f, 0.0f);
	//CBasePrimitive* hitPrimitive = NULL;
	CSpherePrimitive* spherePrimitive = NULL;
	CPlanePrimitive* planePrimitive = NULL;
	CBoxPrimitive* boxPrimitive = NULL;
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
	
	// Boxes intersection
	for(index = 0; index < scene->m_boxCount; ++index)
	{
		if(int res = BoxIntersect(&boxes[index], ray, distance)) // planes intersection
		{
			boxPrimitive = &boxes[index];
			planePrimitive = NULL;
			spherePrimitive = NULL;
			result = res;
		}
	}

	// no hit, terminate ray
	if (!spherePrimitive && !planePrimitive && !boxPrimitive) return make_float3(0.0f, 0.0f, 0.0f);

	// handle intersection with spherical light
	if (spherePrimitive && spherePrimitive->m_isLight)
	{
		// Hit light, return light color
		//return make_float3( 1.0f, 1.0f, 1.0f );
		
		return spherePrimitive->getMaterial()->getColorEx();
		//CVector3 col = spherePrimitive->getMaterial()->getColorEx();
		//return make_float3(col.m_x, col.m_y, col.m_z);
	}

	// determine color at point of intersection
	// get intersection point and normal of intersection
	primitiveIntersection = ray.m_origin + ray.m_direction * distance;
	float3 N;
	if(spherePrimitive) 
		N = spherePrimitive->getNormal(primitiveIntersection);
	else if(planePrimitive)
		N = planePrimitive->getNormal(primitiveIntersection);
	else if(boxPrimitive)
		N = boxPrimitive->getNormal(primitiveIntersection);
	
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
				
				// Boxes
				for ( int s = 0; s < scene->m_boxCount; ++s )
				{
					CBoxPrimitive* pr = &boxes[s];
					if ((BoxIntersect( pr, shadowRay, tdist )))
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
				else if(boxPrimitive && boxPrimitive->getMaterial()->getDiffuse() > 0)
				{
					float dotProduct = dot(shadowRay.m_direction, N);
					if(dotProduct > 0) {
						outColor = outColor + dotProduct * boxPrimitive->getMaterial()->getDiffuse() * shadow * boxPrimitive->getColor(primitiveIntersection) * light->getMaterial()->getColorEx();
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
				else if (boxPrimitive && boxPrimitive->getMaterial()->getSpecular() > 0)
				{
					float3 V = ray.m_direction;
					float3 R = L - 2.0f * dot( L, N ) * N;
					float dotf = dot( V, R );
					if (dotf > 0)
					{
						float spec = powf( dotf, 20 ) * boxPrimitive->getMaterial()->getSpecular() * shadow;
						outColor += spec * light->getMaterial()->getColorEx();
					}
				}
				// END SPECULAR
			}
		}
	}
	
	
	
	
	
	// BEGIN REFLECTION
	if (spherePrimitive)
	{
		float refl = spherePrimitive->getMaterial()->getReflection();
		if (refl > 0.0f && depthLevel < RAYTRACE_DEPTH && result != PRIM_HITIN)
		{
			float3 R = ray.m_direction - 2.0f * dot( ray.m_direction, N ) * N;
			outReflectionRay->m_origin = primitiveIntersection + R * RAYTRACE_EPSILON;
			outReflectionRay->m_direction = R;
			outReflectionRay->m_tmpColor = refl * spherePrimitive->getColor(primitiveIntersection);
		}
		else
			outReflectionRay = NULL;
		
	}
	else if(planePrimitive)
	{
		float refl = planePrimitive->getMaterial()->getReflection();
		if (refl > 0.0f && depthLevel < RAYTRACE_DEPTH && result != PRIM_HITIN)
		{
			float3 R = ray.m_direction - 2.0f * dot( ray.m_direction, N ) * N;
			outReflectionRay->m_origin = primitiveIntersection + R * RAYTRACE_EPSILON;
			outReflectionRay->m_direction = R;
			outReflectionRay->m_tmpColor = refl * planePrimitive->getColor(primitiveIntersection);
		}
		else
			outReflectionRay = NULL;
	}
	else if(boxPrimitive)
	{
		float refl = boxPrimitive->getMaterial()->getReflection();
		if (refl > 0.0f && depthLevel < RAYTRACE_DEPTH && result != PRIM_HITIN)
		{
			float3 R = ray.m_direction - 2.0f * dot( ray.m_direction, N ) * N;
			outReflectionRay->m_origin = primitiveIntersection + R * RAYTRACE_EPSILON;
			outReflectionRay->m_direction = R;
			outReflectionRay->m_tmpColor = refl * boxPrimitive->getColor(primitiveIntersection);
		}
		else
			outReflectionRay = NULL;
	}
	else {
		outReflectionRay = NULL;
	}
	// END REFLECTION




	// BEGIN REFRACTION
	if(spherePrimitive)
	{
		float refraction = spherePrimitive->getMaterial()->getRefraction();
		if ((refraction > 0.0f) && (depthLevel < RAYTRACE_DEPTH))
		{
			float rindex = spherePrimitive->getMaterial()->getRefrIndex();
			float n = 1.0f / rindex;
			float3 N = spherePrimitive->getNormal( primitiveIntersection ) * (float)result;
			float cosI = -dot( N, ray.m_direction );
			float cosT2 = 1.0f - n * n * (1.0f - cosI * cosI);
			if (cosT2 > 0.0f)
			{
				// Beer's Law
				float3 absorbance = spherePrimitive->getColor(primitiveIntersection) * 0.15f * -distance;
				float3 transparency = make_float3( expf( absorbance.x ), expf( absorbance.y ), expf( absorbance.z ) );
				float3 T = (n * ray.m_direction) + (n * cosI - sqrtf( cosT2 )) * N;
				
				outRefractionRay->m_origin = primitiveIntersection + T * RAYTRACE_EPSILON;
				outRefractionRay->m_direction = T;
				outRefractionRay->m_tmpColor = transparency;
			}
			else
				outRefractionRay = NULL;
		}
		else
			outRefractionRay = NULL;
	}
	else if(planePrimitive)
	{
		float refraction = planePrimitive->getMaterial()->getRefraction();
		if ((refraction > 0.0f) && (depthLevel < RAYTRACE_DEPTH))
		{
			float rindex = planePrimitive->getMaterial()->getRefrIndex();
			float n = 1.0f / rindex;
			float3 N = planePrimitive->getNormal( primitiveIntersection ) * (float)result;
			float cosI = -dot( N, ray.m_direction );
			float cosT2 = 1.0f - n * n * (1.0f - cosI * cosI);
			if (cosT2 > 0.0f)
			{
				// Beer's Law
				float3 absorbance = planePrimitive->getColor(primitiveIntersection) * 0.15f * -distance;
				float3 transparency = make_float3( expf( absorbance.x ), expf( absorbance.y ), expf( absorbance.z ) );
				float3 T = (n * ray.m_direction) + (n * cosI - sqrtf( cosT2 )) * N;
				
				outRefractionRay->m_origin = primitiveIntersection + T * RAYTRACE_EPSILON;
				outRefractionRay->m_direction = T;
				outRefractionRay->m_tmpColor = transparency;
			}
			else
				outRefractionRay = NULL;
		}
		else
			outRefractionRay = NULL;
	}
	else if(boxPrimitive)
	{
		float refraction = boxPrimitive->getMaterial()->getRefraction();
		if ((refraction > 0.0f) && (depthLevel < RAYTRACE_DEPTH))
		{
			float rindex = boxPrimitive->getMaterial()->getRefrIndex();
			float n = 1.0f / rindex;
			float3 N = boxPrimitive->getNormal( primitiveIntersection ) * (float)result;
			float cosI = -dot( N, ray.m_direction );
			float cosT2 = 1.0f - n * n * (1.0f - cosI * cosI);
			if (cosT2 > 0.0f)
			{
				// Beer's Law
				float3 absorbance = boxPrimitive->getColor(primitiveIntersection) * 0.15f * -distance;
				float3 transparency = make_float3( expf( absorbance.x ), expf( absorbance.y ), expf( absorbance.z ) );
				float3 T = (n * ray.m_direction) + (n * cosI - sqrtf( cosT2 )) * N;
				
				outRefractionRay->m_origin = primitiveIntersection + T * RAYTRACE_EPSILON;
				outRefractionRay->m_direction = T;
				outRefractionRay->m_tmpColor = transparency;
			}
			else
				outRefractionRay = NULL;
		}
		else
			outRefractionRay = NULL;
	}
	else {
		outRefractionRay = NULL;
	}
	// END REFRACTION
	
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


__global__ void rayTraceKernel(unsigned char* surface, int width, int height, size_t pitch, CUDA_CurrentScene* scene, CSpherePrimitive* spheres, CPlanePrimitive* planes, CBoxPrimitive* boxes)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;
	
	// Calculate new ray
	CUDA_Ray ray;
	float3 outColor = make_float3(0.0f, 0.0f, 0.0f);
	CVector3 cameraPos = scene->camera.m_position;
	ray.m_origin = make_float3(cameraPos.m_x, cameraPos.m_y, cameraPos.m_z);
	
	/*for(float samplingY = y; samplingY < y + 1.0f; samplingY += 0.5f)
	for(float samplingX = x; samplingX < x + 1.0f; samplingX += 0.5f)
	{
		calcRayDir(ray, samplingX, samplingY, &scene->camera);
		CUDA_Ray reflRay;
		CUDA_Ray refrRay;
		CUDA_Ray* reflectionPointer = &reflRay;
		CUDA_Ray* refractionPointer = &refrRay;
		float3 tmpColor = traceRay(ray, 1, reflectionPointer, refractionPointer, scene, spheres, planes, boxes);
	*/
	
	
		calcRayDir(ray, x, y, &scene->camera);
		CUDA_Ray reflRay;
		CUDA_Ray refrRay;
		CUDA_Ray* reflectionPointer = &reflRay;
		CUDA_Ray* refractionPointer = &refrRay;
		outColor = traceRay(ray, 1, reflectionPointer, refractionPointer, scene, spheres, planes, boxes);
		

	#if RAYTRACE_DEPTH == 1
		#define MAX_RAYS 4
	#elif  RAYTRACE_DEPTH == 2
		#define MAX_RAYS 8
	#elif  RAYTRACE_DEPTH == 3
		#define MAX_RAYS 16
	#elif  RAYTRACE_DEPTH == 4
		#define MAX_RAYS 32
	#elif  RAYTRACE_DEPTH == 5
		#define MAX_RAYS 64
	#else 
		#define MAX_RAYS 2
	#endif

		CUDA_Ray currRays[MAX_RAYS];
		CUDA_Ray newRays[MAX_RAYS];
		int currRayCount = 0;
		if(reflectionPointer != NULL && refractionPointer != NULL) {
			currRays[0] = *reflectionPointer;
			currRays[1] = *refractionPointer;
			currRayCount = 2;
		} else if(reflectionPointer != NULL) {
			currRays[0] = *reflectionPointer;
			currRayCount = 1;
		} else if(refractionPointer != NULL) {
			currRays[0] = *refractionPointer;
			currRayCount = 1;
		}

		int depth = 2;
		if(RAYTRACE_DEPTH>1)
		do
		{
			int index = 0;
			int currIndex = 0;
			while(currIndex < currRayCount) {
				currRays[currIndex].init();
				outColor += currRays[currIndex].m_tmpColor * traceRay(currRays[currIndex], depth, reflectionPointer, refractionPointer, scene, spheres, planes, boxes);
				//tmpColor += currRays[currIndex].m_tmpColor * traceRay(currRays[currIndex], depth, reflectionPointer, refractionPointer, scene, spheres, planes, boxes);
				++currIndex;
				
				if(reflectionPointer!=NULL && refractionPointer != NULL) {
					newRays[index] = *reflectionPointer;
					newRays[index+1] = *refractionPointer;
					index+=2;
				}
				else if(reflectionPointer!=NULL) {
					newRays[index] = *reflectionPointer;
					++index;
				}
				else if(refractionPointer != NULL) {
					newRays[index] = *refractionPointer;
					++index;
				}
			}
			
			for(int i = 0; i < index; ++i ) {
				currRays[i] = newRays[i]; 
			}
			currRayCount = index;
			
			++depth;
		} while(depth < RAYTRACE_DEPTH);
	
	
	/*
		outColor += 0.25f * tmpColor;
	} // END OF MULTI SAMPLING FOR
	*/	
	
    // get a pointer to the pixel at (x,y)
    float* pixel = (float*)(surface + y*pitch) + 4*x;
	pixel[0] = outColor.x; // red
	pixel[1] = outColor.y; // green
	pixel[2] = outColor.z; // blue
	pixel[3] = 1; // alpha
}


extern "C" 
void runKernels(void* surface, int width, int height, size_t pitch, CUDA_CurrentScene* scene, CSpherePrimitive* spheres, CPlanePrimitive* planes, CBoxPrimitive* boxes)
{
    cudaError_t error = cudaSuccess;

    dim3 Db = dim3( 16, 16); // block dimensions are fixed to be 256 threads
    dim3 Dg = dim3( (width+Db.x-1)/Db.x, (height+Db.y-1)/Db.y );
    //dim3 Dg = dim3( (width)/Db.x, (height)/Db.y )
    
    cudaThreadSynchronize();
    rayTraceKernel<<<Dg,Db>>>( (unsigned char*)surface, width, height, pitch, scene, spheres, planes, boxes);
    cudaThreadSynchronize();

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("rayTraceKernel() failed to launch error = %d\n", error);
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