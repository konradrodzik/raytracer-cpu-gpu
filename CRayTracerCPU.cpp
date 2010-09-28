////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#include "stdafx.h"

CRayTracerCPU::CRayTracerCPU( int width, int height )
: CRayTracer(width, height)
{

}

E_COMPUTING_TYPE CRayTracerCPU::getType()
{
	return ECT_CPU;
}

void CRayTracerCPU::calculateScene()
{
	// Clear output color
	CColor outputColor = CColor(0.0f, 0.0f, 0.0f);

	// Get current camera
	CCamera* camera = m_currentScene->getCamera();
	camera->initialize();

	int x, y;

	//#pragma omp parallel for private(x, y, outputColor)
	for(y = 0; y < m_height; ++y)
		for(x = 0; x < m_width; ++x)
		{
			float fragmentX = (float)x;
			float fragmentY = (float)y;
			//for(float fragmentX = (float)x; fragmentX < (float)x + 1.0f; fragmentX += 0.5f)
				//for(float fragmentY = (float)y; fragmentY < (float)y + 1.0f; fragmentY += 0.5f)
				{
					CRay ray;
					ray.setOrigin(camera->getPosition());
					camera->calcRayDir(ray, fragmentX, fragmentY);
					CColor resultColor = traceRay(ray, 1);
					//outputColor += 0.25f * resultColor;
					outputColor = resultColor;
				}

				// Set output color to texture
				//pDataSurf[offset * y + x] = D3DCOLOR_XRGB((int)outputColor.m_r, (int)outputColor.m_g, (int)outputColor.m_b);
				m_screenColor[y][x] = outputColor;

				// Clear output color
				outputColor = CColor(0.0f, 0.0f, 0.0f);
		}
}

CColor CRayTracerCPU::traceRay( CRay& ray, int depthLevel )
{
	if (depthLevel > RAYTRACE_DEPTH) 
		return CColor(0.0f, 0.0f, 0.0f);

	float distance = 10000000.0f;
	CColor outColor = CColor(0.0f, 0.0f, 0.0f);
	CBasePrimitive* hitPrimitive = NULL;
	int result;
	CVector3 primitiveIntersection;
	bool isLight = false;

	// Loop through primitives
	for(std::vector<CBasePrimitive*>::iterator itor = m_currentScene->m_primitives.begin(); itor != m_currentScene->m_primitives.end(); ++itor)
	{

		if(int res = (*itor)->intersect(ray, distance))
		{
			hitPrimitive = *itor;
			result = res;
		}
	}

	// no hit, terminate ray
	if (!hitPrimitive) return CColor(0.0f, 0.0f, 0.0f);

	// handle intersection
	if (hitPrimitive->isLight())
	{
		// Hit light, return light color
		//return CColor( 0.0f, 0.0f, 0.0f );
		return hitPrimitive->getMaterial()->getColor();
	}

	// determine color at point of intersection
	primitiveIntersection = ray.getOrigin() + ray.getDirection() * distance;
	CVector3 N = hitPrimitive->getNormal( primitiveIntersection );

	// trace lights
	for ( int l = 0; l < m_currentScene->getPrimitivesCount(); l++ )
	{
		CBasePrimitive* p = m_currentScene->getPrimitive( l );
		if (p->isLight()) 
		{
			CBasePrimitive* light = p;


			// SHADOWS BEGIN
			float shadow = 1.0f;
			CRay shadowRay;
			if (light->getType() == EPT_SPHERE)	// SPHERE
			{
				CVector3 L = ((CSpherePrimitive*)light)->getCenter() - primitiveIntersection;
				float tdist = LENGTH( L );
				L *= (1.0f / tdist);
				shadowRay = CRay( primitiveIntersection + L * RAYTRACE_EPSILON, L );
				for ( int s = 0; s < m_currentScene->getPrimitivesCount(); ++s )
				{
					CBasePrimitive* pr = m_currentScene->getPrimitive( s );
					if ((pr != light) && (pr->intersect( shadowRay, tdist )))
					{
						shadow = 0;
						break;
					}
				}
			}
			// SHADOWS END


			if(shadow > 0.0f)
			{
				// BEGIN DIFFUSE LAMBERT SHADING
				CVector3 L = shadowRay.getDirection();
				if (hitPrimitive->getMaterial()->getDiffuse() > 0)
				{
					float dotProduct = shadowRay.getDirection().dot(N);
					if(dotProduct > 0)
						outColor += dotProduct * hitPrimitive->getMaterial()->getDiffuse() * shadow * hitPrimitive->getColor(primitiveIntersection) * light->getMaterial()->getColor();
				}
				// END DIFFUSE LAMBERT SHADING
				
				// BEGIN SPECULAR
				if (hitPrimitive->getMaterial()->getSpecular() > 0)
				{
					CVector3 V = ray.getDirection();
					CVector3 R = L - 2.0f * DOT( L, N ) * N;
					float dot = DOT( V, R );
					if (dot > 0)
					{
						float spec = powf( dot, 20 ) * hitPrimitive->getMaterial()->getSpecular() * shadow;
						outColor += spec * light->getMaterial()->getColor();
					}
				}
				// END SPECULAR
			}
		}
	}


	// BEGIN REFLECTION
	float refl = hitPrimitive->getMaterial()->getReflection();
	if (refl > 0.0f && depthLevel < RAYTRACE_DEPTH && result != PRIM_HITIN)
	{
		CVector3 R = ray.getDirection() - 2.0f * DOT( ray.getDirection(), N ) * N;
		outColor += refl * traceRay(CRay( primitiveIntersection + R * RAYTRACE_EPSILON, R ), depthLevel + 1) * hitPrimitive->getColor(primitiveIntersection);
	}
	// END REFLECTION




	// BEGIN REFRACTION
	float refraction = hitPrimitive->getMaterial()->getRefraction();
	if ((refraction > 0) && (depthLevel < RAYTRACE_DEPTH))
	{
		float rindex = hitPrimitive->getMaterial()->getRefrIndex();
		float n = 1.0f / rindex;
		CVector3 N = hitPrimitive->getNormal( primitiveIntersection ) * (float)result;
		float cosI = -DOT( N, ray.getDirection() );
		float cosT2 = 1.0f - n * n * (1.0f - cosI * cosI);
		if (cosT2 > 0.0f)
		{
			CVector3 T = (n * ray.getDirection()) + (n * cosI - sqrtf( cosT2 )) * N;
			CColor rcol( 0, 0, 0 );
			rcol = traceRay( CRay( primitiveIntersection + T * RAYTRACE_EPSILON, T ), depthLevel + 1);
			// Beer's Law
			CColor absorbance = hitPrimitive->getColor(primitiveIntersection) * 0.15f * -distance;
			CColor transparency = CColor( expf( absorbance.m_r ), expf( absorbance.m_g ), expf( absorbance.m_b ) );
			outColor += rcol * transparency;
		}
	}
	// END REFRACTION





/*

		if(result != )
			n = scene.GetEnviromentRefraction() / hitPrimitive->getMaterial().getRefraction();
		else
			n = hitPrimitive->getMaterial().getRefraction() / scene.GetEnviromentRefraction();

*/
	

	// return color
	return outColor;
}