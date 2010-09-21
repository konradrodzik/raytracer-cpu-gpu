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

	#pragma omp parallel for private(x, y, outputColor)
	for(y = 0; y < m_height; ++y)
		for(x = 0; x < m_width; ++x)
		{
			//float fragmentX = (float)x;
			//float fragmentY = (float)y;
			for(float fragmentX = (float)x; fragmentX < (float)x + 1.0f; fragmentX += 0.5f)
				for(float fragmentY = (float)y; fragmentY < (float)y + 1.0f; fragmentY += 0.5f)
				{
					CRay ray;
					ray.setOrigin(camera->getPosition());
					camera->calcRayDir(ray, fragmentX, fragmentY);
					CColor resultColor = traceRay(ray, 1);
					outputColor += 0.25f * resultColor;
					//outputColor = resultColor;
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
		//return CColor( 1.0f, 1.0f, 1.0f );
		return hitPrimitive->getMaterial()->getColor();
	}

	// determine color at point of intersection
	primitiveIntersection = ray.getOrigin() + ray.getDirection() * distance;
	// trace lights
	for ( int l = 0; l < m_currentScene->getPrimitivesCount(); l++ )
	{
		CBasePrimitive* p = m_currentScene->getPrimitive( l );
		if (p->isLight()) 
		{
			CBasePrimitive* light = p;


			// Shadows
			float shadow = 1.0f;
			if (light->getType() == EPT_SPHERE)
			{
				CVector3 L = ((CSpherePrimitive*)light)->getCenter() - primitiveIntersection;
				float tdist = LENGTH( L );
				L *= (1.0f / tdist);
				CRay r = CRay( primitiveIntersection + L * RAYTRACE_EPSILON, L );
				for ( int s = 0; s < m_currentScene->getPrimitivesCount(); s++ )
				{
					CBasePrimitive* pr = m_currentScene->getPrimitive( s );
					if ((pr != light) && (pr->intersect( r, tdist )))
					{
						shadow = 0;
						break;
					}
				}
			}



			// calculate diffuse shading
			CVector3 L = ((CSpherePrimitive*)light)->getCenter() - primitiveIntersection;
			NORMALIZE(L);
			CVector3 N = hitPrimitive->getNormal( primitiveIntersection );
			if (hitPrimitive->getMaterial()->getDiffuse() > 0)
			{
				float dot = DOT( N, L );
				if (dot > 0)
				{
					float diff = dot * hitPrimitive->getMaterial()->getDiffuse() * shadow;
					// add diffuse component to ray color
					outColor += diff * hitPrimitive->getColor(primitiveIntersection) * light->getMaterial()->getColor();
				}
			}



			// determine specular component
			if (hitPrimitive->getMaterial()->getSpecular() > 0)
			{
				// point light source: sample once for specular highlight
				CVector3 V = ray.getDirection();
				CVector3 R = L - 2.0f * DOT( L, N ) * N;
				float dot = DOT( V, R );
				if (dot > 0)
				{
					float spec = powf( dot, 20 ) * hitPrimitive->getMaterial()->getSpecular() * shadow;
					// add specular component to ray color
					outColor += spec * light->getMaterial()->getColor();
				}
			}




		}
	}


	// calculate reflection
	float refl = hitPrimitive->getMaterial()->getReflection();
	if (refl > 0.0f)
	{
		CVector3 N = hitPrimitive->getNormal( primitiveIntersection );
		CVector3 R = ray.getDirection() - 2.0f * DOT( ray.getDirection(), N ) * N;
		if (depthLevel < RAYTRACE_DEPTH) 
		{
			CColor rcol( 0.0f, 0.0f, 0.0f );
			rcol = traceRay(CRay( primitiveIntersection + R * RAYTRACE_EPSILON, R ), depthLevel + 1);
			outColor += refl * rcol * hitPrimitive->getColor(primitiveIntersection);
		}
	}


	// return color
	return outColor;
}