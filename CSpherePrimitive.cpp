////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#include "stdafx.h"

CSpherePrimitive::CSpherePrimitive( const CVector3& center, float radius ) 
: m_center(center)
, m_radius(radius)
, m_sqrRadius(radius*radius)
, m_invRadius(1.0f / radius)
{
}

CSpherePrimitive::CSpherePrimitive()
: m_center(CVector3(0.0f, 0.0f, 0.0f))
, m_radius(0.0f)
, m_sqrRadius(0.0f)
, m_invRadius(0.0f)
{

}

E_PRIMITIVE_TYPE CSpherePrimitive::getType()
{
	return EPT_SPHERE;
}

CVector3& CSpherePrimitive::getCenter()
{
	return m_center;
}

float CSpherePrimitive::getRadius()
{
	return m_radius;
}

void CSpherePrimitive::setRadius( float radius )
{
	m_radius = radius;
	m_sqrRadius = radius * radius;
	m_invRadius = 1.0f / radius;
}

float CSpherePrimitive::getSqrRadius()
{
	return m_sqrRadius;
}

float CSpherePrimitive::getInvRadius()
{
	return m_invRadius;
}

CVector3 CSpherePrimitive::getNormal( const CVector3& pos )
{
	return ((pos - m_center) * m_invRadius);
}

int CSpherePrimitive::intersect(CRay& ray, float& distance )
{
	CVector3 v = ray.getOrigin() - m_center;
	float b = -DOT(v, ray.getDirection());
	float det = (b * b) - DOT(v, v) + m_sqrRadius;
	int intResult = PRIM_MISS;

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
					intResult = PRIM_HITIN;
				}
			}
			else
			{
				if(point1 < distance)
				{
					distance = point1;
					intResult = PRIM_HIT;
				}
			}
		}
	}

	return intResult;
}

void CSpherePrimitive::setPosition( const CVector3& pos )
{
	m_center = pos;
}