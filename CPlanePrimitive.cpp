////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#include "stdafx.h"

CPlanePrimitive::CPlanePrimitive( const CVector3& normal, float d_point )
: m_plane(CPlane(normal, d_point))
{

}

CPlanePrimitive::CPlanePrimitive()
: m_plane()
{

}

E_PRIMITIVE_TYPE CPlanePrimitive::getType()
{
	return EPT_PLANE;
}

CVector3& CPlanePrimitive::getNormal()
{
	return m_plane.m_normal;
}

CVector3 CPlanePrimitive::getNormal( const CVector3& pos )
{
	return m_plane.m_normal;
}

float CPlanePrimitive::getDPoint()
{
	return m_plane.m_dPoint;
}

int CPlanePrimitive::intersect( CRay& ray, float& distance )
{
	float d = DOT(m_plane.m_normal, ray.getDirection());

	if(d != 0)
	{
		float dist = -(DOT(m_plane.m_normal, ray.getOrigin()) + m_plane.m_dPoint) / d;
		if(dist > 0 && dist < distance)
		{
			distance = dist;
			return PRIM_HIT;
		}
	}

	return PRIM_MISS;
}

void CPlanePrimitive::setNormal( const CVector3& normal )
{
	m_plane.m_normal = normal;
}

void CPlanePrimitive::setD( float d )
{
	m_plane.m_dPoint = d;
}

void CPlanePrimitive::setPosition( const CVector3& pos )
{
	m_plane.m_normal = pos;
}