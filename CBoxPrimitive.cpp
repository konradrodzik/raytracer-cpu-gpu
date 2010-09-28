////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#include "stdafx.h"

CBoxPrimitive::CBoxPrimitive()
: m_box(CVector3(0.0f, 0.0f, 0.0f), CVector3(0.0f, 0.0f, 0.0f))
{

}

CBoxPrimitive::CBoxPrimitive( CAABBox& box )
: m_box(box)
{

}

E_PRIMITIVE_TYPE CBoxPrimitive::getType()
{
	return EPT_BOX;
}

int CBoxPrimitive::intersect( CRay& ray, float& distance )
{
	float tmin, tmax, tymin, tymax, tzmin, tzmax;
	CVector3 invdir;
	CVector3 direction = ray.getDirection();
	if(direction.m_x != 0.0f)
		invdir.m_x = 1.0f / direction.m_x;
	if(direction.m_y != 0.0f)
		invdir.m_y = 1.0f / direction.m_y;
	if(direction.m_z != 0.0f)
		invdir.m_z = 1.0f / direction.m_z;

	int sign[3];
	sign[0] = invdir.m_x < 0;
	sign[1] = invdir.m_y < 0;
	sign[2] = invdir.m_z < 0;
	CVector3 bounds[2];
	bounds[0] = m_box.getPosition();
	bounds[1] = m_box.getPosition() + m_box.getSize();

	tmin = (bounds[sign[0]].m_x - ray.getOrigin().m_x) * invdir.m_x;
	tmax = (bounds[1-sign[0]].m_x - ray.getOrigin().m_x) * invdir.m_x;
	tymin = (bounds[sign[1]].m_y - ray.getOrigin().m_y) * invdir.m_y;
	tymax = (bounds[1-sign[1]].m_y - ray.getOrigin().m_y) * invdir.m_y;

	if ( tmin > tymax || tymin > tmax)
		return PRIM_MISS;
	if (tymin > tmin)
		tmin = tymin;
	if (tymax < tmax)
		tmax = tymax;

	tzmin = (bounds[sign[2]].m_z - ray.getOrigin().m_z) * invdir.m_z;
	tzmax = (bounds[1-sign[2]].m_z - ray.getOrigin().m_z) * invdir.m_z;
	if ( (tmin > tzmax) || (tzmin > tmax) )
		return PRIM_MISS;
	if (tzmin > tmin)
		tmin = tzmin;
	if (tzmax < tmax)
		tmax = tzmax;

	if(tmin > 0.1f && tmin < distance)
	{
		distance = tmin;
		return PRIM_HIT;
	}

	return PRIM_MISS;
}

CVector3 CBoxPrimitive::getNormal( const CVector3& pos )
{
	float dist[6];
	CVector3 position = m_box.getPosition();
	CVector3 size = m_box.getSize();
	dist[0]  = (pos - position.m_x * CVector3(1.0f, 0.0f, 0.0f)).getLength();
	dist[1]  = (pos - position.m_y * CVector3(0.0f, 1.0f, 0.0f)).getLength();
	dist[2]  = (pos - position.m_z * CVector3(0.0f, 0.0f, 1.0f)).getLength();
	dist[3]  = (pos - size.m_x * CVector3(-1.0f, 0.0f, 0.0f)).getLength();
	dist[4]  = (pos - size.m_y * CVector3(0.0f, -1.0f, 0.0f)).getLength();
	dist[5]  = (pos - size.m_z * CVector3(0.0f, 0.0f, -1.0f)).getLength();

	int best = 0;
	float bestdist = dist[0];
	for(int i = 1; i < 6; i++)
		if(bestdist > dist[i])
		{
			best = i;
			bestdist = dist[i];
		}

		if(best == 0)
			return CVector3(-1.0f, 0.0f, 0.0f);
		else if(best == 1)
			return CVector3(0.0f, -1.0f, 0.0f);
		else if(best == 2)
			return CVector3(0.0f, 0.0f, -1.0f);
		else if(best == 3)
			return CVector3(1.0f, 0.0f, 0.0f);
		else if(best == 4)
			return CVector3(0.0f, 1.0f, 0.0f);
		else
			return CVector3(0.0f, 0.0f, 1.0f);
}

void CBoxPrimitive::setPosition(CVector3& pos )
{
	m_box.setPosition(pos);
}

void CBoxPrimitive::setSize( CVector3& size )
{
	m_box.setSize(size);
}