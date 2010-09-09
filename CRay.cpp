////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#include "stdafx.h"

CRay::CRay() : m_origin(CVector3(0.0f, 0.0f, 0.0f)), m_direction(CVector3(0.0f, 0.0f, 0.0f))
{
}

CRay::CRay( CVector3& origin, CVector3& dir ) : m_origin(origin), m_direction(dir)
{
}

void CRay::setOrigin( CVector3& origin )
{
	m_origin = origin;
}

void CRay::setDirection( CVector3& direction )
{
	m_direction = direction;
}

CVector3& CRay::getOrigin()
{
	return m_origin;
}

CVector3& CRay::getDirection()
{
	return m_direction;
}