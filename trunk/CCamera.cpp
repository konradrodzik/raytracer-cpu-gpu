////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#include "stdafx.h"

CCamera::CCamera()
: m_position(CVector3(0.0f, 0.0f, 0.0f))
, m_direction(CVector3(0.0f, 0.0f, 0.0f))
, m_up(CVector3(0.0f, 1.0f, 0.0f))
, m_screenWidth(0)
, m_screenHeight(0)
, m_fovX(0.0f)
, m_fovY(0.0f)
{

}

CCamera::CCamera( int screenWidth, int screenHeight )
: m_position(CVector3(0.0f, 0.0f, 0.0f))
, m_direction(CVector3(0.0f, 0.0f, 0.0f))
, m_up(CVector3(0.0f, 1.0f, 0.0f))
, m_screenWidth(screenWidth)
, m_screenHeight(screenHeight)
, m_fovX(1)
, m_fovY(1)
//, m_fovY(tanf(m_fovX * ((float)m_screenWidth / (float)m_screenHeight)))
{

}

void CCamera::setPosition( const CVector3& pos )
{
	m_position = pos;
}

CVector3& CCamera::getPosition()
{
	return m_position;
}

void CCamera::setDirection( const CVector3& dir )
{
	m_direction = dir;
}

CVector3& CCamera::getDirection()
{
	return m_direction;
}

void CCamera::setUP( const CVector3& up )
{
	m_up = up;
}

CVector3& CCamera::getUP()
{
	return m_up;
}

void CCamera::initialize()
{
	CVector3 right = CROSS(m_direction, m_up);
	right.normalize();

	CVector3 bottom = CROSS(m_direction, right);
	bottom.normalize();

	m_dx = ((2.0f * tanf(m_fovX)) / (float)m_screenWidth) * right;
	m_dy = ((2.0f * tanf(m_fovY)) / (float)m_screenHeight) * bottom;// * 0.5;

	m_direction.normalize();
}

void CCamera::calcRayDir( CRay& ray, float screenPixelX, float screenPixelY )
{
	CVector3 dir = m_direction + ((2.0f * screenPixelX + 1.0f  - (float)m_screenWidth) * 0.5f) * m_dx
							   + ((2.0f * screenPixelY + 1.0f - (float)m_screenHeight) * 0.5f) * m_dy;

	dir.normalize();
	ray.setDirection(dir);
}