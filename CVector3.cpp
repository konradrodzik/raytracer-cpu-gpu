////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#include "stdafx.h"

/*CVector3::CVector3() : m_x(0.0f), m_y(0.0f), m_z(0.0f)
{
}*/

/*CVector3::CVector3( float x, float y, float z ) : m_x(x), m_y(y), m_z(z)
{
}*/

float CVector3::getSqrLength()
{
	return (m_x*m_x + m_y*m_y + m_z*m_z);
}

void CVector3::operator+=( CVector3& vec )
{
	m_x += vec.m_x;
	m_y += vec.m_y;
	m_z += vec.m_z;
}

void CVector3::operator+=( CVector3* vec )
{
	m_x += vec->m_x;
	m_y += vec->m_y;
	m_z += vec->m_z;
}

void CVector3::operator-=( CVector3& vec )
{
	m_x -= vec.m_x;
	m_y -= vec.m_y;
	m_z -= vec.m_z;
}

void CVector3::operator-=( CVector3* vec )
{
	m_x -= vec->m_x;
	m_y -= vec->m_y;
	m_z -= vec->m_z;
}

void CVector3::operator*=( CVector3& vec )
{
	m_x *= vec.m_x;
	m_y *= vec.m_y;
	m_z *= vec.m_z;
}

void CVector3::operator*=( CVector3* vec )
{
	m_x *= vec->m_x;
	m_y *= vec->m_y;
	m_z *= vec->m_z;
}

void CVector3::operator*=( float scalar )
{
	m_x *= scalar;
	m_y *= scalar;
	m_z *= scalar;
}

CVector3 CVector3::operator-() const
{
	return CVector3(-m_x, -m_y, -m_z);
}

