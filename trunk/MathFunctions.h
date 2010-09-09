////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#ifndef __H_MathFunctions_H__
#define __H_MathFunctions_H__

// Dot product of two vectors
#define DOT(a, b)		(a.m_x*b.m_x + a.m_y*b.m_y + a.m_z*b.m_z)

// Cross product of two vectors
#define CROSS(a, b)		(CVector3(a.m_y*b.m_z - a.m_z*b.m_y, a.m_z*b.m_x - a.m_x*b.m_z, a.m_x*b.m_y - a.m_y*b.m_x))

// Normalize vector
#define NORMALIZE(a)	{float len = 1.0f/sqrtf(a.m_x*a.m_x + a.m_y*a.m_y + a.m_z*a.m_z); a.m_x *= len; a.m_y *= len; a.m_z *= len;}

// Length of vector
#define LENGTH(a)		(sqrtf(a.m_x*a.m_x + a.m_y*a.m_y + a.m_z*a.m_z))

// Square length of vector
#define SQRLENGTH(a)	(a.m_x*a.m_x + a.m_y*a.m_y + a.m_z*a.m_z)

// PI number
#define PI				3.141592653589793238462f

#endif
