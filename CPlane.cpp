////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#include "stdafx.h"

CPlane::CPlane()
: m_normal(0.0f, 0.0f, 0.0f)
, m_dPoint(0.0f)
{

}

CPlane::CPlane( const CVector3& normal, float d_point )
: m_normal(normal)
, m_dPoint(d_point)
{

}