////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#include "stdafx.h"

CPointLight::CPointLight()
: CLight()
, m_position(0.0f, 0.0f, 0.0f)
{
}

CPointLight::CPointLight( CVector3& position )
: CLight()
, m_position(position)
{
}

CPointLight::CPointLight( CVector3* position )
: CLight()
, m_position(*position)
{
}

CPointLight::CPointLight( CVector3& position, CColor& color )
: CLight(color)
, m_position(position)
{
}

CPointLight::CPointLight( CVector3* position, CColor* color )
: CLight(*color)
, m_position(*position)
{
}

E_LIGHT_TYPE CPointLight::getType()
{
	return ELT_POINT;
}

CVector3& CPointLight::getPosition()
{
	return m_position;
}

void CPointLight::setPosition( CVector3& position )
{
	m_position = position;
}

void CPointLight::setPosition( CVector3* position )
{
	m_position = *position;
}

void CPointLight::setPosition( float x, float y, float z )
{
	m_position = CVector3(x, y, z);
}
