////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#include "stdafx.h"

CLight::CLight()
: m_color(CVector3(0.0f, 0.0f, 0.0f))
{
}

CLight::CLight( CColor& color )
: m_color(color)
{
}

CLight::CLight( CColor* color )
: m_color(*color)
{
}

CColor& CLight::getColor()
{
	return m_color;
}

void CLight::setColor( CColor& color )
{
	m_color = m_color;
}

void CLight::setColor( CColor* color )
{
	m_color = *color;
}

void CLight::setColor( float red, float green, float blue )
{
	m_color = CColor(red, green, blue);
}