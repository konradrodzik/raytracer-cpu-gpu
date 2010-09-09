////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#include "stdafx.h"

CMaterial::CMaterial() : m_color(CColor(0.2f, 0.2f, 0.2f)), m_diffuse(0.2f), m_reflection(0.0f)
{
}

void CMaterial::setColor( CColor& color )
{
	m_color = color;
}

CColor CMaterial::getColor()
{
	return m_color;
}

void CMaterial::setDiffuse( float diffuse )
{
	m_diffuse = diffuse;
}

float CMaterial::getDiffuse()
{
	return m_diffuse;
}

void CMaterial::setReflection( float reflection )
{
	m_reflection = reflection;
}

float CMaterial::getReflection()
{
	return m_reflection;
}

float CMaterial::getSpecular()
{
	return (1.0f - m_diffuse);
}