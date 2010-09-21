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

void CMaterial::setTexture( CTexture* tex )
{
	m_texture = tex;
}

CTexture* CMaterial::getTexture()
{
	return m_texture;
}

void CMaterial::setTextureUV( float u, float v )
{
	m_texU = u;
	m_texV = v;
	m_invTexU = 1.0f / u;
	m_invTexV = 1.0f / v;
}

float CMaterial::getTexU()
{
	return m_texU;
}

float CMaterial::getTexV()
{
	return m_texV;
}

float CMaterial::getTexInvU()
{
	return m_invTexU;
}

float CMaterial::getTexInvV()
{
	return m_invTexV;
}