////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#include "stdafx.h"

CBasePrimitive::CBasePrimitive() : m_isLight(false)
{
}

void CBasePrimitive::setMaterial( CMaterial* material )
{
	m_material = *material;
}

CMaterial* CBasePrimitive::getMaterial()
{
	return &m_material;
}

void CBasePrimitive::setLight( bool light )
{
	m_isLight = light;
}

bool CBasePrimitive::isLight()
{
	return m_isLight;
}

void CBasePrimitive::setName( const std::string& name )
{
		m_name = name;
}

std::string CBasePrimitive::getName()
{
	return m_name;
}

CColor CBasePrimitive::getColor( const CVector3& pos )
{
	return m_material.getColor();
}