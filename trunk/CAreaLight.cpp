////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#include "stdafx.h"

CAreaLight::CAreaLight()
: CLight()
, m_areaPrimitive(NULL)
{
	
}

CAreaLight::~CAreaLight()
{
	if(m_areaPrimitive)
	{
		delete m_areaPrimitive;
		m_areaPrimitive = NULL;
	}
}

E_LIGHT_TYPE CAreaLight::getType()
{
	return ELT_AREA;
}