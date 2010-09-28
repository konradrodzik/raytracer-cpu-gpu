////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#include "stdafx.h"

CAABBox::CAABBox()
: m_position(CVector3(0.0f, 0.0f, 0.0f))
, m_size(CVector3(0.0f, 0.0f, 0.0f))
{

}

CAABBox::CAABBox( CVector3& position, CVector3& size )
: m_position(position)
, m_size(size)
{

}

void CAABBox::setPosition( CVector3& position )
{
	m_position = position;
}

CVector3& CAABBox::getPosition()
{
	return m_position;
}

void CAABBox::setSize( CVector3& size )
{
	m_size = size;
}

CVector3& CAABBox::getSize()
{
	return m_size;
}