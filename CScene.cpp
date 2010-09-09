////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#include "stdafx.h"

CScene::CScene()
{

}

CScene::~CScene()
{
	for(unsigned int i = 0; i < m_primitives.size(); ++i)
	{
		CBasePrimitive* prim = m_primitives[i];
		if(prim)
		{
			delete prim;
			prim = NULL;
		}
	}
	m_primitives.clear();
}

int CScene::getPrimitivesCount()
{
	return m_primitives.size();
}

CBasePrimitive* CScene::getPrimitive( unsigned int index )
{
	if(index < 0 && index >= m_primitives.size())
		return NULL;

	return m_primitives[index];
}

void CScene::addPrimitive( CBasePrimitive* prim )
{
	if(prim)
		m_primitives.push_back(prim);
}