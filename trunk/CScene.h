////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#ifndef __H_CScene_H__
#define __H_CScene_H__

// Scene class
class CScene
{
public:
	// Default constructor
	CScene();

	// Destructor
	~CScene();

	// Get primitives count
	int getPrimitivesCount();

	// Get primitive
	CBasePrimitive* getPrimitive(unsigned int index);

	// Add primitive to scene
	void addPrimitive(CBasePrimitive* prim);

public:
	std::vector<CBasePrimitive*> m_primitives;		// Scene primitives
	std::vector<CLight*> m_lights;					// Scene lights
};

#endif