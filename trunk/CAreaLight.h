////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#ifndef __H_CAreaLight_H__
#define __H_CAreaLight_H__

class CAreaLight : public CLight
{
public:
	// Default constructor
	CAreaLight();

	// Destructor
	~CAreaLight();

	// Get light type
	E_LIGHT_TYPE getType();

private:
	CBasePrimitive* m_areaPrimitive;	// Primitive for area light
};

#endif