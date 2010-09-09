////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#ifndef __H_CPointLight_H__
#define __H_CPointLight_H__

class CPointLight : public CLight
{
public:
	// Default constructor
	CPointLight();

	// Initialize constructor (only position)
	CPointLight(CVector3& position);

	// Initialize constructor (only position)
	CPointLight(CVector3* position);

	// Initialize constructor (position & color)
	CPointLight(CVector3& position, CColor& color);

	// Initialize constructor (position & color)
	CPointLight(CVector3* position, CColor* color);

	// Get light type
	E_LIGHT_TYPE getType();

	// Get light position
	CVector3& getPosition();

	// Set position
	void setPosition(CVector3& position);

	// Set position
	void setPosition(CVector3* position);

	// Set position
	void setPosition(float x, float y, float z);

private:
	CVector3 m_position;	// Light position
};

#endif