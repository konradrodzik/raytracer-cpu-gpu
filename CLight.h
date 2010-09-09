////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#ifndef __H_CLight_H__
#define __H_CLight_H__

enum E_LIGHT_TYPE
{
	ELT_POINT = 1,
	ELT_AREA  = 2,
};

class CLight
{
public:
	// Default constructor
	CLight();

	// Initialize constructors
	CLight(CColor& color);

	// Initialize constructors
	CLight(CColor* color);

	// Get light type
	virtual E_LIGHT_TYPE getType() = 0;

	// Get color of light
	CColor& getColor();

	// Set color of light
	void setColor(CColor& color);

	// Set color of light
	void setColor(CColor* color);

	// Set color of light
	void setColor(float red, float green, float blue);

private:
	CColor m_color;		// Light color
};

#endif