////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#ifndef __H_CMaterial_H__
#define __H_CMaterial_H__

class CMaterial
{
public:
	// Default constructor
	CMaterial();

	// Set color
	void setColor(CColor& color);

	// Get color
	CColor getColor();

	// Set diffuse 
	void setDiffuse(float diffuse);

	// Get diffuse
	float getDiffuse();

	// Set reflection
	void setReflection(float reflection);

	// Get reflection
	float getReflection();

	// Get specular
	float getSpecular();

private:
	CColor m_color;
	float m_diffuse;
	float m_reflection;
};

#endif