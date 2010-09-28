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

	// Set specular
	void setSpecular(float specular);

	// Get specular
	float getSpecular();

	// Set refraction
	void setRefraction(float refraction);

	// Get refraction
	float getRefraction();

	// Set refraction index
	void setRefrIndex(float index);

	// Get refraction index
	float getRefrIndex();

	// Set texture
	void setTexture(CTexture* tex);

	// Get texture
	CTexture* getTexture();

	// Set texture UV
	void setTextureUV(float u, float v);

	// Get texture U
	float getTexU();
	// Get texture V
	float getTexV();
	// Get invert texture U
	float getTexInvU();
	// Get invert texture V
	float getTexInvV();

private:
	CColor m_color;			// Material color
	float m_diffuse;		// Material diffuse
	float m_reflection;		// Material reflection
	float m_refraction;		// Material refraction
	float m_specular;		// Material specular
	float m_refractionIndex;	// Refraction index

	CTexture* m_texture;	// Material texture
	float m_texU;			// U texture scale
	float m_texV;			// V texture scale
	float m_invTexU;		// Invert U texture scale
	float m_invTexV;		// Invert V texture scale
};

#endif