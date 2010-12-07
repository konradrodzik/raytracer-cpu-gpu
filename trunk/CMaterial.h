////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#ifndef __H_CMaterial_H__
#define __H_CMaterial_H__

#include "CTexture.h"

class CMaterial
{
public:
	// Default constructor
	CMaterial();

	// Set color
	void setColor(CColor& color);

	// Get color
	__device__ __host__ CColor getColor()
	{
		return m_color;
	}

	__device__ float3 getColorEx()
	{
		return make_float3(m_color.m_x, m_color.m_y, m_color.m_z);
	}

	// Set diffuse 
	void setDiffuse(float diffuse);

	// Get diffuse
	__device__ float getDiffuse()
	{
		return m_diffuse;
	}

	// Set reflection
	void setReflection(float reflection);

	// Get reflection
	__device__ float getReflection()
	{
		return m_reflection;
	}

	// Set specular
	void setSpecular(float specular);

	// Get specular
	__device__ float getSpecular()
	{
		return m_specular;
	}

	// Set refraction
	void setRefraction(float refraction);

	// Get refraction
	__device__ float getRefraction()
	{
		return m_refraction;
	}

	// Set refraction index
	void setRefrIndex(float index);

	// Get refraction index
	__device__ float getRefrIndex()
	{
		return m_refractionIndex;
	}

	// Set texture
	void setTexture(CTexture* tex);

	// Get texture
	CTexture& getTexture()
	{
		return m_texture;
	}

	// Set texture UV
	void setTextureUV(float u, float v);

	// Get texture U
	__device__ __host__ float getTexU()
	{
		return m_texU;
	}
	// Get texture V
	__device__ __host__ float getTexV()
	{
		return m_texV;
	}
	// Get invert texture U
	__device__ __host__ float getTexInvU()
	{
		return m_invTexU;
	}

	// Get invert texture V
	__device__ __host__ float getTexInvV()
	{
		return m_invTexV;
	}

	__device__ __host__ bool isTexture()
	{
		return m_isTexture;
	}

	void setTextureFlag(bool isTexture);

	__device__ __host__ CColor getTexel(float u, float v)
	{
		return m_texture.getTexel(u, v);
	}

private:
	CColor m_color;			// Material color
	float m_diffuse;		// Material diffuse
	float m_reflection;		// Material reflection
	float m_refraction;		// Material refraction
	float m_specular;		// Material specular
	float m_refractionIndex;	// Refraction index

	CTexture m_texture;		// Material texture
	bool m_isTexture;		// Is texture applied to this material
	float m_texU;			// U texture scale
	float m_texV;			// V texture scale
	float m_invTexU;		// Invert U texture scale
	float m_invTexV;		// Invert V texture scale
};

#endif