////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#ifndef __H_CTexture_H__
#define __H_CTexture_H__

class CTexture
{
public:
	// Default constructor
	CTexture();

	// Initialize constructor
	CTexture(CColor* surface, int width, int height);
	CTexture(const char* fileName);

	// Destructor
	~CTexture();

	// Get texture width
	int getWidth();

	// Get texture height
	int getHeight();

	// Get texture surface
	CColor* getSurface();

private:
	void loadTextureFromFile(const char* fileName);

private:
	int m_width;				// Texture width
	int m_height;					// Texture height
	CColor* m_textureSurface;	// Texture surface
};

#endif