////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#include "stdafx.h"

CTexture::CTexture()
: m_textureSurface(NULL)
, m_width(0)
, m_height(0)
{

}

CTexture::CTexture( CColor* surface, int width, int height )
: m_textureSurface(surface)
, m_width(width)
, m_height(height)
{

}

CTexture::CTexture( const char* fileName )
{
	loadTextureFromFile(fileName);
}

CTexture::~CTexture()
{
	if(m_textureSurface)
	{
		delete m_textureSurface;
		m_textureSurface = NULL;
	}
}

int CTexture::getWidth()
{
	return m_width;
}

int CTexture::getHeight()
{
	return m_height;
}

CColor* CTexture::getSurface()
{
	return m_textureSurface;
}

void CTexture::loadTextureFromFile( const char* fileName )
{
	FILE* file = fopen(fileName, "rb");
	if(file)
	{
		unsigned char buffer[20];
		fread(buffer, 1, 20, file);
		m_width = *(buffer + 12) + 256 * *(buffer + 13);
		m_height = *(buffer + 14) + 256 * *(buffer + 15);

		// Rewind file
		rewind(file);

		unsigned char* t = new unsigned char[m_width * m_height * 3 + 1024];
		fread(t, 1, m_width * m_height * 3 + 1024, file);
		fclose(file);

		m_textureSurface = new CColor[m_width * m_height];
		float rec = 1.0f / 256;
		for (int size = m_width * m_height, i = 0; i < size; i++)
			m_textureSurface[i] = CColor( t[i * 3 + 20] * rec, t[i * 3 + 19] * rec, t[i * 3 + 18] * rec );
		delete t;
	}
}