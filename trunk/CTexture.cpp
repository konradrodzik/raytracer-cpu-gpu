////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#include "stdafx.h"

CTexture::CTexture()
: m_width(0)
, m_height(0)
{
	memset(m_textureSurface, 0, sizeof(CColor) * 512 * 512);
}

CTexture::CTexture( CColor* surface, int width, int height )
: m_width(width)
, m_height(height)
{
	memset(m_textureSurface, 0, sizeof(CColor) * 512 * 512);
}

CTexture::CTexture( const char* fileName )
: m_width(0)
, m_height(0)
{
	loadTextureFromFile(fileName);
}

CTexture::~CTexture()
{
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
	// TGA LOADING

	FILE* f = fopen( fileName, "rb" );
	if (f)
	{
		// extract width and height from file header
		unsigned char buffer[20];
		fread( buffer, 1, 20, f );
		m_width = *(buffer + 12) + 256 * *(buffer + 13);
		m_height = *(buffer + 14) + 256 * *(buffer + 15);
		fclose( f );
		// read pixel data
		f = fopen( fileName, "rb" );
		unsigned char* t = new unsigned char[m_width * m_height * 3 + 1024];
		fread( t, 1, m_width * m_height * 3 + 1024, f );
		fclose( f );
		// convert RGB 8:8:8 pixel data to floating point RGB
		//m_textureSurface = new CColor[m_width * m_height];
		memset(m_textureSurface, 0, sizeof(CColor) * m_width * m_height);
		float rec = 1.0f / 256;
		for ( int size = m_width * m_height, i = 0; i < size; i++ )
			m_textureSurface[i] = CColor( t[i * 3 + 20] * rec, t[i * 3 + 19] * rec, t[i * 3 + 18] * rec );
		delete t;
	}

	// BMP LOADING

	/*FILE* file = fopen(fileName, "rb");
	if(file)
	{
		BITMAPFILEHEADER bitmapFileHeader;
		BITMAPINFOHEADER bitmapInfoHeader;
		unsigned char* bitmapImage;

		// Read the bitmap file header
		fread(&bitmapFileHeader, sizeof(BITMAPFILEHEADER),1,file);

		// Verify that this is a bmp file by check bitmap id
		if (bitmapFileHeader.bfType != 0x4D42)
		{
			fclose(file);
			return;
		}

		// Read the bitmap info header
		fread(&bitmapInfoHeader, sizeof(BITMAPINFOHEADER),1,file);

		// Move file point to the begging of bitmap data
		fseek(file, bitmapFileHeader.bfOffBits, SEEK_SET);

		// Allocate enough memory for the bitmap image data
		bitmapImage = (unsigned char*)malloc(bitmapInfoHeader.biSizeImage);

		//verify memory allocation
		if (!bitmapImage)
		{
			free(bitmapImage);
			fclose(file);
			return;
		}

		m_textureSurface = new CColor[bitmapInfoHeader.biSizeImage/3];

		// Swap the r and b values to get RGB (bitmap is BGR)
		for (unsigned int imageIdx = 0;imageIdx < bitmapInfoHeader.biSizeImage;imageIdx+=3)
		{
			unsigned char tempRGB = bitmapImage[imageIdx];
			bitmapImage[imageIdx] = bitmapImage[imageIdx + 2];
			bitmapImage[imageIdx + 2] = tempRGB;

			m_textureSurface[imageIdx/3] = CColor(bitmapImage[imageIdx], bitmapImage[imageIdx+1], bitmapImage[imageIdx+2]);
		}
		m_width = bitmapInfoHeader.biWidth;
		m_height = bitmapInfoHeader.biHeight;

		free(bitmapImage);
	}*/
}

/*__device__ CColor CTexture::getTexel( float u, float v )
{
	float fu = (u + 1000.5f) * m_width;
	float fv = (v + 1000.0f) * m_width;
	int u1 = ((int)fu) % m_width;
	int v1 = ((int)fv) % m_height;
	int u2 = (u1 + 1) % m_width;
	int v2 = (v1 + 1) % m_height;

	float fracu = fu - floorf(fu);
	float fracv = fv - floorf(fv);

	// Calculate weights
	float w1 = (1 - fracu) * (1 - fracv);
	float w2 = fracu * (1 - fracv);
	float w3 = (1 - fracu) * fracv;
	float w4 = fracu *  fracv;

	CColor c1 = m_textureSurface[u1 + v1 * m_width];
	CColor c2 = m_textureSurface[u2 + v1 * m_width];
	CColor c3 = m_textureSurface[u1 + v2 * m_width];
	CColor c4 = m_textureSurface[u2 + v2 * m_width];

	return c1 * w1 + c2 * w2 + c3 * w3 + c4 * w4;
}*/