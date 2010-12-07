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

	// Get texel of texture in u,v
	__device__ __host__ CColor getTexel(float u, float v)
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
	}

private:
	void loadTextureFromFile(const char* fileName);

private:
	int m_width;						// Texture width
	int m_height;						// Texture height
	CColor m_textureSurface[512*512];	// Texture surface
};

/*
// Bitmap header
typedef struct tagBITMAPFILEHEADER
{
	WORD bfType;		//specifies the file type
	DWORD bfSize;		//specifies the size in bytes of the bitmap file
	WORD bfReserved1;   //reserved; must be 0
	WORD bfReserved2;	//reserved; must be 0
	DWORD bOffBits;		//species the offset in bytes from the bitmapfileheader to the bitmap bits
}BITMAPFILEHEADER;

// Bitmap info header
typedef struct tagBITMAPINFOHEADER
{
	DWORD biSize;  //specifies the number of bytes required by the struct
	LONG biWidth;  //specifies width in pixels
	LONG biHeight;  //species height in pixels
	WORD biPlanes; //specifies the number of color planes, must be 1
	WORD biBitCount; //specifies the number of bit per pixel
	DWORD biCompression;//spcifies the type of compression
	DWORD biSizeImage;  //size of image in bytes
	LONG biXPelsPerMeter;  //number of pixels per meter in x axis
	LONG biYPelsPerMeter;  //number of pixels per meter in y axis
	DWORD biClrUsed;  //number of colors used by th ebitmap
	DWORD biClrImportant;  //number of colors that are important
}BITMAPINFOHEADER;*/

#endif