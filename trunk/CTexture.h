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
	CColor getTexel(float u, float v);

private:
	void loadTextureFromFile(const char* fileName);

private:
	int m_width;				// Texture width
	int m_height;				// Texture height
	CColor* m_textureSurface;	// Texture surface
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