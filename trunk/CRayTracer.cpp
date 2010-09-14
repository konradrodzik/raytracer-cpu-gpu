////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#include "stdafx.h"

CRayTracer::CRayTracer( int width, int height)
: m_width(width)
, m_height(height)
, m_currentScene(NULL)
, m_currentSceneIndex(-1)
{
	// Constructor
	m_screenColor = new CColor*[m_height]; 
	for(int i = 0; i < m_height; i++) 
		m_screenColor[i] = new CColor[m_width]; 

	for(int i = 0; i < m_height; i++)
		for(int j = 0; j < m_width; j++)
			m_screenColor[i][j] = CColor(0.0f, 0.0f, 0.0f);

}

CRayTracer::~CRayTracer()
{
	// Delete scenes
	for(unsigned int i = 0; i < m_scenes.size(); ++i)
	{
		CScene* scene = m_scenes[i];
		if(scene)
		{
			delete scene;
			scene = NULL;
		}
	}
	m_scenes.clear();
	m_currentScene = NULL;
	

	// Delete screen color
	for(int i = 0; i < m_height; i++) 
		delete m_screenColor[i];

	delete m_screenColor;
}

void CRayTracer::setWindowSize( int width, int height )
{
	m_width = width;
	m_height = height;
}

CColor** CRayTracer::getScreenColorBuffer()
{
	return m_screenColor;
}

void CRayTracer::saveScene2Image( const char* fileName )
{
	std::ofstream imageFile(fileName, std::ios_base::binary);
	if (!imageFile)
		return;

	// Addition of the TGA header
	imageFile.put(0).put(0);
	imageFile.put(2);        /* RGB not compressed */

	imageFile.put(0).put(0);
	imageFile.put(0).put(0);
	imageFile.put(0);

	imageFile.put(0).put(0); /* origin X */ 
	imageFile.put(0).put(0); /* origin Y */

	imageFile.put((unsigned char)(m_width & 0x00FF)).put((unsigned char)((m_width & 0xFF00) / 256));
	imageFile.put((unsigned char)(m_height & 0x00FF)).put((unsigned char)((m_height & 0xFF00) / 256));
	imageFile.put(24);       /* 24 bit bitmap */ 
	imageFile.put(0); 
	// end of the TGA header

	int x, y;
	for(y = m_height-1; y >= 0; --y)
		for(x = m_width-1; x >= 0; --x)
			imageFile.put((unsigned char)MIN(m_screenColor[y][x].m_b,255.0f)).put((unsigned char)MIN(m_screenColor[y][x].m_g, 255.0f)).put((unsigned char)MIN(m_screenColor[y][x].m_r, 255.0f));

	imageFile.close();
}

bool CRayTracer::loadMaps( const char* benchFile )
{
	FILE* file = fopen(benchFile, "r");
	if(!file)
		return false;

	char buff[255];
	char fileName[255];
	int count = 0;
	if(fscanf(file, "%i\n", &count)!=1)
		return false;
	
	for(int i = 0; i < count; ++i) {
		memset(buff, 0, 255);
		memset(fileName, 0, 255);
		if(fscanf(file, "map: %s\n", buff)!=1)
			return false;


		sprintf(fileName,"maps/%s.rtm", buff);
		CScene* tmp = CScene::loadScene(fileName, m_width, m_height);
		if(!tmp)
			return false;

		m_scenes.push_back(tmp);
	}

	if(!m_scenes.size())
		return false;

	m_currentSceneIndex = 0;
	m_currentScene = m_scenes[m_currentSceneIndex];

	fclose(file);
	return true;
}