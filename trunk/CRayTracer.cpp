////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#include "stdafx.h"

// DirectX vertex structure
struct D3DVERTEX
{
	D3DXVECTOR3 position;	// 3D position
	D3DXVECTOR2 texcoords;	// texture coordinates
};

CRayTracer::CRayTracer( const char* window_title )
: m_width(500)
, m_height(500)
, m_scene(NULL)
{
	// Constructor
	m_screenColor = new CColor*[m_height]; 
	for(int i = 0; i < m_height; i++) 
		m_screenColor[i] = new CColor[m_width]; 

	for(int i = 0; i < m_height; i++)
		for(int j = 0; j < m_width; j++)
			m_screenColor[i][j] = CColor(255.0f, 0.0f, 0.0f);

	m_camera = new CCamera(m_width, m_height);
	m_camera->setPosition(CVector3(320.0f, 220.0f, -380.0f));
	m_camera->setDirection(CVector3(0.0f, 0.0f, 1.0f));
}

CRayTracer::~CRayTracer()
{
	if(m_scene)
	{
		delete m_scene;
		m_scene = NULL;
	}
	
	// delete screen texture, technique, effect, vertex buffer, vertex declaration, screenCOlor, camera


}

bool CRayTracer::loadScene( const char* sceneFile )
{
	// Delete old scene
	if(m_scene)
	{
		delete m_scene;
		m_scene = NULL;
	}

	// Create new one
	m_scene = new CScene;

	FILE* file = fopen(sceneFile, "r");
	if(!file)
		return false;

	// get file size
	fseek(file, 0, SEEK_END);
	int size = ftell(file);
	rewind(file);

	// empty file?
	if(!size)
	{
		fclose(file);
		return false;
	}

	// create buffer
	char* buffer = (char*)malloc(sizeof(char)*size);
	ZeroMemory(buffer, size);
	// read whole file
	fread(buffer, sizeof(char), size, file);
	fclose(file);


	bool result = parseSceneFile(buffer);

	// free our buffer
	if(buffer)
		::free(buffer);

	return result;
}

bool CRayTracer::parseSceneFile( char* buffer )
{
	// Start parsing
	CBasePrimitive* primitive = NULL;
	char *cur = buffer;
	bool sectionOpened = false;
	bool done = false;
	bool error = false;
	while (!done)
	{
		// Skip empty lines
		while (*cur=='\r' || *cur=='\n') cur++;

		// Get line start
		char *start = cur;

		//! Get line end
		while (*cur && *cur!='\r' && *cur!='\n') cur++;

		// End of file ?
		if (*cur==0) done = 1;

		// Cut string
		*cur++ = 0;

		// create primitive section
		if(!primitive && !sectionOpened)
		{
			char *primitiveType = start;
			int offset = 0;
			while (*primitiveType && *primitiveType!=':')
			{
				primitiveType++;
				offset++;
			}

			primitiveType = start;
			primitiveType[offset] = 0;

			if(strcmp(primitiveType, "sphere") == 0)
			{
				primitiveType = start+offset+1;
				while (*primitiveType && *primitiveType==' ') primitiveType++;
				primitive = new CSpherePrimitive();
				primitive->setName(primitiveType);
				m_scene->addPrimitive(primitive);
			}
			else if(strcmp(primitiveType, "plane") == 0)
			{
				primitiveType = start+offset+1;
				while (*primitiveType && *primitiveType==' ') primitiveType++;
				primitive = new CPlanePrimitive();
				primitive->setName(primitiveType);
				m_scene->addPrimitive(primitive);
			}
			else
			{
				// there is error
				error = true;
				break;
			}
		}
		else
		{
			// section opened
			if(*start == '{' && !sectionOpened)
			{
				sectionOpened = true;
				continue;
			}
			else if(*start == '}' && sectionOpened)
			{
				sectionOpened = false;
				primitive = NULL;
				continue;
			}
			else if(sectionOpened && primitive)
			{
				// Parse single field in primitive
				char* field = start;
				if(*field == '[')
				{
					int offset = 0;
					while (*field && *field!=']') 
					{
						field++;
						offset++;
					}

					field = start+1;
					field[offset-1] = 0;

					char *val_buff = start+offset+1;
					while (*val_buff && *val_buff==' ') val_buff++;

					if(strcmp(field, "position") == 0)
					{
						float x=0.0f, y=0.0f, z=0.0f;
						sscanf(val_buff, "%f %f %f\n", &x, &y, &z);
						primitive->setPosition(CVector3(x, y, z));
					}
					else if(strcmp(field, "radius") == 0)
					{
						float radius=0.0f;
						sscanf(val_buff, "%f\n", &radius);
						((CSpherePrimitive*)primitive)->setRadius(radius);
					}
					else if(strcmp(field, "color") == 0)
					{
						float r=0.0f, g=0.0f, b=0.0f;
						sscanf(val_buff, "%f %f %f\n", &r, &g, &b);
						primitive->getMaterial()->setColor(CColor(r, g, b));
					}
					else if(strcmp(field, "diffuse") == 0)
					{
						float diffuse=0.0f;
						sscanf(val_buff, "%f\n", &diffuse);
						primitive->getMaterial()->setDiffuse(diffuse);
					}
					else if(strcmp(field, "reflection") == 0)
					{
						float reflection=0.0f;
						sscanf(val_buff, "%f\n", &reflection);
						primitive->getMaterial()->setReflection(reflection);
					}
					else if(strcmp(field, "light") == 0)
					{
						int light=0;
						sscanf(val_buff, "%i\n", &light);
						primitive->setLight(light);
					}
					else if(strcmp(field, "normal") == 0)
					{
						float x=0.0f, y=0.0f, z=0.0f;
						sscanf(val_buff, "%f %f %f\n", &x, &y, &z);
						((CPlanePrimitive*)primitive)->setNormal(CVector3(x, y, z));
					}
					else if(strcmp(field, "d") == 0)
					{
						float d=0.0f;
						sscanf(val_buff, "%f\n", &d);
						((CPlanePrimitive*)primitive)->setD(d);
					}
				}
			}
			else
			{
				error = true;
				break;
			}
		}
	}

	return !error;
}

void CRayTracer::calculateFrame()
{
	// Lock texture
	D3DLOCKED_RECT lockedRectSurf;
	m_screenTexture->LockRect(0, &lockedRectSurf, NULL, D3DLOCK_DISCARD);
	DWORD* pDataSurf = (DWORD*)(lockedRectSurf.pBits);
	int offset = lockedRectSurf.Pitch / 4;

	// Clear output color
	CColor outputColor = CColor(0.0f, 0.0f, 0.0f);
	float r = 0.0f;
	float g = 0.0f;
	float b = 0.0f;
	m_camera->initialize();

	for(int y = 0; y < m_height; ++y)
	for(int x = 0; x < m_width; ++x)
	{
		//float fragmentX = (float)x;
		//float fragmentY = (float)y;
		for(float fragmentX = (float)x; fragmentX < (float)x + 1.0f; fragmentX += 0.5f)
		for(float fragmentY = (float)y; fragmentY < (float)y + 1.0f; fragmentY += 0.5f)
		{
			CRay ray;
			ray.setOrigin(CVector3(0.0f, 0.0f, -5.0f));
			ray.setOrigin(m_camera->getPosition());
			m_camera->calcRayDir(ray, fragmentX, fragmentY);
			CColor resultColor = traceRay(ray, 1);
			outputColor += 0.25f * resultColor;
			//outputColor = resultColor;
		}
			
		// Set output color to texture
		r = outputColor.m_r*255.0f;
		g = outputColor.m_g*255.0f;
		b = outputColor.m_b*255.0f;

		pDataSurf[offset * y + x] = D3DCOLOR_XRGB((int)r, (int)g, (int)b);
		// Clear output color
		outputColor = CColor(0.0f, 0.0f, 0.0f);
	}

	// Unlock texture
	m_screenTexture->UnlockRect(0);
}


void CRayTracer::setWindowSize( int width, int height )
{
	m_width = width;
	m_height = height;
}

CColor CRayTracer::traceRay( CRay& ray, int depthLevel )
{
	float distance = 10000000.0f;
	CColor outColor = CColor(0.0f, 0.0f, 0.0f);
	CBasePrimitive* hitPrimitive = NULL;
	int result;
	CVector3 pi;

	for(std::vector<CBasePrimitive*>::iterator itor = m_scene->m_primitives.begin(); itor != m_scene->m_primitives.end(); ++itor)
	{
		
		if(int res = (*itor)->intersect(ray, distance))
		{
			hitPrimitive = *itor;
			result = res;
		}
	}

	// no hit, terminate ray
	if (!hitPrimitive) return CColor(0.0f, 0.0f, 0.0f);
	
	// handle intersection
	if (hitPrimitive->isLight())
	{
		// we hit a light, stop tracing
		outColor = CColor( 1.0f, 1.0f, 1.0f );
	}
	else
	{
		// determine color at point of intersection
		pi = ray.getOrigin() + ray.getDirection() * distance;
		// trace lights
		for ( int l = 0; l < m_scene->getPrimitivesCount(); l++ )
		{
			CBasePrimitive* p = m_scene->getPrimitive( l );
			if (p->isLight()) 
			{
				CBasePrimitive* light = p;
				// calculate diffuse shading
				CVector3 L = ((CSpherePrimitive*)light)->getCenter() - pi;
				NORMALIZE(L);
				CVector3 N = hitPrimitive->getNormal( pi );
				if (hitPrimitive->getMaterial()->getDiffuse() > 0)
				{
					float dot = DOT( N, L );
					if (dot > 0)
					{
						float diff = dot * hitPrimitive->getMaterial()->getDiffuse();
						// add diffuse component to ray color
						outColor += diff * hitPrimitive->getMaterial()->getColor() * light->getMaterial()->getColor();
					}
				}
			}
		}
	}

	// return color
	return outColor;
}