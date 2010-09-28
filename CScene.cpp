////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#include "stdafx.h"

CScene::CScene(int width, int height)
: m_camera(NULL)
, m_width(width)
, m_height(height)
{

}

CScene::~CScene()
{
	// Delete primitives
	for(unsigned int i = 0; i < m_primitives.size(); ++i)
	{
		CBasePrimitive* prim = m_primitives[i];
		if(prim)
		{
			delete prim;
			prim = NULL;
		}
	}
	m_primitives.clear();

	// Delete lights
	for(unsigned int i = 0; i < m_lights.size(); ++i)
	{
		CLight* light = m_lights[i];
		if(light)
		{
			delete light;
			light = NULL;
		}
	}
	m_lights.clear();

	// Delete camera
	if(m_camera)
	{
		delete m_camera;
		m_camera = NULL;
	}
}

int CScene::getPrimitivesCount()
{
	return m_primitives.size();
}

CBasePrimitive* CScene::getPrimitive( unsigned int index )
{
	if(index < 0 && index >= m_primitives.size())
		return NULL;

	return m_primitives[index];
}

void CScene::addPrimitive( CBasePrimitive* prim )
{
	if(prim)
		m_primitives.push_back(prim);
}

CScene* CScene::loadScene( const char* sceneFile, int width, int height )
{
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
		return NULL;
	}

	// create buffer
	char* buffer = (char*)malloc(sizeof(char)*size);
	ZeroMemory(buffer, size);
	// read whole file
	fread(buffer, sizeof(char), size, file);
	fclose(file);

	// Create scene
	CScene* scene = new CScene(width, height);
	if(!scene->parseSceneFile(buffer))
	{
		delete scene;
		scene = NULL;
	}


	// free our buffer
	if(buffer)
		::free(buffer);

	return scene;
}

bool CScene::parseSceneFile( char* buffer )
{
	// Start parsing
	CBasePrimitive* primitive = NULL;
	char *cur = buffer;
	bool sectionOpened = false;
	bool done = false;
	bool error = false;
	bool camera = false;

	while (!done)
	{
		// Skip empty lines
		while (*cur=='\r' || *cur=='\n') cur++;

		// Get line start
		char *start = cur;

		//! Get line end
		while (*cur && *cur!='\r' && *cur!='\n') cur++;

		// End of file ?
		if (*cur==0) {
			done = 1;
			continue;
		}

		// Cut string
		*cur++ = 0;

		// create primitive section
		if(!primitive && !sectionOpened && !camera)
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
				addPrimitive(primitive);
			}
			else if(strcmp(primitiveType, "plane") == 0)
			{
				primitiveType = start+offset+1;
				while (*primitiveType && *primitiveType==' ') primitiveType++;
				primitive = new CPlanePrimitive();
				primitive->setName(primitiveType);
				addPrimitive(primitive);
			}
			else if(strcmp(primitiveType, "box") == 0)
			{
				primitiveType = start+offset+1;
				while (*primitiveType && *primitiveType==' ') primitiveType++;
				primitive = new CBoxPrimitive();
				primitive->setName(primitiveType);
				addPrimitive(primitive);
			}
			else if(strcmp(primitiveType, "camera") == 0)
			{
				if(m_camera) delete m_camera;
				m_camera = new CCamera(m_width, m_height);
				camera = true;
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
				if(m_camera)
					camera = false;
				primitive = NULL;
				continue;
			}
			else if(sectionOpened && (primitive || camera))
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

					if(primitive) 
					{
						if(strcmp(field, "texture") == 0)
						{
							char fielName[255];
							memset(fielName, 0, 255);
							sscanf(val_buff, "%s\n", fielName);
							primitive->getMaterial()->setTexture(new CTexture(fielName));

						}
						else if(strcmp(field, "position") == 0)
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
						else if(strcmp(field, "size") == 0)
						{
							float x=0.0f, y=0.0f, z=0.0f;
							sscanf(val_buff, "%f %f %f\n", &x, &y, &z);
							((CBoxPrimitive*)primitive)->setSize(CVector3(x, y, z));
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
						else if(strcmp(field, "refraction") == 0)
						{
							float refraction=0.0f;
							sscanf(val_buff, "%f\n", &refraction);
							primitive->getMaterial()->setRefraction(refraction);
						}
						else if(strcmp(field, "refraction_index") == 0)
						{
							float index=0.0f;
							sscanf(val_buff, "%f\n", &index);
							primitive->getMaterial()->setRefrIndex(index);
						}
						else if(strcmp(field, "specular") == 0)
						{
							float specular=0.0f;
							sscanf(val_buff, "%f\n", &specular);
							primitive->getMaterial()->setSpecular(specular);
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
					else if(m_camera) {
						if(strcmp(field, "position") == 0)
						{
							float x=0.0f, y=0.0f, z=0.0f;
							sscanf(val_buff, "%f %f %f\n", &x, &y, &z);
							m_camera->setPosition(CVector3(x, y, z));
						}
						else if(strcmp(field, "direction") == 0)
						{
							float x=0.0f, y=0.0f, z=0.0f;
							sscanf(val_buff, "%f %f %f\n", &x, &y, &z);
							m_camera->setDirection(CVector3(x, y, z));
						}
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

CCamera* CScene::getCamera()
{
	return m_camera;
}