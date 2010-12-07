////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#ifndef __H_CScene_H__
#define __H_CScene_H__

// Scene class
class CScene
{
public:
	// Default constructor
	CScene(int width, int height);

	// Destructor
	~CScene();

	// Get primitives count
	int getPrimitivesCount();

	// Get primitive
	CBasePrimitive* getPrimitive(unsigned int index);

	// Add primitive to scene
	void addPrimitive(CBasePrimitive* prim);

	// Load scene from file
	static CScene* loadScene(const char* sceneFile, int width, int height);

	// Parse scene file
	bool parseSceneFile(char* buffer);

	// Get camera pointer
	CCamera* getCamera();

	// Set camera pointer
	void setCamera(CCamera* camera)
	{
		m_camera = camera;
	}

	CSpherePrimitive* getSpherePrimitive()
	{
		for(unsigned int i = 0; i < m_primitives.size(); ++i)
		{
			if(m_primitives[i]->getType() == EPT_SPHERE)
			{
				return (CSpherePrimitive*)(m_primitives[i]);
			}
		}
		return NULL;
	}

	int getSphereCount();

	int getPlaneCount();

	int getBoxCount();

	void fillSphereArray(CSpherePrimitive* array);

	void fillPlaneArray(CPlanePrimitive* array);

	void fillBoxArray(CBoxPrimitive* array);

	void setName(char* name);

	const char* getName();

public:
	std::vector<CBasePrimitive*> m_primitives;		// Scene primitives
	std::vector<CLight*> m_lights;					// Scene lights
	CCamera* m_camera;								// Camera on scene
	std::vector<CTexture*> m_textures;				// All loaded textures
	std::string m_sceneName;						// Name of scene

	// Settings
	int m_width;						// Window width
	int m_height;						// Window height
};

#endif