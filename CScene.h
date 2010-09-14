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

public:
	std::vector<CBasePrimitive*> m_primitives;		// Scene primitives
	std::vector<CLight*> m_lights;					// Scene lights
	CCamera* m_camera;								// Camera on scene

	// Settings
	int m_width;						// Window width
	int m_height;						// Window height
};

#endif