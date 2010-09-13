////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#ifndef __H_CRayTracer_H__
#define __H_CRayTracer_H__

// Computing type
enum E_COMPUTING_TYPE
{
	ECT_CPU,	// CPU
	ECT_CUDA,	// GPU = CUDA
};

// Ray tracer class
class CRayTracer
{
public:
	// Initialize constructor
	CRayTracer(const char* window_title);

	// Destructor
	~CRayTracer();

	// Load scene
	bool loadScene(const char* sceneFile);

	// Parse scene file
	bool parseSceneFile(char* buffer);

	// Calculate frame
	void calculateFrame();

	// Set window size
	void setWindowSize(int width, int height);

	// Trace single ray
	CColor traceRay(CRay& ray, int depthLevel);

private:
	
	// Settings
	int m_width;						// Window width
	int m_height;						// Window height

	CScene* m_scene;					// Ray traced scene
	CCamera* m_camera;					// Camera
	CColor** m_screenColor;				// Screen buffer of color
};

#endif