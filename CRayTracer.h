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
	CRayTracer(int width, int height);

	// Destructor
	~CRayTracer();

	// Set window size
	void setWindowSize(int width, int height);

	// Get raytracer type
	virtual E_COMPUTING_TYPE getType() = 0;

	// Calculate scene
	virtual void calculateScene() = 0;

	// Trace single ray
	virtual CColor traceRay(CRay& ray, int depthLevel) = 0;

	// Get screen color buffer
	CColor** getScreenColorBuffer();

	// Save scene to image
	void saveScene2Image(const char* fileName);

	// Load maps from benchmark file
	bool loadMaps(const char* benchFile);

	// Go to next scene in benchmark scenes
	bool nextScene();

	// Get count of scenes in benchmark
	int getScenesCount();

	// Get Acctual scene index
	int getCurrentSceneIndex();

public:
	std::vector<CScene*> m_scenes;		// Benchmark scenes
	int m_currentSceneIndex;			// Current scene index
	CScene* m_currentScene;				// Current scene object
	CColor** m_screenColor;				// Screen buffer of color

	// Settings
	int m_width;						// Window width
	int m_height;						// Window height
};

#endif