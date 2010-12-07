////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#ifndef __H_CRTProfiler_H__
#define __H_CRTProfiler_H__

#include "CRayTracer.h"

// Struct for profiling single scene from raytracer (time in ms)
struct SProfiledScene 
{
	float m_frameTime;		// Whole time needed for scene rendering
	float m_intesectionTime;// Time needed for intersections
	float m_traceLightsTime;// Time needed for whole trace lights
	float m_reflectionTime;	// Time needed for reflections
	float m_refractionTime;	// Time needed for refractions
	float m_texturingTime;	// Time needed for texturing scene
	float m_specularTime;	// Time needed for specular calculation
	float m_lighteningTime;	// Time needed for lightening calculation
	float m_shadowsTime;	// Time needed for shadows calculation
	CScene* m_scene;		// Profiled scene

	// Default constructor
	SProfiledScene()
	{
		m_frameTime = 0.0f;
		m_intesectionTime = 0.0f;
		m_reflectionTime = 0.0f;
		m_refractionTime = 0.0f;
		m_texturingTime = 0.0f;
		m_specularTime = 0.0f;
		m_lighteningTime = 0.0f;
		m_shadowsTime = 0.0f;
		m_scene = NULL;
	}
};

// Class for raytracer profiling scenes
class CRTProfiler
{
public:
	// Default constructor
	CRTProfiler();

	// Default destructor
	~CRTProfiler();

	// Add scene profile
	bool addSceneProfile(SProfiledScene* profile);

	static void saveHardwareInfo(FILE* file);
	static void saveProfile(FILE* f, SProfiledScene* profile);
	static bool saveSceneProfiles(const char* fileName, CRTProfiler* profiler, E_COMPUTING_TYPE ect);
	static bool saveCPUProfiles(const char* fileName, std::vector<SProfiledScene*> profiles);
	static bool saveGPUProfiles(const char* fileName, std::vector<SProfiledScene*> profiles);

private:
	static float getCPUSpeed();


public:
	std::vector<SProfiledScene*> m_sceneProfiles;
};

#endif