////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#ifndef __H_CRayTracerCPU_H__
#define __H_CRayTracerCPU_H__

class CRayTracerCPU : public CRayTracer
{
public:
	// Initialize constructor
	CRayTracerCPU(int width, int height);

	// Destructor
	~CRayTracerCPU();

	// Get raytracer type
	virtual E_COMPUTING_TYPE getType();

	// Calculate scene
	virtual void calculateScene();

private:
	// Trace single ray
	virtual CColor traceRay(CRay& ray, int depthLevel);

	unsigned int m_wholeTimer;			// Main timer for whole raytraced scene
	unsigned int m_intersectionTimer;	// Timer for profiling interections
	unsigned int m_reflectionTimer;		// Timer for profiling reflections
	unsigned int m_refractionTimer;		// Timer for profiling refractions
	unsigned int m_texturingTimer;		// Timer for profiling texturing
	unsigned int m_specularTimer;		// Timer for profiling specular
	unsigned int m_lighteningTimer;		// Timer for profiling lightening
	unsigned int m_shadowsTimer;		// Timer for profiling shadows
	unsigned int m_traceLightsTimer;	// Timer for profiling whole trace lights	
};

#endif;