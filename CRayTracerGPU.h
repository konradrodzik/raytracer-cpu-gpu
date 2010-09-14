////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#ifndef __H_CRayTracerGPU_H__
#define __H_CRayTracerGPU_H__

class CRayTracerGPU : public CRayTracer
{
public:
	// Initialize constructor
	CRayTracerGPU(int width, int height);

	// Get raytracer type
	virtual E_COMPUTING_TYPE getType();

	// Calculate scene
	virtual void calculateScene();

private:
	// Trace single ray
	virtual CColor traceRay(CRay& ray, int depthLevel);
};

#endif;