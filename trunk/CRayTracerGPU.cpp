////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#include "stdafx.h"

CRayTracerGPU::CRayTracerGPU( int width, int height )
: CRayTracer(width, height)
{

}

E_COMPUTING_TYPE CRayTracerGPU::getType()
{
	return ECT_CUDA;
}

void CRayTracerGPU::calculateScene()
{

}

CColor CRayTracerGPU::traceRay( CRay& ray, int depthLevel )
{
	return CColor();
}