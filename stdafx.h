// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#include "targetver.h"

#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers
// Windows Header Files:
#include <windows.h>

// C RunTime Header Files
#include <stdlib.h>
#include <malloc.h>
#include <memory.h>
#include <tchar.h>
#pragma warning(disable:4996)


// TODO: reference additional headers your program requires here

#include <math.h>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>

#include <omp.h>
#include <cuda.h>

#include <d3dx9.h>
#include <dxerr.h>


#define _CRT_SECURE_NO_WARNINGS


// Forward declarations
class CVector3;
class CPlane;
class CAABBox;
class CRay;
class CMaterial;
class CTexture;
class CBasePrimitive;
class CPlanePrimitive;
class CBoxPrimitive;
class CSpherePrimitive;
class CLight;
class CPointLight;
class CAreaLight;
class CCamera;
class CScene;
class CRayTracer;
class CRayTracerCPU;
class CRayTracerGPU;

// Benchmark
#include "CMail.h"
#include "CProcessorInfo.h"

// Math
#include "MathFunctions.h"
#include "CVector3.h"
#include "CPlane.h"
#include "CAABBox.h"
#include "CRay.h"

// Lights
#include "CLight.h"
#include "CPointLight.h"
#include "CAreaLight.h"

// Primitives
#include "CMaterial.h"
#include "CTexture.h"
#include "CBasePrimitive.h"
#include "CPlanePrimitive.h"
#include "CBoxPrimitive.h"
#include "CSpherePrimitive.h"

#include "CCamera.h"
#include "CScene.h"
#include "CRayTracer.h"
#include "CRayTracerCPU.h"
#include "CRayTracerGPU.h"

#include "CFramework.h"

