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



// TODO: reference additional headers your program requires here

#include <math.h>
#include <string>
#include <vector>

#include <d3dx9.h>
#include <dxerr.h>


#define _CRT_SECURE_NO_WARNINGS


// Forward declarations
class CVector3;
class CPlane;
class CRay;
class CMaterial;
class CBasePrimitive;
class CPlanePrimitive;
class CSpherePrimitive;
class CLight;
class CPointLight;
class CCamera;
class CScene;
class CRayTracer;

// Benchmark
#include "CMail.h"

// Math
#include "MathFunctions.h"
#include "CVector3.h"
#include "CPlane.h"
#include "CRay.h"

// Lights
#include "CLight.h"
#include "CPointLight.h"

// Primitives
#include "CMaterial.h"
#include "CBasePrimitive.h"
#include "CPlanePrimitive.h"
#include "CSpherePrimitive.h"

#include "CCamera.h"
#include "CScene.h"
#include "CRayTracer.h"

#include "CFramework.h"

