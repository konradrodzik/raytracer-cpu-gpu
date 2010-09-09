////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#include "stdafx.h"

// DirectX vertex structure
struct D3DVERTEX
{
	D3DXVECTOR3 position;	// 3D position
	D3DXVECTOR2 texcoords;	// texture coordinates
};

CRayTracer::CRayTracer( const char* window_title )
: m_windowTitle(window_title)
, m_width(500)
, m_height(500)
, m_fullscreen(false)
, m_refreshRate(32)
, m_scene(NULL)
, m_D3D9(NULL)
, m_D3D9Dev(NULL)
, m_HWND(NULL)
, m_Instance(NULL)
, m_effect(NULL)
, m_technique(NULL)
, m_screenTexture(NULL)
, m_screenVB(NULL)
, m_vertexDeclaration(NULL)
, m_vertexShader(NULL)
, m_pixelShader(NULL)
, m_constantTable(NULL)
{
	// Constructor
	m_screenColor = new CColor*[m_height]; 
	for(int i = 0; i < m_height; i++) 
		m_screenColor[i] = new CColor[m_width]; 

	for(int i = 0; i < m_height; i++)
		for(int j = 0; j < m_width; j++)
			m_screenColor[i][j] = CColor(255.0f, 0.0f, 0.0f);

	m_camera = new CCamera(m_width, m_height);
	m_camera->setPosition(CVector3(320.0f, 220.0f, -380.0f));
	m_camera->setDirection(CVector3(0.0f, 0.0f, 1.0f));
}

CRayTracer::~CRayTracer()
{
	if(m_scene)
	{
		delete m_scene;
		m_scene = NULL;
	}
	
	// delete screen texture, technique, effect, vertex buffer, vertex declaration, screenCOlor, camera

	destroyDirect3DDevice();
	destroyDirect3D();
}

const char* CRayTracer::getWindowTitle()
{
	return m_windowTitle.c_str();
}

void CRayTracer::setWindowTitle( const char* title )
{
	m_windowTitle = title;
}

E_COMPUTING_TYPE CRayTracer::getComputingType()
{
	return m_computingType;
}

void CRayTracer::setComputingType( E_COMPUTING_TYPE type )
{
	m_computingType = type;
}

bool CRayTracer::loadScene( const char* sceneFile )
{
	// Delete old scene
	if(m_scene)
	{
		delete m_scene;
		m_scene = NULL;
	}

	// Create new one
	m_scene = new CScene;

	FILE* file = fopen(sceneFile, "r");
	if(!file)
		return false;

	// get file size
	fseek(file, 0, SEEK_END);
	int size = ftell(file);
	rewind(file);

	// empty file?
	if(!size)
	{
		fclose(file);
		return false;
	}

	// create buffer
	char* buffer = (char*)malloc(sizeof(char)*size);
	ZeroMemory(buffer, size);
	// read whole file
	fread(buffer, sizeof(char), size, file);
	fclose(file);


	bool result = parseSceneFile(buffer);

	// free our buffer
	if(buffer)
		::free(buffer);

	return result;
}

void CRayTracer::run()
{
	m_D3D9Dev->Clear(0, 0, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, 0xFF808080, 1.0f, 0);
	m_D3D9Dev->BeginScene();


	D3DXMATRIXA16 matWorld, matView, matProj;
	m_D3D9Dev->GetTransform(D3DTS_WORLD, &matWorld);
	m_D3D9Dev->GetTransform(D3DTS_VIEW, &matView);
	m_D3D9Dev->GetTransform(D3DTS_PROJECTION, &matProj);

	D3DXMATRIXA16 matWorldViewProj = matWorld * matView * matProj;
	m_constantTable->SetMatrix(m_D3D9Dev, "WorldViewProj", &matWorldViewProj);

	m_D3D9Dev->SetVertexDeclaration(m_vertexDeclaration);
	m_D3D9Dev->SetVertexShader(m_vertexShader);
	m_D3D9Dev->SetPixelShader(m_pixelShader);
	m_D3D9Dev->SetStreamSource(0, m_screenVB, 0, sizeof(D3DVERTEX));
	m_D3D9Dev->SetTexture(0, m_screenTexture);
	m_D3D9Dev->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0, 2);


	m_D3D9Dev->EndScene();
	m_D3D9Dev->Present(NULL, NULL, NULL , NULL);
}

int CRayTracer::createDirect3D()
{
	if(m_D3D9)
		return -1;

	m_D3D9 = Direct3DCreate9(D3D_SDK_VERSION);
	if(!m_D3D9)
		return -1;

	return 0;
}

int CRayTracer::createDirect3DDevice()
{
	D3DPRESENT_PARAMETERS presentParams;
	generatePresentParams(&presentParams);

	HRESULT hr;

	hr = m_D3D9->CreateDevice(D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, m_HWND, D3DCREATE_HARDWARE_VERTEXPROCESSING, &presentParams, &m_D3D9Dev);
	if(SUCCEEDED(hr))
		return 0;

	hr = m_D3D9->CreateDevice(D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, m_HWND, D3DCREATE_MIXED_VERTEXPROCESSING, &presentParams, &m_D3D9Dev);
	if(SUCCEEDED(hr))
		return 0;

	hr = m_D3D9->CreateDevice(D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, m_HWND, D3DCREATE_SOFTWARE_VERTEXPROCESSING, &presentParams, &m_D3D9Dev);
	if(SUCCEEDED(hr))
		return 0;


	return -1;
}

void CRayTracer::destroyDirect3D()
{
	if(m_D3D9)
	{
		m_D3D9->Release();
		m_D3D9 = NULL;
	}
}

void CRayTracer::destroyDirect3DDevice()
{
	if(m_D3D9Dev)
	{
		m_D3D9Dev->Release();
		m_D3D9Dev = NULL;
	}
}

void CRayTracer::setHWND( HWND hwnd )
{
	m_HWND = hwnd;
}

void CRayTracer::generatePresentParams( D3DPRESENT_PARAMETERS* pp )
{
	pp->BackBufferWidth = m_width;
	pp->BackBufferHeight = m_height;
	pp->BackBufferFormat = D3DFMT_A8R8G8B8;
	pp->BackBufferCount = 1;
	pp->MultiSampleType = D3DMULTISAMPLE_NONE;
	pp->MultiSampleQuality = 0;
	pp->SwapEffect = D3DSWAPEFFECT_DISCARD;
	pp->hDeviceWindow = m_HWND;
	pp->Windowed = !m_fullscreen;
	pp->EnableAutoDepthStencil = TRUE;
	pp->AutoDepthStencilFormat = D3DFMT_D24S8;
	pp->Flags = 0;
	pp->FullScreen_RefreshRateInHz = (m_fullscreen ? m_refreshRate : 0);
	pp->PresentationInterval = D3DPRESENT_INTERVAL_IMMEDIATE;
}

void CRayTracer::createWndClass( const char* class_name )
{
	WNDCLASSEX wcx;
	wcx.cbSize = sizeof(WNDCLASSEX);
	wcx.style = CS_DBLCLKS;
	wcx.lpfnWndProc = &wndProc;
	wcx.cbClsExtra = 0;
	wcx.cbWndExtra = 0;
	wcx.hInstance = m_Instance;
	wcx.hIcon = NULL;
	wcx.hCursor = LoadCursor(0, IDC_ARROW);
	wcx.hbrBackground = 0;
	wcx.lpszMenuName = 0;
	wcx.lpszClassName = class_name;
	wcx.hIconSm = 0;

	// Register class
	RegisterClassEx(&wcx);
}

LRESULT CALLBACK CRayTracer::wndProc( HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam )
{
	switch(msg)
	{
	case WM_DESTROY:
		
		break;

	case WM_KEYDOWN:
		
		break;

	case WM_KEYUP:
		
		break;

	case WM_CHAR:
		
		break;

	case WM_MOUSEWHEEL:
		
		break;

	case WM_MOUSEMOVE:
	
		break;
	}

	return DefWindowProc(hwnd, msg, wParam, lParam);
}

DWORD CRayTracer::getMyWindowStyle()
{
	if(m_fullscreen)
		return WS_POPUP | WS_SYSMENU | WS_VISIBLE;
	else
		return WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX | WS_VISIBLE;
}

int CRayTracer::initialize(const char* title)
{
	// Create Direct3D
	if(createDirect3D() == -1)
		return -1;

	// Register window class
	createWndClass(title);

	// Create window
	createMyWindow(title);

	// Create Direct3D device
	if(createDirect3DDevice())
		return -1;


	if( FAILED( m_D3D9Dev->CreateTexture( m_width, m_height, 0, D3DUSAGE_DYNAMIC, D3DFMT_X8R8G8B8, D3DPOOL_DEFAULT, &m_screenTexture, NULL ) ) )
		return -1;

	// Initialize shaders
	if(initializeShaders() == false)
		return -1;

	// Initialize screen quad vertex buffer and vertex declaration
	initializeScreenVB();

	// Create screen quad texture
	//D3DXCreateTextureFromFile(m_D3D9Dev, "editor.jpg", &m_screenTexture);


	float m_anisotropy = 0;
	float m_fieldOfView = D3DX_PI / 4.0f;
	float m_nearPlane = 1.0f;
	float m_farPlane = 1000.0f;
	float m_aspectRatio = (float)m_width / (float)m_height;
	D3DXMATRIX m_projection;
	D3DXMatrixOrthoOffCenterLH(&m_projection, 0, 800, 600, 0, m_nearPlane, m_farPlane);
	//D3DXMatrixPerspectiveFovLH(&m_projection, m_fieldOfView, m_aspectRatio, m_nearPlane, m_farPlane);
	m_D3D9Dev->SetTransform(D3DTS_PROJECTION, &m_projection);

	for(unsigned i = 0;i < 8;++i)
	{
		m_D3D9Dev->SetSamplerState(i, D3DSAMP_MINFILTER, D3DTEXF_LINEAR);
		m_D3D9Dev->SetSamplerState(i, D3DSAMP_MAGFILTER, D3DTEXF_LINEAR);
		m_D3D9Dev->SetSamplerState(i, D3DSAMP_MIPFILTER, D3DTEXF_ANISOTROPIC);
		m_D3D9Dev->SetSamplerState(i, D3DSAMP_MAXANISOTROPY, m_anisotropy);
	}


	return 0;
}

void CRayTracer::createMyWindow( const char* title )
{
	// Window style
	DWORD style = getMyWindowStyle();

	// Window dimension
	RECT rc = {0, 0, m_width, m_height};

	AdjustWindowRect(&rc, style, FALSE);

	// Create engine window
	m_HWND = CreateWindow(title, title, style, CW_USEDEFAULT, CW_USEDEFAULT, 
		rc.right - rc.left, rc.bottom - rc.top, 0, 0, m_Instance, 0);
}

void CRayTracer::setInstance( HINSTANCE instance )
{
	m_Instance = instance;
}

bool CRayTracer::parseSceneFile( char* buffer )
{
	// Start parsing
	CBasePrimitive* primitive = NULL;
	char *cur = buffer;
	bool sectionOpened = false;
	bool done = false;
	bool error = false;
	while (!done)
	{
		// Skip empty lines
		while (*cur=='\r' || *cur=='\n') cur++;

		// Get line start
		char *start = cur;

		//! Get line end
		while (*cur && *cur!='\r' && *cur!='\n') cur++;

		// End of file ?
		if (*cur==0) done = 1;

		// Cut string
		*cur++ = 0;

		// create primitive section
		if(!primitive && !sectionOpened)
		{
			char *primitiveType = start;
			int offset = 0;
			while (*primitiveType && *primitiveType!=':')
			{
				primitiveType++;
				offset++;
			}

			primitiveType = start;
			primitiveType[offset] = 0;

			if(strcmp(primitiveType, "sphere") == 0)
			{
				primitiveType = start+offset+1;
				while (*primitiveType && *primitiveType==' ') primitiveType++;
				primitive = new CSpherePrimitive();
				primitive->setName(primitiveType);
				m_scene->addPrimitive(primitive);
			}
			else if(strcmp(primitiveType, "plane") == 0)
			{
				primitiveType = start+offset+1;
				while (*primitiveType && *primitiveType==' ') primitiveType++;
				primitive = new CPlanePrimitive();
				primitive->setName(primitiveType);
				m_scene->addPrimitive(primitive);
			}
			else
			{
				// there is error
				error = true;
				break;
			}
		}
		else
		{
			// section opened
			if(*start == '{' && !sectionOpened)
			{
				sectionOpened = true;
				continue;
			}
			else if(*start == '}' && sectionOpened)
			{
				sectionOpened = false;
				primitive = NULL;
				continue;
			}
			else if(sectionOpened && primitive)
			{
				// Parse single field in primitive
				char* field = start;
				if(*field == '[')
				{
					int offset = 0;
					while (*field && *field!=']') 
					{
						field++;
						offset++;
					}

					field = start+1;
					field[offset-1] = 0;

					char *val_buff = start+offset+1;
					while (*val_buff && *val_buff==' ') val_buff++;

					if(strcmp(field, "position") == 0)
					{
						float x=0.0f, y=0.0f, z=0.0f;
						sscanf(val_buff, "%f %f %f\n", &x, &y, &z);
						primitive->setPosition(CVector3(x, y, z));
					}
					else if(strcmp(field, "radius") == 0)
					{
						float radius=0.0f;
						sscanf(val_buff, "%f\n", &radius);
						((CSpherePrimitive*)primitive)->setRadius(radius);
					}
					else if(strcmp(field, "color") == 0)
					{
						float r=0.0f, g=0.0f, b=0.0f;
						sscanf(val_buff, "%f %f %f\n", &r, &g, &b);
						primitive->getMaterial()->setColor(CColor(r, g, b));
					}
					else if(strcmp(field, "diffuse") == 0)
					{
						float diffuse=0.0f;
						sscanf(val_buff, "%f\n", &diffuse);
						primitive->getMaterial()->setDiffuse(diffuse);
					}
					else if(strcmp(field, "reflection") == 0)
					{
						float reflection=0.0f;
						sscanf(val_buff, "%f\n", &reflection);
						primitive->getMaterial()->setReflection(reflection);
					}
					else if(strcmp(field, "light") == 0)
					{
						int light=0;
						sscanf(val_buff, "%i\n", &light);
						primitive->setLight(light);
					}
					else if(strcmp(field, "normal") == 0)
					{
						float x=0.0f, y=0.0f, z=0.0f;
						sscanf(val_buff, "%f %f %f\n", &x, &y, &z);
						((CPlanePrimitive*)primitive)->setNormal(CVector3(x, y, z));
					}
					else if(strcmp(field, "d") == 0)
					{
						float d=0.0f;
						sscanf(val_buff, "%f\n", &d);
						((CPlanePrimitive*)primitive)->setD(d);
					}
				}
			}
			else
			{
				error = true;
				break;
			}
		}
	}

	return !error;
}

void CRayTracer::calculateFrame()
{
	// Lock texture
	D3DLOCKED_RECT lockedRectSurf;
	m_screenTexture->LockRect(0, &lockedRectSurf, NULL, D3DLOCK_DISCARD);
	DWORD* pDataSurf = (DWORD*)(lockedRectSurf.pBits);
	int offset = lockedRectSurf.Pitch / 4;

	// Clear output color
	CColor outputColor = CColor(0.0f, 0.0f, 0.0f);
	float r = 0.0f;
	float g = 0.0f;
	float b = 0.0f;
	m_camera->initialize();

	for(int y = 0; y < m_height; ++y)
	for(int x = 0; x < m_width; ++x)
	{
		//float fragmentX = (float)x;
		//float fragmentY = (float)y;
		for(float fragmentX = (float)x; fragmentX < (float)x + 1.0f; fragmentX += 0.5f)
		for(float fragmentY = (float)y; fragmentY < (float)y + 1.0f; fragmentY += 0.5f)
		{
			CRay ray;
			ray.setOrigin(CVector3(0.0f, 0.0f, -5.0f));
			ray.setOrigin(m_camera->getPosition());
			m_camera->calcRayDir(ray, fragmentX, fragmentY);
			CColor resultColor = traceRay(ray, 1);
			outputColor += 0.25f * resultColor;
			//outputColor = resultColor;
		}
			
		// Set output color to texture
		r = outputColor.m_r*255.0f;
		g = outputColor.m_g*255.0f;
		b = outputColor.m_b*255.0f;

		pDataSurf[offset * y + x] = D3DCOLOR_XRGB((int)r, (int)g, (int)b);
		// Clear output color
		outputColor = CColor(0.0f, 0.0f, 0.0f);
	}

	// Unlock texture
	m_screenTexture->UnlockRect(0);
}

void CRayTracer::initializeScreenVB()
{
	// Set up vertex buffer
	D3DVERTEX screen[4] = {{D3DXVECTOR3(0.0f, 0.0f, 1.0f),
		D3DXVECTOR2(0.0f, 0.0f)},
	{D3DXVECTOR3(800.0f,  0.0f, 1.0f),
	D3DXVECTOR2(1.0f, 0.0f)},
	{D3DXVECTOR3(0.0f, 600.0f, 1.0f),
	D3DXVECTOR2(0.0f, 1.0f)},
	{D3DXVECTOR3( 800.0f,  600.0f, 1.0f),
	D3DXVECTOR2(1.0f, 1.0f)}};

	D3DVERTEX *ptr = NULL;
	m_D3D9Dev->CreateVertexBuffer(sizeof(screen), D3DUSAGE_WRITEONLY, 0, D3DPOOL_MANAGED, &m_screenVB, NULL);
	m_screenVB->Lock(0, 0, (void**)&ptr, 0);
	memcpy((void*)ptr, (void*)screen, sizeof(screen));
	m_screenVB->Unlock();


	// Create vertex declaration
	D3DVERTEXELEMENT9 decl[] = {
							    {0, 0, D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION, 0},
								{0, 12, D3DDECLTYPE_FLOAT2, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 0},
								D3DDECL_END()
							   };
	m_D3D9Dev->CreateVertexDeclaration(decl, &m_vertexDeclaration);
}

bool CRayTracer::initializeShaders()
{
	LPD3DXBUFFER code = NULL;
	HRESULT result;
	ID3DXBuffer* pErrors;

	result = D3DXCompileShaderFromFile( "vertex.shader",     //filepath
		NULL,            //macro's
		NULL,            //includes
		"vs_main",       //main function
		"vs_2_0",        //shader profile
		0,               //flags
		&code,           //compiled operations
		&pErrors,            //errors
		&m_constantTable); //constants
	if(FAILED(result)) {
		MessageBox(m_HWND, (char*)pErrors->GetBufferPointer(), "Error", MB_OK);
		return false;
	}

	m_D3D9Dev->CreateVertexShader((DWORD*)code->GetBufferPointer(), &m_vertexShader);
	code->Release();

	result = D3DXCompileShaderFromFile( "pixel.shader",   //filepath
		NULL,          //macro's            
		NULL,          //includes           
		"ps_main",     //main function      
		"ps_2_0",      //shader profile     
		0,             //flags              
		&code,         //compiled operations
		&pErrors,            //errors
		NULL); //constants
	if(FAILED(result)) {
		MessageBox(m_HWND, (char*)pErrors->GetBufferPointer(), "Error", MB_OK);
		return false;
	}

	m_D3D9Dev->CreatePixelShader((DWORD*)code->GetBufferPointer(), &m_pixelShader);
	code->Release();

	return true;
}

void CRayTracer::setWindowSize( int width, int height )
{
	m_width = width;
	m_height = height;
}

CColor CRayTracer::traceRay( CRay& ray, int depthLevel )
{
	float distance = 10000000.0f;
	CColor outColor = CColor(0.0f, 0.0f, 0.0f);
	CBasePrimitive* hitPrimitive = NULL;
	int result;
	CVector3 pi;

	for(std::vector<CBasePrimitive*>::iterator itor = m_scene->m_primitives.begin(); itor != m_scene->m_primitives.end(); ++itor)
	{
		
		if(int res = (*itor)->intersect(ray, distance))
		{
			hitPrimitive = *itor;
			result = res;
		}
	}

	// no hit, terminate ray
	if (!hitPrimitive) return CColor(0.0f, 0.0f, 0.0f);
	
	// handle intersection
	if (hitPrimitive->isLight())
	{
		// we hit a light, stop tracing
		outColor = CColor( 1.0f, 1.0f, 1.0f );
	}
	else
	{
		// determine color at point of intersection
		pi = ray.getOrigin() + ray.getDirection() * distance;
		// trace lights
		for ( int l = 0; l < m_scene->getPrimitivesCount(); l++ )
		{
			CBasePrimitive* p = m_scene->getPrimitive( l );
			if (p->isLight()) 
			{
				CBasePrimitive* light = p;
				// calculate diffuse shading
				CVector3 L = ((CSpherePrimitive*)light)->getCenter() - pi;
				NORMALIZE(L);
				CVector3 N = hitPrimitive->getNormal( pi );
				if (hitPrimitive->getMaterial()->getDiffuse() > 0)
				{
					float dot = DOT( N, L );
					if (dot > 0)
					{
						float diff = dot * hitPrimitive->getMaterial()->getDiffuse();
						// add diffuse component to ray color
						outColor += diff * hitPrimitive->getMaterial()->getColor() * light->getMaterial()->getColor();
					}
				}
			}
		}
	}

	// return color
	return outColor;
}