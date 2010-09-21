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

CFramework::CFramework()
: m_rayTracer(NULL)
, m_D3D9(NULL)
, m_D3D9Dev(NULL)
, m_HWND(NULL)
, m_Instance(NULL)
, m_screenTexture(NULL)
, m_screenVB(NULL)
, m_vertexDeclaration(NULL)
, m_vertexShader(NULL)
, m_pixelShader(NULL)
, m_constantTable(NULL)
, m_width(500)
, m_height(500)
, m_fullscreen(false)
, m_refreshRate(0)
, m_windowTitle("Simple Title")
{

}

CFramework::CFramework( int width, int height, bool fullscreen, const char* title, HINSTANCE windowInstance )
: m_rayTracer(NULL)
, m_D3D9(NULL)
, m_D3D9Dev(NULL)
, m_HWND(NULL)
, m_Instance(windowInstance)
, m_screenTexture(NULL)
, m_screenVB(NULL)
, m_vertexDeclaration(NULL)
, m_vertexShader(NULL)
, m_pixelShader(NULL)
, m_constantTable(NULL)
, m_width(width)
, m_height(height)
, m_fullscreen(fullscreen)
, m_refreshRate(m_fullscreen?32:0)
, m_windowTitle(title)
{
}

CFramework::~CFramework()
{
	close();
}

int CFramework::createDirect3D()
{
	if(m_D3D9)
		return -1;

	m_D3D9 = Direct3DCreate9(D3D_SDK_VERSION);
	if(!m_D3D9)
		return -1;

	return 0;
}

int CFramework::createDirect3DDevice()
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

void CFramework::destroyDirect3D()
{
	if(m_D3D9)
	{
		m_D3D9->Release();
		m_D3D9 = NULL;
	}
}

void CFramework::destroyDirect3DDevice()
{
	if(m_D3D9Dev)
	{
		m_D3D9Dev->Release();
		m_D3D9Dev = NULL;
	}
}

void CFramework::initializeScreenVB()
{
	// Set up vertex buffer
	D3DVERTEX screen[4] = {
	{D3DXVECTOR3(0.0f, 0.0f, 1.0f),
	D3DXVECTOR2(0.0f, 0.0f)},
	{D3DXVECTOR3((float)m_width,  0.0f, 1.0f),
	D3DXVECTOR2(1.0f, 0.0f)},
	{D3DXVECTOR3(0.0f, (float)m_height, 1.0f),
	D3DXVECTOR2(0.0f, 1.0f)},
	{D3DXVECTOR3( (float)m_width,  (float)m_height, 1.0f),
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

void CFramework::shutdownVB()
{
	// Release vertex declaration
	if(m_vertexDeclaration)
	{
		m_vertexDeclaration->Release();
		m_vertexDeclaration = NULL;
	}

	// Release vertex buffer
	if(m_screenVB)
	{
		m_screenVB->Release();
		m_screenVB = NULL;
	}
}

bool CFramework::initializeShaders()
{
	LPD3DXBUFFER code = NULL;
	HRESULT result;
	ID3DXBuffer* pErrors;

	result = D3DXCompileShaderFromFile( "shaders/vertex.shader",     //filepath
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

	result = D3DXCompileShaderFromFile( "shaders/pixel.shader",   //filepath
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

void CFramework::shutdownShaders()
{
	// Release pixel shader
	if(m_pixelShader)
	{
		m_pixelShader->Release();
		m_pixelShader = NULL;
	}

	// Release vertex shader
	if(m_vertexShader)
	{
		m_vertexShader->Release();
		m_vertexShader = NULL;
	}

	// Release constant table
	if(m_constantTable)
	{
		m_constantTable->Release();
		m_constantTable = NULL;
	}
}

void CFramework::generatePresentParams( D3DPRESENT_PARAMETERS* pp )
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

void CFramework::createWndClass( const char* class_name )
{
	WNDCLASSEX wcx;
	wcx.cbSize = sizeof(WNDCLASSEX);
	wcx.style = CS_DBLCLKS;
	wcx.lpfnWndProc = &CFramework::wndProc;
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

DWORD CFramework::getMyWindowStyle()
{
	if(m_fullscreen)
		return WS_POPUP | WS_SYSMENU | WS_VISIBLE;
	else
		return WS_OVERLAPPEDWINDOW & ~WS_MAXIMIZEBOX & ~WS_THICKFRAME | WS_VISIBLE;
}

void CFramework::createMyWindow( const char* title )
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

LRESULT CALLBACK CFramework::wndProc( HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam )
{
	switch(msg)
	{
	case WM_KEYDOWN:
		if ((wParam & 0xFF) != 27) break;
	case WM_CLOSE:
		SystemParametersInfo( SPI_SETSCREENSAVEACTIVE, 1, 0, 0 );
		ExitProcess( 0 );
		break;
	}

	return DefWindowProc(hwnd, msg, wParam, lParam);
}

int CFramework::initialize(const char* benchmarkFile)
{
	// Create Direct3D
	if(createDirect3D() == -1)
		return -1;

	// Register window class
	createWndClass(m_windowTitle.c_str());

	// Create window
	createMyWindow(m_windowTitle.c_str());

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

	//float m_anisotropy = 0;
	//float m_fieldOfView = D3DX_PI / 4.0f;
	float m_nearPlane = 1.0f;
	float m_farPlane = 1000.0f;
	//float m_aspectRatio = (float)m_width / (float)m_height;
	D3DXMATRIX m_projection;
	D3DXMatrixOrthoOffCenterLH(&m_projection, 0, (float)m_width, (float)m_height, 0, m_nearPlane, m_farPlane);
	//D3DXMatrixPerspectiveFovLH(&m_projection, m_fieldOfView, m_aspectRatio, m_nearPlane, m_farPlane);
	m_D3D9Dev->SetTransform(D3DTS_PROJECTION, &m_projection);

	/*for(unsigned i = 0;i < 8;++i)
	{
		m_D3D9Dev->SetSamplerState(i, D3DSAMP_MINFILTER, D3DTEXF_LINEAR);
		m_D3D9Dev->SetSamplerState(i, D3DSAMP_MAGFILTER, D3DTEXF_LINEAR);
		m_D3D9Dev->SetSamplerState(i, D3DSAMP_MIPFILTER, D3DTEXF_ANISOTROPIC);
		m_D3D9Dev->SetSamplerState(i, D3DSAMP_MAXANISOTROPY, m_anisotropy);
	}*/


	// Create Raytracer
	m_rayTracer = new CRayTracerCPU(m_width, m_height);
	if(!m_rayTracer)
		return -1;

	// Load maps from benchmark
	if(!m_rayTracer->loadMaps(benchmarkFile))
		return -1;


	// Set map raytracing percent
	setWindowTitle(0);

	return 0;
}

void CFramework::close()
{
	shutdownVB();
	shutdownShaders();

	// Release screen texture
	if(m_screenTexture)
	{
		m_screenTexture->Release();
		m_screenTexture = NULL;
	}
	
	destroyDirect3DDevice();
	destroyDirect3D();

	if(m_HWND)
	{
		DestroyWindow( m_HWND );
		m_HWND = NULL;
	}

	// Release raytracer
	if(m_rayTracer)
	{
		delete m_rayTracer;
		m_rayTracer = NULL;
	}
}

void CFramework::draw()
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

void CFramework::sendBenchmarkEmail()
{
	m_mail.setSubject("[INZ] NAZWA KOMPA");
	m_mail.setBody("tresc");
	m_mail.addAttachment("benchmark.log");
	m_mail.send();
}

void CFramework::run()
{
	MSG msg;
	msg.message = NULL;
	while(msg.message != WM_QUIT)
	{
		// Communique
		if(PeekMessage(&msg, 0, 0, 0, PM_REMOVE))
		{
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
		// Draw and calculate actual frame 
		else
		{
			// Lock texture
			D3DLOCKED_RECT lockedRectSurf;
			m_screenTexture->LockRect(0, &lockedRectSurf, NULL, D3DLOCK_DISCARD | D3DLOCK_DONOTWAIT);
			DWORD* pDataSurf = (DWORD*)(lockedRectSurf.pBits);
			int offset = lockedRectSurf.Pitch / 4;

			
			CColor** screenBuffer = m_rayTracer->getScreenColorBuffer();
			for(int y = 0; y < m_height; ++y){
				for(int x = 0; x < m_width; ++x) {
					CColor color = screenBuffer[y][x] * 255;
					pDataSurf[offset * y + x] = D3DCOLOR_XRGB((int)color.m_r, (int)color.m_g, (int)color.m_b);
				}
			}


			// Unlock texture
			m_screenTexture->UnlockRect(0);

			draw();


			m_rayTracer->calculateScene();
		}
	}
}

void CFramework::setWindowTitle( int percent )
{
	char buffer[255];
	memset(buffer, 0, 255);
	int currMap = 1;
	int mapsCount = 10;
	sprintf(buffer, "%s - MAP: %i/%i - %i%%", m_windowTitle.c_str(), currMap, mapsCount, percent);

	SetWindowText(m_HWND, buffer);
}