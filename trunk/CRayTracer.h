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

	// Get window title
	const char* getWindowTitle();

	// Set window title
	void setWindowTitle(const char* title);

	// Get computing type
	E_COMPUTING_TYPE getComputingType();

	// Set computing type
	void setComputingType(E_COMPUTING_TYPE type);

	// Load scene
	bool loadScene(const char* sceneFile);

	// Parse scene file
	bool parseSceneFile(char* buffer);

	// Run ray tracer
	void run();

	// Create Direct3D object
	int createDirect3D();

	// Create Direct3D device object
	int createDirect3DDevice();

	// Destroy Direct3D object
	void destroyDirect3D();

	// Destroy Direct3D device object
	void destroyDirect3DDevice();

	// Set HWND of window
	void setHWND(HWND hwnd);

	// Generate present parameters
	void generatePresentParams(D3DPRESENT_PARAMETERS* pp);

	// Create Wnd Class
	void createWndClass(const char* class_name);

	// Window procedure
	static LRESULT CALLBACK wndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

	// Get my window style
	DWORD getMyWindowStyle();

	// Create my window
	void createMyWindow(const char* title);

	// Initialize ray tracer
	int initialize(const char* title);

	// Set window process handle (instance)
	void setInstance(HINSTANCE instance);

	// Calculate frame
	void calculateFrame();

	// Initialize screen vertex buffer
	void initializeScreenVB();

	// Initialize shaders
	bool initializeShaders();

	// Set window size
	void setWindowSize(int width, int height);

	// Trace single ray
	CColor traceRay(CRay& ray, int depthLevel);

private:
	LPDIRECT3D9 m_D3D9;								  // Direct3D object
	LPDIRECT3DDEVICE9 m_D3D9Dev;					  // Direct3D device object
	HWND m_HWND;									  // HWND of window
	HINSTANCE m_Instance;							  // window process handle
	LPD3DXEFFECT m_effect;							  // Shader effect object
	D3DXHANDLE m_technique;							  // Shader technique object
	IDirect3DTexture9* m_screenTexture;				  // Screen texture using in shader
	LPDIRECT3DVERTEXBUFFER9 m_screenVB;				  // Vertex buffer for screen quad effect
	LPDIRECT3DVERTEXDECLARATION9 m_vertexDeclaration; //VertexDeclaration (NEW)
	LPDIRECT3DVERTEXSHADER9 m_vertexShader;			  // Vertex shader
	LPDIRECT3DPIXELSHADER9 m_pixelShader;			  // Pixel shader
	LPD3DXCONSTANTTABLE m_constantTable;			  // Constant table
	

	// Settings
	int m_width;						// Window width
	int m_height;						// Window height
	bool m_fullscreen;					// Fullscreen mode
	int m_refreshRate;					// Refresh rate in full screen mode

	std::string m_windowTitle;			// Window title
	E_COMPUTING_TYPE m_computingType;	// Ray tracer computing type

	CScene* m_scene;					// Ray traced scene
	CCamera* m_camera;					// Camera
	CColor** m_screenColor;				// Screen buffer of color
};

#endif