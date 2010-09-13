////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#ifndef __H_CFramework_H__
#define __H_CFramework_H__

class CFramework
{
public:
	// Default constructor
	CFramework();

	// Initialize constructor
	CFramework(int width, int height, bool fullscreen, const char* title, HINSTANCE windowInstance);

	// Destructor
	~CFramework();

	int initialize();

	void run();

	void sendBenchmarkEmail();

private:
	// Create Direct3D object
	int createDirect3D();

	// Create Direct3D device object
	int createDirect3DDevice();

	// Destroy Direct3D object
	void destroyDirect3D();

	// Destroy Direct3D device object
	void destroyDirect3DDevice();

	// Initialize screen vertex buffer
	void initializeScreenVB();

	// Initialize shaders
	bool initializeShaders();

	// Generate present parameters
	void generatePresentParams(D3DPRESENT_PARAMETERS* pp);

	// Create Wnd Class
	void createWndClass(const char* class_name);

	// Get my window style
	DWORD getMyWindowStyle();

	// Create my window
	void createMyWindow(const char* title);

	// Window procedure
	static LRESULT CALLBACK wndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

	void draw();

private:
	CRayTracer* m_rayTracer;						 // Raytracer module

	LPDIRECT3D9 m_D3D9;								  // Direct3D object
	LPDIRECT3DDEVICE9 m_D3D9Dev;					  // Direct3D device object
	HWND m_HWND;									  // HWND of window
	HINSTANCE m_Instance;							  // window process handle
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

	// Benchamrk variables
	CMail m_mail;
};

#endif