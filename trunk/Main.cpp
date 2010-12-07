// CUDAPhotonMappingRenderer.cpp : Defines the entry point for the application.
//

#include "stdafx.h"
#include "Main.h"

#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>


#define MAX_LOADSTRING 100

// Global Variables:
HINSTANCE hInst;								// current instance
TCHAR szTitle[MAX_LOADSTRING];					// The title bar text
TCHAR szWindowClass[MAX_LOADSTRING];			// the main window class name

// Forward declarations of functions included in this code module:
ATOM				MyRegisterClass(HINSTANCE hInstance);
BOOL				InitInstance(HINSTANCE, int);
LRESULT CALLBACK	WndProc(HWND, UINT, WPARAM, LPARAM);
INT_PTR CALLBACK	About(HWND, UINT, WPARAM, LPARAM);

int APIENTRY _tWinMain(HINSTANCE hInstance,
                     HINSTANCE hPrevInstance,
                     LPTSTR    lpCmdLine,
                     int       nCmdShow)
{
	UNREFERENCED_PARAMETER(hPrevInstance);
	UNREFERENCED_PARAMETER(lpCmdLine);

	_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
	_CrtDumpMemoryLeaks();
	//_CrtSetBreakAlloc();

	//CProcessor proc;
	//proc.WriteInfoTextFile("tmp.txt");
	//return 0;

	/*CRTProfiler profiler;
	SProfiledScene profile;
	profile.m_frameTime = 500;
	profile.m_lighteningTime = 34;
	profile.m_shadowsTime = 325;
	profiler.addSceneProfile(&profile);
	CRTProfiler::saveSceneProfiles("profilowanie.txt", &profiler, ECT_CPU);

	return 0;*/

	const char* RT_PROFILER = "rxProfiler.txt";


	g_Raytracer = new CFramework(400, 400, false, "rxRT by Konrad Rodzik", hInstance);
	if(g_Raytracer && g_Raytracer->initialize("maps/benchmark.rtb")!=-1) 
	{
		g_Raytracer->initializeRT(ECT_CPU);
		g_Raytracer->run();
		g_Raytracer->finalize(RT_PROFILER, ECT_CPU);
		g_Raytracer->closeRT();

		g_Raytracer->initializeRT(ECT_CUDA);
		g_Raytracer->run();
		g_Raytracer->finalize(RT_PROFILER, ECT_CUDA);
		g_Raytracer->closeRT();
	}

	if(g_Raytracer) {
		g_Raytracer->close();
		delete g_Raytracer;
	}
	
	
	return 0;
}


/*
//
//  FUNCTION: WndProc(HWND, UINT, WPARAM, LPARAM)
//
//  PURPOSE:  Processes messages for the main window.
//
//  WM_COMMAND	- process the application menu
//  WM_PAINT	- Paint the main window
//  WM_DESTROY	- post a quit message and return
//
//
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	int wmId, wmEvent;
	PAINTSTRUCT ps;
	HDC hdc;

	switch (message)
	{
	case WM_COMMAND:
		wmId    = LOWORD(wParam);
		wmEvent = HIWORD(wParam);
		// Parse the menu selections:
		switch (wmId)
		{
		case IDM_ABOUT:
			DialogBox(hInst, MAKEINTRESOURCE(IDD_ABOUTBOX), hWnd, About);
			break;
		case IDM_EXIT:
			DestroyWindow(hWnd);
			break;
		default:
			return DefWindowProc(hWnd, message, wParam, lParam);
		}
		break;
	case WM_PAINT:
		hdc = BeginPaint(hWnd, &ps);
		// TODO: Add any drawing code here...
		EndPaint(hWnd, &ps);
		break;
	case WM_DESTROY:
		PostQuitMessage(0);
		break;
	default:
		return DefWindowProc(hWnd, message, wParam, lParam);
	}
	return 0;
}

// Message handler for about box.
INT_PTR CALLBACK About(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam)
{
	UNREFERENCED_PARAMETER(lParam);
	switch (message)
	{
	case WM_INITDIALOG:
		return (INT_PTR)TRUE;

	case WM_COMMAND:
		if (LOWORD(wParam) == IDOK || LOWORD(wParam) == IDCANCEL)
		{
			EndDialog(hDlg, LOWORD(wParam));
			return (INT_PTR)TRUE;
		}
		break;
	}
	return (INT_PTR)FALSE;
}
*/