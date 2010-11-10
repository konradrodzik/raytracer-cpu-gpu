////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#ifndef __H_CCamera_H__
#define __H_CCamera_H__

struct CUDA_Ray;


// Camera class
class CCamera
{
public:
	// Default constructor
	CCamera();

	// Initialize constructor
	CCamera(int screenWidth, int screenHeight);

	// Set position
	void setPosition(const CVector3& pos);

	// Get position
	CVector3& getPosition();

	// Set direction
	void setDirection(const CVector3& dir);

	// Get direction
	CVector3& getDirection();

	// Set up vector
	void setUP(const CVector3& up);

	// Get up vector
	CVector3& getUP();

	// Initialize camera
	void initialize();

	// Calculate ray direction
	void calcRayDir(CRay& ray, float screenPixelX, float screenPixelY);

	// Calculate ray direction. GPU(CUDA) version
	/*__device__ void calcRayDir(CUDA_Ray& raya, float screenPixelX, float screenPixelY)
	{
	
	float3 dir = make_float3(m_direction.m_x, m_direction.m_y, m_direction.m_z); + 
				((2.0f * screenPixelX + 1.0f  - (float)m_screenWidth) * 0.5f) * m_dx + 
				((2.0f * screenPixelY + 1.0f - (float)m_screenHeight) * 0.5f) * m_dy;
	
	//dir = normalize(dir);
	raya.dir = dir;
	}*/


public:
	CVector3 m_position;	// Camera position
	CVector3 m_direction;	// Camera direction
	CVector3 m_up;			// Camera up vector
	float m_fovX;			// Field of view X
	float m_fovY;			// Field of view Y
	
	CVector3 m_dx;			// Delta x
	CVector3 m_dy;			// Delta y

	int m_screenWidth;		// Screen width
	int m_screenHeight;		// Screen height
};

#endif