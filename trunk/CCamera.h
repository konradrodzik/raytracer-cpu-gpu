////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#ifndef __H_CCamera_H__
#define __H_CCamera_H__

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

private:
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