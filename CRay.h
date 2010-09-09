////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#ifndef __H_CRay_H__
#define __H_CRay_H__

// Ray class
class CRay
{
public:
	// Default constructor
	CRay();

	// Initialize constructor
	CRay(CVector3& origin, CVector3& dir);

	// Set origin vector
	void setOrigin(CVector3& origin);

	// Set direction vector
	void setDirection(CVector3& direction);

	// Get ray origin
	CVector3& getOrigin();

	// Get ray direction
	CVector3& getDirection();

private:
	CVector3 m_origin;			// Ray origin
	CVector3 m_direction;		// Ray direction
};


#endif