////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#ifndef __H_CSpherePrimitive_H__
#define __H_CSpherePrimitive_H__

// Sphere primitive type
class CSpherePrimitive : public CBasePrimitive
{
public:
	// Default constructor
	CSpherePrimitive();

	// Initialize constructor
	CSpherePrimitive(const CVector3& center, float radius);

	// Get type of primitive
	virtual E_PRIMITIVE_TYPE getType();

	// Get center of the sphere
	CVector3& getCenter();

	// Get radius
	float getRadius();

	// Set radius
	void setRadius(float radius);

	// Get squared radius
	float getSqrRadius();

	// Get invert radius
	float getInvRadius();
	
	// Get normal
	virtual CVector3 getNormal(const CVector3& pos);

	// Intersect function
	virtual int intersect(CRay& ray, float& distance);

	// Set position
	virtual void setPosition(const CVector3& pos);

	// Get primitive color at given position
	virtual CColor getColor(const CVector3& pos);

private:
	CVector3 m_center;		// Center of the sphere
	float m_radius;			// Radius of the sphere
	float m_sqrRadius;		// Square radius of the sphere
	float m_invRadius;		// Invert radius of the sphere

	// Additional data
	CVector3 m_vn, m_ve, m_vc;
};

#endif