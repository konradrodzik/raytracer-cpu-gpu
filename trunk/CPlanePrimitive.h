////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#ifndef __H_CPlanePrimitive_H__
#define __H_CPlanePrimitive_H__

// Plane primitive type
class CPlanePrimitive : public CBasePrimitive
{
public:
	// Default constructor
	CPlanePrimitive();

	// Initialize constructor
	CPlanePrimitive(const CVector3& normal, float d_point);

	// Get type of primitive
	virtual E_PRIMITIVE_TYPE getType();

	// Get normal in D point
	virtual CVector3& getNormal();

	// Get normal in specified point
	virtual CVector3 getNormal(const CVector3& pos);

	// Get D point
	float getDPoint();

	// Intersect function
	virtual int intersect(CRay& ray, float& distance);

	// Set normal
	void setNormal(const CVector3& normal);

	// Set D value of plane
	void setD(float d);

	// Set position = Set Normal
	virtual void setPosition(const CVector3& pos);

private:
	CPlane m_plane;		// Plane object
};

#endif