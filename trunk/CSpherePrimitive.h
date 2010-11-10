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
	E_PRIMITIVE_TYPE getType();

	// Get center of the sphere
	CVector3& getCenter();

	__device__ float3 getCenterEx()
	{
		return make_float3(m_center.m_x, m_center.m_y, m_center.m_z);
	}

	// Get radius
	float getRadius();

	// Set radius
	void setRadius(float radius);

	// Get squared radius
	float getSqrRadius();

	// Get invert radius
	float getInvRadius();
	
	// Get normal
	CVector3 getNormal(const CVector3& pos);
	__device__ float3 getNormal(float3 pos)
	{
		CVector3 tmp = ((CVector3(pos.x, pos.y, pos.z) - m_center) * m_invRadius);
		return make_float3(tmp.m_x, tmp.m_y, tmp.m_z);
	}

	// Intersect function
	int intersect(CRay& ray, float& distance);

	// Set position
	void setPosition(CVector3& pos);

	// Get primitive color at given position
	CColor getColor(const CVector3& pos);

	__device__ float3 getColor(float3 pos)
	{
		CColor tmpCol = m_material.getColor();
		return make_float3(tmpCol.m_x, tmpCol.m_y, tmpCol.m_z);
	}

public:
	CVector3 m_center;		// Center of the sphere
	float m_radius;			// Radius of the sphere
	float m_sqrRadius;		// Square radius of the sphere
	float m_invRadius;		// Invert radius of the sphere

	// Additional data
	CVector3 m_vn, m_ve, m_vc;
};

#endif