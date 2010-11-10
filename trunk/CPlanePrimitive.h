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
	E_PRIMITIVE_TYPE getType();

	// Get normal in D point
	CVector3& getNormal();

	// Get normal in specified point
	CVector3 getNormal(const CVector3& pos);
	__device__ float3 getNormal(float3 pos)
	{
		return make_float3(m_plane.m_normal.m_x, m_plane.m_normal.m_y, m_plane.m_normal.m_z);
	}

	// Get D point
	float getDPoint();

	// Intersect function
	int intersect(CRay& ray, float& distance);

	// Set normal
	void setNormal(const CVector3& normal);

	// Set D value of plane
	void setD(float d);

	// Set position = Set Normal
	void setPosition(CVector3& pos);

	__device__ CPlane& getPlane() { return m_plane; }

	__device__ float3 getColor(float3 pos)
	{
		CColor tmpCol = m_material.getColor();
		return make_float3(tmpCol.m_x, tmpCol.m_y, tmpCol.m_z);
	}

private:
	CPlane m_plane;		// Plane object
};

#endif