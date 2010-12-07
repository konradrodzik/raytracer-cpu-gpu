////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#ifndef __H_CBoxPrimitive_H__
#define __H_CBoxPrimitive_H__

class CBoxPrimitive : public CBasePrimitive
{
public:
	// Default constructor
	CBoxPrimitive();

	// Initialize constructor
	CBoxPrimitive(CAABBox& box);

	// Get type of primitive
	E_PRIMITIVE_TYPE getType();

	// Intersect function
	int intersect(CRay& ray, float& distance);

	// Get primitive normal at given position
	CVector3 getNormal(const CVector3& pos);
	__device__ float3 getNormal(float3 pos)
	{
		return make_float3(0, 0, 0);
	}

	// Set position
	void setPosition(CVector3& pos);

	// Set size
	void setSize(CVector3& size);

	// Set rotation angle in Y axis
	void setAngleY(float angle);

	// Get rotation angle in Y axis
	float getAngleY();

	// Get sinus from angle Y
	float getSinusY();

	// Get cosinus from angle Y
	float getCosinusY();

	__device__ CAABBox& getBox() { return m_box; }

	__device__ float getSinusAngle() { return m_sinAngleY; }
	__device__ float getCosinusAngle() { return m_cosAngleY; }

	__device__ float3 getColor(float3 pos)
	{
		//CColor tmpCol = m_material.getColor();
		//return make_float3(tmpCol.m_x, tmpCol.m_y, tmpCol.m_z);
		return m_material.getColorEx();
	}

private:
	CAABBox m_box;		// Axis aligned bounding box
	float m_angleY;		// Rotate angle in Y axis
	float m_cosAngleY;	// Cosinus from rotation angle Y
	float m_sinAngleY;	// Sinus from rotation angle Y
};

#endif