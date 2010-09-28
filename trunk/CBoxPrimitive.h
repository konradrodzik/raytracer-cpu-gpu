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

	// Set position
	void setPosition(CVector3& pos);

	// Set size
	void setSize(CVector3& size);

private:
	CAABBox m_box;
};

#endif