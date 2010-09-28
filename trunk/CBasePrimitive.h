////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#ifndef __H_CBasePrimitive_H__
#define __H_CBasePrimitive_H__

// Primitives type
enum E_PRIMITIVE_TYPE
{
	EPT_SPHERE = 1,
	EPT_PLANE  = 2,
	EPT_BOX	   = 3,
	EPT_LIGHT  = 4,
};

// Intersection flags
#define PRIM_HIT	1
#define PRIM_MISS	0
#define PRIM_HITIN -1

// Base primitive class
class CBasePrimitive
{
public:
	// Default constructor
	CBasePrimitive();

	// Set primitive material
	void setMaterial(CMaterial* material);

	// Get primitive material
	CMaterial* getMaterial();

	// Set primitive light status
	void setLight(bool light);

	// Is this primitive light source?
	bool isLight();

	// Set primitive name
	void setName(const std::string& name);

	// Get primitive name
	std::string getName();

	// Get primitive color at given position
	virtual CColor getColor(const CVector3& pos);

	virtual E_PRIMITIVE_TYPE getType() = 0;

	virtual int intersect(CRay& ray, float& distance) = 0;

	// Get primitive normal at given position
	virtual CVector3 getNormal(const CVector3& pos) = 0;

	virtual void setPosition(CVector3& pos) = 0;

protected:
	CMaterial m_material;	// Primitive material
	bool m_isLight;			// Is this primitive light source?
	std::string m_name;		// Primitive name
};

#endif