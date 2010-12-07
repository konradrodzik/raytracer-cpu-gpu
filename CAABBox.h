////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#ifndef __H_CAABBox_H__
#define __H_CAABBox_H__

class CAABBox
{
public:
	// Default constructor
	CAABBox();

	// Initialize constructor
	CAABBox(CVector3& position, CVector3& size);

	// Set positionw
	void setPosition(CVector3& position);

	// Get positionwwwww
	CVector3& getPosition();

	// Set size
	void setSize(CVector3& size);
	
	// Get size
	CVector3& getSize();

	__device__ float3 getPos()
	{
		return make_float3(m_position.m_x, m_position.m_y, m_position.m_z);
	}

	__device__ float3 getS()
	{
		return make_float3(m_size.m_x, m_size.m_y, m_size.m_z);
	}

public:
	CVector3 m_position;	// Axis aligned bounding box position
	CVector3 m_size;		// Size of AABBox
};

#endif