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

	// Set position
	void setPosition(CVector3& position);

	// Get position
	CVector3& getPosition();

	// Set size
	void setSize(CVector3& size);
	
	// Get size
	CVector3& getSize();

private:
	CVector3 m_position;	// Axis aligned bounding box position
	CVector3 m_size;		// Size of AABBox
};

#endif