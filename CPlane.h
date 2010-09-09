////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#ifndef __H_CPlane_H__
#define __H_CPlane_H__

class CPlane
{
public:
	// Default constructor
	CPlane();

	// Initialize constructor
	CPlane(const CVector3& normal, float d_point);

public:
	union
	{
		struct
		{
			CVector3 m_normal;
			float m_dPoint;
		};

		float k[4];
	};
};

#endif