////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#ifndef __H_CVector3_H__
#define __H_CVector3_H__

__device__ class CVector3
{
public:
	// Default constructor
	__device__ CVector3() : m_x(0.0f), m_y(0.0f), m_z(0.0f)
	{
	}

	// Initialize constructor
	__device__ CVector3(float x, float y, float z) : m_x(x), m_y(y), m_z(z)
	{
	}

	// Set vector
	inline void set(float x, float y, float z)
	{
		m_x = x;
		m_y = y;
		m_z = z;
	}

	// Normalize vector
	inline void normalize()
	{
		float length = 1.0f / getLength();
		m_x *= length;
		m_y *= length;
		m_z *= length;
	}

	// Get length of vector
	inline float getLength()
	{
		return sqrtf(m_x*m_x + m_y*m_y + m_z*m_z);
	}

	// Get square length of vector
	inline float getSqrLength();

	// Dot product
	inline float dot(CVector3 vec) { return (m_x*vec.m_x + m_y*vec.m_y + m_z*vec.m_z); }

	// Cross product
	inline CVector3 cross(CVector3 vec) { return CVector3(m_y*vec.m_z - m_z*vec.m_y, m_z*vec.m_x - m_x*vec.m_z, m_x*vec.m_y - m_y*vec.m_x); }

	// OPERATORS
	
	void operator+=(CVector3& vec);
	void operator+=(CVector3* vec);
	void operator-=(CVector3& vec);
	void operator-=(CVector3* vec);
	void operator*=(CVector3& vec);
	void operator*=(CVector3* vec);
	void operator*=(float scalar);
	CVector3 operator-() const;

	friend CVector3 operator+( const CVector3& vec1, const CVector3& vec2 )
	{
		return CVector3(vec1.m_x + vec2.m_x, vec1.m_y + vec2.m_y, vec1.m_z + vec2.m_z);
	}

	friend CVector3 operator-( const CVector3& vec1, const CVector3& vec2 )
	{
		return CVector3(vec1.m_x - vec2.m_x, vec1.m_y - vec2.m_y, vec1.m_z - vec2.m_z);
	}

	friend CVector3 operator*( const CVector3& vec, float scalar )
	{
		return CVector3(vec.m_x * scalar, vec.m_y * scalar, vec.m_z * scalar);
	}

	friend CVector3 operator*( const CVector3& vec1, CVector3& vec2 )
	{
		return CVector3(vec1.m_x * vec2.m_x, vec1.m_y * vec2.m_y, vec1.m_z * vec2.m_z);
	}

	friend CVector3 operator*( float scalar, const CVector3& vec )
	{
		return CVector3(vec.m_x * scalar, vec.m_y * scalar, vec.m_z * scalar);
	}

public:
	union
	{
		struct  
		{
			float m_x, m_y, m_z;
		};
		struct  
		{
			float m_r, m_g, m_b;
		};
		struct  
		{
			float k[3];
		};
	};
};

typedef CVector3 CColor;

#endif