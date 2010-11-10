////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#include "stdafx.h"

void Tetrahedron(CScene* scene, const CVector3& pos, int layers, float distance_r, float sphere_r) {
	int z = 0;
	CMaterial* material = new CMaterial;
	while (z < layers)
	{
		int x = z;
		while (x < layers)
		{
			CSpherePrimitive *s = new CSpherePrimitive(CVector3(pos.m_x + x * 2 * distance_r - z * distance_r, pos.m_y + distance_r, pos.m_z + z * distance_r * sqrtf(3)), sphere_r);
			scene->addPrimitive(s);

			float r = 0.5f + 0.49f * sinf(1+x+z);
			float g = 0.5f + 0.49f * sinf(2+x+z);
			float b = 0.5f + 0.49f * sinf(3+x+z);
			CColor color;
			color.set(r, g, b);
			material->setColor(color);

			CVector3 c;
			//c.randomizeRGBColor();
			//c.interpolateLinear(c, colorGray, 0.2f);

			s->setMaterial(material);

			x++;
		}

		z++;
	}

	if (layers > 1)
	{
		CVector3 new_pos(pos.m_x + distance_r, pos.m_y + 2 * distance_r * sqrtf(2.0f / 3.0f), pos.m_z + distance_r * sqrtf(3) / 3.0f);

		Tetrahedron(scene, new_pos, layers - 1, distance_r, sphere_r);
	}
}
