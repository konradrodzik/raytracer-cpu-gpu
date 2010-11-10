////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#include "stdafx.h"

void FibonacciCurve(CScene* scene) {
	CColor color;
	const int COUNT = 100;
	const double TWO_PI = PI * 2.0;

	CMaterial* material = new CMaterial();


	for (int i = 1; i < COUNT; i++)
	{
		const float VALUE = 0.5f * (1 + sqrtf(5));
		const float ANGLE = TWO_PI * VALUE;

		float angle = i * ANGLE;
		float distance = sqrtf(i);

		float factor = (float)i / COUNT;
		float factorpi = 0.5f + factor * TWO_PI;

		float r = 0.5f + 0.49f * sinf(1+factorpi);
		float g = 0.5f + 0.49f * sinf(2+factorpi);
		float b = 0.5f + 0.49f * sinf(3+factorpi);
		color.set(r, g, b);
		//color.interpolateLinear(color, colorYellow, factor);

		float x = distance * cosf(angle);
		float z = distance * sinf(angle);

		float rad = 0.6f;//1.2f * ((COUNT - i) / (float)COUNT);

		float uu = 0 * 8 * rad * rad;

		CSpherePrimitive* sphere = new CSpherePrimitive(CVector3(x, uu + 0.9f + rad, z), 0.9f + rad);
		material->setColor(color);
		sphere->setMaterial(material);
		scene->addPrimitive(sphere);

		//scene->add(s = new Sphere(x, uu + 0.9f + rad, z, 0.9f + rad));
		//s->setMaterial(new Diffusor(color));
	}

	CSpherePrimitive* sphere = new CSpherePrimitive(CVector3(0.0, 10.0, -2.0), 0.1);
	sphere->setLight(true);
	CMaterial* mat = new CMaterial;
	mat->setColor(CVector3(1.0,1.0,1.0));
	sphere->setMaterial(mat);
	scene->addPrimitive(sphere);


	CPlanePrimitive* plane = new CPlanePrimitive(CVector3(0.0, 0.0, -1.0), 12.4);
	CMaterial* planeMat = new CMaterial;
	planeMat->setColor(CVector3(0.4, 0.3, 0.3));
	planeMat->setDiffuse(1.0);
	planeMat->setReflection(0.0);
	planeMat->setRefraction(0.0);
	plane->setMaterial(planeMat);
	//scene->addPrimitive(plane);


}
