////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#include "stdafx.h"

CRTProfiler::CRTProfiler()
{

}

CRTProfiler::~CRTProfiler()
{
	// Delete all scene profiles...
	for(unsigned int i = 0; i < m_sceneProfiles.size(); ++i)
	{
		SProfiledScene* profile = m_sceneProfiles[i];
		if(profile)
		{
			delete profile;
			profile = NULL;
		}
	}
	m_sceneProfiles.clear();
}

bool CRTProfiler::saveSceneProfiles( const char* fileName, CRTProfiler* profiler, E_COMPUTING_TYPE ect )
{
	if(ect == ECT_CPU)
	{
		return CRTProfiler::saveCPUProfiles(fileName, profiler->m_sceneProfiles);
	}
	else if(ect == ECT_CUDA)
	{
		return CRTProfiler::saveGPUProfiles(fileName, profiler->m_sceneProfiles);
	}
}

bool CRTProfiler::saveCPUProfiles( const char* fileName, std::vector<SProfiledScene*> profiles )
{
	FILE* file = fopen(fileName, "a+");
	if(file)
	{
		CRTProfiler::saveHardwareInfo(file);

		fprintf(file, "CPU CALCULATIONS \n\n");

		for(unsigned int i = 0; i < profiles.size(); ++i)
		{
			SProfiledScene* profile = profiles[i];
			if(profile)
			{
				if(profile->m_scene)
					fprintf(file, "MAP: %s\n{\n", profile->m_scene->getName());
				else
					fprintf(file, "MAP: %s\n{\n", "nazwa_mapy");
				CRTProfiler::saveProfile(file, profile);
				fprintf(file, "}\n\n");
			}
		}
		fclose(file);
		return true;
	}

	return false;
}

bool CRTProfiler::saveGPUProfiles( const char* fileName, std::vector<SProfiledScene*> profiles )
{
	FILE* file = fopen(fileName, "a");
	if(file)
	{
		//CRTProfiler::saveHardwareInfo(file);
		fprintf(file, "GPU CALCULATIONS \n\n");

		for(unsigned int i = 0; i < profiles.size(); ++i)
		{
			SProfiledScene* profile = profiles[i];
			if(profile)
			{
				if(profile->m_scene)
					fprintf(file, "MAP: %s\n{\n", profile->m_scene->getName());
				else
					fprintf(file, "MAP: %s\n{\n", "nazwa_mapy");
				CRTProfiler::saveProfile(file, profile);
				fprintf(file, "}\n\n");
			}
		}
		fclose(file);
		return true;
	}

	return false;
}

bool CRTProfiler::addSceneProfile( SProfiledScene* profile )
{
	if(profile)
	{
		m_sceneProfiles.push_back(profile);
		return true;
	}
	return false;
}

void CRTProfiler::saveProfile( FILE* f, SProfiledScene* profile )
{
	if(f && profile)
	{
		fprintf(f, "\tRT Calculation Time ---> %.2f ms\n", profile->m_frameTime);
		/*
		fprintf(f, "\tIntesection Calculation Time ---> %.2f ms\n", profile->m_intesectionTime);

		fprintf(f, "\tTrace lights Time            ---> %.2f ms\n", profile->m_traceLightsTime);
		fprintf(f, "\t{\n");
		fprintf(f, "\t\tLight Calculation Time    ---> %.2f ms\n", profile->m_lighteningTime);
		fprintf(f, "\t\tShadows Calculation Time  ---> %.2f ms\n", profile->m_shadowsTime);
		fprintf(f, "\t\tSpecular Calculation Time ---> %.2f ms\n", profile->m_specularTime);
		fprintf(f, "\t}\n");

		fprintf(f, "\tReflection Calculation Time  ---> %.2f ms\n", profile->m_reflectionTime);
		fprintf(f, "\tRefraction Calculation Time  ---> %.2f ms\n", profile->m_refractionTime);
		fprintf(f, "\tTexturing Calculation Time   ---> %.2f ms\n", profile->m_texturingTime);
		*/
	}
}

void CRTProfiler::saveHardwareInfo( FILE* file )
{
	SYSTEM_INFO sysInfo;
	memset(&sysInfo, 0, sizeof(SYSTEM_INFO));
	GetSystemInfo(&sysInfo);

	MEMORYSTATUS memoryStatus;
	memset(&memoryStatus, 0, sizeof(MEMORYSTATUS));
	GlobalMemoryStatus(&memoryStatus);

	float CPUSpeed = CRTProfiler::getCPUSpeed();
	string deviceModel;
	DISPLAY_DEVICE dd;
	dd.cb = sizeof(DISPLAY_DEVICE);
	int i = 0;
	string id;
	// locate primary display device
	while (EnumDisplayDevices(NULL, i, &dd, 0))
	{
		if (dd.StateFlags & DISPLAY_DEVICE_PRIMARY_DEVICE)
		{
			deviceModel = dd.DeviceString;
			break;
		}
		i++;
	}

	fprintf(file, "Hardware information: \n{\n");
	fprintf(file, "\tNumber of processors: %i\n", sysInfo.dwNumberOfProcessors);
	fprintf(file, "\tProcessor speed: %.2f MHz\n", CPUSpeed);
	fprintf(file, "\tRAM Memory: %.0f MB\n", memoryStatus.dwTotalPhys / (1024.0f * 1024.0f));
	fprintf(file, "\tGraphic card model: %s\n", deviceModel.c_str());
	fprintf(file, "}\n\n");
}

float CRTProfiler::getCPUSpeed()
{
	#define RdTSC __asm _emit 0x0f __asm _emit 0x31

	// variables for the clock-cycles:

	__int64 cyclesStart = 0, cyclesStop = 0;
	// variables for the High-Res Preformance Counter:

	unsigned __int64 nCtr = 0, nFreq = 0, nCtrStop = 0;


	// retrieve performance-counter frequency per second:

	if(!QueryPerformanceFrequency((LARGE_INTEGER *) &nFreq)) return 0;

	// retrieve the current value of the performance counter:

	QueryPerformanceCounter((LARGE_INTEGER *) &nCtrStop);

	// add the frequency to the counter-value:

	nCtrStop += nFreq;


	_asm
	{// retrieve the clock-cycles for the start value:

		RdTSC
			mov DWORD PTR cyclesStart, eax
			mov DWORD PTR [cyclesStart + 4], edx
	}

	do{
		// retrieve the value of the performance counter

		// until 1 sec has gone by:

		QueryPerformanceCounter((LARGE_INTEGER *) &nCtr);
	}while (nCtr < nCtrStop);

	_asm
	{// retrieve again the clock-cycles after 1 sec. has gone by:

		RdTSC
			mov DWORD PTR cyclesStop, eax
			mov DWORD PTR [cyclesStop + 4], edx
	}

	// stop-start is speed in Hz divided by 1,000,000 is speed in MHz

	return    ((float)cyclesStop-(float)cyclesStart) / 1000000;
}