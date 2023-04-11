/*
	THIS FILE USED TO BE A PART OF GTA V SCRIPT HOOK SDK
				http://dev-c.com
			(C) Alexander Blade 2015
*/

#define _CRT_SECURE_NO_WARNINGS

#include "main.h"
#include "script.h"
#include "keyboard.h"
#include "capture.h"

#include <MinHook.h>
#include <stdio.h>
#include <dxgi.h>
#include <d3d11.h>
#include <d3d11_4.h>
#include <system_error>
#include <export.h>
using Microsoft::WRL::ComPtr;




BOOL APIENTRY DllMain(HMODULE hInstance, DWORD reason, LPVOID lpReserved)
{
	MH_STATUS res;
	auto f = fopen(logFilePath, "a");
	switch (reason)
	{
	case DLL_PROCESS_ATTACH:
		res = MH_Initialize();
		if (res != MH_OK) fprintf(f, "Could not init Minihook\n");
		//scriptRegister(hinstance, scriptMain);
		presentCallbackRegister(presentCallback);
		keyboardHandlerRegister(OnKeyboardMessage);
		scriptRegister(hInstance, ScriptMain);
		break;
	case DLL_PROCESS_DETACH:
		res = MH_Uninitialize();
		if (res != MH_OK) fprintf(f, "Could not deinit MiniHook\n");
		presentCallbackUnregister(presentCallback);
		keyboardHandlerUnregister(OnKeyboardMessage);
		scriptUnregister(hInstance);
		break;
	}
	fclose(f);
	return TRUE;
}


