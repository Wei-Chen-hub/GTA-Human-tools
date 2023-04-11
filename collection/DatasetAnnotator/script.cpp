/*
	THIS FILE IS A PART OF GTA V SCRIPT HOOK SDK
				http://dev-c.com			
			(C) Alexander Blade 2015
*/

#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING

#include "scenario.h"
#include <string.h>
#include <direct.h>
#include <fstream>
#include "files.h"
#include <list>
#include <unordered_set>
#include <experimental/filesystem>
#include "keyboard.h"
#include <windows.h>
//#include <process.h>
//#include <Tlhelp32.h>
//#include <winbase.h>

DWORD	vehUpdateTime;
DWORD	pedUpdateTime;
using namespace std;
namespace fs = std::experimental::filesystem;

//void killProcessByName(const char* filename)
//{
//	HANDLE hSnapShot = CreateToolhelp32Snapshot(TH32CS_SNAPALL, NULL);
//	PROCESSENTRY32 pEntry;
//	pEntry.dwSize = sizeof(pEntry);
//	BOOL hRes = Process32First(hSnapShot, &pEntry);
//	while (hRes)
//	{
//		if (strcmp(pEntry.szExeFile, filename) == 0)
//		{
//			HANDLE hProcess = OpenProcess(PROCESS_TERMINATE, 0,
//				(DWORD)pEntry.th32ProcessID);
//			if (hProcess != NULL)
//			{
//				TerminateProcess(hProcess, 9);
//				CloseHandle(hProcess);
//			}
//		}
//		hRes = Process32Next(hSnapShot, &pEntry);
//	}
//	CloseHandle(hSnapShot);
//}

void record() {
	char path[] = "MTA\\";
	char scenarios_path[] = "MTA-Scenarios\\";

	char buffer[80];
	std::ofstream task_logger;
	time_t t = time(0);
	struct tm now;
	localtime_s(&now, &t);
	strftime(buffer, 80, "mta_task_log_%Y_%m_%d_%H_%M.log", &now);
	task_logger.open(buffer);

	_mkdir(path);

	int max_samples = 12+1;

	DatasetAnnotator* S;

	loadAllPedAnimsList();

	int seq_number = 0;

	// Day sequences
	while (true) {
		for (auto& p : fs::recursive_directory_iterator(scenarios_path)) {
			if (fs::is_regular_file(p)) {
				task_logger << "P" << ':' << p << std::endl;
				task_logger.flush();
				std::string output_path = std::string(path) + std::string("seq_") + p.path().filename().string().substr(4, 8);
				try {
					int nsamples = 0;
					_mkdir(output_path.c_str());
					fs::copy(p, fs::path(output_path), fs::copy_options::overwrite_existing); //copy config file to result folder

					S = new DatasetAnnotator(output_path, p.path().string().c_str(), max_samples, 0);
					WAIT(1000);
					//while (nsamples < max_samples) {
					//	nsamples = (*S).update();
					//	WAIT(0);
					//}
					//Sleep(100);
					delete static_cast <DatasetAnnotator*>(S);
					WAIT(0);
					seq_number++;
					fs::remove(p);
				}
				catch (const std::exception& e) {
					task_logger << "F" << ':' << p << std::endl;
					task_logger << "E" << ':' << e.what() << std::endl;
					task_logger.flush();
					ofstream summary_file;
					summary_file.open(output_path + "\\summary.txt");
					summary_file << "Done with Failure\n";
				}
			}
		}
		WAIT(1);
	}

	// killProcessByName("GTAV.exe");

}


void main()
{	
	std::ofstream strm("logme.txt");
	while (true) {
		if (IsKeyJustUp(VK_F8)){ 
//			while (true) {
				record();
//			}
		}
		WAIT(0);

	}
}

void ScriptMain()
{	
	srand(GetTickCount());
	main();
}
