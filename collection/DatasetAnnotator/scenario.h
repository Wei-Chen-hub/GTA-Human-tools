#ifndef SCENARIO_H
#define SCENARIO_H

#include "script.h"
#include <string>
#include "dictionary.h"
#include <vector>
#include <cstdlib>
#include <ctime>
#include <Windows.h>
#include <gdiplus.h>
#include <iostream>
#include <cmath>
#include <fstream>
#include "keyboard.h"
#include <unordered_set>
#include <string>


#pragma comment (lib,"Gdiplus.lib")

const int max_number_of_peds = 1024;					// size of the pedestrians array
const int number_of_joints = 98;					    // size of the joint_ID subset
const float JOINT_DELTA = 0;
const int max_wpeds = 300;

typedef struct wPed {
	Ped ped;
	bool goingTo = true;
	Vector3 from, to;
	int stopTime;
	int timeFix = -1;
	float speed;
} wPed;

class DatasetAnnotator
{
public:
	DatasetAnnotator(std::string output_path, const char* file_scenario, int max_samples, int is_night);
	int update();
	~DatasetAnnotator();
	static Vector3 ForceGroundZ(Vector3 v);

private:
	std::string output_path;
	int sequence_index;
	Player player;
	Ped playerPed;
	std::string line;								// string use the safe the fram data-line
	std::string log;
	Cam camera;										// camera
	Vector3 cam_coords;								// coordinates of the camera
	Vector3 cam_rot;
	Vector3 wanderer;
	Ped entity_cam;
	bool SHOW_JOINT_RECT;							// bool used to switch the rectangle drawing around the joint
	Ped ped_spawned[max_number_of_peds];
	int ped_spawned_action[max_number_of_peds];
	int ped_spawned_num;
	std::unordered_set<Ped> set_spawned;
	int max_samples;
	std::vector<const char*> bad_scenarios; // 36
	int ped_with_cam;
	const char* file_scenario;
	wPed wPeds[max_wpeds];
	wPed wPeds_scenario[max_wpeds];
	int nwPeds = 0;
	int nwPeds_scenario = 0;
	int moving;
	Vector3 A, B, C;

	int windowWidth;
	int windowHeight;
	int captureFreq;
	int joint_int_codes[number_of_joints];
	int fov;
	int max_waiting_time = 0;
	int is_night;

	HWND hWnd;
	HDC hWindowDC;
	HDC hCaptureDC;
	HBITMAP hCaptureBitmap;

	float recordingPeriod;
	std::clock_t lastRecordingTime;
	long lastRecGameTime;
	int nsample;
	std::ofstream peds_file;								// file used to save joints coordinates data
	std::ofstream depth_file;                               // file used to save depth info
	std::ofstream test_file;                               // file used to test program
	std::ofstream log_file;									// file used to save joints coordinates data
	std::ofstream summary_file;									// file used to mark ending of rendering

	CLSID pngClsid;

	void get_2D_from_3D(Vector3 v, float *x, float *y);
	void save_frame();																		// function used to capture frames internally, then private
	void save_frame_multi_cam(int camera_number);
	void save_labels(Ped ped, std::string anim_dict, std::string anim_name);
	void setCameraFixed(Vector3 coords, Vector3 rot, float cam_z, int fov);
	void genRandomFixedCamera(Vector3 pos);
	void setCameraMoving(Vector3 A, Vector3 B, Vector3 C, int fov);							// function used to set the camera stuff
	//void spawnPed(Vector3 spawnAreaCenter, int numPed);										// function used to spawn pedestrians at the beginning of the scenario
	Vector3 teleportPlayer(Vector3 pos);													// function used to teleport the player to a random position returned by the function
	void draw_text(char* text, float x, float y, float scale);
	void spawn_multi_peds_with_animation(Vector3 pos, int peds_num, BOOL save_depth, BOOL save_stencil);
	void play_multi_peds_animation(std::vector<Ped> peds, int peds_num, BOOL save_depth, BOOL save_stencil);
	void play_multi_peds_multi_cam_animation(std::vector<Ped> peds, int peds_num, BOOL save_depth, BOOL save_stencil, Vector3 pos);
	//void addwPed(Ped p, Vector3 from, Vector3 to, int stop, float spd);
	//void spawn_peds_flow(Vector3 pos, Vector3 goFrom, Vector3 goTo, int npeds, int ngroup, int currentBehaviour, int task_time, int type, int radius, 
	//	int min_lenght, int time_between_walks, int spawning_radius, float speed);
	//void spawn_peds(Vector3 pos, Vector3 goFrom, Vector3 goTo, int npeds, int ngroup, int currentBehaviour,
	//	int task_time, int type, int radius, int min_lenght, int time_between_walks, int spawning_radius, float speed);
	void play_peds_animation(std::vector<Ped> peds, std::string anim_dict, std::string anim_name, int anim_time);
	void Get_depth_info();
	void spawn_peds_with_animation(Vector3 pos, Vector3 goFrom, Vector3 goTo, int npeds, int currnetBehavirour);
	//void spawn_peds_with_animation_v0(Vector3 pos, Vector3 goFrom, Vector3 goTo, int npeds, int currnetBehavirour);
	//void spawn_peds_climb(Vector3 pos, Vector3 goFrom, Vector3 goTo, int npeds, int currnetBehavirour);
	void loadScenario(const char* fname);
	void load_new_multi_Scenario(const char* fname);
	//void walking_peds();
	int myreadLine(FILE *f, Vector3 *pos, int *nPeds, int *ngroup, int *currentBehaviour, float *speed, Vector3 *goFrom, Vector3 *goTo, int *task_time,
		int *type, int *radius, int *min_lenght, int *time_between_walks, int *spawning_radius);
	
	//void addwPed_scenario(Ped p);
	Cam lockCam(Vector3 pos, Vector3 rot);

	//new added variables below

	int hash;
	std::string model_key;
};

int GetEncoderClsid(const WCHAR* format, CLSID* pClsid);
void loadAllPedAnimsList();

#endif // !SCENARIO_H
