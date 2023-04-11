#define _CRT_SECURE_NO_WARNINGS

#include "scenario.h"
#include <vector>
#include <direct.h>
#include <string.h>
#include <filesystem>
#include <string>
#include <sstream>
#include <fstream>
#include <ctime>
#include <iostream>
#include <Eigen/Dense>
#include <list>
using Eigen::MatrixXd;

#define _USE_MATH_DEFINES
#include <math.h>
#include <time.h>
#include <list>
#include <unordered_set>
#include <capture.h>
using namespace std;

#define SCREEN_WIDTH 1920
#define SCREEN_HEIGHT 1080
#define CAM_FOV 50
#define TIME_FACTOR 2.0
#define RECORD_FPS 15 
#define DISPLAY_FLAG FALSE
//#define DISPLAY_FLAG TRUE
#define WANDERING_RADIUS 10.0
#define MAX_PED_TO_CAM_DISTANCE 10.0
#define DEMO FALSE

static char scenarioTypes[14][40]{
	"NEAREST",
	"RANDOM",
	"WORLD_HUMAN_MUSICIAN",
	"WORLD_HUMAN_SMOKING",
	"WORLD_HUMAN_BINOCULARS",
	"WORLD_HUMAN_CHEERING",
	"WORLD_HUMAN_DRINKING",
	"WORLD_HUMAN_PARTYING",
	"WORLD_HUMAN_PICNIC",
	"WORLD_HUMAN_STUPOR",
	"WORLD_HUMAN_PUSH_UPS",
	"WORLD_HUMAN_LEANING",
	"WORLD_HUMAN_MUSCLE_FLEX",
	"WORLD_HUMAN_YOGA"
};

static LPCSTR weaponNames[] = {
	"WEAPON_KNIFE", "WEAPON_NIGHTSTICK", "WEAPON_HAMMER", "WEAPON_BAT", "WEAPON_GOLFCLUB",
	"WEAPON_CROWBAR", "WEAPON_PISTOL", "WEAPON_COMBATPISTOL", "WEAPON_APPISTOL", "WEAPON_PISTOL50",
	"WEAPON_MICROSMG", "WEAPON_SMG", "WEAPON_ASSAULTSMG", "WEAPON_ASSAULTRIFLE",
	"WEAPON_CARBINERIFLE", "WEAPON_ADVANCEDRIFLE", "WEAPON_MG", "WEAPON_COMBATMG", "WEAPON_PUMPSHOTGUN",
	"WEAPON_SAWNOFFSHOTGUN", "WEAPON_ASSAULTSHOTGUN", "WEAPON_BULLPUPSHOTGUN", "WEAPON_STUNGUN", "WEAPON_SNIPERRIFLE",
	"WEAPON_HEAVYSNIPER", "WEAPON_GRENADELAUNCHER", "WEAPON_GRENADELAUNCHER_SMOKE", "WEAPON_RPG", "WEAPON_MINIGUN",
	"WEAPON_GRENADE", "WEAPON_STICKYBOMB", "WEAPON_SMOKEGRENADE", "WEAPON_BZGAS", "WEAPON_MOLOTOV",
	"WEAPON_FIREEXTINGUISHER", "WEAPON_PETROLCAN", "WEAPON_FLARE", "WEAPON_SNSPISTOL", "WEAPON_SPECIALCARBINE",
	"WEAPON_HEAVYPISTOL", "WEAPON_BULLPUPRIFLE", "WEAPON_HOMINGLAUNCHER", "WEAPON_PROXMINE", "WEAPON_SNOWBALL",
	"WEAPON_VINTAGEPISTOL", "WEAPON_DAGGER", "WEAPON_FIREWORK", "WEAPON_MUSKET", "WEAPON_MARKSMANRIFLE",
	"WEAPON_HEAVYSHOTGUN", "WEAPON_GUSENBERG", "WEAPON_HATCHET", "WEAPON_RAILGUN", "WEAPON_COMBATPDW",
	"WEAPON_KNUCKLE", "WEAPON_MARKSMANPISTOL", "WEAPON_FLASHLIGHT", "WEAPON_MACHETE", "WEAPON_MACHINEPISTOL",
	"WEAPON_SWITCHBLADE", "WEAPON_REVOLVER", "WEAPON_COMPACTRIFLE", "WEAPON_DBSHOTGUN", "WEAPON_FLAREGUN",
	"WEAPON_AUTOSHOTGUN", "WEAPON_BATTLEAXE", "WEAPON_COMPACTLAUNCHER", "WEAPON_MINISMG", "WEAPON_PIPEBOMB",
	"WEAPON_POOLCUE", "WEAPON_SWEEPER", "WEAPON_WRENCH"
};

static LPCSTR meleeWeapons[] = {
	"WEAPON_KNIFE", "WEAPON_NIGHTSTICK", "WEAPON_HAMMER", "WEAPON_BAT", "WEAPON_GOLFCLUB",
	"WEAPON_CROWBAR", "WEAPON_BOTTLE", "WEAPON_DAGGER", "WEAPON_HATCHET",
	"WEAPON_KNUCKLEDUSTER", "WEAPON_FLASHLIGHT", "WEAPON_MACHETE", "WEAPON_POOLCUE",
	"WEAPON_SWITCHBLADE", "WEAPON_WRENCH", "WEAPON_BATTLEAXE"
};

static const int weaponCnt = 16;

std::map<std::string, std::vector<std::pair<std::string, float>>> AllPedAnims;
std::vector<std::string> AnimDictlist;
std::vector<std::tuple<std::string, std::string, float>> animlist;
std::vector<std::tuple<std::string, std::string, float>> scene_anim;
std::vector<int> AnimHashlist;

int nAnims = 0;
void loadAllPedAnimsList()
{
	std::ifstream fin("PedAnimList.txt");

	if (fin.is_open())
	{
		AllPedAnims.clear();
		AnimDictlist.clear();
		animlist.clear();

		//size_t space;
		//std::string lineLeft, lineRight;

		//for (std::string line; std::getline(fin, line);)
		//{
		//	if (line.length() > 2)
		//	{
		//		space = line.find(' ');
		//		if (space != std::string::npos)
		//		{
		//			lineLeft = line.substr(0, space);
		//			lineRight = line.substr(space + 1);
		//			AllPedAnims[lineLeft].push_back(lineRight);
		//		}
		//	}
		//}
		for (std::string line; std::getline(fin, line);) {
			std::istringstream ss(line);
			std::string ad, an;
			float at;
			ss >> ad >> an >> at;
			AllPedAnims[ad].push_back(std::make_pair(an, at));
			animlist.push_back(std::make_tuple(ad, an, at));
		}
		for (auto item : AllPedAnims) AnimDictlist.push_back(item.first);
		fin.close();
	}
}

void loadAllPedHashList()
{
	std::ifstream fin("PedHashList.txt");
	if (fin.is_open()) {
		AnimHashlist.clear();
		for (std::string line; std::getline(fin, line);) {
			std::istringstream ss(line);
			int hash_num;
			ss >> hash_num;
			AnimHashlist.push_back(hash_num);
		}
		fin.close();
	}
}

void set_status_text(std::string text)
{
	UI::_SET_NOTIFICATION_TEXT_ENTRY((char*)"STRING");
	UI::_ADD_TEXT_COMPONENT_STRING((LPSTR)text.c_str());
	UI::_DRAW_NOTIFICATION(1, 1);
}

void display_clear() {
	UI::DISPLAY_HUD(FALSE);
	UI::DISPLAY_RADAR(FALSE);
	UI::DISPLAY_AREA_NAME(FALSE);
	UI::DISPLAY_AMMO_THIS_FRAME(FALSE);
	UI::DISPLAY_CASH(FALSE);	
}

float random_float(float min, float max) {
	return min + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (max - min)));
}

int random_int(int min, int max) {
	return min + rand() % (max - min + 1);
}

Vector3 coordsToVector(float x, float y, float z)
{
	Vector3 v;
	v.x = x;
	v.y = y;
	v.z = z;
	return v;
}

int GetEncoderClsid(const WCHAR* format, CLSID* pClsid)
{
	UINT  num = 0;          // number of image encoders
	UINT  size = 0;         // size of the image encoder array in bytes

	Gdiplus::ImageCodecInfo* pImageCodecInfo = NULL;

	Gdiplus::GetImageEncodersSize(&num, &size);
	if (size == 0)
		return -1;  // Failure

	pImageCodecInfo = (Gdiplus::ImageCodecInfo*)(malloc(size));
	if (pImageCodecInfo == NULL)
		return -1;  // Failure

	GetImageEncoders(num, size, pImageCodecInfo);

	for (UINT j = 0; j < num; ++j)
	{
		if (wcscmp(pImageCodecInfo[j].MimeType, format) == 0)
		{
			*pClsid = pImageCodecInfo[j].Clsid;
			free(pImageCodecInfo);
			return j;  // Success
		}
	}

	free(pImageCodecInfo);
	return -1;  // Failure
}

int StringToWString(std::wstring &ws, const std::string &s)
{
	std::wstring wsTmp(s.begin(), s.end());
	ws = wsTmp;
	return 0;
}

DatasetAnnotator::DatasetAnnotator(std::string _output_path, const char* _file_scenario, int _max_samples, int _is_night)
{
	lastRecordingTime = std::clock();

	// reset game time scale
	GAMEPLAY::SET_TIME_SCALE(1);

	this->output_path = _output_path;
	this->file_scenario = _file_scenario;
	this->max_samples = _max_samples;
	this->is_night = _is_night;

	std::srand((unsigned int)time(NULL)); //Initialises randomiser or sum' like that

	// joint codes
	for (int i = 0; i < 98; i++) {
		joint_int_codes[i] = m.find(jointNames[i])->second;
	}

	// inizialize the peds_file used to storage coords data
	log_file.open(output_path + "\\log.txt");
	peds_file.open(output_path + "\\peds.csv");
	peds_file << "frame,ped_id,ped_action,joint_type,2D_x,2D_y,3D_x,3D_y,3D_z,occluded,self_occluded,";
	peds_file << "cam_3D_x,cam_3D_y,cam_3D_z,cam_rot_x,cam_rot_y,cam_rot_z,fov,is_male\n";

	this->player = PLAYER::PLAYER_ID();
	this->playerPed = PLAYER::PLAYER_PED_ID();
	this->line = "";
	this->log = "";
	this->captureFreq = (int)(RECORD_FPS / TIME_FACTOR);
	this->SHOW_JOINT_RECT = DISPLAY_FLAG;
	this->fov = CAM_FOV;
	this->ped_spawned_num = 0;



	


	//Screen capture buffer
	GRAPHICS::_GET_SCREEN_ACTIVE_RESOLUTION(&windowWidth, &windowHeight);
	hWnd = ::FindWindow(NULL, "Compatitibility Theft Auto V");
	hWindowDC = GetDC(hWnd);
	hCaptureDC = CreateCompatibleDC(hWindowDC);
	hCaptureBitmap = CreateCompatibleBitmap(hWindowDC, SCREEN_WIDTH, SCREEN_HEIGHT);
	SelectObject(hCaptureDC, hCaptureBitmap);
	SetStretchBltMode(hCaptureDC, COLORONCOLOR);

	// used to decide how often save the sample
	recordingPeriod = 1.0f / captureFreq;

	// initialize recording stuff
	Gdiplus::GdiplusStartupInput gdiplusStartupInput;
	ULONG_PTR gdiplusToken;
	GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, NULL);
	GetEncoderClsid(L"image/jpeg", &pngClsid);

	// inizialize the int used to count the saved frame
	nsample = 0;


	// clear player state
	//Avoid bad things such as getting killed by the police, robbed, dying in car accidents or other horrible stuff
	PLAYER::SET_EVERYONE_IGNORE_PLAYER(player, TRUE);
	PLAYER::SET_POLICE_IGNORE_PLAYER(player, TRUE);
	PLAYER::CLEAR_PLAYER_WANTED_LEVEL(player);
	PLAYER::SET_PLAYER_INVINCIBLE(player, TRUE);
	PLAYER::SPECIAL_ABILITY_FILL_METER(player, 1);
	PLAYER::SET_PLAYER_NOISE_MULTIPLIER(player, 0.0);
	PLAYER::SET_SWIM_MULTIPLIER_FOR_PLAYER(player, 1.49f);
	PLAYER::SET_RUN_SPRINT_MULTIPLIER_FOR_PLAYER(player, 1.49f);
	PLAYER::DISABLE_PLAYER_FIRING(player, TRUE);
	PLAYER::SET_DISABLE_AMBIENT_MELEE_MOVE(player, TRUE);
	ENTITY::SET_ENTITY_ALPHA(playerPed, 0, 0);
	ENTITY::SET_ENTITY_CAN_BE_DAMAGED(playerPed, FALSE);
	//ENTITY::SET_ENTITY_COLLISION(playerPed, FALSE, TRUE); //this line can cause texture loss every 8s

	log_file << "LOG: Init time " << std::clock() - lastRecordingTime << std::endl;
	log_file << "LOG: Frame Time " << GAMEPLAY::GET_FRAME_TIME() << std::endl;

	lastRecordingTime = std::clock();
	lastRecGameTime = GAMEPLAY::GET_GAME_TIMER() + 100;
	

	load_new_multi_Scenario(file_scenario);
	//loadScenario(file_scenario);
	

}

DatasetAnnotator::~DatasetAnnotator()	
{
	// todo: implement a destroyer
	ReleaseDC(hWnd, hWindowDC);
	DeleteDC(hCaptureDC);
	DeleteObject(hCaptureBitmap);
	peds_file.close();
	log_file.close();
	summary_file.close();	
}

void DatasetAnnotator::get_2D_from_3D(Vector3 v, float *x2d, float *y2d) {

	// translation
	float x = v.x - cam_coords.x;
	float y = v.y - cam_coords.y;
	float z = v.z - cam_coords.z;

	// rotation
	float cam_x_rad = cam_rot.x * (float)M_PI / 180.0f;
	float cam_y_rad = cam_rot.y * (float)M_PI / 180.0f;
	float cam_z_rad = cam_rot.z * (float)M_PI / 180.0f;

	// cos
	float cx = cos(cam_x_rad);
	float cy = cos(cam_y_rad);
	float cz = cos(cam_z_rad);

	// sin
	float sx = sin(cam_x_rad);
	float sy = sin(cam_y_rad);
	float sz = sin(cam_z_rad);	

	Vector3 d;
	d.x = cy*(sz*y + cz*x) - sy*z;
	d.y = sx*(cy*z + sy*(sz*y + cz*x)) + cx*(cz*y - sz*x);
	d.z = cx*(cy*z + sy*(sz*y + cz*x)) - sx*(cz*y - sz*x);

	float fov_rad = fov * (float)M_PI / 180;
	float f = (SCREEN_HEIGHT / 2.0f) * cos(fov_rad / 2.0f) / sin(fov_rad / 2.0f);

	*x2d = ((d.x * (f / d.y)) / SCREEN_WIDTH + 0.5f);
	*y2d = (0.5f - (d.z * (f / d.y)) / SCREEN_HEIGHT);
}

void DatasetAnnotator::save_frame() {
	StretchBlt(hCaptureDC, 0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, hWindowDC, 0, 0, windowWidth, windowHeight, SRCCOPY | CAPTUREBLT);
	Gdiplus::Bitmap image(hCaptureBitmap, (HPALETTE)0);
	std::wstring ws;
	StringToWString(ws, output_path);
	char idx[9];
	snprintf(idx, 9, "%08dc%02d", nsample);
	std::wstring idx_str;
	StringToWString(idx_str, std::string(idx));
	image.Save((ws + L"\\" + idx_str + L".jpeg").c_str(), &pngClsid, NULL);
}

void DatasetAnnotator::save_frame_multi_cam(int camera_number) {
	StretchBlt(hCaptureDC, 0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, hWindowDC, 0, 0, windowWidth, windowHeight, SRCCOPY | CAPTUREBLT);
	Gdiplus::Bitmap image(hCaptureBitmap, (HPALETTE)0);
	std::wstring ws;
	StringToWString(ws, output_path);
	char idx[12];
	snprintf(idx, 12, "%08dc%02d", nsample, camera_number);
	std::wstring idx_str;
	StringToWString(idx_str, std::string(idx));
	image.Save((ws + L"\\" + idx_str + L".jpeg").c_str(), &pngClsid, NULL);
}

void DatasetAnnotator::setCameraMoving(Vector3 A, Vector3 B, Vector3 C, int fov) {
	
	CAM::DESTROY_ALL_CAMS(TRUE);
	this->camera = CAM::CREATE_CAM((char *)"DEFAULT_SCRIPTED_CAMERA", TRUE);
	//this->ped_with_cam = PED::CREATE_RANDOM_PED(A.x, A.y, A.z);
	this->ped_with_cam = PLAYER::PLAYER_PED_ID();
	ENTITY::SET_ENTITY_COORDS_NO_OFFSET(ped_with_cam, A.x, A.y, A.z, 0, 0, 1);
	//AI::TASK_WANDER_IN_AREA(this->ped_with_cam, coords.x, coords.y, coords.z, WANDERING_RADIUS, 1.0, 1.0);
	float z_offset = ((float)((rand() % (6)) - 2)) / 10;
	CAM::ATTACH_CAM_TO_ENTITY(camera, this->ped_with_cam, 0, 0, z_offset, TRUE);
	CAM::SET_CAM_ACTIVE(camera, TRUE);
	CAM::SET_CAM_FOV(camera, (float)fov);
	CAM::RENDER_SCRIPT_CAMS(TRUE, FALSE, 0, TRUE, TRUE);
	//CAM::SET_CAM_MOTION_BLUR_STRENGTH(camera, 10.0);

	//ENTITY::SET_ENTITY_HEALTH(ped_with_cam, 0);
	WAIT(500);
	//AI::CLEAR_PED_TASKS_IMMEDIATELY(ped_with_cam);
	//PED::RESURRECT_PED(ped_with_cam);
	//PED::REVIVE_INJURED_PED(ped_with_cam);
	//PED::SET_PED_CAN_RAGDOLL(ped_with_cam, TRUE);

	ENTITY::SET_ENTITY_COLLISION(ped_with_cam, TRUE, TRUE);
	ENTITY::SET_ENTITY_VISIBLE(ped_with_cam, FALSE, FALSE);
	ENTITY::SET_ENTITY_ALPHA(ped_with_cam, 0, FALSE);
	ENTITY::SET_ENTITY_CAN_BE_DAMAGED(ped_with_cam, FALSE);
	PED::SET_BLOCKING_OF_NON_TEMPORARY_EVENTS(ped_with_cam, TRUE);
	PED::SET_PED_COMBAT_ATTRIBUTES(ped_with_cam, 1, FALSE);

	Object seq;
	AI::OPEN_SEQUENCE_TASK(&seq);
	//AI::TASK_USE_MOBILE_PHONE_TIMED(0, max_waiting_time + 10000);
	AI::TASK_STAND_STILL(0, max_waiting_time + 10000);
	AI::TASK_GO_TO_COORD_ANY_MEANS(0, A.x, A.y, A.z, 1.0, 0, 0, 786603, 0xbf800000);
	AI::TASK_GO_TO_COORD_ANY_MEANS(0, B.x, B.y, B.z, 1.0, 0, 0, 786603, 0xbf800000);
	AI::TASK_GO_TO_COORD_ANY_MEANS(0, C.x, C.y, C.z, 1.0, 0, 0, 786603, 0xbf800000);
	AI::TASK_GO_TO_COORD_ANY_MEANS(0, B.x, B.y, B.z, 1.0, 0, 0, 786603, 0xbf800000);
	AI::TASK_GO_TO_COORD_ANY_MEANS(0, A.x, A.y, A.z, 1.0, 0, 0, 786603, 0xbf800000);
	AI::TASK_GO_TO_COORD_ANY_MEANS(0, B.x, B.y, B.z, 1.0, 0, 0, 786603, 0xbf800000);
	AI::TASK_GO_TO_COORD_ANY_MEANS(0, C.x, C.y, C.z, 1.0, 0, 0, 786603, 0xbf800000);
	AI::TASK_GO_TO_COORD_ANY_MEANS(0, B.x, B.y, B.z, 1.0, 0, 0, 786603, 0xbf800000);
	AI::TASK_GO_TO_COORD_ANY_MEANS(0, A.x, A.y, A.z, 1.0, 0, 0, 786603, 0xbf800000);
	AI::CLOSE_SEQUENCE_TASK(seq);
	AI::TASK_PERFORM_SEQUENCE(ped_with_cam, seq);
	AI::CLEAR_SEQUENCE_TASK(&seq);

	// set the cam_coords used on update() function
	this->cam_coords = CAM::GET_CAM_COORD(camera);
	this->cam_rot = CAM::GET_CAM_ROT(camera, 2);
}

void DatasetAnnotator::setCameraFixed(Vector3 coords, Vector3 rot, float cam_z, int fov) {

	CAM::DESTROY_ALL_CAMS(TRUE);
	this->camera = CAM::CREATE_CAM((char *)"DEFAULT_SCRIPTED_CAMERA", TRUE);
	CAM::SET_CAM_COORD(camera, coords.x, coords.y, coords.z+cam_z);
	CAM::SET_CAM_ROT(camera, rot.x, rot.y, rot.z, 2);
	CAM::SET_CAM_ACTIVE(camera, TRUE);
	CAM::SET_CAM_FOV(camera, (float)fov);
	CAM::RENDER_SCRIPT_CAMS(TRUE, FALSE, 0, TRUE, TRUE);


	// set the cam_coords used on update() function
	this->cam_coords = CAM::GET_CAM_COORD(camera);
	this->cam_rot = CAM::GET_CAM_ROT(camera, 2);
	this->fov = (int)CAM::GET_CAM_FOV(camera);
}

void DatasetAnnotator::genRandomFixedCamera(Vector3 pos) {
	float distance = random_float(5, 8);
	float new_rx = random_float(-0.5 * M_PI, 0.5 * M_PI);
	float new_ry = 0;
	float new_rz = random_float(0, 0.25 * M_PI);

	Vector3 new_cam_coord, new_cam_rot;
	new_cam_rot.x = new_rx;
	new_cam_rot.y = new_ry;
	new_cam_rot.z = new_rz;

	float x_off = distance * cos(new_rx) * sin(new_rz);
	float y_off = -distance * cos(new_rx) * cos(new_rz);
	float z_off = distance * sin(new_rx);

	new_cam_coord.x = pos.x + x_off;
	new_cam_coord.y = pos.y + y_off;
	new_cam_coord.z = pos.z + z_off;

	cam_coords = new_cam_coord;
	cam_rot = new_cam_rot;
}


Vector3 DatasetAnnotator::teleportPlayer(Vector3 pos){
												
	// set the heading
	float heading = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / 360));

	// teleport the player to the previously selected coordinates
	PLAYER::START_PLAYER_TELEPORT(this->player, pos.x, pos.y, pos.z, heading, 0, 0, 0);
	while (PLAYER::IS_PLAYER_TELEPORT_ACTIVE()) WAIT(0);

	return pos;
}

void DatasetAnnotator::draw_text(char *text, float x, float y, float scale) {
	UI::SET_TEXT_FONT(0);
	UI::SET_TEXT_SCALE(scale, scale);
	UI::SET_TEXT_COLOUR(255, 255, 255, 245);
	UI::SET_TEXT_WRAP(0.0, 1.0);
	UI::SET_TEXT_CENTRE(0);
	UI::SET_TEXT_DROPSHADOW(2, 2, 0, 0, 0);
	UI::SET_TEXT_EDGE(1, 0, 0, 0, 205);
	UI::_SET_TEXT_ENTRY((char *)"STRING");
	UI::_ADD_TEXT_COMPONENT_STRING(text);
	UI::_DRAW_TEXT(y, x);
}

int DatasetAnnotator::myreadLine(FILE *f, Vector3 *pos, int *nPeds, int *ngroup, int *currentBehaviour, float *speed, Vector3 *goFrom, Vector3 *goTo, int *task_time, int *type, 
	int *radius, int *min_lenght, int *time_between_walks, int *spawning_radius)
{
	int result = fscanf_s(f, "%d %f %f %f %d %d %f %f %f %f %f %f %f %d %d %d %d %d %d \n", nPeds, &(*pos).x, &(*pos).y, &(*pos).z,
		ngroup, currentBehaviour, speed, 
		&(*goFrom).x, &(*goFrom).y, &(*goFrom).z, &(*goTo).x, &(*goTo).y, &(*goTo).z, 
		task_time, type, radius, min_lenght, time_between_walks, spawning_radius);

	return result;
}

Cam DatasetAnnotator::lockCam(Vector3 pos, Vector3 rot) {
	CAM::DESTROY_ALL_CAMS(true);
	Cam lockedCam = CAM::CREATE_CAM_WITH_PARAMS((char *)"DEFAULT_SCRIPTED_CAMERA", pos.x, pos.y, pos.z, rot.x, rot.y, rot.z, 50, true, 2);
	CAM::SET_CAM_ACTIVE(lockedCam, true);
	CAM::RENDER_SCRIPT_CAMS(1, 0, 3000, 1, 0);
	return lockedCam;
}

void DatasetAnnotator::loadScenario(const char* fname)
{
	FILE *f = fopen(fname, "r");
	Vector3 cCoords, cRot;
	Vector3 vTP1, vTP2, vTP1_rot, vTP2_rot, safe, tmpTP;
	int stop;
	float tmp;

	fscanf_s(f, "%d ", &moving);
	if (moving == 0) 
		fscanf_s(f, "%f %f %f %d %f %f %f %d\n", &cCoords.x, &cCoords.y, &cCoords.z, &stop, &cRot.x, &cRot.y, &cRot.z, &nAnims);
	else 
		fscanf_s(f, "%f %f %f %d %f %f %f %f %f %f\n", &A.x, &A.y, &A.z, &stop, &B.x, &B.y, &B.z, &C.x, &C.y, &C.z);

	fscanf_s(f, "%f %f %f %f %f %f\n", &vTP1.x, &vTP1.y, &vTP1.z, &vTP1_rot.x, &vTP1_rot.y, &vTP1_rot.z);
	fscanf_s(f, "%f %f %f %f %f %f\n", &vTP2.x, &vTP2.y, &vTP2.z, &vTP2_rot.x, &vTP2_rot.y, &vTP2_rot.z);
	Entity e = PLAYER::PLAYER_PED_ID();
	
	bool finded;	
	float new_z;
	finded = FALSE;
	int count = 0;
	Ped nearby_ped;
	while (TRUE) {
		//srand(time(NULL) + (81 + count)^2);
		//finded = PATHFIND::GET_RANDOM_VEHICLE_NODE(random_float(-1000.0, 1000.0), random_float(-2000.0, 500.0), 150.0, 1200.f, 1, 1, 1, &vTP1, &tmp);
		finded = PATHFIND::GET_SAFE_COORD_FOR_PED(random_float(-3000.0, 4000.0), random_float(-3000.0, 4000.0), 50.0, TRUE, &vTP1, 0);
		//finded = PATHFIND::GET_SAFE_COORD_FOR_PED(vTP1.x, vTP1.y, 50.0, TRUE, &vTP1, 0);
		//finded = GAMEPLAY::GET_GROUND_Z_FOR_3D_COORD(vTP1.x, vTP1.y, 50.0, &new_z, 0);
		//finded = PED::GET_CLOSEST_PED(vTP1.x, vTP1.y, 50.0, 4000, 0, 0, &nearby_ped, 1, 1, -1);
		//vTP1 = ENTITY::GET_ENTITY_COORDS(nearby_ped, 1);
		//GAMEPLAY::GET_GROUND_Z_FOR_3D_COORD(vTP1.x, vTP1.y, 50.0, &new_z, 0);
		//vTP1.z = new_z;
		break;
		if (finded) {
			break;
		}
	}

	ENTITY::SET_ENTITY_COORDS_NO_OFFSET(e, vTP1.x, vTP1.y, vTP1.z, 0, 0, 1);
	if (DEMO)
		WAIT(3000);
	else
		WAIT(1000);
	this->camera = CAM::GET_RENDERING_CAM();
	
	//if (random_float(0, 1) > 0.5) {
	//	PATHFIND::GET_SAFE_COORD_FOR_PED(vTP1.x, vTP1.y, vTP1.z, TRUE, &safe, 16);
	//	if (safe.x != 0 || safe.y != 0 || safe.z != 0)
	//		if (safe.x > -1000 && safe.x < 1000 && safe.y > -2000 && safe.y < 500) {
	//			vTP1 = safe;
	//			ENTITY::SET_ENTITY_COORDS_NO_OFFSET(e, vTP1.x, vTP1.y, vTP1.z, 0, 0, 1);
	//		}
	//}
	//vTP1.z += 25; //prevent tree/leaf occlusion (raycast fail)
	log_file << "CONFIG" << "," << "POSITION" << "," << vTP1.x << "," << vTP1.y << ","<< vTP1.z << std::endl;

	float new_cz = 0;
	cCoords.x += vTP1.x;	
	cCoords.y += vTP1.y;
	cCoords.z += vTP1.z;
	GAMEPLAY::GET_GROUND_Z_FOR_3D_COORD(cCoords.x, cCoords.y, cCoords.z, &new_cz, 0);
	if (cCoords.z < new_cz) {
		cCoords.z += 0.2;
	}

	DatasetAnnotator::setCameraFixed(cCoords, cRot, 0, fov);
	ENTITY::SET_ENTITY_COORDS_NO_OFFSET(e, cCoords.x, cCoords.y, cCoords.z+10, 0, 0, 1);
	ENTITY::FREEZE_ENTITY_POSITION(e, true);

	Vector3 pos, goFrom, goTo;
	int npeds, ngroup, currentBehaviour, task_time, type, radius, min_lenght, time_between_walks, spawning_radius;
	float speed;
	while (myreadLine(f, &pos, &npeds, &ngroup, &currentBehaviour, &speed, 
		&goFrom, &goTo, &task_time, &type, &radius, &min_lenght, 
		&time_between_walks, &spawning_radius) >= 0) {
		//if (currentBehaviour == 9) {
		//	spawn_peds_climb(pos, goFrom, goTo, npeds, currentBehaviour);
		//	continue;

		//}
		if (currentBehaviour == -1) {
			fclose(f);
			f = NULL;
			pos = vTP1;
			
			spawn_peds_with_animation(pos, goFrom, goTo, npeds, currentBehaviour);
			//spawn_multi_peds_with_animation(pos, npeds);
			break;
		}
		tmpTP = pos;
		pos = vTP1;
		goTo.x += pos.x;
		goTo.y += pos.y;
		goTo.z += pos.z;
		goFrom.x += pos.x;
		goFrom.y += pos.y;
		goFrom.z += pos.z;
		pos.x += tmpTP.x - vTP2.x;
		pos.y += tmpTP.y - vTP2.y;
		//if (currentBehaviour == 8) {
		//	spawn_peds_flow(pos, goFrom, goTo, npeds, ngroup,
		//		currentBehaviour, task_time, type, radius,
		//		min_lenght, time_between_walks, spawning_radius, speed);
		//}
		//else {
		//	spawn_peds(pos, goFrom, goTo, npeds, ngroup,
		//		currentBehaviour, task_time, type, radius,
		//		min_lenght, time_between_walks, spawning_radius, speed);
		//}
			
	}

	if (f!=NULL) fclose(f);

	set_spawned.insert(ped_spawned, ped_spawned + ped_spawned_num);

	//cCoords.x += vTP1.x;
	//cCoords.y += vTP1.y;
	//cCoords.z += vTP1.z;
	//DatasetAnnotator::setCameraFixed(cCoords, cRot, 0, fov);
		
}

void DatasetAnnotator::load_new_multi_Scenario(const char* fname)
{
	//FILE* f = fopen(fname, "r");
	Vector3 pos, cam_pos{}, cRot{};
	int stop, peds_num{};
	int time_h, time_m, time_s;
	// char* locale_type = char*("city_street");
	std::string locale_type;
	std::string weather_type;
	float tmp;
	BOOL camera_moving = FALSE;
	BOOL save_depth = FALSE;
	BOOL save_stencil = FALSE;
	BOOL random_ped = FALSE;
	

	std::ifstream fin(fname);

	if (fin.is_open())
	{
		int count = 1;

		for (std::string line; std::getline(fin, line);) {
			std::istringstream ss(line);
			if (count == 1) {
				ss >> camera_moving >> save_depth >> save_stencil >> random_ped;
			}
			if (count == 2) {
				ss >> weather_type;
			}
			if (count == 3) {
				ss >> time_h >> time_m >> time_s;
			}
			if (count == 4) {
				ss >> locale_type;
			}
			if (count == 5) {
				ss >> pos.x >> pos.y >> pos.z;
			}
			if (count == 6) {
				ss >> cRot.x >> cRot.y >> cRot.z;
			}
			if (count == 7) {
				ss >> cam_pos.x >> cam_pos.y >> cam_pos.z;
			}
			if (count == 8) {
				ss >> peds_num;
			}

			if (count > 9) {
				break;
			}
			count ++;
		}

		fin.close();
	}

	save_stencil = TRUE; // In this batch all save stencil

	log_file << "CONFIG" << "," << "WEATHER" << "," << weather_type << std::endl;
	log_file << "CONFIG" << "," << "DAYTIME" << "," << time_h << "," << time_m << "," << time_s << std::endl;


	// set weather & time
	char* weather = &weather_type[0];
	GAMEPLAY::SET_RANDOM_WEATHER_TYPE();
	GAMEPLAY::CLEAR_WEATHER_TYPE_PERSIST();
	GAMEPLAY::SET_OVERRIDE_WEATHER(weather);
	GAMEPLAY::SET_WEATHER_TYPE_NOW(weather);
	TIME::SET_CLOCK_TIME(time_h, time_m, time_s);

	//pos.z = 0;
	//int refresh_times = 0;
	//BOOL finded = FALSE;
	//while (pos.z == 0) {
	//	pos = ForceGroundZ(pos);
	//	refresh_times += 1;
	//	log_file << "Mesh loading......." << refresh_times << std::endl;
	//	if (refresh_times >= 10) {
	//		break;
	//	}
	//}
	//log_file << "Mesh loaded after refreshing " << refresh_times << " times" << std::endl;

	cRot.x = -cRot.x;
	cam_coords.x = pos.x + cam_pos.x;
	cam_coords.y = pos.y + cam_pos.y;
	cam_coords.z = pos.z - cam_pos.z;
	
	log_file << "Peds number: " << peds_num << std::endl;
	log_file << "Position: " << pos.x << " " << pos.y << " " << pos.z << std::endl;
	log_file << "Camera position: " << cam_coords.x << " " << cam_coords.y << " " << cam_coords.z << std::endl;
	log_file << "Camera rotation: " << cRot.x << " " << cRot.y << " " << cRot.z << std::endl;
	

	Entity e = PLAYER::PLAYER_PED_ID();

	float new_z;

	ENTITY::SET_ENTITY_COORDS_NO_OFFSET(e, pos.x, pos.y, pos.z + 10, 0, 0, 1);
	if (DEMO)
		WAIT(3000);
	else
		WAIT(1000);
	this->camera = CAM::GET_RENDERING_CAM();

	DatasetAnnotator::setCameraFixed(cam_coords, cRot, 0, fov);
	ENTITY::SET_ENTITY_COORDS_NO_OFFSET(e, pos.x, pos.y, pos.z + 10, 0, 0, 1);
	ENTITY::FREEZE_ENTITY_POSITION(e, true);


	spawn_multi_peds_with_animation(pos, peds_num, save_depth, save_stencil);
	//if (f != NULL) fclose(f);

	set_spawned.insert(ped_spawned, ped_spawned + ped_spawned_num);
	log_file << "Scenario created" << std::endl;

	//cCoords.x += vTP1.x;
	//cCoords.y += vTP1.y;
	//cCoords.z += vTP1.z;
	//DatasetAnnotator::setCameraFixed(cCoords, cRot, 0, fov);

}


string getRandLine(const string& fileName, int flag) {
	ifstream inf(fileName.c_str());
	string lineData;
	int i = 1;
	string tmpLine;
	std::srand((unsigned int)time(NULL) + flag);
	while (getline(inf, tmpLine))
	{
		if (rand() % i == 0)
			lineData = tmpLine;
		++i;
	}
	inf.close();
	return lineData;
}
string getHash(const string& String) {
	std::string input = String;
	vector<string> result;
	stringstream s_stream(input);
	int item_count = 0;
	while (s_stream.good()) {
		string substr;
		getline(s_stream, substr, ',');
		result.push_back(substr);
	}
	for (int i = 0; i < result.size(); i++) {
		if (i == 0) {
			return result.at(i);
		}
	}
}

string getType(const string& String) {
	std::string input = String;
	vector<string> result;
	stringstream s_stream(input);
	int item_count = 0;
	while (s_stream.good()) {
		string substr;
		getline(s_stream, substr, ',');
		result.push_back(substr);
	}
	for (int i = 0; i < result.size(); i++) {
		if (i == 1) {
			return result.at(i);
		}
	}
}
void DatasetAnnotator::spawn_peds_with_animation(Vector3 pos, Vector3 goFrom, Vector3 goTo, int npeds, int currentBehaviour) {
	Ped ped[100];
	std::vector<Ped> peds;
	int ped_hash_num;// = -1745486195; // random_float(0,99999999);
	int ped_type_num;// = 5; // random_int(1, 10);
	int pedn = 0;
	// npeds 4

	//tring hash_type_info;
	//hash_type_info = getRandLine("hash_type.txt");

	//ped_hash_num = stod(getHash(hash_type_info));
	//ped_type_num = stod(getType(hash_type_info));

	float step = 2.0;
	float r = GAMEPLAY::GET_DISTANCE_BETWEEN_COORDS(pos.x, pos.y, pos.z, cam_coords.x, cam_coords.y, cam_coords.z, 1);
	float heading = 0; //random_float(-90, 90);
	for (int i = 0; i < npeds; i++) {
		int gen_count = 0;
		while (gen_count < 100) {
			ped[pedn] = PED::CREATE_RANDOM_PED(pos.x, pos.y, pos.z);
			ped_hash_num = (int) ENTITY::GET_ENTITY_MODEL(ped[pedn]);
			ped_type_num = PED::GET_PED_TYPE(ped[pedn]);
			//ped[pedn] = PED::CREATE_PED(ped_type_num, ped_hash_num, pos.x, pos.y, pos.z, heading, FALSE, FALSE);
			if ((ped_hash_num != 0)&&(ped_type_num != 0)) {
				break;
			}
			else {
				pos.x += random_float(-0.5, 0.5);
				pos.y += random_float(-0.5, 0.5);
				pos.z += 0.01;
				gen_count += 1;
			}
		}
		 // add pos off if needed

		ENTITY::SET_ENTITY_HEADING(ped[pedn], heading);
		//std::string ped_str = "ig_abigail";
		//char* ped_name = (char*)ped_str.c_str();
		//Hash h_key = GAMEPLAY::GET_HASH_KEY(ped_name);
		//log_file << "Ped hash:  " << h_key << std::endl;
		
		//ped[pedn] = PED::CREATE_PED(ped_type_num, ped_hash_num, pos.x, pos.y, pos.z, heading, FALSE, FALSE);
		log_file << "Ped hash:  " << ped_hash_num << std::endl;
		log_file << "Ped type:  " << ped_type_num << std::endl;
		peds.push_back(ped[pedn]);
		pedn++;
		WAIT(25);
	}

	WAIT(250);
	npeds = pedn;
	for (int i = 0; i < npeds; i++) {
		ENTITY::SET_ENTITY_HEALTH(ped[i], 0);
	}
	for (int i = 0; i < npeds; i++) {
		AI::CLEAR_PED_TASKS_IMMEDIATELY(ped[i]);
		PED::RESURRECT_PED(ped[i]);
		PED::REVIVE_INJURED_PED(ped[i]);
		ENTITY::SET_ENTITY_COLLISION(ped[i], TRUE, TRUE);
		PED::SET_PED_CAN_RAGDOLL(ped[i], TRUE);
		PED::SET_BLOCKING_OF_NON_TEMPORARY_EVENTS(ped[i], TRUE);
		PED::SET_PED_COMBAT_ATTRIBUTES(ped[i], 1, FALSE);
	}
	WAIT(1000);
	std::string anim_dict, anim_name;
	//anim_dict = AnimDictlist[nAnims];
	//anim_name = AllPedAnims[anim_dict].front().first;
    //int anim_time = (int)AllPedAnims[anim_dict].front().second;
	anim_dict = std::get<0>(animlist[nAnims]);
	anim_name = std::get<1>(animlist[nAnims]);
	int anim_time = (int)std::get<2>(animlist[nAnims]);
	play_peds_animation(peds, anim_dict, anim_name, anim_time);
	// Get_depth_info(); // for depth map
	for (int i = 0; i < npeds; i++) PED::DELETE_PED(&ped[i]);

	WAIT(50);
}

void DatasetAnnotator::play_peds_animation(std::vector<Ped> peds, std::string anim_dict, std::string anim_name, int anim_time) {
	char* anim_dict_str = (char*)anim_dict.c_str();
	char* anim_name_str = (char*)anim_name.c_str();

	int load_flag = 1;
	DWORD ticksStart = GetTickCount();
	if (!STREAMING::HAS_ANIM_DICT_LOADED(anim_dict_str)) {
		STREAMING::REQUEST_ANIM_DICT(anim_dict_str);
		while (!STREAMING::HAS_ANIM_DICT_LOADED(anim_dict_str))
		{
			WAIT(0);
			if (GetTickCount() > ticksStart + 1000) {
				//set_status_text("Could not load animation with code " + anim_dict);
				log_file << "anim fail: " << anim_dict << std::endl;
				load_flag = 0;
				break;
			}
		}
	}

	if (!load_flag) return;

	std::map<Ped, Vector3> ped_init_pos;
	for (auto ped : peds) ped_init_pos[ped] = ENTITY::GET_ENTITY_COORDS(ped, TRUE);

	int anim_time_acc = 0;
	//for (auto anim_item : AllPedAnims[anim_dict]) {
	//	anim_name = anim_item.first;
	//	anim_name_str = (char*)anim_name.c_str();
	//	int anim_time = (int)anim_item.second;
	//	anim_time_acc += anim_time;
	//	
	//	ticksStart = GetTickCount();
	//	for (auto ped : peds) AI::TASK_PLAY_ANIM(ped, anim_dict_str, anim_name_str, 8.0, -8.0, anim_time, 1, 8.0, 0, 0, 0);

	//	Ped ped = peds.back();

	//	while (!ENTITY::IS_ENTITY_PLAYING_ANIM(ped, anim_dict_str, anim_name_str, 1)) {
	//		WAIT(0);
	//		//log_file << "wait anim," << anim_dict << "," << anim_name << std::endl;
	//		if (GetTickCount() > ticksStart + 5000) break;
	//	}

	//	log_file << "CONFIG" << "," << "ANIM" << "," << anim_dict << "," << anim_name << "," << anim_time << std::endl;
	//	GAMEPLAY::SET_TIME_SCALE(1.0f / (float)TIME_FACTOR);
	//	PED::SET_PED_DENSITY_MULTIPLIER_THIS_FRAME(0);
	//	VEHICLE::SET_VEHICLE_DENSITY_MULTIPLIER_THIS_FRAME(0);

	//	while (ENTITY::IS_ENTITY_PLAYING_ANIM(ped, anim_dict_str, anim_name_str, 1)) {
	//		if (nsample == 0)
	//			log_file << "START TIME: " << GAMEPLAY::GET_GAME_TIMER() << std::endl;
	//		long delay = GAMEPLAY::GET_GAME_TIMER() - lastRecGameTime;
	//		if (delay >= (long)(1000 / RECORD_FPS)) {
	//			log_file << "recording in delay = " << delay << std::endl;
	//			lastRecGameTime = GAMEPLAY::GET_GAME_TIMER();
	//		}
	//		else {
	//			log_file << "delay = " << delay << std::endl;
	//			WAIT(0);
	//			continue;
	//		}
	//		display_clear();
	//		WAIT(0);
	//		GAMEPLAY::SET_GAME_PAUSED(true);
	//		WAIT(50);
	//		save_frame();
	//		for (auto _ped: peds) save_labels(_ped, anim_dict, anim_name);
	//		nsample++;
	//		// WAIT(1000);
	//		GAMEPLAY::SET_GAME_PAUSED(false);
	//		WAIT(0);
	//		if (nsample > (RECORD_FPS * anim_time_acc / 1000.0)) break; // anim may be blocked by env object
	//	}
	//	for (auto ped : peds) ENTITY::SET_ENTITY_COORDS_NO_OFFSET(ped,
	//		ped_init_pos[ped].x, ped_init_pos[ped].y, ped_init_pos[ped].z, 0, 0, 1);
	//}

	anim_time_acc += anim_time;

	ticksStart = GetTickCount();
	for (auto ped : peds) AI::TASK_PLAY_ANIM(ped, anim_dict_str, anim_name_str, 8.0, -8.0, anim_time, 1, 8.0, 0, 0, 0);

	Ped ped = peds.back();

	while (!ENTITY::IS_ENTITY_PLAYING_ANIM(ped, anim_dict_str, anim_name_str, 1)) {
		WAIT(0);
		//log_file << "wait anim," << anim_dict << "," << anim_name << std::endl;
		if (GetTickCount() > ticksStart + 10000) break;
	}

	log_file << "CONFIG" << "," << "ANIM" << "," << anim_dict << "," << anim_name << "," << anim_time << std::endl;
	GAMEPLAY::SET_TIME_SCALE(1.0f / (float)TIME_FACTOR);
	PED::SET_PED_DENSITY_MULTIPLIER_THIS_FRAME(0);
	VEHICLE::SET_VEHICLE_DENSITY_MULTIPLIER_THIS_FRAME(0);

	while (ENTITY::IS_ENTITY_PLAYING_ANIM(ped, anim_dict_str, anim_name_str, 1) && nsample <= 30) {
		if (nsample == 0)
			log_file << "START TIME: " << GAMEPLAY::GET_GAME_TIMER() << std::endl;
		long delay = GAMEPLAY::GET_GAME_TIMER() - lastRecGameTime;
		if (delay >= (long)(1000 / RECORD_FPS)) {
			log_file << "recording in delay = " << delay << std::endl;
			lastRecGameTime = GAMEPLAY::GET_GAME_TIMER();
		}
		else {
			log_file << "delay = " << delay << std::endl;
			WAIT(0);
			continue;
		}
		display_clear();
		WAIT(0);
		GAMEPLAY::SET_GAME_PAUSED(true);
		// WAIT(50);
		save_frame();
		for (auto _ped : peds) save_labels(_ped, anim_dict, anim_name);
		nsample++;
		// WAIT(1000);
		GAMEPLAY::SET_GAME_PAUSED(false);
		WAIT(0);
		if (nsample > (RECORD_FPS * anim_time_acc / 1000.0)) break; // anim may be blocked by env object
	}

	//log_file << "done anim," << anim_dict << "," << anim_name << std::endl;
	STREAMING::REMOVE_ANIM_DICT(anim_dict_str);
	summary_file.open(output_path + "\\summary.txt");
	summary_file << "Done\n";
}


void DatasetAnnotator::spawn_multi_peds_with_animation(Vector3 pos, int peds_num, BOOL save_depth, BOOL save_stencil) {
	Ped ped[100];
	std::vector<Ped> peds;
	Vector3 temp_pos;
	int ped_hash_num;// = -1745486195; // random_float(0,99999999);
	int ped_type_num;// = 5; // random_int(1, 10);
	int pedn = 0;
	// npeds 4
	float step = 2.0;
	float r = GAMEPLAY::GET_DISTANCE_BETWEEN_COORDS(pos.x, pos.y, pos.z, cam_coords.x, cam_coords.y, cam_coords.z, 1);
	float heading = 0;
	// list<float> flo;
	float flox, floy, floz;
	int ped_pos_line = 0;



	for (int i = 0; i < peds_num; i++) {
		int gen_count = 0;
		FILE* scene = fopen(file_scenario, "r");
		std::ifstream fin(scene);

		if (fin.is_open())
		{
			int count = 1;
			int temp = 0;
			for (std::string line; std::getline(fin, line);) {
				std::istringstream ss(line);
				if (temp >= peds_num) {
					break;
				}
				if (count == 2 * i + 10) {

					ss >> flox >> floy >> floz >> heading;
					log_file << "flox1 = " << flox << "  floy1 = " << floy << std::endl;

					temp++;
				}
				count++;
			}
			fin.close();
		}

		
	
		while (gen_count < 100) {

			srand(time(NULL) + (101 * i + 73) ^ 2 + (83 - i) ^ 2 + gen_count);

			//srand(time(0) + i * 100 + gen_count + 1000);
			ped_pos_line = 2 * i + 9;
			

			ped[pedn] = PED::CREATE_RANDOM_PED(pos.x + flox, pos.y + floy, pos.z + floz);
			ped_hash_num = (int)ENTITY::GET_ENTITY_MODEL(ped[pedn]);
			ped_type_num = PED::GET_PED_TYPE(ped[pedn]);

			//string hash_type_info;
			//hash_type_info = getRandLine("hash_type.txt", i);
			//ped_hash_num = stod(getHash(hash_type_info));
			//ped_type_num = stod(getType(hash_type_info));

			//ped_hash_num = 1745486195;//1224306523; // random_float(0,99999999);
			//ped_type_num = 5; // random_int(1, 10);

			//ped[pedn] = PED::CREATE_PED(26, ped_hash_num, pos.x + flox * ((-1) ^ (i)), pos.y + floy * ((-1) ^ (i + 1)), pos.z, heading, FALSE, FALSE);

			log_file << "flox = " << flox << "  floy = " << floy << std::endl;

			if ((ped_hash_num != 0) && (ped_type_num != 0)) {
				log_file << "Ped hash:  " << ped_hash_num << std::endl;
				log_file << "Ped type:  " << ped_type_num << std::endl;
				break;
			}
			else {
				//temp_pos.x += 0.1*flox;
				//temp_pos.y += 0.1*floy;
				//temp_pos.z += 0.01;
				gen_count += 1;
			}
		}
		//call random float, for the other random_float to be really random
		//log_file << "Dump: " << dump1 << std::endl;
		std::srand(time(NULL) + (10 * i + 73) ^ 2 + (83 - i) ^ 2);
		float dump1 = random_float(-1, 1);
		
		if (flox >= 0.0 && floy >= 0.0) {
			heading = random_float(-180, -90);
		}
		else if(flox <= 0.0 && floy >= 0.0) {
			heading = random_float(-90, 0);
		}
		else if (flox <= 0.0 && floy <= 0.0) {	
			heading = random_float(0, 90);
		}
		else {
			heading = random_float(-180, -90);
		}
		//heading = heading * (-1);
		//heading = -180;
		log_file << "heading:  " << heading << std::endl;
		//heading = random_float(-180, 180);
		// add pos off if needed
		ENTITY::SET_ENTITY_HEADING(ped[pedn], heading);
		//std::string ped_str = "ig_abigail";
		//char* ped_name = (char*)ped_str.c_str();
		//Hash h_key = GAMEPLAY::GET_HASH_KEY(ped_name);
		//log_file << "Ped hash:  " << h_key << std::endl;

		peds.push_back(ped[pedn]);
		pedn++;
		WAIT(25);
	}

	WAIT(250);
	peds_num = pedn;
	for (int i = 0; i < peds_num; i++) {
		ENTITY::SET_ENTITY_HEALTH(ped[i], 0);
	}
	for (int i = 0; i < peds_num; i++) {
		AI::CLEAR_PED_TASKS_IMMEDIATELY(ped[i]);
		PED::RESURRECT_PED(ped[i]);
		PED::REVIVE_INJURED_PED(ped[i]);
		ENTITY::SET_ENTITY_COLLISION(ped[i], TRUE, TRUE);
		PED::SET_PED_CAN_RAGDOLL(ped[i], TRUE);
		//PED::SET_BLOCKING_OF_NON_TEMPORARY_EVENTS(ped[i], TRUE);
		PED::SET_PED_COMBAT_ATTRIBUTES(ped[i], 1, TRUE);
	}
	WAIT(3000);

	//anim_dict = AnimDictlist[nAnims];
	//anim_name = AllPedAnims[anim_dict].front().first;
	//int anim_time = (int)AllPedAnims[anim_dict].front().second;


	play_multi_peds_animation(peds, peds_num, save_depth, save_stencil);
	//play_multi_peds_multi_cam_animation(peds, peds_num, save_depth, save_stencil, pos);

	for (int i = 0; i < peds_num; i++) PED::DELETE_PED(&ped[i]);

	WAIT(50);
}


void DatasetAnnotator::play_multi_peds_animation(std::vector<Ped> peds, int peds_num, BOOL save_depth, BOOL save_stencil)//std::string anim_dict, std::string anim_name, int anim_time) 
{
	list<string> multi_anim_dict;
	list<string> multi_anim_name;
	list<int> multi_anim_time;
	string anim_dict, anim_name;
	int anim_time;
	char* anim_dict_str;
	char* anim_name_str;
	int anim_time_acc = 0;
	FILE *scene = fopen(file_scenario, "r");

	std::ifstream fin(scene);

	if (fin.is_open())
	{

		scene_anim.clear();

		for (std::string line; std::getline(fin, line);) {
			std::istringstream ss(line);
			std::string ad, an;
			float at;
			ss >> ad >> an >> at;
			scene_anim.push_back(std::make_tuple(ad, an, at));
		}
		for (auto item : AllPedAnims) AnimDictlist.push_back(item.first);
		fin.close();
	}
	float min_anim_time = 10000.0;
	for (int i = 0; i < peds_num; i++) {

		int action_line = 2 * i + 8;
		multi_anim_dict.push_back(std::get<0>(scene_anim[action_line]));
		multi_anim_name.push_back(std::get<1>(scene_anim[action_line]));
		multi_anim_time.push_back((int)std::get<2>(scene_anim[action_line]));

		anim_dict = multi_anim_dict.back();
		anim_name = multi_anim_name.back();
		anim_time = multi_anim_time.back();

		anim_dict_str = (char*)anim_dict.c_str();
		anim_name_str = (char*)anim_name.c_str();

		int load_flag = 1;
		DWORD ticksStart = GetTickCount64();
		if (!STREAMING::HAS_ANIM_DICT_LOADED(anim_dict_str)) {
			STREAMING::REQUEST_ANIM_DICT(anim_dict_str);
			while (!STREAMING::HAS_ANIM_DICT_LOADED(anim_dict_str))
			{
				WAIT(0);
				if (GetTickCount64() > double(ticksStart) + 1000) {
					//set_status_text("Could not load animation with code " + anim_dict);
					log_file << "anim fail: no anim loaded or (action)" << anim_dict << "not loaded" <<std::endl;
					load_flag = 0;
					break;
				}
			}
		}
		if (!load_flag) return;

		anim_time_acc += anim_time;
		ticksStart = GetTickCount64();
		std::map<Ped, Vector3> ped_init_pos;
		if (min_anim_time < anim_time) {
			min_anim_time = anim_time;
		}
		AI::TASK_PLAY_ANIM(peds[i], anim_dict_str,
			anim_name_str, 8.0, -8.0,
			anim_time, 1, 8.0, 0, 0, 0);
		
		//Ped ped = peds.front();

		log_file << "CONFIG" << "," << "ANIM" << "," << anim_dict << "," << anim_name << "," << anim_time * 10 << std::endl;
		GAMEPLAY::SET_TIME_SCALE(1.0f / (float)TIME_FACTOR);
	}
	// PED::SET_PED_DENSITY_MULTIPLIER_THIS_FRAME(random_int(0,5));
	PED::SET_PED_DENSITY_MULTIPLIER_THIS_FRAME(2.0);
	VEHICLE::SET_VEHICLE_DENSITY_MULTIPLIER_THIS_FRAME(1.0);
	

	while (nsample <= (min_anim_time / 1000)){//ENTITY::IS_ENTITY_PLAYING_ANIM(ped, anim_dict_str, anim_name_str, 1)) {
		if (nsample == 0)
			log_file << "START TIME: " << GAMEPLAY::GET_GAME_TIMER() << std::endl;
		long delay = GAMEPLAY::GET_GAME_TIMER() - lastRecGameTime;
		if (delay >= (long)(1000 / RECORD_FPS)) {
			log_file << "recording in delay = " << delay << std::endl;
			lastRecGameTime = GAMEPLAY::GET_GAME_TIMER();
		}
		else {
			log_file << "delay = " << delay << std::endl;
			WAIT(0);
			continue;
		}
		display_clear();
		WAIT(0);
		GAMEPLAY::SET_GAME_PAUSED(true);
		WAIT(50);

		save_frame();

		char sample_num[9];
		snprintf(sample_num, 9, "%08d", nsample);
		capture_depth(output_path + std::string("\\raws"), std::string(sample_num), save_depth, save_stencil);


		WAIT(50);

		for (auto _ped : peds) {
			anim_dict = multi_anim_dict.front();
			anim_name = multi_anim_name.front();

			save_labels(_ped, anim_dict, anim_name);

			multi_anim_dict.pop_front();
			multi_anim_name.pop_front();
			multi_anim_dict.push_back(anim_dict);
			multi_anim_name.push_back(anim_name);
		}
		nsample++;
		// WAIT(1000);
		GAMEPLAY::SET_GAME_PAUSED(false);
		WAIT(0);
		if (nsample > (RECORD_FPS * double(anim_time_acc) / 1000.0)) break; // anim may be blocked by env object
	}

	//log_file << "done anim," << anim_dict << "," << anim_name << std::endl;
	//STREAMING::REMOVE_ANIM_DICT(anim_dict_str);
	summary_file.open(output_path + "\\summary.txt");
	summary_file << "Done\n";
}



void DatasetAnnotator::play_multi_peds_multi_cam_animation(std::vector<Ped> peds, int peds_num, BOOL save_depth, BOOL save_stencil, Vector3 pos)
{
	list<string> multi_anim_dict;
	list<string> multi_anim_name;
	list<int> multi_anim_time;
	string anim_dict, anim_name;
	int anim_time;
	char* anim_dict_str;
	char* anim_name_str;
	int anim_time_acc = 0;
	FILE* scene = fopen(file_scenario, "r");

	std::ifstream fin(scene);

	if (fin.is_open())
	{

		scene_anim.clear();

		for (std::string line; std::getline(fin, line);) {
			std::istringstream ss(line);
			std::string ad, an;
			float at;
			ss >> ad >> an >> at;
			scene_anim.push_back(std::make_tuple(ad, an, at));
		}
		for (auto item : AllPedAnims) AnimDictlist.push_back(item.first);
		fin.close();
	}

	for (int i = 0; i < peds_num; i++) {

		int action_line = 2 * i + 8;
		multi_anim_dict.push_back(std::get<0>(scene_anim[action_line]));
		multi_anim_name.push_back(std::get<1>(scene_anim[action_line]));
		multi_anim_time.push_back((int)std::get<2>(scene_anim[action_line]));

		anim_dict = multi_anim_dict.back();
		anim_name = multi_anim_name.back();
		anim_time = multi_anim_time.back();

		anim_dict_str = (char*)anim_dict.c_str();
		anim_name_str = (char*)anim_name.c_str();

		int load_flag = 1;
		DWORD ticksStart = GetTickCount64();
		if (!STREAMING::HAS_ANIM_DICT_LOADED(anim_dict_str)) {
			STREAMING::REQUEST_ANIM_DICT(anim_dict_str);
			while (!STREAMING::HAS_ANIM_DICT_LOADED(anim_dict_str))
			{
				WAIT(0);
				if (GetTickCount64() > double(ticksStart) + 1000) {
					//set_status_text("Could not load animation with code " + anim_dict);
					log_file << "anim fail: no anim loaded or (action)" << anim_dict << "not loaded" << std::endl;
					load_flag = 0;
					break;
				}
			}
		}
		if (!load_flag) return;

		anim_time_acc += anim_time;
		ticksStart = GetTickCount64();
		std::map<Ped, Vector3> ped_init_pos;

		AI::TASK_PLAY_ANIM(peds[i], anim_dict_str,
			anim_name_str, 8.0, -8.0,
			10000, 1, 8.0, 0, 0, 0);

		//Ped ped = peds.front();

		log_file << "CONFIG" << "," << "ANIM" << "," << anim_dict << "," << anim_name << "," << anim_time * 10 << std::endl;
		GAMEPLAY::SET_TIME_SCALE(1.0f / (float)TIME_FACTOR);
	}
	// PED::SET_PED_DENSITY_MULTIPLIER_THIS_FRAME(random_int(0,5));
	PED::SET_PED_DENSITY_MULTIPLIER_THIS_FRAME(2.0);
	VEHICLE::SET_VEHICLE_DENSITY_MULTIPLIER_THIS_FRAME(1.0);


	while (nsample <= 50) {//ENTITY::IS_ENTITY_PLAYING_ANIM(ped, anim_dict_str, anim_name_str, 1)) {
		if (nsample == 0)
			log_file << "START TIME: " << GAMEPLAY::GET_GAME_TIMER() << std::endl;
		long delay = GAMEPLAY::GET_GAME_TIMER() - lastRecGameTime;
		if (delay >= (long)(1000 / RECORD_FPS)) {
			log_file << "recording in delay = " << delay << std::endl;
			lastRecGameTime = GAMEPLAY::GET_GAME_TIMER();
		}
		else {
			log_file << "delay = " << delay << std::endl;
			WAIT(0);
			continue;
		}
		display_clear();
		WAIT(0);

		Vector3 temp_cam_coord, temp_cam_rot;
		int camera_num = 2;

		for (int cam_c = 1; cam_c <= camera_num; cam_c++) {
			
			// genRandomFixedCamera(pos);
			if (cam_c == 1) {
				temp_cam_rot.x = -1.276338098355437;
				temp_cam_rot.y = 0.0;
				temp_cam_rot.z = 21.91148493589303;
				temp_cam_coord.x = pos.x + 2.9846493836027257; 
				temp_cam_coord.y = pos.y - 7.420250289377103;
				temp_cam_coord.z = pos.z + 0.17819567884459658;
			}
			if (cam_c == 2){
				temp_cam_rot.x = -24.147510495021542;
				temp_cam_rot.y = 0.0;
				temp_cam_rot.z = -26.632972855625134;
				temp_cam_coord.x = pos.x - 3.27238021410871;
				temp_cam_coord.y = pos.y - 6.52540996850987;
				temp_cam_coord.z = pos.z + 3.272698042469218;

			}
			DatasetAnnotator::setCameraFixed(temp_cam_coord, temp_cam_rot, 0, fov);
			this->camera = CAM::GET_RENDERING_CAM();

			GAMEPLAY::SET_GAME_PAUSED(true);
			WAIT(200);
			save_frame_multi_cam(cam_c);
			char sample_num[12];
			snprintf(sample_num, 12, "%08dc%02d", nsample, cam_c);
			capture_depth(output_path + std::string("\\raws"), std::string(sample_num), save_depth, save_stencil);
			WAIT(200);

			for (auto _ped : peds) {
				anim_dict = multi_anim_dict.front();
				anim_name = multi_anim_name.front();

				save_labels(_ped, anim_dict, anim_name);

				multi_anim_dict.pop_front();
				multi_anim_name.pop_front();
				multi_anim_dict.push_back(anim_dict);
				multi_anim_name.push_back(anim_name);
			}

			GAMEPLAY::SET_GAME_PAUSED(false);
			WAIT(0);
		}



		nsample++;
		// WAIT(1000);

		WAIT(0);
		if (nsample > (RECORD_FPS * double(anim_time_acc) / 1000.0)) break; // anim may be blocked by env object
	}

	//log_file << "done anim," << anim_dict << "," << anim_name << std::endl;
	//STREAMING::REMOVE_ANIM_DICT(anim_dict_str);
	summary_file.open(output_path + "\\summary.txt");
	summary_file << "Done\n";
}



void DatasetAnnotator::save_labels(Ped ped, std::string anim_dict, std::string anim_name) {
	// for each pedestrians scan all the joint_ID we choose on the subset
	for (int n = -1; n < number_of_joints; n++) {
		Vector3 joint_coords;
		if (n == -1) {
			Vector3 head_coords = ENTITY::GET_WORLD_POSITION_OF_ENTITY_BONE(ped, PED::GET_PED_BONE_INDEX(ped, joint_int_codes[0]));
			Vector3 neck_coords = ENTITY::GET_WORLD_POSITION_OF_ENTITY_BONE(ped, PED::GET_PED_BONE_INDEX(ped, joint_int_codes[1]));
			float head_neck_norm = GAMEPLAY::GET_DISTANCE_BETWEEN_COORDS(neck_coords.x, neck_coords.y, neck_coords.z, head_coords.x, head_coords.y, head_coords.z, 1);
			float dx = (head_coords.x - neck_coords.x) / head_neck_norm;
			float dy = (head_coords.y - neck_coords.y) / head_neck_norm;
			float dz = (head_coords.z - neck_coords.z) / head_neck_norm;

			joint_coords.x = head_coords.x + head_neck_norm * dx;
			joint_coords.y = head_coords.y + head_neck_norm * dy;
			joint_coords.z = head_coords.z + head_neck_norm * dz;
		}
		else 
			joint_coords = ENTITY::GET_WORLD_POSITION_OF_ENTITY_BONE(ped, PED::GET_PED_BONE_INDEX(ped, joint_int_codes[n]));
		
		// finding the versor (dx, dy, dz) pointing from the joint to the cam
		float joint2cam_distance = GAMEPLAY::GET_DISTANCE_BETWEEN_COORDS(
			joint_coords.x, joint_coords.y, joint_coords.z, 
			cam_coords.x, cam_coords.y, cam_coords.z, 1
		);
		float dx = (cam_coords.x - joint_coords.x) / joint2cam_distance;
		float dy = (cam_coords.y - joint_coords.y) / joint2cam_distance;
		float dz = (cam_coords.z - joint_coords.z) / joint2cam_distance;
		
		// ray #1: from joint to cam_coords (ignoring the pedestrian to whom the joint belongs and intersecting only pedestrian (8))
		// ==> useful for detecting occlusions of pedestrian
		Vector3 end_coords1, surface_norm1;
		BOOL occlusion_ped;
		Entity entityHit1 = 0;

		int ray_ped_occlusion = WORLDPROBE::_CAST_RAY_POINT_TO_POINT(
			joint_coords.x, joint_coords.y, joint_coords.z,
			cam_coords.x, cam_coords.y, cam_coords.z, 8, ped, 7 );
		WORLDPROBE::_GET_RAYCAST_RESULT(ray_ped_occlusion, &occlusion_ped, &end_coords1, &surface_norm1, &entityHit1);

		// ray #2: from joint to camera (without ignoring the pedestrian to whom the joint belongs and intersecting only pedestrian (8))
		// ==> useful for detecting self-occlusions
		Vector3 endCoords2, surfaceNormal2;
		BOOL occlusion_self;
		Entity entityHit2 = 0;
		int ray_joint2cam = WORLDPROBE::_CAST_RAY_POINT_TO_POINT(
			joint_coords.x + 0.1f*dx, joint_coords.y + 0.1f*dy, joint_coords.z + 0.1f*dz,
			cam_coords.x, cam_coords.y, cam_coords.z, 
			8, 0, 7
		);
		WORLDPROBE::_GET_RAYCAST_RESULT(ray_joint2cam, &occlusion_self, &endCoords2, &surfaceNormal2, &entityHit2);

		// ray #3: from camera to joint (ignoring the pedestrian to whom the joint belongs and intersecting everything but peds (4 and 8))
		// ==> useful for detecting occlusions with objects
		Vector3 endCoords3, surfaceNormal3;
		BOOL occlusion_object;
		Entity entityHit3 = 0;
		int ray_joint2cam_obj = WORLDPROBE::_CAST_RAY_POINT_TO_POINT(
			cam_coords.x, cam_coords.y, cam_coords.z,
			joint_coords.x, joint_coords.y, joint_coords.z, (~0 ^ (8|4)), ped, 7 );
		//int ray_joint2cam_obj = WORLDPROBE::_CAST_3D_RAY_POINT_TO_POINT(
		//	cam_coords.x, cam_coords.y, cam_coords.z,
		//	joint_coords.x, joint_coords.y, joint_coords.z, 0.1, (~0 ^ (8|4)), ped, 7 );
		WORLDPROBE::_GET_RAYCAST_RESULT(ray_joint2cam_obj, &occlusion_object, &endCoords3, &surfaceNormal3, &entityHit3);
		
		BOOL occluded = occlusion_ped || occlusion_object;

		float x, y;
		get_2D_from_3D(joint_coords, &x, &y);
		x = x * SCREEN_WIDTH;
		y = y * SCREEN_HEIGHT;
		std::string ped_action = anim_dict +"#"+anim_name;
		peds_file << nsample;					  // frame number
		peds_file << "," << ped;			  // pedestrian ID
		peds_file << "," << ped_action; // pedestrain action
		peds_file << "," << n+1;				  // joint type
		peds_file << "," << x;				  // camera 2D x [px]
		peds_file << "," << y;	              // camera 2D y [px]
		peds_file << "," << joint_coords.x;	  // joint 3D x [m]
		peds_file << "," << joint_coords.y;	  // joint 3D y [m]
		peds_file << "," << joint_coords.z;	  // joint 3D z [m]
		peds_file << "," << occluded;			  // is joint occluded?
		peds_file << "," << occlusion_self;	  // is joint self-occluded?
		peds_file << "," << cam_coords.x;		  // camera 3D x [m]
		peds_file << "," << cam_coords.y;	      // camera 3D y [m]
		peds_file << "," << cam_coords.z;	      // camera 3D z [m]
		peds_file << "," << cam_rot.x;		  // camera 3D rotation x [degrees]
		peds_file << "," << cam_rot.y;	      // camera 3D rotation y [degrees]
		peds_file << "," << cam_rot.z;	      // camera 3D rotation z [degrees]
		peds_file << "," << fov;
		peds_file << "," << PED::IS_PED_MALE(ped);				  

		peds_file << "\n";
	}
}


/// <summary>
  ///     Forces Ground Z position even when the location doesn't have collisions loaded
  /// </summary>
Vector3 DatasetAnnotator::ForceGroundZ(Vector3 v)
{
	float zcoord = 0.0f;

	float firstCheck[] = { 0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000 };
	float secondCheck[] = { 1000, 900, 800, 700, 600, 500, 400, 300, 200, 100, 0, -100, -200, -300, -400, -500 };
	float thirdCheck[] = { -500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000 };

	GAMEPLAY::GET_GROUND_Z_FOR_3D_COORD(v.x, v.y, 1000.f, &zcoord, 0);

	if (zcoord == 0)
	{
		for (int i = 0; i < 11; i++)
		{
			STREAMING::REQUEST_COLLISION_AT_COORD(v.x, v.y, firstCheck[i]);
			WAIT(0);
		}

		GAMEPLAY::GET_GROUND_Z_FOR_3D_COORD(v.x, v.y, 1000.f, &zcoord, 0);
	}

	if (zcoord == 0)
	{
		for (int i = 0; i < 16; i++)
		{
			STREAMING::REQUEST_COLLISION_AT_COORD(v.x, v.y, secondCheck[i]);
			WAIT(0);
		}

		GAMEPLAY::GET_GROUND_Z_FOR_3D_COORD(v.x, v.y, 1000.f, &zcoord, 0);
	}


	if (zcoord == 0)
	{
		for (int i = 0; i < 16; i++)
		{
			STREAMING::REQUEST_COLLISION_AT_COORD(v.x, v.y, thirdCheck[i]);
			WAIT(0);
		}

		GAMEPLAY::GET_GROUND_Z_FOR_3D_COORD(v.x, v.y, 1000.f, &zcoord, 0);
	}
	v.z = zcoord;
	return v;
}