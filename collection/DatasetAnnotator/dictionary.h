#ifndef DICTIONARY_H
#define DICTIONARY_H

#include <string>
#include <vector>
#include <map>


// ref: https://wiki.gtanet.work/index.php?title=Bones
static const std::map< std::string, int > m = {
	// original 21 joints defined in JTA-dataset
	// note "head_top" (index 0) generated from "head_center"("SKEL_Head") and "neck"("SKEL_Neck_1")
	// so JTA-dataset has 22 joints in total
	{"SKEL_Head",			31086}, // 01
	{"SKEL_Neck_1",			39317}, // 02
	{"SKEL_R_Clavicle",		10706}, // 03
	{"SKEL_R_UpperArm",		40269}, // 04
	{"SKEL_R_Forearm",		28252}, // 05
	{"SKEL_R_Hand",			57005}, // 06
	{"SKEL_L_Clavicle",		64729}, // 07
	{"SKEL_L_UpperArm",		45509}, // 08
	{"SKEL_L_Forearm",		61163}, // 09
	{"SKEL_L_Hand",			18905}, // 10
	{"SKEL_Spine3",			24818}, // 11
	{"SKEL_Spine2",			24817}, // 12
	{"SKEL_Spine1",			24816}, // 13
	{"SKEL_Spine0",			23553}, // 14
	{"SKEL_Spine_Root",		57597}, // 15
	{"SKEL_R_Thigh",		51826}, // 16
	{"SKEL_R_Calf",			36864}, // 17
	{"SKEL_R_Foot",			52301}, // 18
	{"SKEL_L_Thigh",		58271}, // 19
	{"SKEL_L_Calf",			63931}, // 20
	{"SKEL_L_Foot",			14201}, // 21

	// additional joints available
	{"SKEL_ROOT",			0	}, // 22
	{"FB_R_Brow_Out_000",	1356}, // 23
	{"SKEL_L_Toe0",			2108}, // 24
	{"MH_R_Elbow",			2992}, // 25
	{"SKEL_L_Finger01",		4089}, // 26
	{"SKEL_L_Finger02",		4090}, // 27
	{"SKEL_L_Finger31",		4137}, // 28
	{"SKEL_L_Finger32",		4138}, // 29
	{"SKEL_L_Finger41",		4153}, // 30
	{"SKEL_L_Finger42",		4154}, // 31
	{"SKEL_L_Finger11",		4169}, // 32
	{"SKEL_L_Finger12",		4170}, // 33
	{"SKEL_L_Finger21",		4185}, // 34
	{"SKEL_L_Finger22",		4186}, // 35
	{"RB_L_ArmRoll",		5232}, // 36
	{"IK_R_Hand",			6286}, // 37
	{"RB_R_ThighRoll",		6442}, // 38
	{"FB_R_Lip_Corner_000",	11174}, // 39
	{"SKEL_Pelvis",			11816}, // 40
	{"IK_Head",				12844}, // 41
	{"MH_R_Knee",			16335}, // 42
	{"FB_LowerLipRoot_000",	17188}, // 43
	{"FB_R_Lip_Top_000",	17719}, // 44
	{"FB_R_CheekBone_000",	19336}, // 45
	{"FB_UpperLipRoot_000",	20178}, // 46
	{"FB_L_Lip_Top_000",	20279}, // 47
	{"FB_LowerLip_000",		20623}, // 48
	{"SKEL_R_Toe0",			20781}, // 49
	{"FB_L_CheekBone_000",	21550}, // 50
	{"MH_L_Elbow",			22711}, // 51
	{"RB_L_ThighRoll",		23639}, // 52
	{"PH_R_Foot",			24806}, // 53
	{"FB_L_Eye_000",		25260}, // 54
	{"SKEL_L_Finger00",		26610}, // 55
	{"SKEL_L_Finger10",		26611}, // 56
	{"SKEL_L_Finger20",		26612}, // 57
	{"SKEL_L_Finger30",		26613}, // 58
	{"SKEL_L_Finger40",		26614}, // 59
	{"FB_R_Eye_000",		27474}, // 60
	{"PH_R_Hand",			28422}, // 61
	{"FB_L_Lip_Corner_000",	29868}, // 62
	{"IK_R_Foot",			35502}, // 63
	{"RB_Neck_1",			35731}, // 64
	{"IK_L_Hand",			36029}, // 65
	{"RB_R_ArmRoll",		37119}, // 66
	{"FB_Brow_Centre_000",	37193}, // 67
	{"FB_R_Lid_Upper_000",	43536}, // 68
	{"RB_R_ForeArmRoll",	43810}, // 69
	{"FB_L_Lid_Upper_000",	45750}, // 70
	{"MH_L_Knee",			46078}, // 71
	{"FB_Jaw_000",			46240}, // 72
	{"FB_L_Lip_Bot_000",	47419}, // 73
	{"FB_Tongue_000",		47495}, // 74
	{"FB_R_Lip_Bot_000",	49979}, // 75
	{"IK_Root",				56604}, // 76
	{"PH_L_Foot",			57717}, // 77
	{"FB_L_Brow_Out_000",	58331}, // 78
	{"SKEL_R_Finger00",		58866}, // 79
	{"SKEL_R_Finger10",		58867}, // 80
	{"SKEL_R_Finger20",		58868}, // 81
	{"SKEL_R_Finger30",		58869}, // 82
	{"SKEL_R_Finger40",		58870}, // 83
	{"PH_L_Hand",			60309}, // 84
	{"RB_L_ForeArmRoll",	61007}, // 85
	{"FB_UpperLip_000",		61839}, // 86
	{"SKEL_R_Finger01",		64016}, // 87
	{"SKEL_R_Finger02",		64017}, // 88
	{"SKEL_R_Finger31",		64064}, // 89
	{"SKEL_R_Finger32",		64065}, // 90
	{"SKEL_R_Finger41",		64080}, // 91
	{"SKEL_R_Finger42",		64081}, // 92
	{"SKEL_R_Finger11",		64096}, // 93
	{"SKEL_R_Finger12",		64097}, // 94
	{"SKEL_R_Finger21",		64112}, // 95
	{"SKEL_R_Finger22",		64113}, // 96
	{"FACIAL_facialRoot",	65068}, // 97
	{"IK_L_Foot",			65245}, // 98
};


static const std::vector< std::string > jointNames = {
	"SKEL_Head",			// 01
	"SKEL_Neck_1",			// 02
	"SKEL_R_Clavicle",		// 03
	"SKEL_R_UpperArm",		// 04
	"SKEL_R_Forearm",		// 05
	"SKEL_R_Hand",			// 06
	"SKEL_L_Clavicle",		// 07
	"SKEL_L_UpperArm",		// 08
	"SKEL_L_Forearm",		// 09
	"SKEL_L_Hand",			// 10
	"SKEL_Spine3",			// 11
	"SKEL_Spine2",			// 12
	"SKEL_Spine1",			// 13
	"SKEL_Spine0",			// 14
	"SKEL_Spine_Root",		// 15
	"SKEL_R_Thigh",			// 16
	"SKEL_R_Calf",			// 17
	"SKEL_R_Foot",			// 18
	"SKEL_L_Thigh",			// 19
	"SKEL_L_Calf",			// 20
	"SKEL_L_Foot",			// 21
	"SKEL_ROOT",			// 22
	"FB_R_Brow_Out_000",	// 23
	"SKEL_L_Toe0",			// 24
	"MH_R_Elbow",			// 25
	"SKEL_L_Finger01",		// 26
	"SKEL_L_Finger02",		// 27
	"SKEL_L_Finger31",		// 28
	"SKEL_L_Finger32",		// 29
	"SKEL_L_Finger41",		// 30
	"SKEL_L_Finger42",		// 31
	"SKEL_L_Finger11",		// 32
	"SKEL_L_Finger12",		// 33
	"SKEL_L_Finger21",		// 34
	"SKEL_L_Finger22",		// 35
	"RB_L_ArmRoll",			// 36
	"IK_R_Hand",			// 37
	"RB_R_ThighRoll",		// 38
	"FB_R_Lip_Corner_000",	// 39
	"SKEL_Pelvis",			// 40
	"IK_Head",				// 41
	"MH_R_Knee",			// 42
	"FB_LowerLipRoot_000",	// 43
	"FB_R_Lip_Top_000",		// 44
	"FB_R_CheekBone_000",	// 45
	"FB_UpperLipRoot_000",	// 46
	"FB_L_Lip_Top_000",		// 47
	"FB_LowerLip_000",		// 48
	"SKEL_R_Toe0",			// 49
	"FB_L_CheekBone_000",	// 50
	"MH_L_Elbow",			// 51
	"RB_L_ThighRoll",		// 52
	"PH_R_Foot",			// 53
	"FB_L_Eye_000",			// 54
	"SKEL_L_Finger00",		// 55
	"SKEL_L_Finger10",		// 56
	"SKEL_L_Finger20",		// 57
	"SKEL_L_Finger30",		// 58
	"SKEL_L_Finger40",		// 59
	"FB_R_Eye_000",			// 60
	"PH_R_Hand",			// 61
	"FB_L_Lip_Corner_000",	// 62
	"IK_R_Foot",			// 63
	"RB_Neck_1",			// 64
	"IK_L_Hand",			// 65
	"RB_R_ArmRoll",			// 66
	"FB_Brow_Centre_000",	// 67
	"FB_R_Lid_Upper_000",	// 68
	"RB_R_ForeArmRoll",		// 69
	"FB_L_Lid_Upper_000",	// 70
	"MH_L_Knee",			// 71
	"FB_Jaw_000",			// 72
	"FB_L_Lip_Bot_000",		// 73
	"FB_Tongue_000",		// 74
	"FB_R_Lip_Bot_000",		// 75
	"IK_Root",				// 76
	"PH_L_Foot",			// 77
	"FB_L_Brow_Out_000",	// 78
	"SKEL_R_Finger00",		// 79
	"SKEL_R_Finger10",		// 80
	"SKEL_R_Finger20",		// 81
	"SKEL_R_Finger30",		// 82
	"SKEL_R_Finger40",		// 83
	"PH_L_Hand",			// 84
	"RB_L_ForeArmRoll",		// 85
	"FB_UpperLip_000",		// 86
	"SKEL_R_Finger01",		// 87
	"SKEL_R_Finger02",		// 88
	"SKEL_R_Finger31",		// 89
	"SKEL_R_Finger32",		// 90
	"SKEL_R_Finger41",		// 91
	"SKEL_R_Finger42",		// 92
	"SKEL_R_Finger11",		// 93
	"SKEL_R_Finger12",		// 94
	"SKEL_R_Finger21",		// 95
	"SKEL_R_Finger22",		// 96
	"FACIAL_facialRoot",	// 97
	"IK_L_Foot",			// 98
};

#endif // DICTIONARY_H