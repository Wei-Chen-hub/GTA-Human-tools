#pragma once
#include <wrl.h>
#include <string>

void capture_depth(std::string depth_path, std::string img_seq, BOOL capture_depth, BOOL capture_stencil);

void draw_hook_impl();

void presentCallback(void* chain);

static int draw_indexed_count = 0;

static const char* logFilePath = "GTANativePlugin.log";