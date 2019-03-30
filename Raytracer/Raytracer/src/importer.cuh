#pragma once

#include <string>

class Scene;
struct Material;

namespace importer
{
	void import_scene(const char * path, Scene * scene);
	Material import_material(const std::string & line);
}
