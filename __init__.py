bl_info = {
    "name": "Minecraft JSON Import/Export",
    "author": "phonon",
    "version": (0, 2, 0),
    "blender": (2, 83, 0),
    "location": "View3D",
    "description": "Minecraft JSON import/export",
    "warning": "",
    "tracker_url": "https://github.com/phonon/blender-minecraft-json",
    "category": "Minecraft",
}

from . import io_scene_minecraft_json
from . import minecraft_utils

# reload imported modules
import importlib
importlib.reload(io_scene_minecraft_json) 
importlib.reload(minecraft_utils)

def register():
    io_scene_minecraft_json.register()
    minecraft_utils.register()

def unregister():
    minecraft_utils.unregister()
    io_scene_minecraft_json.unregister()