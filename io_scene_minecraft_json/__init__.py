# ExportHelper is a helper class, defines filename and
# invoke() function which calls the file selector.
import bpy
from bpy_extras.io_utils import ImportHelper, ExportHelper
from bpy.props import StringProperty, BoolProperty, EnumProperty
from bpy.types import Operator

from . import export_minecraft_json
from . import import_minecraft_json

# reload imported modules
import importlib
importlib.reload(export_minecraft_json) 
importlib.reload(import_minecraft_json)

class ImportMinecraftJSON(Operator, ImportHelper):
    """Import Minecraft .json file"""
    bl_idname = "minecraft.import_json"
    bl_label = "Import a Minecraft .json model"

    # ImportHelper mixin class uses this
    filename_ext = ".json"

    filter_glob: StringProperty(
        default="*.json",
        options={'HIDDEN'},
        maxlen=255,  # Max internal buffer length, longer would be clamped.
    )

    # applies default shift from minecraft origin
    translate_origin_by_8: BoolProperty(
        name="Translate by (-8, -8, -8)",
        description="Recenter model with (-8, -8, -8) translation (Minecraft origin)",
        default=False,
    )

    recenter_to_origin: BoolProperty(
        name="Recenter to Origin",
        description="Recenter model median to origin",
        default=True,
    )

    def execute(self, context):
        args = self.as_keywords()
        return import_minecraft_json.load(context, **args)

class ExportMinecraftJSON(Operator, ExportHelper):
    """Exports scene cuboids as minecraft .json object"""
    bl_idname = "minecraft.export_json"
    bl_label = "Export as Minecraft .json"

    # ExportHelper mixin class uses this
    filename_ext = ".json"

    filter_glob: StringProperty(
        default="*.json",
        options={'HIDDEN'},
        maxlen=255,  # Max internal buffer length, longer would be clamped.
    )

    selection_only: BoolProperty(
        name="Selection Only",
        description="Export selection",
        default=False,
    )

    recenter_coords: BoolProperty(
        name="Recenter Coordinates",
        description="Recenter model in [-16, 32] so origin = (8,8,8)",
        default=True,
    )

    rescale_to_max: BoolProperty(
        name="Rescale to Max",
        description="Rescale model to fit entire volume [-16, 32]",
        default=True,
    )

    # export texture filename
    # TEMPORARY property until
    # texture uv editing/export
    texture_name: StringProperty(
        name="Texture Name",
        description="Export texture filename, applied to all cuboids",
        default="",
    )

    def execute(self, context):
        args = self.as_keywords()
        return export_minecraft_json.save(context, **args)

# add io to menu
def menu_func_import(self, context):
    self.layout.operator(ImportMinecraftJSON.bl_idname, text="Minecraft (.json)")

def menu_func_export(self, context):
    self.layout.operator(ExportMinecraftJSON.bl_idname, text="Minecraft (.json)")

# register
classes = [
    ImportMinecraftJSON,
    ExportMinecraftJSON,
]

def register():
    from bpy.utils import register_class
    for cls in classes:
        register_class(cls)

    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)

    from bpy.utils import unregister_class
    for cls in reversed(classes):
        unregister_class(cls)

if __name__ == "__main__":
    register()