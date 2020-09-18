# ExportHelper is a helper class, defines filename and
# invoke() function which calls the file selector.
import bpy
from bpy_extras.io_utils import ImportHelper, ExportHelper
from bpy.props import StringProperty, BoolProperty, EnumProperty, IntProperty
from bpy.types import Operator

from . import export_minecraft_json
from . import export_minecraft_json_by_bones
from . import import_minecraft_json

# reload imported modules
import importlib
importlib.reload(export_minecraft_json) 
importlib.reload(export_minecraft_json_by_bones) 
importlib.reload(import_minecraft_json)

class ImportMinecraftJSON(Operator, ImportHelper):
    """Import Minecraft .json file"""
    bl_idname = "minecraft.import_json"
    bl_label = "Import a Minecraft .json model"
    bl_options = {"REGISTER", "UNDO"}

    # ImportHelper mixin class uses this
    filename_ext = ".json"

    filter_glob: StringProperty(
        default="*.json",
        options={"HIDDEN"},
        maxlen=255,  # Max internal buffer length, longer would be clamped.
    )

    import_uvs: BoolProperty(
        name="Import UVs",
        description="Import UVs",
        default=True,
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
        options={"HIDDEN"},
        maxlen=255,  # Max internal buffer length, longer would be clamped.
    )

    selection_only: BoolProperty(
        name="Selection Only",
        description="Export selection",
        default=False,
    )

    translate_coords: BoolProperty(
        name="Translate Coordinates",
        description="Translate into [-16, 32] space so origin = (8,8,8)",
        default=True,
    )

    rescale_to_max: BoolProperty(
        name="Rescale to Max",
        description="Rescale model to fit entire volume [-16, 32]",
        default=True,
    )

    recenter_to_origin: BoolProperty(
        name="Recenter Model to Origin",
        description="Recenter model so its center is at the origin",
        default=True,
    )

    # ================================
    # texture options
    texture_folder: StringProperty(
        name="Texture Subfolder",
        description="Subfolder in resourcepack: assets/minecraft/textures/[folder]",
        default="item",
    )

    texture_filename: StringProperty(
        name="Texture Name",
        description="Export texture filename, applied to all cuboids",
        default="",
    )

    export_uvs: BoolProperty(
        name="Export UVs",
        description="Export UVs",
        default=True,
    )
    
    generate_texture: BoolProperty(
        name="Generate Color Texture",
        description="Generate texture image from all material colors",
        default=False,
    )
    
    use_only_exported_object_colors: BoolProperty(
        name="Only Use Exported Object Colors",
        description="Generate texture from material colors only on exported objects",
        default=False,
    )

    # ================================
    # minify options
    minify: BoolProperty(
        name="Minify .json",
        description="Enables minification options to reduce .json file size",
        default=False,
    )

    decimal_precision: IntProperty(
        name="Decimal Precision",
        description="Number of digits after decimal point (use -1 to disable)",
        min=-1,
        max=16,
        soft_min=-1,
        soft_max=16,
        default=8,
    )

    # ================================
    # animation options EXPERIMENTAL
    export_bones: BoolProperty(
        name="Export bones",
        description="Export model in clusters attached to bones",
        default=False,
    )

    export_animation: BoolProperty(
        name="Export animations",
        description="Export bone animation keyframes into .json file",
        default=False,
    )

    def execute(self, context):
        args = self.as_keywords()
        if args["export_bones"] == True or args["export_animation"] == True:
            return export_minecraft_json_by_bones.save(context, **args)
        else:
            return export_minecraft_json.save(context, **args)
    
    def draw(self, context):
        pass


# export options panel for geometry
class JSON_PT_export_geometry(bpy.types.Panel):
    bl_space_type = "FILE_BROWSER"
    bl_region_type = "TOOL_PROPS"
    bl_label = "Geometry"
    bl_parent_id = "FILE_PT_operator"

    @classmethod
    def poll(cls, context):
        sfile = context.space_data
        operator = sfile.active_operator
        return operator.bl_idname == "MINECRAFT_OT_export_json"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False  # No animation.

        sfile = context.space_data
        operator = sfile.active_operator

        layout.prop(operator, "selection_only")
        layout.prop(operator, "translate_coords")
        layout.prop(operator, "rescale_to_max")
        layout.prop(operator, "recenter_to_origin")


# export options panel for textures
class JSON_PT_export_textures(bpy.types.Panel):
    bl_space_type = "FILE_BROWSER"
    bl_region_type = "TOOL_PROPS"
    bl_label = "Textures"
    bl_parent_id = "FILE_PT_operator"

    @classmethod
    def poll(cls, context):
        sfile = context.space_data
        operator = sfile.active_operator
        return operator.bl_idname == "MINECRAFT_OT_export_json"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False  # No animation.

        sfile = context.space_data
        operator = sfile.active_operator

        layout.prop(operator, "texture_folder")
        layout.prop(operator, "texture_filename")
        layout.prop(operator, "export_uvs")
        layout.prop(operator, "generate_texture")
        layout.prop(operator, "use_only_exported_object_colors")


# export options panel for minifying .json output
class JSON_PT_export_minify(bpy.types.Panel):
    bl_space_type = "FILE_BROWSER"
    bl_region_type = "TOOL_PROPS"
    bl_label = "Minify"
    bl_parent_id = "FILE_PT_operator"

    @classmethod
    def poll(cls, context):
        sfile = context.space_data
        operator = sfile.active_operator
        return operator.bl_idname == "MINECRAFT_OT_export_json"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False  # No animation.

        sfile = context.space_data
        operator = sfile.active_operator

        layout.prop(operator, "minify")
        layout.prop(operator, "decimal_precision")

# export options panel for animation
class JSON_PT_export_animation(bpy.types.Panel):
    bl_space_type = "FILE_BROWSER"
    bl_region_type = "TOOL_PROPS"
    bl_label = "Animation"
    bl_parent_id = "FILE_PT_operator"

    @classmethod
    def poll(cls, context):
        sfile = context.space_data
        operator = sfile.active_operator
        return operator.bl_idname == "MINECRAFT_OT_export_json"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False  # No animation.

        sfile = context.space_data
        operator = sfile.active_operator

        layout.prop(operator, "export_bones")
        layout.prop(operator, "export_animation")

# add io to menu
def menu_func_import(self, context):
    self.layout.operator(ImportMinecraftJSON.bl_idname, text="Minecraft (.json)")

def menu_func_export(self, context):
    self.layout.operator(ExportMinecraftJSON.bl_idname, text="Minecraft (.json)")

# register
classes = [
    ImportMinecraftJSON,
    ExportMinecraftJSON,
    JSON_PT_export_geometry,
    JSON_PT_export_textures,
    JSON_PT_export_minify,
    JSON_PT_export_animation,
]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()