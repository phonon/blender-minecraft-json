import bpy
import math

# solid octagonal:
#    _
#  /   \     height = (1 + sqrt(2)) * edge 
# |     |
#  \ _ /
class PrimitiveAddOctagonal(bpy.types.Operator):
    bl_idname = "minecraft.primitive_add_octagonal"
    bl_label = "Add Octagonal (4 Cuboids)"

    def execute(self, context):
        size = 1
        edge = size / (1 + math.sqrt(2))

        cubes = []

        bpy.ops.mesh.primitive_cube_add()
        cube = bpy.context.active_object
        bpy.ops.transform.resize(value=(edge, 1, 1))
        bpy.ops.object.transform_apply(location=False, scale=True, rotation=False)
        cubes.append(cube)

        bpy.ops.mesh.primitive_cube_add()
        cube = bpy.context.active_object
        bpy.ops.transform.resize(value=(1, 1, edge))
        bpy.ops.object.transform_apply(location=False, scale=True, rotation=False)
        cubes.append(cube)

        bpy.ops.mesh.primitive_cube_add()
        cube = bpy.context.active_object
        bpy.ops.transform.resize(value=(edge, 1, 1))
        bpy.ops.transform.rotate(value=math.pi/4, orient_axis='Y')
        bpy.ops.object.transform_apply(location=False, scale=True, rotation=False)
        cubes.append(cube)

        bpy.ops.mesh.primitive_cube_add()
        cube = bpy.context.active_object
        bpy.ops.transform.resize(value=(edge, 1, 1))
        bpy.ops.transform.rotate(value=-math.pi/4, orient_axis='Y')
        bpy.ops.object.transform_apply(location=False, scale=True, rotation=False)
        cubes.append(cube)

        # select all
        for c in cubes:
            c.select_set(True)
        
        return {'FINISHED'}

# hollow octagonal (from 8 cuboids)
#    _
#  / _ \     height = (1 + sqrt(2)) * edge 
# | |_| |
#  \ _ /
class PrimitiveAddOctagonalHollow(bpy.types.Operator):
    bl_idname = "minecraft.primitive_add_octagonal_hollow"
    bl_label = "Add Hollow Octagonal (8 Cuboids)"

    def execute(self, context):
        size_outer_half_extent = 1
        size_inner_half_extent = 0.7
        edge = size_outer_half_extent / (1 + math.sqrt(2))
        thick = 0.5 * (size_outer_half_extent - size_inner_half_extent)

        cubes = []

        for i in range(0,8):
            bpy.ops.mesh.primitive_cube_add()
            cube = bpy.context.active_object
            bpy.ops.transform.resize(value=(edge, 1, thick))
            bpy.ops.transform.translate(value=(0, 0, size_outer_half_extent - thick))
            bpy.ops.object.transform_apply(location=True, scale=True, rotation=False)
            bpy.ops.transform.rotate(value=i * math.pi/4, orient_axis='Y')
            cubes.append(cube)

        # select all
        for c in cubes:
            c.select_set(True)
        
        return {'FINISHED'}

# solid hexadecagon (16-sided polygon) (uses 8 cuboids)
# height = sin(7pi/16) / sin(pi/16) * edge
class PrimitiveAddHexadecagon(bpy.types.Operator):
    bl_idname = "minecraft.primitive_add_hexadecagon"
    bl_label = "Add Hexadecagon (8 Cuboids)"

    def execute(self, context):
        size_outer_half_extent = 1
        thick = size_outer_half_extent * math.sin(math.pi / 16.0) / math.sin(7.0 * math.pi / 16)

        cubes = []

        for i in range(0,16):
            bpy.ops.mesh.primitive_cube_add()
            cube = bpy.context.active_object
            bpy.ops.transform.resize(value=(1, 1, thick))
            bpy.ops.object.transform_apply(location=False, scale=True, rotation=False)
            bpy.ops.transform.rotate(value=i * math.pi/8, orient_axis='Y')
            cubes.append(cube)

        # select all
        for c in cubes:
            c.select_set(True)
        
        return {'FINISHED'}

# hollow hexadecagon (16-sided polygon) (uses 16 cuboids)
# height = sin(7pi/16) / sin(pi/16) * edge
class PrimitiveAddHexadecagonHollow(bpy.types.Operator):
    bl_idname = "minecraft.primitive_add_hexadecagon_hollow"
    bl_label = "Add Hollow Hexadecagon (16 Cuboids)"

    def execute(self, context):
        size_outer_half_extent = 1
        size_inner_half_extent = 0.7
        edge = size_outer_half_extent * math.sin(math.pi / 16.0) / math.sin(7.0 * math.pi / 16)
        thick = 0.5 * (size_outer_half_extent - size_inner_half_extent)

        cubes = []

        for i in range(0,16):
            bpy.ops.mesh.primitive_cube_add()
            cube = bpy.context.active_object
            bpy.ops.transform.resize(value=(edge, 1, thick))
            bpy.ops.transform.translate(value=(0, 0, size_outer_half_extent - thick))
            bpy.ops.object.transform_apply(location=True, scale=True, rotation=False)
            bpy.ops.transform.rotate(value=i * math.pi/8, orient_axis='Y')
            cubes.append(cube)

        # select all
        for c in cubes:
            c.select_set(True)
        
        return {'FINISHED'}

class VIEW3D_MT_minecraft_submenu(bpy.types.Menu):
    bl_idname = "VIEW3D_MT_minecraft_submenu"
    bl_label = "Minecraft"
    
    def draw(self, context):
        layout = self.layout
        layout.operator(
            PrimitiveAddOctagonal.bl_idname,
            text="Octagonal",
            icon="MESH_CYLINDER")
        layout.operator(
            PrimitiveAddOctagonalHollow.bl_idname,
            text="Octagonal (Hollow)",
            icon="MESH_TORUS")
        layout.operator(
            PrimitiveAddHexadecagon.bl_idname,
            text="Hexadecagon",
            icon="MESH_CYLINDER")
        layout.operator(
            PrimitiveAddHexadecagonHollow.bl_idname,
            text="Hexadecagon (Hollow)",
            icon="MESH_TORUS")
    
def add_submenu(self, context):
    self.layout.separator()
    self.layout.menu(VIEW3D_MT_minecraft_submenu.bl_idname, icon="MESH_CUBE")

# register
classes = [
    PrimitiveAddOctagonal,
    PrimitiveAddOctagonalHollow,
    PrimitiveAddHexadecagon,
    PrimitiveAddHexadecagonHollow,
    VIEW3D_MT_minecraft_submenu,
]

def register():
    from bpy.utils import register_class
    for cls in classes:
        register_class(cls)

    # add minecraft primitives to object add menu
    bpy.types.VIEW3D_MT_add.append(add_submenu)

def unregister():
    # remove minecraft primitives from object add menu
    bpy.types.VIEW3D_MT_add.remove(add_submenu)
    
    from bpy.utils import unregister_class
    for cls in reversed(classes):
        unregister_class(cls)

if __name__ == "__main__":
    register()
