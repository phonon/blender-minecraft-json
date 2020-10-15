import bpy
from bpy import context
from mathutils import Vector, Euler, Matrix
import math
import numpy as np
from math import inf
import posixpath # need "/" separator
import os
import json

# minecraft model coordinates must be from [-16, 32] (48x48x48 volume)
# -> 24 along each axis
MAX_SIZE = 24

# minecraft single axis rotations with discrete values
# [-45, -22.5, 0, 22.5, 45]
ROTATIONS = [
    ("X", -45.0),
    ("X", -22.5),
    ("X",   0.0),
    ("X",  22.5),
    ("X",  45.0),
    ("Y", -45.0),
    ("Y", -22.5),
#    ("Y",   0.0), 0 rotation, default to x
    ("Y",  22.5),
    ("Y",  45.0),
    ("Z", -45.0),
    ("Z", -22.5),
#    ("Z",   0.0), 0 rotation, default to x
    ("Z",  22.5),
    ("Z",  45.0),
]

# generate all rotation matrices (15x3x3)
MAT_ROTATIONS = np.zeros((len(ROTATIONS), 3, 3))
for i, r in enumerate(ROTATIONS):
    MAT_ROTATIONS[i,:,:] = np.array(Matrix.Rotation(math.radians(r[1]), 3, r[0]))

# direction names for minecraft cube face UVs
DIRECTIONS = np.array([
    "north",
    "east",
    "west",
    "south",
    "up",
    "down",
])

# normals for minecraft directions in BLENDER world space
# e.g. blender (-1, 0, 0) is minecraft north (0, 0, -1)
# shape (f,n) = (6,3)
#   f = 6: number of cuboid faces to test
#   v = 3: vertex coordinates (x,y,z)
DIRECTION_NORMALS = np.array([
    [-1.,  0.,  0.],
    [ 0.,  1.,  0.],
    [ 0., -1.,  0.],
    [ 1.,  0.,  0.],
    [ 0.,  0.,  1.],
    [ 0.,  0., -1.],
])
# DIRECTION_NORMALS = np.tile(DIRECTION_NORMALS[np.newaxis,...], (6,1,1))

# blender counterclockwise uv -> minecraft uv rotation lookup table
# (these values experimentally determined)
# access using [uv_loop_start_index][vert_loop_start_index]
COUNTERCLOCKWISE_UV_ROTATION_LOOKUP = [
    [0, 270, 180, 90],
    [90, 0, 270, 180],
    [180, 90, 0, 270],
    [270, 180, 90, 0],
]

# blender clockwise uv -> minecraft uv rotation lookup table
# (these values experimentally determined)
# access using [uv_loop_start_index][vert_loop_start_index]
# Note: minecraft uv must also be x-flipped
CLOCKWISE_UV_ROTATION_LOOKUP = [
    [90, 0, 270, 180],
    [0, 270, 180, 90],
    [270, 180, 90, 0],
    [180, 90, 0, 270],
]

# directed_distance: find closest distances between points 
# in each set and sum results
#    d(s1, s2) = SUM ( for v in s1 ( for u in s2 min ||v-u||^2 ) )
# for two point cloud matrices in format
#        [ x0 x1 ... ]
#    s = [ y0 y1 ... ]
#        [ z0 z1 ... ]
def directed_distance(s1, s2):    
    s1_stacked = np.transpose(s1[..., np.newaxis], (1, 0, 2))
    s2_stacked = np.tile(s2[np.newaxis, ...], (s1.shape[1], 1, 1))
 
    d = s1_stacked - s2_stacked
    d = np.sum(d * d, axis=1)
    d = np.amin(d, axis=1)
    d = np.sum(d)
    
    return d

# return chamfer distance between two point sets:
# sum of directed distance in both directions
#   cd(s1, s2) = d(s1, s2) + d(s2, s1)
def chamfer_distance(s1, s2):
    dist_s1_s2 = directed_distance(s1, s2)
    dist_s2_s1 = directed_distance(s2, s1)
    return dist_s1_s2 + dist_s2_s1

# return sign of value as 1 or -1
def sign(x):
    if x >= 0:
        return 1
    else:
        return -1

# convert blender to minecraft axis
# X -> Z
# Y -> X
# Z -> Y
def to_minecraft_axis(ax):
    if "X" in ax:
        return "Z"
    elif "Y" in ax:
        return "X"
    elif "Z" in ax:
        return "Y"

# convert blender to minecraft axis
# X -> Z
# Y -> X
# Z -> Y
def to_y_up(arr):
    return np.array([arr[1], arr[2], arr[0]])

# round to tick [-45, -22.5, 0, 22.5, 45]
# used to handle floating point errors that cause slight
# offsets from these ticks.
# only apply if angle within this range, else return input angle
def round_rotation(angle, eps = 1e-4):
    if angle > 0:
        if abs(angle - math.pi/4) < eps:
            return math.pi/4
        if abs(angle - math.pi/8) < eps:
            return math.pi/8
        if abs(angle) < eps:
            return 0.0
    if angle < 0:
        if abs(angle + math.pi/4) < eps:
            return -math.pi/4
        if abs(angle + math.pi/8) < eps:
            return -math.pi/8
        if abs(angle) < eps:
            return 0.0
    
    return angle


def add_to_group(groups, name, id):
    """Add integer id to group `name` in groups dict.
    Used for grouping blender collection objects.
    """
    if name in groups:
        groups[name].append(id)
    else:
        groups[name] = [id]


def get_material_color(mat):
    """Get material color as tuple (r, g, b, a). Return None if no
    material node has color property.
    Inputs:
    - mat: Material
    Returns:
    - color tuple (r, g, b,a ) if a basic color,
      "texture_path" string if a texture,
      or None if no color/path
    """
    # get first node with valid color
    if mat.node_tree is not None:
        for n in mat.node_tree.nodes:
            # principled BSDF
            if "Base Color" in n.inputs:
                node_color = n.inputs["Base Color"]
                # check if its a texture path
                for link in node_color.links:
                    from_node = link.from_node
                    if isinstance(from_node, bpy.types.ShaderNodeTexImage):
                        if from_node.image is not None and from_node.image.filepath != "":
                            return from_node.image.filepath
                # else, export color tuple
                color = node_color.default_value
                color = (color[0], color[1], color[2], color[3])
                return color
            # most other materials with color
            elif "Color" in n.inputs:
                color = n.inputs["Color"].default_value
                color = (color[0], color[1], color[2], color[3])
                return color
    
    return None


def get_object_color(obj, material_index, default_color = (0.0, 0.0, 0.0, 1.0)):
    """Get obj material color in index as either 
    - tuple (r, g, b, a) if using a default color input
    - texture file name string "path" if using a texture input
    """
    if material_index < len(obj.material_slots):
        slot = obj.material_slots[material_index]
        material = slot.material
        if material is not None:
            color = get_material_color(material)
            if color is not None:
                return color
    
    return default_color


def loop_is_clockwise(coords):
    """Detect if loop of 2d coordinates is clockwise or counterclockwise.
    Inputs:
    - coords: List of 2d array indexed coords, [p0, p1, p2, ... pN]
              where each is array indexed as p[0] = p0.x, p[1] = p0.y
    Returns:
    - True if clockwise, False if counterclockwise
    """
    num_coords = len(coords)
    area = 0
    
    # use polygon winding area to detect if loop is clockwise or counterclockwise
    for i in range(num_coords):
        # next index
        k = i + 1 if i < num_coords - 1 else 0
        area += (coords[k][0] - coords[i][0]) * (coords[k][1] + coords[i][1])
    
    # clockwise if area positive
    return area > 0


def create_color_texture(
    colors,
    min_size = 16,
):
    """Create a packed square texture from list of input colors. Each color
    is a distinct RGB tuple given a 3x3 pixel square in the texture. These
    must be 3x3 pixels so that there is no uv bleeding near the face edges.
    Also includes a tile for a default color for faces with no material.
    This is the next unfilled 3x3 tile.

    Inputs:
    - colors: Iterable of colors. Each color should be indexable like an rgb
              tuple c = (r, g, b), just so that r = c[0], b = c[1], g = c[2].
    - min_size: Minimum size of texture (must be power 2^n). By default
                16 because Minecraft needs min sized 16 textures for 4 mipmap levels.'
    
    Returns:
    - tex_pixels: Flattened array of texture pixels.
    - tex_size: Size of image texture.
    - color_tex_uv_map: Dict map from rgb tuple color to minecraft format uv coords
                        (r, g, b) -> (xmin, ymin, xmax, ymax)
    - default_color_uv: Default uv coords for unmapped materials (xmin, ymin, xmax, ymax).
    """
    # blender interprets (r,g,b,a) in sRGB space
    def linear_to_sRGB(v):
        if v < 0.0031308:
            return v * 12.92
        else:
            return 1.055 * (v ** (1/2.4)) - 0.055
    
    # fit textures into closest (2^n,2^n) sized texture
    # each color takes a (3,3) pixel chunk to avoid color
    # bleeding at UV edges seams
    # -> get smallest n to fit all colors, add +1 for a default color tile
    color_grid_size = math.ceil(math.sqrt(len(colors) + 1)) # colors on each axis
    tex_size = max(min_size, 2 ** math.ceil(math.log2(3 * color_grid_size))) # fit to (2^n, 2^n) image
    
    # composite colors into white RGBA grid
    tex_colors = np.ones((color_grid_size, color_grid_size, 4))
    color_tex_uv_map = {}
    for i, c in enumerate(colors):
        # convert color to sRGB
        c_srgb = (linear_to_sRGB(c[0]), linear_to_sRGB(c[1]), linear_to_sRGB(c[2]), c[3])

        tex_colors[i // color_grid_size, i % color_grid_size, :] = c_srgb
        
        # uvs: [x1, y1, x2, y2], each value from [0, 16] as proportion of image
        # map each color to a uv
        x1 = ( 3*(i % color_grid_size) + 1 ) / tex_size * 16
        x2 = ( 3*(i % color_grid_size) + 2 ) / tex_size * 16
        y1 = ( 3*(i // color_grid_size) + 1 ) / tex_size * 16
        y2 = ( 3*(i // color_grid_size) + 2 ) / tex_size * 16
        color_tex_uv_map[c] = [x1, y1, x2, y2]
    
    # default color uv coord (last coord + 1)
    idx = len(colors)
    default_color_uv = [
        ( 3*(idx % color_grid_size) + 1 ) / tex_size * 16,
        ( 3*(idx // color_grid_size) + 1 ) / tex_size * 16,
        ( 3*(idx % color_grid_size) + 2 ) / tex_size * 16,
        ( 3*(idx // color_grid_size) + 2 ) / tex_size * 16
    ]

    # triple colors into 3x3 pixel chunks
    tex_colors = np.repeat(tex_colors, 3, axis=0)
    tex_colors = np.repeat(tex_colors, 3, axis=1)
    tex_colors = np.flip(tex_colors, axis=0)

    # pixels as flattened array (for blender Image api)
    tex_pixels = np.ones((tex_size, tex_size, 4))
    tex_pixels[-tex_colors.shape[0]:, 0:tex_colors.shape[1], :] = tex_colors
    tex_pixels = tex_pixels.flatten("C")

    return tex_pixels, tex_size, color_tex_uv_map, default_color_uv


def save_objects(
    filepath,
    objects,
    rescale_to_max = False,
    use_head_upscaling_compensation = False,
    translate_coords = True,
    recenter_to_origin = False,
    blender_origin = [0., 0., 0.],
    generate_texture=True,
    use_only_exported_object_colors=False,
    texture_folder="",
    texture_filename="",
    export_uvs=True,
    minify=False,
    decimal_precision=-1,
    **kwargs):
    """Main exporter function. Parses Blender objects into Minecraft
    cuboid format, uvs, and handles texture read and generation.
    Will save .json file to output paths.

    Inputs:
    - filepath: Output file path name.
    - object: Iterable collection of Blender objects
    - rescale_to_max: Scale exported model to max volume allowed.
    - use_head_upscaling_compensation:
            Clamp over-sized models to max,
            but add upscaling in minecraft head slot (experimental,
            used with animation export, can allow up to 4x max size).
            Overridden by rescale_to_max (these are incompatible).
    - translate_coords: Translate into Minecraft [-16, 32] space so origin = (8,8,8)
    - recenter_to_origin: Recenter model so that its center is at Minecraft origin (8,8,8)
    - blender_origin: Origin in Blender coordinates (in Blender XYZ space).
            Use this to translate exported model.
    - generate_texture: Generate texture from solid material colors. By default, creates
            a color texture from all materials in file (so all groups of
            objects can share the same texture file).
    - use_only_exported_object_colors:
            Generate texture colors from only exported objects instead of default using
            all file materials.
    - texture_folder: Output texture subpath, for typical "item/texture_name" the
            texture folder would be "item".
    - texture_file_name: Name of texture file. TODO: auto-parse textures from materials.
    - export_uvs: Export object uvs.
    - minify: Minimize output file size (write into single line, remove spaces, ...)
    - decimal_precision: Number of digits after decimal to keep in numbers.
            Requires `minify = True`. Set to -1 to disable.
    """

    # output json model
    model_json = {
        "texture_size": [16, 16], # default, will be overridden
        "textures": {},
    }

    elements = []
    groups = {}

    # re-used buffers for every object
    v_world = np.zeros((3, 8))
    v_local = np.zeros((3, 8))
    face_normals = np.zeros((3,6))
    face_uvs = np.zeros((4,))
    face_colors = [None for _ in range(6)]

    # model bounding box vector
    # when re-centering to origin, these will clamp to true model bounding box
    # when not re-centering, bounding box must be relative to blender origin
    # so model min/max starts at (0,0,0) to account for origin point
    if recenter_to_origin:
        model_v_min = np.array([inf, inf, inf])
        model_v_max = np.array([-inf, -inf, -inf])
    else:
        model_v_min = np.array([0., 0., 0.,])
        model_v_max = np.array([0., 0., 0.,])
    
    # all material colors tuples from all object faces
    if use_only_exported_object_colors:
        model_colors = set()
    else:
        model_colors = None
    
    # all material texture str paths
    model_textures = set()
    
    for obj in objects:
        mesh = obj.data
        if not isinstance(mesh, bpy.types.Mesh):
            continue
        
        # object properties
        origin = np.array(obj.location)
        mat_world = obj.matrix_world
        
        # count number of vertices, ignore if not cuboid
        num_vertices = len(mesh.vertices)
        if num_vertices != 8:
            continue
        
        # get world space and local mesh coordinates
        for i, v in enumerate(mesh.vertices):
            v_local[0:3,i] = v.co
            v_world[0:3,i] = mat_world @ v.co
        
        # ================================
        # first reduce rotation to [-45, 45] by applying all 90 deg
        # rotations directly to vertices
        # ================================
        rotation = obj.rotation_euler
        
        rot_reduced = rotation.copy()
        rot_to_reduce_x = 0
        rot_to_reduce_y = 0
        rot_to_reduce_z = 0
        
        # get sign of euler angles
        rot_x_sign = sign(rotation.x)
        rot_y_sign = sign(rotation.y)
        rot_z_sign = sign(rotation.z)
        
        eps = 1e-6 # angle error range for floating point precision issues
        
        # included sign() in loop condition to avoid cases where angle
        # overflows to other polarity and infinite loops
        while abs(rot_reduced.x) > math.pi/4 + eps and sign(rot_reduced.x) == rot_x_sign:
            angle = rot_x_sign * math.pi/2
            rot_reduced.x = round_rotation(rot_reduced.x - angle)
            rot_to_reduce_x += rot_x_sign
        while abs(rot_reduced.y) > math.pi/4 + eps and sign(rot_reduced.y) == rot_y_sign:
            angle = rot_y_sign * math.pi/2
            rot_reduced.y = round_rotation(rot_reduced.y - angle)
            rot_to_reduce_y += rot_y_sign
        while abs(rot_reduced.z) > math.pi/4 + eps and sign(rot_reduced.z) == rot_z_sign:
            angle = rot_z_sign * math.pi/2
            rot_reduced.z = round_rotation(rot_reduced.z - angle)
            rot_to_reduce_z += rot_z_sign
        
        rot_to_reduce = Euler((rot_to_reduce_x * math.pi/2, rot_to_reduce_y * math.pi/2, rot_to_reduce_z * math.pi/2), "XYZ")

        mat_rot_reducer = np.array(rot_to_reduce.to_matrix())
        v_local_transformed = mat_rot_reducer @ v_local
        
        # ================================
        # determine best rotation:
        # 1. transform on all possible minecraft rotations
        # 2. select rotation with min chamfer distance
        # ================================
        # transform with all possible rotation matrices
        v_transformed = origin[...,np.newaxis] + MAT_ROTATIONS @ v_local_transformed[np.newaxis, ...]
        distance_metric = np.array([chamfer_distance(v_transformed[i,:,:], v_world) for i in range(v_transformed.shape[0])])
        
        # get best rotation
        idx_best_rot = np.argmin(distance_metric)
        rot_best = ROTATIONS[idx_best_rot]
        
        # ================================
        # find bounding box edge vertices
        # of transformed local coordinates
        # ================================
        v_local_transformed_translated = origin[...,np.newaxis] + v_local_transformed
        v_min = np.amin(v_local_transformed_translated, axis=1)
        v_max = np.amax(v_local_transformed_translated, axis=1)

        # ================================
        # update global bounding box
        # -> consider world coords, local coords, origin
        # ================================
        model_v_min = np.amin(np.append(v_world, model_v_min[...,np.newaxis], axis=1), axis=1)
        model_v_max = np.amax(np.append(v_world, model_v_max[...,np.newaxis], axis=1), axis=1)
        model_v_min = np.amin(np.append(v_local_transformed_translated, model_v_min[...,np.newaxis], axis=1), axis=1)
        model_v_max = np.amax(np.append(v_local_transformed_translated, model_v_max[...,np.newaxis], axis=1), axis=1)
        model_v_min = np.amin(np.append(origin[...,np.newaxis], model_v_min[...,np.newaxis], axis=1), axis=1)
        model_v_max = np.amax(np.append(origin[...,np.newaxis], model_v_max[...,np.newaxis], axis=1), axis=1)
        
        # ================================
        # output coordinate values
        # ================================
        # change axis to minecraft y-up axis
        v_min = to_y_up(v_min)
        v_max = to_y_up(v_max)
        origin = to_y_up(origin)
        rot_best = (to_minecraft_axis(rot_best[0]), rot_best[1])
        
        # ================================
        # texture/uv generation
        # 
        # NOTE: BLENDER VS MINECRAFT UV AXIS
        # - blender: uvs origin is bottom-left (0,0) to top-right (1, 1)
        # - minecraft: uvs origin is top-left (0,0) to bottom-right (16, 16)
        # minecraft uvs: [x1, y1, x2, y2], each value from [0, 16] as proportion of image
        # as well as 0, 90, 180, 270 degree uv rotation

        # uv loop to export depends on:
        # - clockwise/counterclockwise order
        # - uv starting coordinate (determines rotation) relative to face
        #   vertex loop starting coordinate
        # 
        # Assume "natural" index order of face vertices and uvs without
        # any rotations in local mesh space is counterclockwise loop:
        #   3___2      ^ +y
        #   |   |      |
        #   |___|      ---> +x
        #   0   1
        # 
        # uv, vertex starting coordinate is based on this loop.
        # Use the uv rotation lookup tables constants to determine rotation.
        # ================================

        # initialize faces
        faces = {
            "north": {"uv": [0, 0, 4, 4], "texture": "#0"},
            "east": {"uv": [0, 0, 4, 4], "texture": "#0"},
            "south": {"uv": [0, 0, 4, 4], "texture": "#0"},
            "west": {"uv": [0, 0, 4, 4], "texture": "#0"},
            "up": {"uv": [0, 0, 4, 4], "texture": "#0"},
            "down": {"uv": [0, 0, 4, 4], "texture": "#0"}
        }
        
        uv_layer = mesh.uv_layers.active.data

        for i, face in enumerate(mesh.polygons):
            if i > 5: # should be 6 faces only
                print(f"WARNING: {obj} has >6 faces")
                break
            
            face_normal = face.normal
            face_normal_world = mat_rot_reducer @ face_normal

            # stack + reshape to (6,3)
            face_normal_stacked = np.transpose(face_normal_world[..., np.newaxis], (1,0))
            face_normal_stacked = np.tile(face_normal_stacked, (6,1))

            # get face direction string
            face_direction_index = np.argmax(np.sum(face_normal_stacked * DIRECTION_NORMALS, axis=1), axis=0)
            d = DIRECTIONS[face_direction_index]
            
            face_color = get_object_color(obj, face.material_index)
            
            # solid color tuple
            if isinstance(face_color, tuple) and generate_texture:
                faces[d] = face_color # replace face with color
                if model_colors is not None:
                    model_colors.add(face_color)
            # texture
            elif isinstance(face_color, str):
                faces[d]["texture"] = face_color
                model_textures.add(face_color)

                if export_uvs:
                    # uv loop
                    loop_start = face.loop_start
                    face_uv_0 = uv_layer[loop_start].uv
                    face_uv_1 = uv_layer[loop_start+1].uv
                    face_uv_2 = uv_layer[loop_start+2].uv
                    face_uv_3 = uv_layer[loop_start+3].uv

                    uv_min_x = min(face_uv_0[0], face_uv_2[0])
                    uv_max_x = max(face_uv_0[0], face_uv_2[0])
                    uv_min_y = min(face_uv_0[1], face_uv_2[1])
                    uv_max_y = max(face_uv_0[1], face_uv_2[1])

                    uv_clockwise = loop_is_clockwise([face_uv_0, face_uv_1, face_uv_2, face_uv_3])

                    # vertices loop (using vertices transformed by all 90 deg angles)
                    # project 3d vertex loop onto 2d loop based on face normal,
                    # minecraft uv mapping starting corner experimentally determined
                    verts = [ v_local_transformed[:,v] for v in face.vertices ]
                    #face_normal_world = mat_rot_reducer @ face_normal
                    
                    if face_normal_world[0] > 0.5: # normal = (1, 0, 0)
                        verts = [ (v[1], v[2]) for v in verts ]
                    elif face_normal_world[0] < -0.5: # normal = (-1, 0, 0)
                        verts = [ (-v[1], v[2]) for v in verts ]
                    elif face_normal_world[1] > 0.5: # normal = (0, 1, 0)
                        verts = [ (-v[0], v[2]) for v in verts ]
                    elif face_normal_world[1] < -0.5: # normal = (0, -1, 0)
                        verts = [ (v[0], v[2]) for v in verts ]
                    elif face_normal_world[2] > 0.5: # normal = (0, 0, 1)
                        verts = [ (v[1], -v [0]) for v in verts ]
                    elif face_normal_world[2] < -0.5: # normal = (0, 0, -1)
                        verts = [ (v[1], v[0]) for v in verts ]
                    
                    vert_min_x = min(verts[0][0], verts[2][0])
                    vert_max_x = max(verts[0][0], verts[2][0])
                    vert_min_y = min(verts[0][1], verts[2][1])
                    vert_max_y = max(verts[0][1], verts[2][1])

                    vert_clockwise = loop_is_clockwise(verts)
                    
                    # get uv, vert loop starting corner index 0..3 in face loop

                    # uv start corner index
                    uv_start_x = face_uv_0[0]
                    uv_start_y = face_uv_0[1]
                    if uv_start_y < uv_max_y:
                        # start coord 0
                        if uv_start_x < uv_max_x:
                            uv_loop_start_index = 0
                        # start coord 1
                        else:
                            uv_loop_start_index = 1
                    else:
                        # start coord 2
                        if uv_start_x > uv_min_x:
                            uv_loop_start_index = 2
                        # start coord 3
                        else:
                            uv_loop_start_index = 3
                    
                    # vert start corner index
                    vert_start_x = verts[0][0]
                    vert_start_y = verts[0][1]
                    if vert_start_y < vert_max_y:
                        # start coord 0
                        if vert_start_x < vert_max_x:
                            vert_loop_start_index = 0
                        # start coord 1
                        else:
                            vert_loop_start_index = 1
                    else:
                        # start coord 2
                        if vert_start_x > vert_min_x:
                            vert_loop_start_index = 2
                        # start coord 3
                        else:
                            vert_loop_start_index = 3

                    # set uv flip and rotation based on
                    # 1. clockwise vs counterclockwise loop
                    # 2. relative starting corner difference between vertex loop and uv loop
                    # NOTE: if face normals correct, vertices should always be counterclockwise...
                    if uv_clockwise == False and vert_clockwise == False:
                        face_uvs[0] = uv_min_x
                        face_uvs[1] = uv_max_y
                        face_uvs[2] = uv_max_x
                        face_uvs[3] = uv_min_y
                        face_uv_rotation = COUNTERCLOCKWISE_UV_ROTATION_LOOKUP[uv_loop_start_index][vert_loop_start_index]
                    elif uv_clockwise == True and vert_clockwise == False:
                        # invert x face uvs
                        face_uvs[0] = uv_max_x
                        face_uvs[1] = uv_max_y
                        face_uvs[2] = uv_min_x
                        face_uvs[3] = uv_min_y
                        face_uv_rotation = CLOCKWISE_UV_ROTATION_LOOKUP[uv_loop_start_index][vert_loop_start_index]
                    elif uv_clockwise == False and vert_clockwise == True:
                        # invert y face uvs, case should not happen
                        face_uvs[0] = uv_max_x
                        face_uvs[1] = uv_max_y
                        face_uvs[2] = uv_min_x
                        face_uvs[3] = uv_min_y
                        face_uv_rotation = CLOCKWISE_UV_ROTATION_LOOKUP[uv_loop_start_index][vert_loop_start_index]
                    else: # uv_clockwise == True and vert_clockwise == True:
                        # case should not happen
                        face_uvs[0] = uv_min_x
                        face_uvs[1] = uv_max_y
                        face_uvs[2] = uv_max_x
                        face_uvs[3] = uv_min_y
                        face_uv_rotation = COUNTERCLOCKWISE_UV_ROTATION_LOOKUP[uv_loop_start_index][vert_loop_start_index]
                    
                    xmin = face_uvs[0] * 16
                    ymin = (1.0 - face_uvs[1]) * 16
                    xmax = face_uvs[2] * 16
                    ymax = (1.0 - face_uvs[3]) * 16
                    faces[d]["uv"] = [ xmin, ymin, xmax, ymax ]
                    
                    if face_uv_rotation != 0 and face_uv_rotation != 360:
                        faces[d]["rotation"] = face_uv_rotation if face_uv_rotation >= 0 else 360 + face_uv_rotation
        
        # ================================
        # get collection
        # ================================
        collection = obj.users_collection[0]
        if collection is not None:          
            add_to_group(groups, collection.name, len(elements))
        
        # add object to output
        elements.append({
            "name": obj.name,
            "from": v_min,
            "to": v_max,
            "rotation": {
                "angle": rot_best[1],
                "axis": rot_best[0].lower(),
                "origin": origin,
            },
            "faces": faces,
        })
    
    # tranpose model bbox to minecraft axes
    model_v_min = to_y_up(model_v_min)
    model_v_max = to_y_up(model_v_max)
    model_center = 0.5 * (model_v_min + model_v_max)
    
    # get rescaling factors
    if rescale_to_max:
        if recenter_to_origin:
            rescale_factor = np.min(MAX_SIZE / (model_v_max - model_center))
        # absolute scale relative to (0,0,0), min scaling of MAX/v_max and -MAX/v_min
        else:
            rescale_factor = np.min(np.abs(MAX_SIZE / np.concatenate((-model_v_min - blender_origin, model_v_max - blender_origin))))
    # rescaling, but add head display scaling to compensate for downscaling
    elif use_head_upscaling_compensation:
        if recenter_to_origin:
            rescale_factor = np.min(MAX_SIZE / (model_v_max - model_center))
        else:
            rescale_factor = np.min(np.abs(MAX_SIZE / np.concatenate((-model_v_min, model_v_max))))

        # clamp if not re-scaling to max
        if rescale_factor >= 1.0:
            if rescale_to_max == False:
                rescale_factor = 1.0
        # rescale < 1.0, model too large, inject head display scaling
        else:
            display_head_scale = np.clip(1.0 / rescale_factor, 0, 4)
            model_json["display"] = {
                "head": {
                    "scale": [display_head_scale, display_head_scale, display_head_scale]
                }
            }
    else:
        rescale_factor = 1.0
        
    # debug
    print("RESCALE", rescale_factor)
    print("BBOX MIN/MAX", model_v_min, "/", model_v_max)
    print("CENTER", model_center)
    print("BLENDER ORIGIN", blender_origin)
    print("")
    
    # model post-processing (recenter, rescaling coordinates)
    minecraft_origin = np.array([8, 8, 8])

    # re-center coordinates to minecraft origin and model bounding box center
    if recenter_to_origin:
        model_origin_shift = -model_center
    else:
        model_origin_shift = -to_y_up(blender_origin)
    
    # ===========================
    # generate texture images
    # ===========================
    if generate_texture:
        # default, get colors from all materials in file
        if model_colors is None:
            model_colors = set()
            for mat in bpy.data.materials:
                color = get_material_color(mat)
                if isinstance(color, tuple):
                    model_colors.add(color)
        
        tex_pixels, tex_size, color_tex_uv_map, default_color_uv = create_color_texture(model_colors)

        # texture output filepaths
        if texture_filename == "":
            current_dir = os.path.dirname(filepath)
            filepath_name = os.path.splitext(os.path.basename(filepath))[0]
            texture_save_path = os.path.join(current_dir, filepath_name + ".png")
            texture_model_path = posixpath.join(texture_folder, filepath_name)
        else:
            current_dir = os.path.dirname(filepath)
            texture_save_path = os.path.join(current_dir, texture_filename + ".png")
            texture_model_path = posixpath.join(texture_folder, texture_filename)
        
        # create + save texture
        tex = bpy.data.images.new("tex_colors", alpha=True, width=tex_size, height=tex_size)
        tex.file_format = "PNG"
        tex.pixels = tex_pixels
        tex.filepath_raw = texture_save_path
        tex.save()

        # write texture info to output model
        model_json["texture_size"] = [tex_size, tex_size]
        model_json["textures"]["0"] = texture_model_path
    
    # if not generating texture, just write texture path to json file
    # TODO: scan materials for textures, then update output size
    elif texture_filename != "":
        model_json["texture_size"] = [16, 16]
        model_json["textures"]["0"] = posixpath.join(texture_folder, texture_filename)
    
    # ===========================
    # process face texture paths
    # convert blender path names "//folder\tex.png" -> "item/tex"
    # add textures indices for textures, and create face mappings like "#1"
    # note: #0 id reserved for generated color texture
    # ===========================
    texture_refs = {} # maps blender path name -> #n identifiers
    texture_id = 1    # texture id in "#1" identifier
    for texture_path in model_textures:
        texture_out_path = texture_path
        if texture_out_path[0:2] == "//":
            texture_out_path = texture_out_path[2:]
        texture_out_path = texture_out_path.replace("\\", "/")
        texture_out_path = os.path.splitext(texture_out_path)[0]
        
        texture_refs[texture_path] = "#" + str(texture_id)
        model_json["textures"][str(texture_id)] = posixpath.join(texture_folder, texture_out_path)
        texture_id += 1

    # ===========================
    # final object + face processing
    # 1. recenter/rescale object
    # 2. map solid color face uv -> location in generated texture
    # 3. rewrite path textures -> texture name reference
    # ===========================
    for obj in elements:
        # re-center model
        obj["to"] = model_origin_shift + obj["to"]
        obj["from"] = model_origin_shift + obj["from"]
        obj["rotation"]["origin"] = model_origin_shift + obj["rotation"]["origin"]
        
        # re-scale objects
        obj["to"] = rescale_factor * obj["to"]
        obj["from"] = rescale_factor * obj["from"]
        obj["rotation"]["origin"] = rescale_factor * obj["rotation"]["origin"]

        # re-center coordinates to minecraft origin
        if translate_coords:
            obj["to"] = minecraft_origin + obj["to"]
            obj["from"] = minecraft_origin + obj["from"]
            obj["rotation"]["origin"] = minecraft_origin + obj["rotation"]["origin"]     

        # convert numpy to python list
        obj["to"] = obj["to"].tolist()
        obj["from"] = obj["from"].tolist()
        obj["rotation"]["origin"] = obj["rotation"]["origin"].tolist()

        faces = obj["faces"]
        for d, f in faces.items():
            if isinstance(f, tuple):
                color_uv = color_tex_uv_map[f] if f in color_tex_uv_map else default_color_uv
                faces[d] = {
                    "uv": color_uv,
                    "texture": "#0",
                }
            elif isinstance(f, dict):
                face_texture = f["texture"]
                if face_texture in texture_refs:
                    f["texture"] = texture_refs[face_texture]
                else:
                    face_texture = "#0"
    
    # ===========================
    # convert groups
    # ===========================
    groups_export = []
    for g in groups:
        groups_export.append({
            "name": g,
            "origin": [0, 0, 0],
            "children": groups[g],
        })

    # save
    model_json["elements"] = elements
    model_json["groups"] = groups_export

    # minification options to reduce .json file size
    if minify == True:
        # go through json dict and replace all float with rounded strings
        if decimal_precision >= 0:
            def round_float(x):
                return round(x, decimal_precision)
            
            for elem in model_json["elements"]:
                elem["from"] = [round_float(x) for x in elem["from"]]
                elem["to"] = [round_float(x) for x in elem["to"]]
                elem["rotation"]["origin"] = [round_float(x) for x in elem["rotation"]["origin"]]
                for face in elem["faces"].values():
                    face["uv"] = [round_float(x) for x in face["uv"]]
    
    # save json
    with open(filepath, "w") as f:
        json.dump(model_json, f, separators=(",", ":"))


def save(context,
         filepath,
         selection_only = False,
         **kwargs):
    
    objects = bpy.context.selected_objects if selection_only else bpy.context.scene.objects 
    save_objects(filepath, objects, **kwargs)
    
    print("SAVED", filepath)

    return {"FINISHED"}