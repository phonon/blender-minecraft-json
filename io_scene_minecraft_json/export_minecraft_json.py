import bpy
from bpy import context
from mathutils import Vector, Euler, Matrix
import math
import numpy as np
from math import inf
import posixpath # need '/' separator
import os
import json

# minecraft model coordinates must be from [-16, 32] (48x48x48 volume)
# -> 24 along each axis
MAX_SIZE = 24

# minecraft single axis rotations with discrete values
# [-45, -22.5, 0, 22.5, 45]
ROTATIONS = [
    ('X', -45.0),
    ('X', -22.5),
    ('X',   0.0),
    ('X',  22.5),
    ('X',  45.0),
    ('Y', -45.0),
    ('Y', -22.5),
#    ('Y',   0.0), 0 rotation, default to x
    ('Y',  22.5),
    ('Y',  45.0),
    ('Z', -45.0),
    ('Z', -22.5),
#    ('Z',   0.0), 0 rotation, default to x
    ('Z',  22.5),
    ('Z',  45.0),
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
# shape (f,n,v) = (6,6,3)
#   f = 6: number of cuboid faces to test
#   n = 6: number of normal directions
#   v = 3: vertex coordinates (x,y,z)
DIRECTION_NORMALS = np.array([
    [-1.,  0.,  0.],
    [ 0.,  1.,  0.],
    [ 0., -1.,  0.],
    [ 1.,  0.,  0.],
    [ 0.,  0.,  1.],
    [ 0.,  0., -1.],
])
DIRECTION_NORMALS = np.tile(DIRECTION_NORMALS[np.newaxis,...], (6,1,1))

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
    if name in groups:
        groups[name].append(id)
    else:
        groups[name] = [id]

# default color for objects with no material
DEFAULT_COLOR = (0.0, 0.0, 0.0, 1.0)

# get obj material color in index as tuple (r, g, b, a)
def get_material_color(obj, material_index):
    if material_index < len(obj.material_slots):
        slot = obj.material_slots[material_index]
        material = slot.material
        if material is not None:
            nodes = material.node_tree.nodes

            # get first node with valid color
            for n in nodes:
                # principled BSDF
                if 'Base Color' in n.inputs:
                    color = n.inputs['Base Color'].default_value
                    color = (color[0], color[1], color[2], color[3])
                    return color
                # most other materials with color
                elif 'Color' in n.inputs:
                    color = n.inputs['Color'].default_value
                    color = (color[0], color[1], color[2], color[3])
                    return color
            
    return DEFAULT_COLOR

# detect if uv loop direction from mesh uv_layer and start index, returns
# True - clockwise
# False - counterclockwise
def uv_loop_is_clockwise(uv_layer, loop_start):
    face_uv_0 = uv_layer[loop_start].uv
    face_uv_1 = uv_layer[loop_start+1].uv
    face_uv_2 = uv_layer[loop_start+2].uv
    face_uv_3 = uv_layer[loop_start+3].uv

    # use polygon winding to detect if uv loop order is clockwise or counterclockwise
    area = (face_uv_1.x - face_uv_0.x) * (face_uv_1.y - face_uv_0.y)
    area += (face_uv_2.x - face_uv_1.x) * (face_uv_2.y - face_uv_1.y)
    area += (face_uv_3.x - face_uv_2.x) * (face_uv_3.y - face_uv_2.y)
    area += (face_uv_0.x - face_uv_3.x) * (face_uv_0.y - face_uv_3.y)

    # clockwise if area positive
    return area > 0


# main exporter function:
# parses objects and outputs json in filepath
def write_file(
    filepath,
    objects,
    rescale_to_max = False,
    recenter_coords = True,
    generate_texture=True,
    texture_folder="",
    texture_filename="",
    export_uvs=True,
    minify=False,            # minimize output .json size
    decimal_precision=-1,    # float decimal precision (used if minify=True, -1 to disable)
    **kwargs):

    # output json model
    model_json = {
        "texture_size": [16, 16], # default, will be overridden
        "textures": {
            "0": ""
        },
    }

    elements = []
    groups = {}

    # re-used buffers for every object
    v_world = np.zeros((3, 8))
    v_local = np.zeros((3, 8))
    face_normals = np.zeros((3,6))
    face_uvs = np.zeros((6,4))
    face_colors = [None for _ in range(6)]

    # model bounding box vector
    model_v_min = np.array([inf, inf, inf])
    model_v_max = np.array([-inf, -inf, -inf])

    # all material colors from all object faces
    model_colors = set()

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
        
        rot_to_reduce = Euler((rot_to_reduce_x * math.pi/2, rot_to_reduce_y * math.pi/2, rot_to_reduce_z * math.pi/2), 'XYZ')

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
            face_normals[0:3,i] = face.normal
            face_colors[i] = get_material_color(obj, face.material_index)

            # determine min/max corners of uv loop
            loop_start = face.loop_start
            face_uv_0 = uv_layer[loop_start].uv
            face_uv_2 = uv_layer[loop_start+2].uv

            # set uv min/max depending on clockwise or counterclockwise order
            if uv_loop_is_clockwise(uv_layer, loop_start):
                uv_min = face_uv_0
                uv_max = face_uv_2
            else:
                uv_min = face_uv_2
                uv_max = face_uv_0
            
            face_uvs[i][0] = uv_min[0]
            face_uvs[i][1] = uv_min[1]
            face_uvs[i][2] = uv_max[0]
            face_uvs[i][3] = uv_max[1]

        # add face colors to overall model set
        model_colors.update(face_colors)

        # apply all 90 deg rots to face normals
        face_normals_transformed = mat_rot_reducer @ face_normals

        # reshape to (6,1,3)
        face_normals_transformed = np.transpose(face_normals_transformed, (1,0))
        face_normals_transformed = np.reshape(face_normals_transformed[...,np.newaxis], (6,1,3))

        # get face direction strings, set face colors
        face_directions = np.argmax(np.sum(face_normals_transformed * DIRECTION_NORMALS, axis=2), axis=1)
        face_directions = DIRECTIONS[face_directions]

        # replace faces with material colors
        if generate_texture:
            for i, d in enumerate(face_directions):
                faces[d] = face_colors[i]
        # set uvs
        elif export_uvs:
            for i, d in enumerate(face_directions):
                xmin = face_uvs[i][0] * 16
                ymin = (1.0 - face_uvs[i][1]) * 16
                xmax = face_uvs[i][2] * 16
                ymax = (1.0 - face_uvs[i][3]) * 16
                faces[d]["uv"] = [ xmin, ymin, xmax, ymax ]

        # ================================
        # get collection
        # ================================
        collection = obj.users_collection[0]
        if collection is not None:          
            add_to_group(groups, collection.name, len(elements))
        
        # add object to output
        elements.append({
            "name": obj.name,
            "from": v_min.tolist(),
            "to": v_max.tolist(),
            "rotation": {
                "angle": rot_best[1],
                "axis": rot_best[0].lower(),
                "origin": origin.tolist(),
            },
            "faces": faces,
        })
    
    # tranpose model bbox to minecraft axes
    model_v_min = to_y_up(model_v_min)
    model_v_max = to_y_up(model_v_max)
    model_center = 0.5 * (model_v_min + model_v_max)
    
    # get rescaling factors
    if rescale_to_max:
        rescale_factor = np.min(MAX_SIZE / (model_v_max - model_center))
    else:
        rescale_factor = 1.0
    
    model_center_rescaled = rescale_factor * model_center
    
    # debug
    print('RESCALE', rescale_factor)
    print('BBOX MIN/MAX', model_v_min, '/', model_v_max)
    print('CENTER', model_center)

    # model post-processing (recenter, rescaling coordinates)
    minecraft_origin = np.array([8, 8, 8])
    new_origin = minecraft_origin - model_center_rescaled
    for obj in elements:
        # re-scale to max
        if rescale_to_max:
            obj["to"] = (rescale_factor * np.array(obj["to"])).tolist()
            obj["from"] = (rescale_factor * np.array(obj["from"])).tolist()
            obj["rotation"]["origin"] = (rescale_factor * np.array(obj["rotation"]["origin"])).tolist()

        # re-center coordinates
        if recenter_coords:
            obj["to"] = (new_origin + np.array(obj["to"])).tolist()
            obj["from"] = (new_origin + np.array(obj["from"])).tolist()
            obj["rotation"]["origin"] = (new_origin + np.array(obj["rotation"]["origin"])).tolist()
    
    # ===========================
    # generate texture images
    # ===========================
    if generate_texture:
        # fit textures into closest (2^n,2^n) sized texture
        # each color takes a (3,3) pixel chunk to avoid color
        # bleeding at UV edges seams
        # -> get smallest n to fit all colors
        color_grid_size = math.ceil(math.sqrt(len(model_colors))) # colors on each axis
        tex_size = 2 ** math.ceil(math.log2(3 * color_grid_size)) # fit to (2^n, 2^n) image

        # blender interprets (r,g,b,a) in sRGB space
        def linear_to_sRGB(v):
            if v < 0.0031308:
                return v * 12.92
            else:
                return 1.055 * (v ** (1/2.4)) - 0.055

        # composite colors into white RGBA grid
        tex_colors = np.ones((color_grid_size, color_grid_size, 4))
        color_tex_uv_map = {}
        for i, c in enumerate(model_colors):
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
        
        # triple colors into 3x3 pixel chunks
        tex_colors = np.repeat(tex_colors, 3, axis=0)
        tex_colors = np.repeat(tex_colors, 3, axis=1)
        tex_colors = np.flip(tex_colors, axis=0)

        # pixels as flattened array (for blender Image api)
        tex_pixels = np.ones((tex_size, tex_size, 4))
        tex_pixels[-tex_colors.shape[0]:, 0:tex_colors.shape[1], :] = tex_colors
        tex_pixels = tex_pixels.flatten('C')

        # texture output filepaths
        if texture_filename == "":
            current_dir = os.path.dirname(filepath)
            filepath_name = os.path.splitext(os.path.basename(filepath))[0]
            texture_save_path = os.path.join(current_dir, filepath_name + '.png')
            texture_model_path = posixpath.join(texture_folder, filepath_name)
        else:
            current_dir = os.path.dirname(filepath)
            texture_save_path = os.path.join(current_dir, texture_filename + '.png')
            texture_model_path = posixpath.join(texture_folder, texture_filename)
        
        # create + save texture
        tex = bpy.data.images.new("tex_colors", alpha=True, width=tex_size, height=tex_size)
        tex.file_format = 'PNG'
        tex.pixels = tex_pixels
        tex.filepath_raw = texture_save_path
        tex.save()

        # re-write UVs on all elements
        for obj in elements:
            faces = obj["faces"]
            for f in faces:
                color = faces[f]
                if isinstance(color, tuple):
                    faces[f] = {
                        "uv": color_tex_uv_map[color],
                        "texture": "#0",
                    }

        # write texture info to output model
        model_json["texture_size"] = [tex_size, tex_size]
        model_json["textures"]["0"] = texture_model_path
    
    # if not generating texture, just write texture path to json file
    # TODO: scan materials for textures, then update output size
    elif texture_filename != "":
        model_json["texture_size"] = [16, 16]
        model_json["textures"]["0"] = posixpath.join(texture_folder, texture_filename)
    
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
    with open(filepath, 'w') as f:
        json.dump(model_json, f, separators=(",", ":"))


def save(context,
         filepath,
         selection_only = False,
         **kwargs):
    
    objects = bpy.context.selected_objects if selection_only else bpy.context.scene.objects 
    write_file(filepath, objects, **kwargs)
    
    print('SAVED', filepath)

    return {'FINISHED'}