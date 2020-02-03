import bpy
from bpy import context
from mathutils import Vector, Euler, Matrix
import math
import numpy as np
from math import inf
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

def write_file(
    filepath,
    objects,
    texture_name="",
    texture_size=[32, 32],
    rescale_to_max = False,
    recenter_coords = True,
    **kwargs):

    # output json model
    model_json = {
        "texture_size": texture_size,
        "textures": {
            "0": texture_name,
            "particle": texture_name,
        },
    }

    elements = []
    groups = {}

    # model bounding box vector
    model_v_min = np.array([inf, inf, inf])
    model_v_max = np.array([-inf, -inf, -inf])

    for obj in objects:
        mesh = obj.data
        if not isinstance(mesh, bpy.types.Mesh):
            continue
        
        # object properties
        origin = np.array(obj.location)
        scale = np.array(obj.scale)
        mat_world = obj.matrix_world
        
        # count number of vertices, ignore if not cuboid
        num_vertices = len(mesh.vertices)
        if num_vertices != 8:
            continue
        
        # get world space and local mesh coordinates
        v_world = np.zeros((3, 8))
        v_local = np.zeros((3, 8))
        for i, v in enumerate(mesh.vertices):
            v_local[0:3,i] = v.co
            v_world[0:3,i] = mat_world @ v.co
        
        v_local_translated = origin[...,np.newaxis] + v_local
        
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
        # change axis to minecraft format
        v_min = np.array([v_min[1], v_min[2], v_min[0]])
        v_max = np.array([v_max[1], v_max[2], v_max[0]])
        origin = np.array([origin[1], origin[2], origin[0]])
        rot_best = (to_minecraft_axis(rot_best[0]), rot_best[1])
        
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
            "faces": {
                "north": {"uv": [0, 0.5, 0.5, 1], "texture": "#0"},
                "east": {"uv": [0, 0.5, 0.5, 1], "texture": "#0"},
                "south": {"uv": [0, 0.5, 0.5, 1], "texture": "#0"},
                "west": {"uv": [0, 0.5, 0.5, 1], "texture": "#0"},
                "up": {"uv": [0, 0.5, 0.5, 1], "texture": "#0"},
                "down": {"uv": [0, 0.5, 0.5, 1], "texture": "#0"}
            }
        })
    
    # tranpose model bbox to minecraft axes
    model_v_min = np.array([model_v_min[1], model_v_min[2], model_v_min[0]])
    model_v_max = np.array([model_v_max[1], model_v_max[2], model_v_max[0]])
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

    # convert groups
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

    with open(filepath, 'w') as f:
        json.dump(model_json, f)
    
def save(context,
         filepath,
         selection_only = False,
         **kwargs):
    
    objects = bpy.context.selected_objects if selection_only else bpy.context.scene.objects 
    write_file(filepath, objects, **kwargs)
    
    print('SAVED', filepath)

    return {'FINISHED'}