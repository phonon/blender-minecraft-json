"""
Export model in object "clusters" associated with bones (experimental).
- For each bone with associated objects, export separate
  .json model file with associated objects.
- Export .json bone hierarchy and data file (with bone matrices)

Notes:
- Blender quaternion format is (w,x,y,z) -> converted to (x,y,z,w) on export
- Blender bones use local y-up coordinate system (so do not need to change bone coords on export)
- Blender calls the bone world-space transform "matrix_local".
  This is instead exported as "matrix_world".
"""
import bpy
from bpy import context
from mathutils import Vector, Euler, Quaternion, Matrix
import math
import numpy as np
from math import inf
import posixpath # need '/' separator
import os
import json

from . import export_minecraft_json

# matrix for converting from blender XYZ to minecraft y-up format
# X -> -Z
# Y -> -X
# Z -> Y
MATRIX_TO_Y_UP = Matrix([
    [ 0., -1., 0., 0.],
    [ 0.,  0., 1., 0.],
    [-1.,  0., 0., 0.],
    [ 0.,  0., 0., 1.],
])

QUAT_TO_Y_UP = MATRIX_TO_Y_UP.to_quaternion()

# scale model coordinates to minecraft world space units
# = world_units_per_model_units * armor_stand_head_scale
# = 1/16 * 5/8
MODEL_SCALE = 0.0390625

# axis-aligned bounding box
class AABB():
    def __init__(self):
        self.min = np.array([inf, inf, inf])
        self.max = np.array([-inf, -inf, -inf])
    
    def __str__(self):
        return "AABB(min={min}, max={max}) ".format(
            min=self.min,
            max=self.max,
        )
    
    # update min, max from input world space vertices
    # as np.array of shape = (3, n)
    # n - number of vertices
    def update(self, vertices):
        self.min = np.amin(np.append(vertices, self.min[...,np.newaxis], axis=1), axis=1)
        self.max = np.amax(np.append(vertices, self.max[...,np.newaxis], axis=1), axis=1)


# Bone tree hierarchy node backed by a dict
# For easily serialization to json
# Note matrix dimensions:
# - matrix = 3x3, converted to 4x4 for export
# - matrix_local = 4x4
class BoneNode(dict):
    def __init__(self, name, matrix, matrix_local, parent):
        super(BoneNode, self).__init__()

        # convert matrices:
        # 1. convert to y-up
        matrix_transformed = matrix.to_4x4() if matrix is not None else Matrix.Identity(4)
        matrix_local_transformed = matrix_local if matrix_local is not None else Matrix.Identity(4)

        matrix_transformed = MATRIX_TO_Y_UP @ matrix_transformed
        matrix_local_transformed = MATRIX_TO_Y_UP @ matrix_local_transformed

        # convert to numpy
        matrix_transformed = np.array(matrix_transformed)
        matrix_local_transformed = np.array(matrix_local_transformed)

        # 2. scale model coordinates to minecraft world space units
        matrix_transformed[0:3,3] = MODEL_SCALE * matrix_transformed[0:3,3] 
        matrix_local_transformed[0:3,3] = MODEL_SCALE * matrix_local_transformed[0:3,3] 

        self["name"] = name
        self["matrix"] = matrix_transformed.flatten(order="C").tolist()
        self["matrix_world"] = matrix_local_transformed.flatten(order="C").tolist()
        self["parent"] = parent
        self["children"] = []
    
    def __str__(self):
        return "Bone(name={name}, children={children})".format(
            name=self["name"],
            children=self["children"],
        )
    
    def name(self):
        return self["name"]
    
    def children(self):
        return self["children"]
    
    def add(self, child):
        self["children"].append(child)


# Group together bones -> objects
class BoneObjectGroup():
    def __init__(self, bone):
        self.bone = bone
        self.objects = []

    def add(self, obj):
        self.objects.append(obj)


def build_bone_hierarchy(objects):
    """Get serializable bone tree and map bones -> object clusters.
    Inputs:
    - objects: Objects to consider
    Returns:
    - bone_tree: BoneNode dict tree of skeleton tree
    - bone_object_groups: dict of bone name -> BoneObjectGroup
    - bbox: bounding box of objects to export
    """

    skeleton = bpy.data.armatures[0]

    # initialize empty bone object groups from bones in first skeleton in scene
    bone_object_groups = {}
    for bone in skeleton.bones:
        bone_object_groups[bone.name] = BoneObjectGroup(bone)

    # model bounding box
    bbox = AABB()

    # re-used buffers for every object
    v_world = np.zeros((3, 8))
    
    # iterate over input mesh objects:
    # - group objects with their bone (export each bone as separate model)
    # - find bounding box of model
    for obj in objects:
        if not isinstance(obj.data, bpy.types.Mesh):
            continue
        
        # map vertex group -> sum of vertex weights in that group
        v_group_weight_sum = {}
        
        for v in obj.data.vertices:
            for vg_element in v.groups:
                weight = vg_element.weight
                group = obj.vertex_groups[vg_element.group]
                
                if group.name in v_group_weight_sum:
                    v_group_weight_sum[group.name] += weight
                else:
                    v_group_weight_sum[group.name] = weight
            
        # no vertex groups, skip object        
        if len(v_group_weight_sum) == 0:
            continue
        
        # find group with highest total weight
        vg_iter = iter(v_group_weight_sum.keys())
        strongest_group = next(vg_iter)
        for g in vg_iter:
            if v_group_weight_sum[g] > v_group_weight_sum[strongest_group]:
                strongest_group = g
        
        # add object to group
        bone_object_groups[strongest_group].add(obj)
        
        # update overall model bounding box
        mat_world = obj.matrix_world
        for i, v in enumerate(obj.data.vertices):
            v_world[0:3,i] = mat_world @ v.co
            
        bbox.update(v_world)
    
    # get root bones
    roots = []
    for bone in skeleton.bones:
        if bone.parent is None:
            roots.append(bone)

    # recursively parse bone tree
    def parse_bones(parent_node, bone):
        node = BoneNode(
            bone.name,
            bone.matrix,
            bone.matrix_local,
            parent_node.name(),
        )
        parent_node.add(node)
        
        for child in bone.children:
            parse_bones(node, child)

    bone_tree = BoneNode("__root__", None, None, None)
    for bone in roots:
        parse_bones(bone_tree, bone)

    return bone_tree, bone_object_groups, bbox


def get_animations():
    # matrix for converting from blender bone local space
    # to minecraft y-up format
    # note: this coordinate frame is different than blender world space
    # X -> -X
    # Y -> Y
    # Z -> X
    BONE_LOCAL_BLENDER_TO_MINECRAFT = Matrix([
        [ 0., 0., -1., 0.],
        [ 0.,  1., 0., 0.],
        [-1.,  0., 0., 0.],
        [ 0.,  0., 0., 1.],
    ])

    FCURVE_DATA_PATH_TO_PROPERTY = {
        "location": "position",
        "rotation_quaternion": "quaternion",
    }

    ARRAY_INDEX_TO_PROPERTY = {
        0: "x",
        1: "y",
        2: "z",
        3: "w",
    }

    class Keyframe:
        def __init__(self, frame, interpolation, value):
            self.frame = frame
            self.interpolation = interpolation
            self.value = value

    """
    Format of temporary bone keyframes dict to store tracks:
    {
        "bone1": {
            "position": {
                t0: Keyframe(t0, INTERPOLATION, value),
                t1: Keyframe(t1, INTERPOLATION, value),
                ...
            },
            "quaternion": {
                t0: Keyframe(t0, INTERPOLATION, value),
                t1: Keyframe(t1, INTERPOLATION, value),
                ...
            }
        },
        "bone2": {
            ...
        },
        ...
    }

    - t0, t1, ... : frame time integers, not ordered
    - INTERPOLATION : interpolation type
    - value : object value, either Vector or Quaternion
    """
    def add_property(keyframes_dict, bone_name, property_name, interpolation, frame, value, array_index):
        if bone_name not in keyframes_dict:
            keyframes_dict[bone_name] = {}
        
        if property_name not in keyframes_dict[bone_name]:
            keyframes_dict[bone_name][property_name] = {}
        
        if frame not in keyframes_dict[bone_name][property_name]:
            if property_name == "position":
                default_value = Vector()
            elif property_name == "quaternion":
                default_value = Quaternion()
            else:
                # invalid property
                return
            
            keyframes_dict[bone_name][property_name][frame] = Keyframe(frame, interpolation, default_value)
        
        # setting value
        keyframes_dict[bone_name][property_name][frame].value[array_index] = value

    # skeleton
    skeleton = bpy.data.armatures[0]
    bones = skeleton.bones

    # get animation actions
    animations = {}

    for a in bpy.data.actions:
       
        # get all actions
        fcurves = a.fcurves
        
        # skip empty actions
        if len(fcurves) == 0:
            continue
        
        bone_keyframes = {}
        
        # gather fcurve track data into position, quaternion
        # bone keyframes
        for fcu in fcurves:
            # read bone name in format: path.bones["name"].property
            data_path = fcu.data_path
            if not data_path.startswith("pose.bones"):
                continue
            
            bone_name = data_path[data_path.find("[")+2 : data_path.find("]")-1]
            
            # skip if bone not found
            if bone_name not in bones:
                continue
            
            bone = bones[bone_name]
            
            # match data_path property to export name
            export_property_name = None
            for data_path_name, export_name in FCURVE_DATA_PATH_TO_PROPERTY.items():
                if data_path.endswith(data_path_name):
                    export_property_name = export_name
                    break
            
            # invalid property
            if export_property_name == None:
                continue
            
            # make sure keyframes are in order
            fcu.update()
            
            # gather individual property keyframe into object keyframes
            for keyframe in fcu.keyframe_points:
                co = keyframe.co
                add_property(bone_keyframes, bone_name, export_property_name, keyframe.interpolation, int(co[0]), co[1], fcu.array_index)
        
        # transform to minecraft y-up coordinate space
        for bone_name, object_keyframes in bone_keyframes.items():
            if "position" in object_keyframes:
                bone = bones[bone_name]
                for keyframe in object_keyframes["position"].values():
                    keyframe.value = bone.matrix @ keyframe.value
                    keyframe.value = keyframe.value * MODEL_SCALE
            # bone quaternions already in y-up format?
            # if "quaternion" in object_keyframes:
            #     for keyframe in object_keyframes["quaternion"].values():
            #     keyframe.value = QUAT_TO_Y_UP @ keyframe.value
        
        # write animation json dict
        animations[a.name] = {}
        
        for bone_name, object_keyframes in bone_keyframes.items():
            animations[a.name][bone_name] = {}
            
            if "position" in object_keyframes:
                sorted_keyframes = sorted(list(object_keyframes["position"].values()), key=lambda x: x.frame)
                animations[a.name][bone_name]["position"] = [ [ k.frame, k.interpolation, k.value[0], k.value[1], k.value[2] ] for k in sorted_keyframes ]
            
            if "quaternion" in object_keyframes:
                sorted_keyframes = sorted(list(object_keyframes["quaternion"].values()), key=lambda x: x.frame)
                animations[a.name][bone_name]["quaternion"] = [ [ k.frame, k.interpolation, k.value[1], k.value[2], k.value[3], k.value[0] ] for k in sorted_keyframes ]
    
    return animations


def save(context,
         filepath,
         selection_only = False,
         export_bones = True,
         export_animations = True,
         **kwargs):
    
    out_dir = os.path.dirname(filepath)
    filepath_name = os.path.splitext(os.path.basename(filepath))[0]
    path_animation_data = os.path.join(out_dir, filepath_name + ".data.json")

    objects = bpy.context.selected_objects if selection_only else bpy.context.scene.objects 

    # json data to save
    json_animation_data = {
        "name": filepath_name,   # model name
    }

    # ==================
    # Bones
    # exports bones and exports model as chunks 
    # ==================
    if export_bones:
        bone_tree, bone_object_groups, bbox = build_bone_hierarchy(objects)

        # remove overridden parameters
        chunk_kwargs = kwargs.copy()
        del chunk_kwargs["rescale_to_max"]
        del chunk_kwargs["translate_coords"]
        del chunk_kwargs["recenter_to_origin"]

        # save object parts
        for bone_name, group in bone_object_groups.items():
            # skip empty groups
            if len(group.objects) == 0:
                continue
            
            # get bone origin in blender world space
            bone_origin = group.bone.head_local

            chunk_filepath = os.path.join(out_dir, filepath_name + "." + bone_name + ".json")
            
            print("chunk " + filepath_name + "." + bone_name + ": " + chunk_filepath)

            export_minecraft_json.save_objects(
                chunk_filepath,
                group.objects,
                rescale_to_max=False,
                use_head_upscaling_compensation=True,
                translate_coords=True,
                recenter_to_origin=False,
                blender_origin=bone_origin,
                **chunk_kwargs,
            )
        
        # save to json data export
        json_animation_data["bones"] = bone_tree
    
    # ==================
    # Animations
    # ==================
    if export_animations:
        animations = get_animations()
        json_animation_data["animation"] = animations
    
    # save data
    with open(path_animation_data, "w") as f:
        json.dump(json_animation_data, f)
    
    print("SAVED:", filepath)

    return {"FINISHED"}