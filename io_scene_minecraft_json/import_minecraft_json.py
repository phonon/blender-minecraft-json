import os
import json
import numpy as np
import math
from math import inf
import bpy
from mathutils import Vector

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
#   v = 3: vector coordinates (x,y,z)
DIRECTION_NORMALS = np.array([
    [-1.,  0.,  0.],
    [ 0.,  1.,  0.],
    [ 0., -1.,  0.],
    [ 1.,  0.,  0.],
    [ 0.,  0.,  1.],
    [ 0.,  0., -1.],
])
DIRECTION_NORMALS = np.tile(DIRECTION_NORMALS[np.newaxis,...], (6,1,1))


def index_of(val, in_list):
    """Return index of value in in_list"""
    try:
        return in_list.index(val)
    except ValueError:
        return -1 


def merge_dict_properties(dict_original, d):
    """Merge inner dict properties"""
    for k in d:
        if k in dict_original and isinstance(dict_original[k], dict):
            dict_original[k].update(d[k])
        else:
            dict_original[k] = d[k]
    
    return dict_original


def get_full_file_path(filepath_parts, path, merge_point=None):
    """"Typical path formats in json files are like:
            "parent": "block/cube",
            "textures": {
                "top": "block/top"
            }
    This checks in filepath_parts of the blender file,
    matches the base path, then merges the input path, e.g.
        path = "block/cube"
        merge_point = "models"
        filepath_parts = ["C:", "minecraft", "resources", "models", "block", "cobblestone.json"]
                                                             |
                                                     Matched merge point
        
        joined parts = ["C:", "minecraft", "resources", "models"] + ["block", "cube"]
    """
    path_chunks = os.path.split(path)

    # match base path
    if merge_point is not None:
        idx_base_path = index_of(merge_point, filepath_parts)
    else:
        idx_base_path = index_of(path_chunks[0], filepath_parts)

    if idx_base_path != -1:
        # system agnostic path join
        joined_path = os.path.join(os.sep, filepath_parts[0] + os.sep, *filepath_parts[1:idx_base_path+1], *path.split("/"))
        return joined_path
    else:
        return path # failed



def create_textured_principled_bsdf(mat_name, tex_path):
    """Create new material with `mat_name` and texture path `tex_path`
    """
    mat = bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    node_tree = mat.node_tree
    nodes = node_tree.nodes
    bsdf = nodes.get("Principled BSDF") 

    # add texture node
    if bsdf is not None:
        if "Base Color" in bsdf.inputs:
            tex_input = nodes.new(type="ShaderNodeTexImage")
            tex_input.interpolation = "Closest"

            # load image, if fail make a new image with filepath set to tex path
            try:
                img = bpy.data.images.load(tex_path, check_existing=True)
            except:
                print("FAILED TO LOAD IMAGE:", tex_path)
                img = bpy.data.images.new(os.path.split(tex_path)[-1], width=16, height=16)
                img.filepath = tex_path
        
            tex_input.image = img
            node_tree.links.new(tex_input.outputs[0], bsdf.inputs["Base Color"])
        
        # disable shininess
        if "Specular" in bsdf.inputs:
            bsdf.inputs["Specular"].default_value = 0.0
    
    return mat


def load(context,
         filepath,
         import_uvs = True,               # import face uvs
         import_textures = True,          # import textures into materials
         translate_origin_by_8 = False,   # shift model by (-8, -8, -8)
         recenter_to_origin = True,       # recenter model to origin, overrides translate origin
         **kwargs):
    """Main import function"""

    with open(filepath, "r") as f:
        data = json.load(f)
    
    # chunks of import file path, to get base directory
    filepath_parts = filepath.split(os.path.sep)
    
    # check if parent exists, need to merge data into parent
    if "parent" in data:
        # build stack of data dicts, then write backwards:
        # data_hierarchy = [this, parent1, parent2, ...]
        data_hierarchy = [data]
        curr_data = data
        
        # in case circular dependency...
        curr_level = 0
        MAX_PARENT_LEVEL = 10

        # find path models merge point:
        # - normal model location is "assets/models"
        # - if "models" does not exist, check for "model"
        path_models_merge_point = None
        for p in reversed(filepath_parts):
            if p == "models" or p == "model":
                path_models_merge_point = p
                break

        while True:
            # parent path without namespacing, e.g. "minecraft:block/cobblestone" -> "block/cobblestone"
            parent_path = curr_data["parent"].split(":")[-1]
            # parent_path_chunks = os.path.split(parent_path)
            
            # # match base path
            # idx_base_path = index_of(parent_path_chunks[0], filepath_parts)            
            # filepath_parent = os.path.join(*filepath_parts[0:idx_base_path], parent_path + ".json")
            filepath_parent = get_full_file_path(filepath_parts, parent_path, merge_point=path_models_merge_point) + ".json"

            with open(filepath_parent, "r") as f:
                data_parent = json.load(f)
                data_hierarchy.append(data_parent)
                curr_data = data_parent

            curr_level += 1

            if "parent" not in curr_data or curr_level > MAX_PARENT_LEVEL:
                break
    
        # merge together data, need to specially merge inner dict values
        data = {}
        for d in reversed(data_hierarchy):
            data = merge_dict_properties(data, d)

    # main object elements
    elements = data["elements"]

    # check if groups in .json
    # not a minecraft .json spec, used by this exporter + Blockbench
    # as additional data to group models together
    if "groups" in data:
        groups = data["groups"]
    else:
        groups = {}
    
    # objects created
    objects = []

    # model bounding box vector
    model_v_min = np.array([inf, inf, inf])
    model_v_max = np.array([-inf, -inf, -inf])

    # minecraft coordinate system origin
    if translate_origin_by_8:
        minecraft_origin = np.array([8., 8., 8.])
    else:
        # ignore if not translating
        minecraft_origin = np.array([0., 0., 0.])

    # set scene collection as active
    scene_collection = bpy.context.view_layer.layer_collection
    bpy.context.view_layer.active_layer_collection = scene_collection
    
    # re-used buffers for every object
    v_world = np.zeros((3, 8))
    face_normals = np.zeros((6,1,3))

    # =============================================
    # import textures, create map of material name => material
    # =============================================
    """Note two type of texture formats:
        "textures:" {
            "down": "#bottom",                         # texture alias to another texture
            "bottom": "minecraft:block/cobblestone",   # actual texture image
        }

    Loading textures is two pass:
        1. load all actual texture images
        2. map aliases to loaded texture images
    """
    textures = {}
    if import_textures and "textures" in data:
        # get textures path for models, replace "model" or "models" with "textures"
        filepath_textures = filepath_parts
        idx = -1
        for i, p in enumerate(filepath_parts):
            if p == "models" or p == "model":
                idx = i
                break
        if idx != -1:
            filepath_textures[idx] = "textures"
        
        # load texture images
        for tex_name, tex_path in data["textures"].items():
            # skip aliases
            if tex_path[0] == "#":
                continue

            tex_path = tex_path.split(":")[-1] # strip out namespace, like "minecraft:block/name"
            filepath_tex = get_full_file_path(filepath_textures, tex_path, merge_point="textures") + ".png"
            textures[tex_name] = create_textured_principled_bsdf(tex_name, filepath_tex)

        # map texture aliases
        for tex_name, tex_path in data["textures"].items():
            if tex_path[0] == "#":
                tex_path = tex_path[1:]
                if tex_path in textures:
                    textures[tex_name] = textures[tex_path]

    # =============================================
    # import geometry, uvs
    # =============================================
    for i, e in enumerate(elements):
        # get cube min/max
        v_min = np.array([e["from"][2], e["from"][0], e["from"][1]])
        v_max = np.array([e["to"][2], e["to"][0], e["to"][1]])

        # get rotation + origin
        rot = e.get("rotation")
        if rot is not None:
            rot_axis = rot["axis"]
            rot_angle = rot["angle"] * math.pi / 180
            location = np.array([rot["origin"][2], rot["origin"][0], rot["origin"][1]]) - minecraft_origin
            
            if rot_axis == "x":
                rot_euler = (0.0, rot_angle, 0.0)
            if rot_axis == "y":
                rot_euler = (0.0, 0.0, rot_angle)
            if rot_axis == "z":
                rot_euler = (rot_angle, 0.0, 0.0)
        else:
            # default location to center of mass
            location = 0.5 * (v_min + v_max)
            rot_euler = (0.0, 0.0, 0.0)
        
        # create cube
        bpy.ops.mesh.primitive_cube_add(location=location, rotation=rot_euler)
        obj = bpy.context.active_object
        mesh = obj.data
        mesh_materials = {} # tex_name => material_index

        # center local mesh coordiantes
        v_min = v_min - minecraft_origin - location
        v_max = v_max - minecraft_origin - location
        
        # set vertices
        mesh.vertices[0].co[:] = v_min[0], v_min[1], v_min[2]
        mesh.vertices[1].co[:] = v_min[0], v_min[1], v_max[2]
        mesh.vertices[2].co[:] = v_min[0], v_max[1], v_min[2]
        mesh.vertices[3].co[:] = v_min[0], v_max[1], v_max[2]
        mesh.vertices[4].co[:] = v_max[0], v_min[1], v_min[2]
        mesh.vertices[5].co[:] = v_max[0], v_min[1], v_max[2]
        mesh.vertices[6].co[:] = v_max[0], v_max[1], v_min[2]
        mesh.vertices[7].co[:] = v_max[0], v_max[1], v_max[2]

        # set face uvs
        uv = e.get("faces")
        if uv is not None:
            if import_uvs:
                for i, face in enumerate(mesh.polygons):
                    face_normals[i,0,0:3] = face.normal
                
                # map face normal -> face name
                # NOTE: this process may not be necessary since new blender
                # objects are created with the same face normal order,
                # so could directly map index -> minecraft face name.
                # keeping this in case the order changes in future
                face_directions = np.argmax(np.sum(face_normals * DIRECTION_NORMALS, axis=2), axis=1)
                face_directions = DIRECTIONS[face_directions]

                # set uvs face order in blender loop:
                #   2___1    Minecraft space:
                #   |   |      uv_min = 2
                #   |___|      uv_max = 0
                #   3   0
                uv_layer = mesh.uv_layers.active.data
                for uv_direction, face in zip(face_directions, mesh.polygons):
                    face_uv = uv.get(uv_direction)
                    if face_uv is not None:
                        if "uv" in face_uv:
                            # unpack uv coords from minecraft [xmin, ymin, xmax, ymax]
                            # transform from minecraft [0, 16] space +x,-y space to blender [0,1] +x,+y
                            face_uv_coords = face_uv["uv"]
                            xmin = face_uv_coords[0] / 16.0
                            ymin = 1.0 - face_uv_coords[3] / 16.0
                            xmax = face_uv_coords[2] / 16.0
                            ymax = 1.0 - face_uv_coords[1] / 16.0
                        else:
                            xmin = 0.0
                            ymin = 0.0
                            xmax = 1.0
                            ymax = 1.0
                        
                        # swap axes according to the rotation, if specified
                        if "rotation" in face_uv:
                            if face_uv["rotation"] == 90:
                                xmax,ymax = ymax,xmax
                                xmin,ymin = ymin,xmin
                            if face_uv["rotation"] == 180:
                                xmin,xmax = xmax,xmin
                                ymin,ymax = ymax,ymin
                            if face_uv["rotation"] == 270:
                                xmax,ymin = ymin,xmax
                                xmin,ymax = ymax,xmin
                        
                        # write 4 uv face loop coords
                        k = face.loop_start
                        uv_layer[k].uv[0:2] = xmax, ymin
                        uv_layer[k+1].uv[0:2] = xmax, ymax
                        uv_layer[k+2].uv[0:2] = xmin, ymax
                        uv_layer[k+3].uv[0:2] = xmin, ymin

                        # assign material
                        if "texture" in face_uv:
                            tex_name = face_uv["texture"][1:] # remove the "#" in start
                            if tex_name in mesh_materials:
                                face.material_index = mesh_materials[tex_name]
                            elif tex_name in textures: # need new mapping
                                idx = len(obj.data.materials)
                                obj.data.materials.append(textures[tex_name])
                                mesh_materials[tex_name] = idx
                                face.material_index = idx

        # set name (choose whatever is available or "cube" if no name or comment is given)
        obj.name = e.get("name") or e.get("__comment") or "cube"

        # save created object
        objects.append(obj)

        # ================================
        # update global model bounding box
        # ================================
        # get world coordinates
        mat_world = obj.matrix_world
        for i, v in enumerate(mesh.vertices):
            v_world[0:3,i] = mat_world @ v.co
        
        model_v_min = np.amin(np.append(v_world, model_v_min[...,np.newaxis], axis=1), axis=1)
        model_v_max = np.amax(np.append(v_world, model_v_max[...,np.newaxis], axis=1), axis=1)

    # model post-processing
    if recenter_to_origin:
        mean = 0.5 * (model_v_min + model_v_max)
        mean = Vector((mean[0], mean[1], mean[2]))
        for o in objects:
            o.location = o.location - mean
    
    # import groups as collections
    for g in groups:
        name = g["name"]
        if name == "Master Collection":
            continue
        
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
        for index in g["children"]:
            col.objects.link(objects[index])
            bpy.context.scene.collection.objects.unlink(objects[index])
    
    # select newly imported objects
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    for obj in objects:
        obj.select_set(True)
    
    return {"FINISHED"}
