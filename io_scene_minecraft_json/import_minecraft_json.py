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

def load(context,
         filepath,
         import_uvs = True,               # import face uvs
         translate_origin_by_8 = False,   # shift model by (-8, -8, -8)
         recenter_to_origin = True,       # recenter model to origin, overrides translate origin
         **kwargs):
    
    with open(filepath, "r") as f:
        data = json.load(f)

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
        
        bpy.ops.mesh.primitive_cube_add(location=location, rotation=rot_euler)
        obj = bpy.context.active_object
        mesh = obj.data

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
        if import_uvs:
            uv = e.get("faces")
            if uv is not None:
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
                        # unpack uv coords from minecraft [xmin, ymin, xmax, ymax]
                        # transform from minecraft [0, 16] space +x,-y space to blender [0,1] +x,+y
                        face_uv_coords = face_uv["uv"]
                        xmin = face_uv_coords[0] / 16.0
                        ymin = 1.0 - face_uv_coords[3] / 16.0
                        xmax = face_uv_coords[2] / 16.0
                        ymax = 1.0 - face_uv_coords[1] / 16.0

                        # write 4 uv face loop coords
                        k = face.loop_start
                        uv_layer[k].uv[0:2] = xmax, ymin
                        uv_layer[k+1].uv[0:2] = xmax, ymax
                        uv_layer[k+2].uv[0:2] = xmin, ymax
                        uv_layer[k+3].uv[0:2] = xmin, ymin
        
        # set name
        obj.name = e["name"]

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