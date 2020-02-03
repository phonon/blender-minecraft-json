import json
import numpy as np
import math
from math import inf
import bpy
from mathutils import Vector

def load(context,
         filepath,
         translate_origin_by_8 = False,   # shift model by (-8, -8, -8)
         recenter_to_origin = True,       # recenter model to origin, overrides translate origin
         **kwargs):
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    groups = data['groups']
    elements = data['elements']

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
    
    for i, e in enumerate(elements):
        # get cube min/max
        v_min = np.array([e['from'][2], e['from'][0], e['from'][1]])
        v_max = np.array([e['to'][2], e['to'][0], e['to'][1]])

        # get rotation + origin
        rot = e.get('rotation')
        if rot is not None:
            rot_axis = rot['axis']
            rot_angle = rot['angle'] * math.pi / 180
            location = np.array([rot['origin'][2], rot['origin'][0], rot['origin'][1]]) - minecraft_origin
            
            if rot_axis == 'x':
                rot_euler = (0.0, rot_angle, 0.0)
            if rot_axis == 'y':
                rot_euler = (0.0, 0.0, rot_angle)
            if rot_axis == 'z':
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

        # set name
        obj.name = e['name']

        # save created object
        objects.append(obj)

        # ================================
        # update global model bounding box
        # ================================
        # get world coordinates
        mat_world = obj.matrix_world
        v_world = np.zeros((3, 8))
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
        name = g['name']
        if name == 'Master Collection':
            continue
        
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
        for index in g['children']:
            col.objects.link(objects[index])
            bpy.context.scene.collection.objects.unlink(objects[index])
    
    return {'FINISHED'}