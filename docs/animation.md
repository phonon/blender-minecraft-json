Animation Export System
======================
This describes the export format for the experimental skeletal
animation system using ArmorStand entity head items.
This is **not** a default Minecraft .json model standard.
This system requires additional server plugins to implement it.

Overview of system:

1. Assign Blender scene mesh objects into **bone groups** based
on their bone with the strongest weight.
2. Export each bone group into separate .json models.
3. Export bone tree structure and animation keyframes separately
into an additional `.data.json` file used by the server.
4. Reconstruct the model in Minecraft using a separate ArmorStand
with a custom item model for each bone group.
5. Animate ArmorStand transforms to simulate skeletal animation.

Below describes the output `.data.json` format.

*(Note: The popular Blockbench program outputs `.animation.json`
files for Minecraft Bedrock animation data. The `.data.json`
convention used here is to avoid name conflict with that.)*


Notes/Limitations
----------------------
- Requires Minecraft server-side plugin to implement this system. 
- Only rigid body bone transforms (position and quaternion),
no bone scaling allowed.
- Animation speed is 20 fps (corresponds to 20 ticks/second update
speed of servers).
- Exporter will transform coordinates from Blender's z-up space
to Minecraft's y-up space.
- Armature Blender origin should be at (0, 0, 0)

Json Structure Overview
----------------------
File: `model_name.data.json`

    {
        "name": "model_name",
        "bones": { ... },
        "animations": { ... }
    }

Bone Tree Structure
----------------------
The `bones` field describes the model's bone tree. Each node in
the tree is a .json object in format:

    {
        "name": "bone",
        "matrix": [
            0.0, -1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            -1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        ],
        "matrix_world": [
            0.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 1.57,
            -1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        ],
        "parent": "body",
        "children": []
    }

Fields:
- **name**: Name of this bone.
- **matrix**: Local bone transform relative to parent's transform, 4x4 matrix in row-major as length 16 array.
- **matrix_world**: World space bone tranform, 4x4 matrix in row-major
as length 16 array.
- **parent**: Name of parent bone.
- **children**: Array of these bone objects.

The first node is always a default-inserted `__root__` node. All parent-level Blender
skeleton bones are inserted as children of this `__root__` bone.
Example of a portion of the bone tree block:

    "bones": {
        "name": "__root__",
        "matrix": [
            0.0, -1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            -1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0],
        "matrix_world": [
            0.0, -1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            -1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0],
        "parent": null,
        "children": [
            {
                "name": "body",
                "matrix": [
                    0.0, -1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    -1.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 1.0],
                "matrix_world": [
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 1.0, 0.0, 0.88,
                    -1.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 1.0],
                "parent": "root",
                "children": [
                    {
                        "name": "head",
                        "matrix": [
                            0.0, -1.0, 0.0, 0.0,
                            0.0, 0.0, 1.0, 0.0,
                            -1.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 1.0],
                        "matrix_world": [
                            0.0, 0.0, 1.0, 0.0,
                            0.0, 1.0, 0.0, 1.58,
                            -1.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 1.0],
                        "parent": "body",
                        "children": []
                    }
                ]
            }
        ]
    }


Animation Structure
----------------------
The animation block consists of a dictionary containing all
Blender animation actions associated with an armature.
Inside each action are keyframe lists for each bone's
position and/or quaternion.

    "animation": {
        "action0": {
            "bone0": {
                "position": [
                    [frame, interpolation, x, y, z],
                    [frame, interpolation, x, y, z],
                    ...
                ],
                "quaternion": [
                    [frame, interpolation, x, y, z, w],
                    [frame, interpolation, x, y, z, w],
                    ...
                ]
            },
            "bone1": { ... },
            "bone2": { ... },
            ...
        },
        "action1": { ... },
        "action2": { ... }
    }

The keyframe formats are:
- **position**: `[frame, interpolation, x, y, z]`
- **quaternion**: `[frame, interpolation, x, y, z, w]`


Example of filled animations:

    "animation": {
        "action0": {
            "body": {
                "position": [
                    [0, "LINEAR", 0.0, 0.0, 0.0],
                    [2, "LINEAR", -0.0438, 0.0, 0.0],
                    [6, "LINEAR", 0.0, 0.0, 0.0],
                    [8, "LINEAR", -0.0438, 0.0, 0.0]
                ],
                "quaternion": [
                    [0, "LINEAR", -1.20e-07, -8.88e-16, 0.50, 0.87],
                    [2, "LINEAR", -1.20e-07, -8.88e-16, 0.46, 0.89],
                    [6, "LINEAR", -1.20e-07, -8.88e-16, 0.50, 0.87],
                    [8, "LINEAR", -1.20e-07, -8.88e-16, 0.46, 0.89]
                ]
            },
            "head": {
                "quaternion": [
                    [0, "LINEAR", -5.85e-15, -1.01e-14, -0.37, 0.93],
                    [2, "LINEAR", 4.27e-10, 2.07e-09, -0.36, 0.93],
                    [6, "LINEAR", -6.74e-15, -1.16e-14, -0.43, 0.90],
                    [8, "LINEAR", 4.27e-10, 2.07e-09, -0.36, 0.93]
                ]
            },
        }
    }