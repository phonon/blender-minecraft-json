Blender Minecraft JSON Import/Export
=======================================
Import/export cuboid geometry between Blender and Minecraft .json model format. The Blender model must follow very specific restrictions for the exporter to work (read **Export Guide** below).

Currently can auto-generate and export solid material colors packed into an image texture. However, **there is no full UV editing yet.**

Tested on Blender 2.81a.


Installation
---------------------------------------
1. `git clone` or copy this repository into your `scripts/addons` or custom scripts folder.
2. Enable in **Edit > Preferences > Add-ons** (search for *Minecraft JSON Import/Export*)


Export Guide 
---------------------------------------
- **Only exports cuboid objects** e.g. Object meshes must be rectangular prisms (8 vertices and 6 faces). The local mesh coordinates must be aligned to global XYZ axis. **Do not rotate the mesh vertices in edit mode.**
- **All cuboids must be separate objects.**
- **Apply rotations at the Object level (outside of mesh edit).** Only one axis can have a **net rotation** with 5 possible degree values: [-45, -22.5, 0, 22.5, 45]. What **net rotation** means is that the object can have any multiple of 90 deg rotations on any axis (because the object is still a cuboid), but only one axis can have a further rotation of values listed. Objects with invalid rotations will be rounded to their closest valid rotation. Examples of valid/invalid rotations:

|x   | y   | z  |       |
|----|---- |----|------ |
|0   |22.5 |0   | valid |
|90  |270  |45  | valid |
|90  |270  |135 | valid |
|45  |135  |0   | invalid (two net rotations) |
|30  |0    |0   | invalid (value not allowed) |

- **Apply all scale to objects before exporting.** Use `ctrl + A` to bring up Apply menu then hit `Apply > Scale`. (Also found in `Object > Apply > Scale` tab in viewport)

Example `.blend` files and generated `.json` models are in **examples** folder.


Export Options
---------------------------------------
|  Option  |  Default   | Description  |
|----------|------------|------------- |
| Selection Only | False | If True, export only selected objects|
| Recenter Coordinates | True | Recenters Blender origin `(0,0,0)` to Minecraft origin `(8,8,8)`|
| Rescale to Max | True | Rescale exported model so the largest axis fits the 48x48x48 Minecraft model volume. (Models can be scaled down but not up ingame.)
| Generate Color Texture| True | Auto-textures solid material colors and generates a `.png` image texture exported alongside model |
| Texture Subfolder | "item" | Subfolder for model texture path: `assets/minecraft/textures/[subfolder]` |
| Texture Name | | Name of texture generated `[name].png` (blank defaults to `.json` filename) |


Import Options
---------------------------------------
|  Option  |  Default   | Description  |
|----------|------------|------------- |
| Translate by (-8, -8, -8) | False | Fixed `(-8,-8,-8)` translation to convert from Minecraft `[-16,32]` space |
| Recenter to Origin | True | Centers model on Blender world origin (overrides translate option) |


Minecraft Geometry Restrictions Overview
---------------------------------------
Detailed overview: https://minecraft.gamepedia.com/Model
- Uses a Y-up coordinate system (Blender uses Z-up, exporter handles the conversion).
- Coordinates must be from [-16, 32]
- Single axis rotations with 5 possible values: [-45, -22.5, 0, 22.5, 45]
- Each face UVs specified as % of texture image area


TODO
---------------------------------------
- Collection hierarchy export (currently no nested collections, only immediate collection exported)
- Texture/UV import/export
