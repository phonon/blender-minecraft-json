Blender Minecraft JSON Import/Export
=======================================
Import/export cuboid geometry between Blender and Minecraft .json model format. The Blender model must follow very specific restrictions for the exporter to work (read **Export Guide** below).

Supports import/export uvs and can also auto-generate and export solid material colors packed into an image texture.

Tested on Blender 2.83.


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
- **Recalculate normals if textures are on wrong faces.** Applying negative scales can sometimes flip face normals inside, which causes incorrect auto-generated texture colors. Recalculate normals on all meshes to fix (`Select All` > `Edit Mode` > `Select All` > `ctrl + shift + N` to recalculate and **uncheck inside**).

Example `.blend` files and generated `.json` models are in **examples** folder.


Export Options
---------------------------------------
|  Option  |  Default   | Description  |
|----------|------------|------------- |
| Selection Only | False | If True, export only selected objects|
| Recenter Coordinates | True | Recenters Blender origin `(0,0,0)` to Minecraft origin `(8,8,8)`|
| Rescale to Max | True | Rescale exported model so the largest axis fits the 48x48x48 Minecraft model volume.
| Recenter Model to Origin | True | Recenter model so its center is at the origin.
| Texture Subfolder | "item" | Subfolder for model texture path: `assets/minecraft/textures/[subfolder]` |
| Texture Name | | Name of texture generated `[name].png` (blank defaults to `.json` filename) |
| Export UVs | True | Exports face UVs |
| Generate Color Texture| False | Auto-textures solid material colors and generates a `.png` image texture exported alongside model (overwrites UVs). By default will get colors from all materials in the Blender file. |
| Only Use Exported Object Colors | False | When exporting auto-generated color texture, only use materials colors on exported objects (instead of all materials in file). |
| Minify .json | False | Enable options to reduce .json file size |
| Decimal Precision | 8 | Number of digits after decimal point in output .json (-1 to disable) |
| Export Bones | False | Export model in chunks by associated bones (experimental) |
| Export Animations | False | Export bone animations into a separate .json file (see docs, experimental) |


Import Options
---------------------------------------
|  Option  |  Default   | Description  |
|----------|------------|------------- |
| Import UVs | True | Import face UVs |
| Translate by (-8, -8, -8) | False | Fixed `(-8,-8,-8)` translation to convert from Minecraft `[-16,32]` space |
| Recenter to Origin | True | Centers model on Blender world origin (overrides translate option) |


Minecraft Geometry Restrictions Overview
---------------------------------------
Detailed overview: https://minecraft.gamepedia.com/Model
- Uses a Y-up coordinate system (Blender uses Z-up, exporter handles the conversion).
- Coordinates must be from [-16, 32]
- Single axis rotations with 5 possible values: [-45, -22.5, 0, 22.5, 45]
- Minecraft UV axis is top-left (0, 0) to bottom-right (16, 16) while Blender is bottom-left (0, 0) to top-right (1, 1)  
- Each face UVs specified as in format [xmin, ymin, xmax, ymax] where each value in range [0, 16]

TODO
---------------------------------------
- Collection hierarchy export (currently no nested collections, only immediate collection exported)
- Exporting multiple textures on objects