There are two main steps to convexify the model of reachy mini for collision detection.
1. Merge all the stls of the head - see head/README.md
2. Cut the body top in two parts - see body_top/README.md 
    - this might not be necessary depending on teh convexification algorithm you use 
    - I found it necessary in my case

Once you have done that, you can convexify the resulting stl files using your favorite convexification algorithm.

#### Requirements
You will need three pip packages:
- trimesh
- coacd
- shapely


#### Convexify command

```bash
python convexify.py -i <input_stl_file> -o <output_stl_file>  <options>
```
See coacd git for more details on the options you can use.


##### Head convexification
After merging the head stls as per head/README.md, you can convexify the resulting
head_one_3dprint.stl file:

```bash
python convexify.py -i head_one_3dprint.stl -o head_one_3dprint_collider  -t 1 -d -dt 70
```
This will create a file:
- head_one_3dprint_collider.stl

Options
    - t 1 : generate 1 convex object 
    - d : simplify the mesh before after convexification
    - dt 70 : maximal number of vertices in the simplified mesh - the higher the number, the more detailed the mesh

##### Body top convexification
After cutting the body top stl as per body_top/README.md, you can convexify
each part separately:

```bash
python convexify.py -i body_top_3dprint_front.stl -o body_top_3dprint_front_collider  -c3 -d -dt 50
```
Options
    - c3 : generate 3 convex objects
    - d : simplify the mesh before after convexification
    - dt 50 : maximal number of vertices in the simplified mesh - the higher the number, the more detailed the mesh

This command will create three files:
- body_top_3dprint_front_collider_0.stl
- body_top_3dprint_front_collider_1.stl
- body_top_3dprint_front_collider_2.stl

```bash
python convexify.py -i body_top_3dprint_back.stl -o body_top_3dprint_back_collider  -c4 -d -dt 50
```
Options
    - c4 : generate 4 convex objects
    - d : simplify the mesh before after convexification
    - dt 50 : maximal number of vertices in the simplified mesh - the higher the number, the more detailed the mesh

This command will create four files:
- body_top_3dprint_back_collider_0.stl
- body_top_3dprint_back_collider_1.stl
- body_top_3dprint_back_collider_2.stl
- body_top_3dprint_back_collider_3.stl



### Using the convexified files

Then put these files together to the `src/reachy_mini/description/reachy_mini/mjcf/assets/collision/coarse/` folder. 
The `fine`folder was generated the same way but without the simplification options (`-d -dt`).
