import trimesh
import numpy as np

# arguments parsing
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--inputs",
    type=str,
    required=True,
    help="input model loaded by trimesh. Supported formats: stl"
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    required=True,
    help="output model exported by trimesh. Supported formats: stl"
)

args = parser.parse_args()
input_files = args.inputs

input_files = input_files.split(',')

output_file = args.output
# load input mesh
meshes = [trimesh.load(file, force='mesh') for file in input_files]


scene = trimesh.Scene()
np.random.seed(0)
for p in meshes:    
    p.visual.vertex_colors[:, :3] = (np.random.rand(3) * 255).astype(np.uint8)
    scene.add_geometry(p)
scene.export(output_file)