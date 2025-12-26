import numpy as np
import trimesh

# arguments parsing
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--input",
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
input_file = args.input
output_file = args.output

# load input mesh
mesh = trimesh.load(input_file, force='mesh')

# cut mesh by plane z=0
plane_origin = [0., -0.015, 0]
plane_normal = np.array([0, 1, 0])
sliced = mesh.slice_plane(plane_origin, plane_normal)
sliced.export(f"{output_file}_part1.stl")


sliced = mesh.slice_plane(plane_origin, -plane_normal)
sliced.export(f"{output_file}_part2.stl")