I found that for best convexifying results, I needed to cut the STL in two front and back. Basically at the lowest point of the body top. This way, I could convexify each half separately, and then re-join them.

RUn the following commands from the root of the repository:

```bash
python cut_stl.py -i body_top_3dprint.stl -o body_top_3dprint
```

And it will create two files:
- body_top_3dprint_part1.stl
- body_top_3dprint_part2.stl