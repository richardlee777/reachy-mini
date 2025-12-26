In order convexify a head model, I needed to merge multiple stl files together.

Run the following commands from the root of the repository:

```bash
python merge_stls.py -i "head_front_3dprint.stl,head_back_3dprint.stl,neck_reference_3dprint.stl" -o head_one_3dprint.stl
```

This will create a single file:
- head_one_3dprint.stl