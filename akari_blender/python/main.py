import os
import sys
if __name__ == "__main__":
    argv = sys.argv
    if len(argv) < 3:
        print("Usage: main.py <blend> <out_dir>")
        sys.exit(1)
    blend = argv[1]
    args = argv[2:]
    cur_dir = os.path.dirname(__file__)
    blend_abs_path = os.path.abspath(blend)
    os.system(f"blender -b {blend} -P {os.path.join(cur_dir, 'exporter.py')} -- {blend_abs_path} {' '.join(args)}")