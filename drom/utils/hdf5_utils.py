import os
import drom.demos

def choose_hdf5_from_args(args):
    if args.mimicgen:
        demo_path = os.path.join(drom.demos.hdf5_root, "mimicgen", args.directory)
        if args.dmp:
            dir_path = os.path.join(demo_path, "dmp")
        else:
            dir_path = demo_path
    else:
        demo_path = os.path.join(drom.demos.hdf5_root, args.directory)
        if args.dmp:
            dir_path = os.path.join(demo_path, "dmp")
        else:
            dir_path = demo_path
        
    hdf5_files = [f for f in os.listdir(dir_path) if f.endswith(".hdf5")]
    if not hdf5_files:
        raise FileNotFoundError(f"No .hdf5 files found in {dir_path}")

    print("\nAvailable HDF5 files:")
    for i, file in enumerate(hdf5_files, 1):
        print(f"{i}: {file}")

    selection = input("Select file [default=1]: ").strip()
    idx = int(selection) - 1 if selection.isdigit() else 0
    return os.path.join(dir_path, hdf5_files[idx])