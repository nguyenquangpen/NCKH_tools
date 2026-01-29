import h5py

file_path = "user_prompt_pool.h5"

with h5py.File(file_path, "r") as f:
    def print_structure(name, obj):
        print(name, obj)

    f.visititems(print_structure)


