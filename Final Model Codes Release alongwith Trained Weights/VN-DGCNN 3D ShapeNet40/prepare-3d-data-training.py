import os
import h5py
import numpy as np

def merge_h5_files(folder_path, output_file):
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.h5')]
    all_files.sort()  # Optional: Ensure consistent order

    merged_data = {}
    
    # Read all datasets into memory
    for file in all_files:
        file_path = os.path.join(folder_path, file)
        with h5py.File(file_path, 'r') as h5f:
            for key in h5f.keys():
                data = h5f[key][:]
                if key not in merged_data:
                    merged_data[key] = [data]
                else:
                    merged_data[key].append(data)

    # Concatenate all collected data
    for key in merged_data:
        merged_data[key] = np.concatenate(merged_data[key], axis=0)

    # Write to a single output HDF5 file
    with h5py.File(output_file, 'w') as h5f_out:
        for key, data in merged_data.items():
            h5f_out.create_dataset(key, data=data)

    print(f"✅ Merged {len(all_files)} files into: {output_file}")

# Example usage
merge_h5_files('test_data', 'shapenet40_test.h5') # Your folder which contains the split h5 files will be mapped here.
