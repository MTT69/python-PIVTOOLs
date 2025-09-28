import os

import scipy.io

# Path to the original .mat file
src_path = (
    "/Users/morgan/Downloads/bailey/calibrated_piv/500/Cam2/instantaneous/Co_ords.mat"
)

# Load the .mat file
mat_data = scipy.io.loadmat(src_path)

# Extract the struct
co_ords_struct = mat_data["Co_ords"]

# Rename the variable
coordinates = co_ords_struct

# Prepare new file path
dst_path = os.path.join(os.path.dirname(src_path), "coordinates.mat")

# Save the new variable
scipy.io.savemat(dst_path, {"coordinates": coordinates})
