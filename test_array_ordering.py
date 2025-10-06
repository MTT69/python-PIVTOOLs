"""
Simple diagnostic to understand array indexing.
"""
import numpy as np

# Simulate our setup
n_x = 15  # windows along width (x-direction)
n_y = 17  # windows along height (y-direction)

# Create a test array in Fortran order like C library returns
test_array_f = np.zeros((n_x, n_y), dtype=np.float32, order='F')

# Fill with a pattern that makes orientation obvious
# Pattern: value = ii * 100 + jj where ii is x-index, jj is y-index  
for ii in range(n_x):
    for jj in range(n_y):
        test_array_f[ii, jj] = ii * 100 + jj

print("="*80)
print("FORTRAN-ORDER ARRAY (as returned by C library)")
print("="*80)
print(f"Shape: {test_array_f.shape} = (n_x={n_x}, n_y={n_y})")
print(f"Memory order: {'F' if test_array_f.flags['F_CONTIGUOUS'] else 'C'}")
print()
print("Corner values (value = ii*100 + jj):")
print(f"  test_array_f[0, 0] = {test_array_f[0, 0]} (ii=0, jj=0, bottom-left window)")
print(f"  test_array_f[{n_x-1}, 0] = {test_array_f[n_x-1, 0]} (ii={n_x-1}, jj=0, bottom-right window)")
print(f"  test_array_f[0, {n_y-1}] = {test_array_f[0, n_y-1]} (ii=0, jj={n_y-1}, top-left window)")
print(f"  test_array_f[{n_x-1}, {n_y-1}] = {test_array_f[n_x-1, n_y-1]} (ii={n_x-1}, jj={n_y-1}, top-right window)")
print()

# Now transpose
test_array_t = test_array_f.T
print("="*80)
print("AFTER TRANSPOSE: test_array_f.T")
print("="*80)
print(f"Shape: {test_array_t.shape} = (n_y={n_y}, n_x={n_x})")
print(f"Memory order: {'F' if test_array_t.flags['F_CONTIGUOUS'] else 'C'}")
print()
print("Corner values:")
print(f"  test_array_t[0, 0] = {test_array_t[0, 0]} (was test_array_f[0, 0])")
print(f"  test_array_t[0, {n_x-1}] = {test_array_t[0, n_x-1]} (was test_array_f[{n_x-1}, 0])")
print(f"  test_array_t[{n_y-1}, 0] = {test_array_t[n_y-1, 0]} (was test_array_f[0, {n_y-1}])")
print(f"  test_array_t[{n_y-1}, {n_x-1}] = {test_array_t[n_y-1, n_x-1]} (was test_array_f[{n_x-1}, {n_y-1}])")
print()

# What about double transpose?
test_array_tt = test_array_t.T
print("="*80)
print("AFTER DOUBLE TRANSPOSE: test_array_f.T.T")
print("="*80)
print(f"Shape: {test_array_tt.shape} = (n_x={n_x}, n_y={n_y})")
print()
print("Corner values:")
print(f"  test_array_tt[0, 0] = {test_array_tt[0, 0]}")
print(f"  test_array_tt[{n_x-1}, 0] = {test_array_tt[n_x-1, 0]}")
print(f"  test_array_tt[0, {n_y-1}] = {test_array_tt[0, n_y-1]}")
print(f"  test_array_tt[{n_x-1}, {n_y-1}] = {test_array_tt[n_x-1, n_y-1]}")
print()

# Print a small section to see the pattern
print("="*80)
print("VISUAL COMPARISON")
print("="*80)
print("\nOriginal Fortran array [0:3, 0:5] (first 3 x-indices, first 5 y-indices):")
print(test_array_f[0:3, 0:5])
print("\nTransposed [0:5, 0:3] (first 5 rows, first 3 cols):")
print(test_array_t[0:5, 0:3])
print()
print("Interpretation:")
print("- If we want arr[i,j] where i=row (y-position), j=col (x-position)")
print("- And C library gives us arr_c[ii, jj] where ii=x-index, jj=y-index")
print("- Then arr[i,j] = arr_c.T[i,j] = arr_c[j, i]")
print("- This means we're correctly mapping (x,y) → (col,row) = (j,i)")
