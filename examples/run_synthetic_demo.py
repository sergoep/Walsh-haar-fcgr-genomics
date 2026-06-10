"""Small beginner-friendly example."""

from walsh_haar_fcgr.descriptors import DescriptorConfig, theory_aligned_descriptor
from walsh_haar_fcgr.fcgr import fcgr_matrix

sequence = "ACGTGCAATGCCGTTAACCGGTTACGTACGATCGATCGTACGATCGTTAGCA" * 2
config = DescriptorConfig(scales=(3, 4), reverse_complement_average=True)

for k in config.scales:
    result = fcgr_matrix(sequence, k)
    print(f"k={k}: shape={result.matrix.shape}, valid_fraction={result.valid_fraction:.3f}")

features = theory_aligned_descriptor(sequence, config)
print(f"descriptor length: {features.size}")
print("first ten coordinates:", features[:10])
