import numpy as np

# sRGB to linear function
def srgb_to_linear(srgb):
	return np.where(
		srgb <= 0.04045,
		srgb / 12.92,
		((srgb + 0.055) / 1.055) ** 2.4,
	)

# Linear to PQ (ITU-R BT.2100)
def linear_to_pq(linear):
	m1 = 0.1593
	m2 = 78.8438
	c1 = 0.8359
	c2 = 18.8516
	c3 = 18.6875

	L = np.clip(linear, 0, 1)  # Ensure values are in [0, 1]
	Lp = (c1 + c2 * (L ** m1)) / (1 + c3 * (L ** m1))
	return np.clip(Lp ** m2, 0, 1)

# Transform matrix from sRGB to Rec. 2020
srgb_to_rec2020 = np.array([
	[0.6274, 0.3293, 0.0433],
	[0.0691, 0.9195, 0.0114],
	[0.0164, 0.0880, 0.8956],
])

# Example conversion function
def convert_srgb_to_rec2020_pq(image_srgb):
	# Decode sRGB to linear
	linear_rgb = srgb_to_linear(image_srgb) * 207 / 10000
	
	# Transform to Rec. 2020 primaries
	rec2020_linear = np.dot(linear_rgb, srgb_to_rec2020.T)
	
	# Encode with PQ
	rec2020_pq = linear_to_pq(rec2020_linear)
	return rec2020_pq

# Inverse PQ function
def pq_to_linear(pq):
	c1 = 0.8359
	c2 = 18.8516
	c3 = 18.6875
	m = 0.1593
	n = 78.8438

	pq = np.clip(pq, 0, 1)  # Clamp PQ input to valid range
	L = ((np.maximum(pq, 0) ** (1 / n)) - c1) / (c2 - c3 * (np.maximum(pq, 0) ** (1 / n)))
	return np.clip(L, 0, 1) ** (1 / m)

# Transformation matrix: Rec. 2020 to sRGB
rec2020_to_srgb = np.array([
	[1.7167, -0.3557, -0.2534],
	[-0.6667,  1.6165,  0.0158],
	[0.0176, -0.0428,  0.9421],
])

# sRGB gamma encoding
def linear_to_srgb(linear):
	return np.where(
		linear <= 0.0031308,
		12.92 * linear,
		1.055 * (linear ** (1 / 2.4)) - 0.055,
	)

# Full conversion: Rec. 2020 PQ to sRGB
def convert_rec2020_pq_to_srgb(image_pq):
	# Step 1: Decode PQ to linear Rec. 2020
	linear_rec2020 = pq_to_linear(image_pq) / 207 * 10000
	
	# Step 2: Convert Rec. 2020 linear to sRGB linear
	linear_srgb = linear_rec2020#np.dot(linear_rec2020, rec2020_to_srgb.T)
	
	# Step 3: Encode sRGB linear to sRGB gamma
	srgb = linear_to_srgb(np.clip(linear_srgb, 0, 1))
	return srgb
