import numpy as np
from noise import pnoise2


def generate_base_terrain(width, height, scale=500.0):
	terrain = np.zeros((height, width))
	for y in range(height):
		for x in range(width):
			terrain[y][x] = pnoise2(
				x / scale, y / scale, octaves=3, persistence=0.5, lacunarity=2.0
			)
	return terrain * 5.0


def generate_surface_roughness(width, height, scale=5.0):
	roughness = np.zeros((height, width))
	for y in range(height):
		for x in range(width):
			roughness[y][x] = pnoise2(
				x / scale, y / scale, octaves=6, persistence=0.3, lacunarity=3.0
			)

	return roughness * 0.1


def generate_test_terrain(*args, **kwargs):
	test_heightmap = np.zeros((256, 256))
	for i in range(256):
		test_heightmap[i, :] = i * 0.1

	return test_heightmap


def generate_terrain(width, height, scale=50.0):
	return generate_base_terrain(
		width, height, scale
	) + generate_surface_roughness(width, height)
