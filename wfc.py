import numpy as np
import random
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set
import pybullet as p
from noise import pnoise2


class TerrainType(Enum):
	GRASS = 0
	FOREST = 1
	DIRT_PATH = 2
	PAVED_ROAD = 3
	STREAM = 4
	STEEP_HILL = 5
	VALLEY = 6
	ROCK = 7


@dataclass
class TerrainProperties:
	height_base: float  # Base height offset
	height_noise: float  # Noise amplitude
	friction: float  # Traction coefficient
	color: Tuple[float, float, float]  # RGB color for visualization


# Define terrain properties
TERRAIN_PROPS = {
	TerrainType.GRASS: TerrainProperties(0.0, 0.1, 0.8, (0.2, 0.6, 0.2)),
	TerrainType.FOREST: TerrainProperties(0.2, 0.3, 0.6, (0.1, 0.4, 0.1)),
	TerrainType.DIRT_PATH: TerrainProperties(0.0, 0.05, 1.0, (0.6, 0.4, 0.2)),
	TerrainType.PAVED_ROAD: TerrainProperties(0.0, 0.02, 1.2, (0.3, 0.3, 0.3)),
	TerrainType.STREAM: TerrainProperties(-0.3, 0.1, 0.2, (0.2, 0.4, 0.8)),
	TerrainType.STEEP_HILL: TerrainProperties(1.0, 0.4, 0.7, (0.5, 0.4, 0.3)),
	TerrainType.VALLEY: TerrainProperties(-0.5, 0.2, 0.8, (0.3, 0.5, 0.3)),
	TerrainType.ROCK: TerrainProperties(0.5, 0.6, 0.5, (0.6, 0.6, 0.6)),
}

# Adjacency rules - what terrain types can be next to each other
ADJACENCY_RULES = {
	TerrainType.GRASS: {
		TerrainType.GRASS,
		TerrainType.FOREST,
		TerrainType.DIRT_PATH,
		TerrainType.STEEP_HILL,
		TerrainType.VALLEY,
		TerrainType.ROCK,
	},
	TerrainType.FOREST: {
		TerrainType.GRASS,
		TerrainType.FOREST,
		TerrainType.DIRT_PATH,
		TerrainType.STEEP_HILL,
		TerrainType.ROCK,
	},
	TerrainType.DIRT_PATH: {
		TerrainType.GRASS,
		TerrainType.FOREST,
		TerrainType.DIRT_PATH,
		TerrainType.PAVED_ROAD,
		TerrainType.VALLEY,
	},
	TerrainType.PAVED_ROAD: {
		TerrainType.DIRT_PATH,
		TerrainType.PAVED_ROAD,
		TerrainType.GRASS,
	},
	TerrainType.STREAM: {
		TerrainType.GRASS,
		TerrainType.VALLEY,
		TerrainType.STREAM,
	},
	TerrainType.STEEP_HILL: {
		TerrainType.GRASS,
		TerrainType.FOREST,
		TerrainType.ROCK,
		TerrainType.STEEP_HILL,
	},
	TerrainType.VALLEY: {
		TerrainType.GRASS,
		TerrainType.STREAM,
		TerrainType.DIRT_PATH,
		TerrainType.VALLEY,
	},
	TerrainType.ROCK: {
		TerrainType.STEEP_HILL,
		TerrainType.ROCK,
		TerrainType.GRASS,
		TerrainType.FOREST,
	},
}


class WFCTerrainGenerator:
	def __init__(self, width: int, height: int):
		self.width = width
		self.height = height
		self.grid = np.full((height, width), None, dtype=object)
		self.possible_states = {}
		self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # N, E, S, W

		# Initialize all cells with all possible states
		for y in range(height):
			for x in range(width):
				self.possible_states[(x, y)] = set(TerrainType)

	def get_entropy(self, x: int, y: int) -> int:
		"""Get entropy (number of possible states) for a cell"""
		if self.grid[y, x] is not None:
			return 0  # Already collapsed
		return len(self.possible_states[(x, y)])

	def get_lowest_entropy_cell(self) -> Tuple[int, int]:
		"""Find cell with lowest entropy > 0"""
		min_entropy = float("inf")
		candidates = []

		for y in range(self.height):
			for x in range(self.width):
				entropy = self.get_entropy(x, y)
				if 0 < entropy < min_entropy:
					min_entropy = entropy
					candidates = [(x, y)]
				elif entropy == min_entropy and entropy > 0:
					candidates.append((x, y))

		return random.choice(candidates) if candidates else None

	def collapse_cell(self, x: int, y: int):
		"""Collapse a cell to a specific state"""
		possible = list(self.possible_states[(x, y)])

		# Weight probabilities based on terrain type
		weights = self._get_terrain_weights(x, y, possible)
		chosen_type = random.choices(possible, weights=weights)[0]

		self.grid[y, x] = chosen_type
		self.possible_states[(x, y)] = {chosen_type}

	def _get_terrain_weights(
		self, x: int, y: int, possible: List[TerrainType]
	) -> List[float]:
		"""Get weighted probabilities for terrain types based on position and context"""
		weights = []

		for terrain_type in possible:
			weight = 1.0

			# Edge biases
			if x == 0 or x == self.width - 1 or y == 0 or y == self.height - 1:
				if terrain_type == TerrainType.GRASS:
					weight *= 2.0
				elif terrain_type == TerrainType.PAVED_ROAD:
					weight *= 0.1

			# Stream preference for lower areas (conceptually)
			if terrain_type == TerrainType.STREAM:
				if y > self.height * 0.6:  # Lower part of map
					weight *= 3.0
				else:
					weight *= 0.3

			# Hills prefer higher areas
			if terrain_type == TerrainType.STEEP_HILL:
				if y < self.height * 0.4:  # Upper part of map
					weight *= 2.0

			# Roads prefer to be continuous (check neighbors)
			if terrain_type == TerrainType.PAVED_ROAD:
				road_neighbors = self._count_neighbor_type(
					x, y, TerrainType.PAVED_ROAD
				)
				if road_neighbors > 0:
					weight *= 3.0
				else:
					weight *= 0.5

			weights.append(weight)

		return weights

	def _count_neighbor_type(
		self, x: int, y: int, terrain_type: TerrainType
	) -> int:
		"""Count how many neighbors are of a specific type"""
		count = 0
		for dx, dy in self.directions:
			nx, ny = x + dx, y + dy
			if 0 <= nx < self.width and 0 <= ny < self.height:
				if self.grid[ny, nx] == terrain_type:
					count += 1
		return count

	def propagate_constraints(self, x: int, y: int):
		"""Update possible states of neighboring cells based on collapsed cell"""
		stack = [(x, y)]

		while stack:
			cx, cy = stack.pop()
			current_type = self.grid[cy, cx]

			if current_type is None:
				continue

			# Check all neighbors
			for dx, dy in self.directions:
				nx, ny = cx + dx, cy + dy

				if not (0 <= nx < self.width and 0 <= ny < self.height):
					continue

				if self.grid[ny, nx] is not None:
					continue  # Already collapsed

				# Remove impossible states from neighbor
				old_possible = self.possible_states[(nx, ny)].copy()
				new_possible = set()

				for possible_type in old_possible:
					if current_type in ADJACENCY_RULES[possible_type]:
						new_possible.add(possible_type)

				self.possible_states[(nx, ny)] = new_possible

				# If we reduced possibilities, add to stack for further propagation
				if (
					len(new_possible) < len(old_possible)
					and len(new_possible) > 0
				):
					stack.append((nx, ny))
				elif len(new_possible) == 0:
					# Contradiction! Handle by backtracking or restart
					print(f"Contradiction at ({nx}, {ny})")

	def generate(self) -> np.ndarray:
		"""Generate terrain using WFC algorithm"""
		iteration = 0
		max_iterations = self.width * self.height

		while iteration < max_iterations:
			# Find cell with lowest entropy
			cell = self.get_lowest_entropy_cell()
			if cell is None:
				break  # All cells collapsed

			x, y = cell

			# Collapse the cell
			self.collapse_cell(x, y)

			# Propagate constraints
			self.propagate_constraints(x, y)

			iteration += 1

		return self.grid

	def to_heightmap(
		self, detail_scale: float = 0.1
	) -> Tuple[np.ndarray, np.ndarray]:
		"""Convert terrain grid to heightmap and friction map"""
		heightmap = np.zeros((self.height, self.width))
		friction_map = np.zeros((self.height, self.width))

		for y in range(self.height):
			for x in range(self.width):
				terrain_type = self.grid[y, x]
				if terrain_type is None:
					terrain_type = TerrainType.GRASS  # Default fallback

				props = TERRAIN_PROPS[terrain_type]

				# Base height
				base_height = props.height_base

				# Add noise for surface detail
				noise_value = pnoise2(
					x * detail_scale,
					y * detail_scale,
					octaves=4,
					persistence=0.5,
					lacunarity=2.0,
				)
				surface_height = base_height + (
					noise_value * props.height_noise
				)

				heightmap[y, x] = surface_height
				friction_map[y, x] = props.friction

		# Smooth transitions between different terrain types
		heightmap = self._smooth_heightmap(heightmap)

		return heightmap, friction_map

	def _smooth_heightmap(
		self, heightmap: np.ndarray, iterations: int = 2
	) -> np.ndarray:
		"""Apply smoothing to reduce harsh transitions"""
		kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0

		smoothed = heightmap.copy()
		for _ in range(iterations):
			# Apply convolution manually to avoid scipy dependency
			temp = np.zeros_like(smoothed)
			for y in range(1, self.height - 1):
				for x in range(1, self.width - 1):
					region = smoothed[y - 1 : y + 2, x - 1 : x + 2]
					temp[y, x] = np.sum(region * kernel)
			smoothed = temp

		return smoothed


def create_wfc_terrain(
	width: int = 128, height: int = 128
) -> Tuple[int, np.ndarray]:
	"""Create a PyBullet terrain using WFC algorithm"""

	# Generate terrain
	generator = WFCTerrainGenerator(width, height)
	terrain_grid = generator.generate()

	# Convert to heightmap and friction map
	heightmap, friction_map = generator.to_heightmap()

	# Create PyBullet terrain
	terrain_shape = p.createCollisionShape(
		shapeType=p.GEOM_HEIGHTFIELD,
		meshScale=[0.1, 0.1, 1.0],
		heightfieldData=heightmap.flatten(),
		heightfieldTextureScaling=width - 1,
		numHeightfieldRows=height,
		numHeightfieldColumns=width,
	)

	terrain_body = p.createMultiBody(
		baseMass=0,
		baseCollisionShapeIndex=terrain_shape,
		basePosition=[0, 0, 0],
	)

	return terrain_body, friction_map


if __name__ == "__main__":
	p.connect(p.GUI)

	terrain_body, friction_map = create_wfc_terrain(256, 256)

	input()
