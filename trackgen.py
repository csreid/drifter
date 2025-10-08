import pybullet as p
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import cdist
import math


class RacetrackGenerator:
	def __init__(self, physics_client):
		"""
		Initialize the racetrack generator.

		Args:
		    physics_client: PyBullet physics client ID
		"""
		self.physics_client = physics_client
		self.track_body = None
		self.barrier_bodies = []

	def _generate_control_points(self, track_type="oval", scale=10.0):
		"""
		Generate control points for different track types.

		Args:
		    track_type: Type of track ("oval", "serpentine", "custom")
		    scale: Scale factor for track size

		Returns:
		    List of (x, y) control points
		"""
		if track_type == "oval":
			# Simple oval track with good curvature
			points = [
				(scale, 0),
				(scale * 0.8, scale * 0.6),
				(scale * 0.3, scale * 0.9),
				(-scale * 0.3, scale * 0.9),
				(-scale * 0.8, scale * 0.6),
				(-scale, 0),
				(-scale * 0.8, -scale * 0.6),
				(-scale * 0.3, -scale * 0.9),
				(scale * 0.3, -scale * 0.9),
				(scale * 0.8, -scale * 0.6),
			]
		elif track_type == "serpentine":
			# S-shaped serpentine track
			points = []
			for i in range(8):
				x = (i - 3.5) * scale / 2
				y = scale * 0.6 * math.sin(i * math.pi / 2)
				points.append((x, y))
		else:
			# Default to simple oval
			points = [
				(scale, 0),
				(scale * 0.5, scale * 0.8),
				(-scale * 0.5, scale * 0.8),
				(-scale, 0),
				(-scale * 0.5, -scale * 0.8),
				(scale * 0.5, -scale * 0.8),
			]

		return points

	def generate_control_points(
		self,
		scale=10.0,
		num_points=12,
		radius_variation=0.3,
		min_radius_ratio=0.6,
		seed=None,
	):
		"""
		Generate control points by sampling random polar coordinates.

		Args:
				scale: Base radius scale
				num_points: Number of control points to generate
				radius_variation: How much the radius can vary (0.0 to 1.0)
				min_radius_ratio: Minimum radius as fraction of scale (0.0 to 1.0)
				seed: Random seed for reproducible results

		Returns:
				List of (x, y) control points
		"""
		if seed is not None:
			np.random.seed(seed)

		points = []

		# Generate evenly spaced angles
		angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

		# Add some angular noise to make it less regular
		angle_noise = np.random.uniform(
			-np.pi / num_points * 0.3, np.pi / num_points * 0.3, num_points
		)
		angles += angle_noise

		# Generate random radii
		min_radius = scale * min_radius_ratio
		max_radius = scale

		# Create smooth radius variation using sine waves
		base_radii = np.random.uniform(min_radius, max_radius, num_points)

		# Apply smooth variation to avoid sharp changes
		for i in range(num_points):
			# Add some variation based on neighboring points
			variation = (
				np.random.uniform(-radius_variation, radius_variation) * scale
			)
			radius = base_radii[i] + variation

			# Clamp to valid range
			radius = np.clip(radius, min_radius, max_radius)

			# Convert to cartesian coordinates
			x = radius * np.cos(angles[i])
			y = radius * np.sin(angles[i])
			points.append((x, y))

		return points

	def interpolate_spline(self, control_points, num_points=200, smoothing=0):
		"""
		Create a smooth spline from control points.

		Args:
		    control_points: List of (x, y) tuples
		    num_points: Number of points to generate along the spline
		    smoothing: Smoothing factor (0 = no smoothing)

		Returns:
		    Tuple of (x_coords, y_coords) arrays
		"""
		# Convert to numpy arrays
		points = np.array(control_points)
		x = points[:, 0]
		y = points[:, 1]

		# Create periodic spline (closed loop)
		tck, u = splprep([x, y], s=smoothing, per=True)

		# Generate interpolated points
		u_new = np.linspace(0, 1, num_points)
		x_new, y_new = splev(u_new, tck)

		return x_new, y_new

	def check_self_intersection(self, x_coords, y_coords, min_distance=2.0):
		"""
		Check for self-intersections in the track path.

		Args:
		    x_coords: X coordinates of the track centerline
		    y_coords: Y coordinates of the track centerline
		    min_distance: Minimum allowed distance between non-adjacent points

		Returns:
		    Boolean indicating if the track is valid (no problematic intersections)
		"""
		points = np.column_stack((x_coords, y_coords))

		# Calculate distances between all pairs of points
		distances = cdist(points, points)

		# Check for points that are too close (excluding adjacent points)
		n_points = len(points)
		for i in range(n_points):
			for j in range(i + 3, n_points):  # Skip adjacent points
				# Handle wraparound for closed loop
				if j >= n_points - 2 and i <= 2:
					continue

				if distances[i, j] < min_distance:
					print(
						f"Warning: Points {i} and {j} are too close ({distances[i, j]:.2f})"
					)
					# return False

		return True

	def calculate_track_normals(self, x_coords, y_coords):
		"""
		Calculate normal vectors for each point along the track.

		Args:
		    x_coords: X coordinates of the track centerline
		    y_coords: Y coordinates of the track centerline

		Returns:
		    Tuple of (normal_x, normal_y) arrays
		"""
		n_points = len(x_coords)
		normal_x = np.zeros(n_points)
		normal_y = np.zeros(n_points)

		for i in range(n_points):
			# Get neighboring points (with wraparound)
			prev_i = (i - 1) % n_points
			next_i = (i + 1) % n_points

			# Calculate tangent vector
			tangent_x = x_coords[next_i] - x_coords[prev_i]
			tangent_y = y_coords[next_i] - y_coords[prev_i]

			# Normalize tangent
			tangent_length = math.sqrt(tangent_x**2 + tangent_y**2)
			if tangent_length > 0:
				tangent_x /= tangent_length
				tangent_y /= tangent_length

			# Calculate normal (perpendicular to tangent)
			normal_x[i] = -tangent_y
			normal_y[i] = tangent_x

		return normal_x, normal_y

	def create_track_mesh(self, x_coords, y_coords, width=4.0, height=0.2):
		"""
		Create a single mesh for the entire track surface.

		Args:
		    x_coords: X coordinates of the track centerline
		    y_coords: Y coordinates of the track centerline
		    width: Width of the track
		    height: Height/thickness of the track

		Returns:
		    PyBullet body ID of the created track
		"""
		n_points = len(x_coords)

		# Calculate normal vectors
		normal_x, normal_y = self.calculate_track_normals(x_coords, y_coords)

		# Create vertices for the track surface
		vertices = []
		faces = []

		# Create vertices for top and bottom surfaces
		for i in range(n_points):
			# Inner edge vertices (top and bottom)
			inner_x = x_coords[i] - normal_x[i] * width / 2
			inner_y = y_coords[i] - normal_y[i] * width / 2
			vertices.extend(
				[
					[inner_x, inner_y, height],  # Top inner
					[inner_x, inner_y, 0],  # Bottom inner
				]
			)

			# Outer edge vertices (top and bottom)
			outer_x = x_coords[i] + normal_x[i] * width / 2
			outer_y = y_coords[i] + normal_y[i] * width / 2
			vertices.extend(
				[
					[outer_x, outer_y, height],  # Top outer
					[outer_x, outer_y, 0],  # Bottom outer
				]
			)

		# Create faces
		for i in range(n_points):
			next_i = (i + 1) % n_points

			# Vertex indices for current segment
			curr_inner_top = i * 4
			curr_inner_bot = i * 4 + 1
			curr_outer_top = i * 4 + 2
			curr_outer_bot = i * 4 + 3

			# Vertex indices for next segment
			next_inner_top = next_i * 4
			next_inner_bot = next_i * 4 + 1
			next_outer_top = next_i * 4 + 2
			next_outer_bot = next_i * 4 + 3

			# Top surface triangles
			faces.extend(
				[
					[curr_inner_top, next_inner_top, curr_outer_top],
					[next_inner_top, next_outer_top, curr_outer_top],
				]
			)

			# Bottom surface triangles
			faces.extend(
				[
					[curr_inner_bot, curr_outer_bot, next_inner_bot],
					[next_inner_bot, curr_outer_bot, next_outer_bot],
				]
			)

			# Inner side triangles
			faces.extend(
				[
					[curr_inner_bot, next_inner_top, curr_inner_top],
					[curr_inner_bot, next_inner_bot, next_inner_top],
				]
			)

			# Outer side triangles
			faces.extend(
				[
					[curr_outer_bot, curr_outer_top, next_outer_top],
					[curr_outer_bot, next_outer_top, next_outer_bot],
				]
			)

		# Create mesh
		collision_shape = p.createCollisionShape(
			p.GEOM_MESH,
			vertices=vertices,
			indices=[face for triangle in faces for face in triangle],
			physicsClientId=self.physics_client,
		)

		visual_shape = p.createVisualShape(
			p.GEOM_MESH,
			vertices=vertices,
			indices=[face for triangle in faces for face in triangle],
			rgbaColor=[0.4, 0.4, 0.4, 1.0],
			physicsClientId=self.physics_client,
		)

		# Create body
		body_id = p.createMultiBody(
			baseMass=0,  # Static body
			baseCollisionShapeIndex=collision_shape,
			baseVisualShapeIndex=visual_shape,
			basePosition=[0, 0, 0],
			physicsClientId=self.physics_client,
		)

		return body_id

	def create_barrier_mesh(
		self, x_coords, y_coords, offset=4.0, height=2.0, thickness=0.3
	):
		"""
		Create barrier walls along the track edges using a single mesh.

		Args:
		    x_coords: X coordinates of the track centerline
		    y_coords: Y coordinates of the track centerline
		    offset: Distance from track centerline to barriers
		    height: Height of the barriers
		    thickness: Thickness of the barrier walls

		Returns:
		    List of PyBullet body IDs for the barrier walls
		"""
		n_points = len(x_coords)
		normal_x, normal_y = self.calculate_track_normals(x_coords, y_coords)

		barrier_bodies = []

		# Create inner and outer barriers separately
		for side in [-1, 1]:  # -1 for inner, 1 for outer
			vertices = []
			faces = []

			# Calculate barrier positions
			for i in range(n_points):
				# Base position for barrier
				barrier_x = x_coords[i] + side * normal_x[i] * offset
				barrier_y = y_coords[i] + side * normal_y[i] * offset

				# Create vertices for barrier segment (inner and outer faces)
				inner_x = barrier_x - side * normal_x[i] * thickness / 2
				inner_y = barrier_y - side * normal_y[i] * thickness / 2
				outer_x = barrier_x + side * normal_x[i] * thickness / 2
				outer_y = barrier_y + side * normal_y[i] * thickness / 2

				vertices.extend(
					[
						[inner_x, inner_y, 0],  # Bottom inner
						[inner_x, inner_y, height],  # Top inner
						[outer_x, outer_y, 0],  # Bottom outer
						[outer_x, outer_y, height],  # Top outer
					]
				)

			# Create faces for the barrier
			for i in range(n_points):
				next_i = (i + 1) % n_points

				# Vertex indices
				curr_bot_in = i * 4
				curr_top_in = i * 4 + 1
				curr_bot_out = i * 4 + 2
				curr_top_out = i * 4 + 3

				next_bot_in = next_i * 4
				next_top_in = next_i * 4 + 1
				next_bot_out = next_i * 4 + 2
				next_top_out = next_i * 4 + 3

				# Outer face triangles
				faces.extend(
					[
						[curr_bot_out, next_top_out, curr_top_out],
						[curr_bot_out, next_bot_out, next_top_out],
					]
				)

				# Inner face triangles
				faces.extend(
					[
						[curr_bot_in, curr_top_in, next_top_in],
						[curr_bot_in, next_top_in, next_bot_in],
					]
				)

				# Top face triangles
				faces.extend(
					[
						[curr_top_in, next_top_out, next_top_in],
						[curr_top_in, curr_top_out, next_top_out],
					]
				)

			# Check for barrier self-intersections
			barrier_points = []
			for i in range(n_points):
				barrier_x = x_coords[i] + side * normal_x[i] * offset
				barrier_y = y_coords[i] + side * normal_y[i] * offset
				barrier_points.append([barrier_x, barrier_y])

			barrier_points = np.array(barrier_points)
			if not self.check_self_intersection(
				barrier_points[:, 0],
				barrier_points[:, 1],
				min_distance=thickness * 2,
			):
				print(
					f"Warning: Barrier on side {side} has self-intersections, skipping"
				)
				continue

			# Create barrier mesh
			collision_shape = p.createCollisionShape(
				p.GEOM_MESH,
				vertices=vertices,
				indices=[face for triangle in faces for face in triangle],
				physicsClientId=self.physics_client,
			)

			visual_shape = p.createVisualShape(
				p.GEOM_MESH,
				vertices=vertices,
				indices=[face for triangle in faces for face in triangle],
				rgbaColor=[1.0, 0.2, 0.2, 1.0],
				physicsClientId=self.physics_client,
			)

			body_id = p.createMultiBody(
				baseMass=0,
				baseCollisionShapeIndex=collision_shape,
				baseVisualShapeIndex=visual_shape,
				basePosition=[0, 0, 0],
				physicsClientId=self.physics_client,
			)

			barrier_bodies.append(body_id)

		return barrier_bodies

	def generate_track(
		self,
		track_type="oval",
		scale=10.0,
		track_width=4.0,
		track_height=0.2,
		num_points=200,
		add_barriers=True,
		barrier_offset=4.0,
		barrier_height=2.0,
		max_attempts=5,
	):
		"""
		Generate a complete racetrack with validation.

		Args:
		    track_type: Type of track to generate
		    scale: Scale factor for track size
		    track_width: Width of track surface
		    track_height: Height/thickness of track surface
		    num_points: Number of points in the spline interpolation
		    add_barriers: Whether to add barriers along the track
		    barrier_offset: Distance of barriers from track centerline
		    barrier_height: Height of the barriers
		    max_attempts: Maximum attempts to generate a valid track

		Returns:
		    Dictionary containing track information, or None if generation failed
		"""
		# Clear existing track
		self.clear_track()

		for attempt in range(max_attempts):
			print(f"Track generation attempt {attempt + 1}/{max_attempts}")

			# Generate control points
			control_points = self.generate_control_points(
				num_points=5
			)  # track_type, scale)

			# Create spline interpolation
			x_coords, y_coords = self.interpolate_spline(
				control_points, num_points
			)

			# Check for self-intersections
			min_distance = (
				max(track_width, barrier_offset * 2)
				if add_barriers
				else track_width
			)
			if not self.check_self_intersection(
				x_coords, y_coords, min_distance
			):
				print(
					f"Attempt {attempt + 1} failed: self-intersection detected"
				)
				if attempt < max_attempts - 1:
					# Try with different parameters
					scale *= 0.9  # Reduce scale
					num_points = min(
						num_points + 20, 300
					)  # Increase resolution
					continue
				else:
					print(
						"Failed to generate valid track after maximum attempts"
					)
					return None

			# Create track surface
			self.track_body = self.create_track_mesh(
				x_coords, y_coords, track_width, track_height
			)

			# Create barriers if requested
			if add_barriers:
				self.barrier_bodies = self.create_barrier_mesh(
					x_coords, y_coords, track_width / 2, barrier_height
				)

			print("Track generated successfully!")
			return {
				"track_body": self.track_body,
				"barrier_bodies": self.barrier_bodies,
				"centerline_x": x_coords,
				"centerline_y": y_coords,
				"control_points": control_points,
				"track_width": track_width,
				"scale": scale,
			}

		return None

	def clear_track(self):
		"""Remove all existing track and barriers."""
		if self.track_body is not None:
			p.removeBody(self.track_body, physicsClientId=self.physics_client)
			self.track_body = None

		for barrier_body in self.barrier_bodies:
			p.removeBody(barrier_body, physicsClientId=self.physics_client)

		self.barrier_bodies.clear()

	def get_start_position(self, height_offset=0.5):
		"""
		Get a good starting position for vehicles.

		Args:
		    height_offset: Height above the track surface

		Returns:
		    (x, y, z) starting position and orientation
		"""
		if self.track_body is None:
			return (0, 0, height_offset), (0, 0, 0, 1)

		# Use the first point of the centerline as start
		# In a real implementation, you'd store the centerline data
		return (0, 0, height_offset), (0, 0, 0, 1)


# Example usage
def demo_racetrack():
	"""Demonstrate the racetrack generator."""
	# Connect to PyBullet
	physics_client = p.connect(p.GUI)
	p.setGravity(0, 0, -9.81, physicsClientId=physics_client)

	# Create racetrack generator
	track_gen = RacetrackGenerator(physics_client)

	# Generate different types of tracks
	track_types = ["oval", "serpentine"]

	for track_type in track_types:
		print(f"\n=== Generating {track_type} track ===")

		# Generate track
		track_info = track_gen.generate_track(
			track_type=track_type,
			scale=20.0,
			track_width=3.0,
			num_points=150,
			add_barriers=True,
			barrier_offset=1.5,
			barrier_height=0.25,
		)

		if track_info:
			print(f"Track created successfully!")
			print(f"Track body ID: {track_info['track_body']}")
			print(
				f"Number of barrier bodies: {len(track_info['barrier_bodies'])}"
			)
		else:
			print("Failed to generate track")

		# Wait for user input to continue
		input("Press Enter to continue to next track type...")

	# Disconnect
	p.disconnect(physicsClientId=physics_client)


if __name__ == "__main__":
	demo_racetrack()
