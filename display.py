import tkinter as tk
import math
import time
import pybullet as p


class DataDisplay:
	def __init__(self, max_speed=50.0):
		self.root = tk.Tk()
		self.root.title("Vehicle Telemetry")
		self.root.geometry("400x800")
		self.root.configure(bg="#1a1a1a")
		self.root.attributes("-type", "dialog")

		self.max_speed = max_speed

		# Create main frame
		main_frame = tk.Frame(self.root, bg="#1a1a1a")
		main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

		# Top row - Speedometer and E-brake
		top_frame = tk.Frame(main_frame, bg="#1a1a1a")
		top_frame.pack(fill=tk.X, pady=(0, 10))

		# Speedometer
		self.speedometer_frame = tk.Frame(
			top_frame, bg="#2a2a2a", relief=tk.RAISED, bd=2
		)
		self.speedometer_frame.pack(side=tk.LEFT, padx=(0, 10))

		tk.Label(
			self.speedometer_frame,
			text="SPEEDOMETER",
			bg="#2a2a2a",
			fg="white",
			font=("Arial", 10, "bold"),
		).pack(pady=5)

		self.speedometer_canvas = tk.Canvas(
			self.speedometer_frame,
			width=200,
			height=200,
			bg="black",
			highlightthickness=0,
		)
		self.speedometer_canvas.pack(padx=10, pady=10)

		# E-brake indicator
		ebrake_frame = tk.Frame(top_frame, bg="#2a2a2a", relief=tk.RAISED, bd=2)
		ebrake_frame.pack(side=tk.LEFT, padx=(0, 10))

		tk.Label(
			ebrake_frame,
			text="E-BRAKE",
			bg="#2a2a2a",
			fg="white",
			font=("Arial", 10, "bold"),
		).pack(pady=5)

		self.ebrake_canvas = tk.Canvas(
			ebrake_frame,
			width=100,
			height=100,
			bg="black",
			highlightthickness=0,
		)
		self.ebrake_canvas.pack(padx=10, pady=10)

		# Middle row - IMU visualization
		imu_frame = tk.Frame(main_frame, bg="#2a2a2a", relief=tk.RAISED, bd=2)
		imu_frame.pack(fill=tk.X, pady=(0, 10))

		tk.Label(
			imu_frame,
			text="6-AXIS IMU",
			bg="#2a2a2a",
			fg="white",
			font=("Arial", 10, "bold"),
		).pack(pady=5)

		imu_content = tk.Frame(imu_frame, bg="#2a2a2a")
		imu_content.pack(padx=10, pady=10)

		# Accelerometer (left side)
		accel_frame = tk.Frame(imu_content, bg="#2a2a2a")
		accel_frame.pack(side=tk.LEFT, padx=(0, 20))

		tk.Label(
			accel_frame,
			text="ACCELEROMETER",
			bg="#2a2a2a",
			fg="white",
			font=("Arial", 9, "bold"),
		).pack()

		self.accel_canvas = tk.Canvas(
			accel_frame, width=150, height=150, bg="black", highlightthickness=0
		)
		self.accel_canvas.pack()

		# Gyroscope (right side)
		gyro_frame = tk.Frame(imu_content, bg="#2a2a2a")
		gyro_frame.pack(side=tk.LEFT)

		tk.Label(
			gyro_frame,
			text="GYROSCOPE",
			bg="#2a2a2a",
			fg="white",
			font=("Arial", 9, "bold"),
		).pack()

		self.gyro_canvas = tk.Canvas(
			gyro_frame, width=150, height=150, bg="black", highlightthickness=0
		)
		self.gyro_canvas.pack()

		# Bottom row - Input controls
		input_frame = tk.Frame(main_frame, bg="#1a1a1a")
		input_frame.pack(fill=tk.X)

		# Throttle
		throttle_frame = tk.Frame(
			input_frame, bg="#2a2a2a", relief=tk.RAISED, bd=2
		)
		throttle_frame.pack(
			side=tk.LEFT, padx=(0, 10), fill=tk.BOTH, expand=True
		)

		tk.Label(
			throttle_frame,
			text="THROTTLE",
			bg="#2a2a2a",
			fg="white",
			font=("Arial", 10, "bold"),
		).pack(pady=5)

		self.throttle_canvas = tk.Canvas(
			throttle_frame,
			width=80,
			height=200,
			bg="black",
			highlightthickness=0,
		)
		self.throttle_canvas.pack(padx=10, pady=10)

		# Steering
		steering_frame = tk.Frame(
			input_frame, bg="#2a2a2a", relief=tk.RAISED, bd=2
		)
		steering_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

		tk.Label(
			steering_frame,
			text="STEERING",
			bg="#2a2a2a",
			fg="white",
			font=("Arial", 10, "bold"),
		).pack(pady=5)

		self.steering_canvas = tk.Canvas(
			steering_frame,
			width=200,
			height=80,
			bg="black",
			highlightthickness=0,
		)
		self.steering_canvas.pack(padx=10, pady=10)

		# Initialize all displays
		self.draw_speedometer(0)
		self.draw_ebrake(False)
		self.draw_imu([0, 0, 0], [0, 0, 0])  # accel, gyro
		self.draw_throttle(0)
		self.draw_steering(0)

	def draw_speedometer(self, speed):
		canvas = self.speedometer_canvas
		canvas.delete("all")

		# Draw outer circle
		canvas.create_oval(10, 10, 190, 190, outline="white", width=3)

		# Draw speed markings
		center_x, center_y = 100, 100
		radius = 80

		for i in range(
			0, int(self.max_speed) + 1, max(1, int(self.max_speed / 10))
		):
			angle = math.radians(
				225 - (i / self.max_speed) * 270
			)  # 225° to -45°
			x1 = center_x + (radius - 15) * math.cos(angle)
			y1 = center_y + (radius - 15) * math.sin(angle)
			x2 = center_x + radius * math.cos(angle)
			y2 = center_y + radius * math.sin(angle)

			canvas.create_line(x1, y1, x2, y2, fill="white", width=2)

			# Add numbers
			text_x = center_x + (radius - 25) * math.cos(angle)
			text_y = center_y + (radius - 25) * math.sin(angle)
			canvas.create_text(
				text_x, text_y, text=str(i), fill="white", font=("Arial", 8)
			)

		# Draw needle
		needle_angle = math.radians(
			225 - (min(speed, self.max_speed) / self.max_speed) * 270
		)
		needle_x = center_x + (radius - 20) * math.cos(needle_angle)
		needle_y = center_y + (radius - 20) * math.sin(needle_angle)

		canvas.create_line(
			center_x, center_y, needle_x, needle_y, fill="red", width=3
		)
		canvas.create_oval(
			center_x - 5, center_y - 5, center_x + 5, center_y + 5, fill="red"
		)

		# Speed text
		canvas.create_text(
			center_x,
			center_y + 30,
			text=f"{speed:.1f} m/s",
			fill="white",
			font=("Arial", 12, "bold"),
		)

	def draw_ebrake(self, engaged):
		canvas = self.ebrake_canvas
		canvas.delete("all")

		color = "#ff0000" if engaged else "#333333"
		text_color = "white" if engaged else "#666666"

		# Draw brake light
		canvas.create_oval(20, 20, 80, 80, fill=color, outline="white", width=2)
		canvas.create_text(
			50, 50, text="E", fill=text_color, font=("Arial", 24, "bold")
		)

		status = "ENGAGED" if engaged else "OFF"
		canvas.create_text(
			50, 90, text=status, fill=text_color, font=("Arial", 8, "bold")
		)

	def draw_imu(self, accel, gyro):
		# Draw accelerometer (3D vector visualization)
		canvas = self.accel_canvas
		canvas.delete("all")

		center_x, center_y = 75, 75
		max_accel = 20.0  # m/s²

		# Draw axes
		canvas.create_line(
			center_x - 60,
			center_y,
			center_x + 60,
			center_y,
			fill="#333333",
			width=1,
		)
		canvas.create_line(
			center_x,
			center_y - 60,
			center_x,
			center_y + 60,
			fill="#333333",
			width=1,
		)

		# Draw acceleration vector
		ax, ay, az = accel[:3] if len(accel) >= 3 else (0, 0, 0)

		# Scale to fit canvas
		scale = 50 / max_accel
		x_end = center_x + ax * scale
		y_end = center_y - ay * scale  # Flip Y for screen coordinates

		# Draw vector
		canvas.create_line(
			center_x, center_y, x_end, y_end, fill="#00ff00", width=3
		)
		canvas.create_oval(
			x_end - 3, y_end - 3, x_end + 3, y_end + 3, fill="#00ff00"
		)

		# Draw Z component as circle size
		z_radius = max(5, min(20, abs(az * scale / 2)))
		z_color = "#0080ff" if az > 0 else "#ff8000"
		canvas.create_oval(
			center_x + 40,
			center_y - 40,
			center_x + 40 + z_radius,
			center_y - 40 + z_radius,
			fill=z_color,
			outline="white",
		)

		# Labels
		canvas.create_text(
			center_x,
			10,
			text=f"X:{ax:.1f} Y:{ay:.1f} Z:{az:.1f}",
			fill="white",
			font=("Arial", 8),
		)

		# Draw gyroscope (rotation visualization)
		canvas = self.gyro_canvas
		canvas.delete("all")

		center_x, center_y = 75, 75
		max_gyro = 5.0  # rad/s

		# Draw rings for each axis
		gx, gy, gz = gyro[:3] if len(gyro) >= 3 else (0, 0, 0)

		# X-axis (red ring)
		x_angle = (gx / max_gyro) * 180
		canvas.create_arc(
			center_x - 40,
			center_y - 40,
			center_x + 40,
			center_y + 40,
			start=0,
			extent=x_angle,
			outline="red",
			width=3,
			style="arc",
		)

		# Y-axis (green ring)
		y_angle = (gy / max_gyro) * 180
		canvas.create_arc(
			center_x - 30,
			center_y - 30,
			center_x + 30,
			center_y + 30,
			start=90,
			extent=y_angle,
			outline="green",
			width=3,
			style="arc",
		)

		# Z-axis (blue ring)
		z_angle = (gz / max_gyro) * 180
		canvas.create_arc(
			center_x - 20,
			center_y - 20,
			center_x + 20,
			center_y + 20,
			start=180,
			extent=z_angle,
			outline="blue",
			width=3,
			style="arc",
		)

		# Labels
		canvas.create_text(
			center_x,
			10,
			text=f"X:{gx:.2f} Y:{gy:.2f} Z:{gz:.2f}",
			fill="white",
			font=("Arial", 8),
		)

	def draw_throttle(self, throttle_pct):
		canvas = self.throttle_canvas
		canvas.delete("all")

		# Draw throttle bar background
		canvas.create_rectangle(20, 20, 60, 180, outline="white", width=2)

		# Draw throttle level
		throttle_height = int(160 * (throttle_pct / 100.0))
		if throttle_height > 0:
			canvas.create_rectangle(
				22, 180 - throttle_height, 58, 178, fill="#00ff00", outline=""
			)

		# Draw percentage text
		canvas.create_text(
			40,
			190,
			text=f"{throttle_pct:.0f}%",
			fill="white",
			font=("Arial", 10, "bold"),
		)

		# Draw tick marks
		for i in range(0, 101, 25):
			y = 180 - int(160 * i / 100)
			canvas.create_line(15, y, 20, y, fill="white", width=1)
			canvas.create_text(
				10, y, text=str(i), fill="white", font=("Arial", 6)
			)

	def draw_steering(self, steering_angle):
		canvas = self.steering_canvas
		canvas.delete("all")

		center_x, center_y = 100, 40
		max_angle = 45  # degrees

		# Draw steering wheel background
		canvas.create_oval(
			center_x - 30,
			center_y - 30,
			center_x + 30,
			center_y + 30,
			outline="white",
			width=2,
		)

		# Draw steering indicator
		angle_rad = math.radians(steering_angle)
		indicator_x = center_x + 25 * math.sin(angle_rad)
		indicator_y = center_y - 25 * math.cos(angle_rad)

		canvas.create_line(
			center_x, center_y, indicator_x, indicator_y, fill="yellow", width=4
		)

		# Draw angle scale
		canvas.create_line(20, center_y, 180, center_y, fill="#333333", width=1)

		# Draw angle marker
		scale_x = center_x + (steering_angle / max_angle) * 80
		canvas.create_line(
			scale_x, center_y - 5, scale_x, center_y + 5, fill="yellow", width=3
		)

		# Draw angle text
		canvas.create_text(
			center_x,
			70,
			text=f"{steering_angle:.1f}°",
			fill="white",
			font=("Arial", 12, "bold"),
		)

		# Draw scale labels
		canvas.create_text(
			20,
			center_y + 15,
			text=f"-{max_angle}",
			fill="white",
			font=("Arial", 8),
		)
		canvas.create_text(
			center_x, center_y + 15, text="0", fill="white", font=("Arial", 8)
		)
		canvas.create_text(
			180,
			center_y + 15,
			text=f"+{max_angle}",
			fill="white",
			font=("Arial", 8),
		)

	def update(
		self,
		speed=0,
		ebrake=False,
		accel=[0, 0, 0],
		gyro=[0, 0, 0],
		throttle=0,
		steering=0,
	):
		try:
			self.draw_speedometer(speed)
			self.draw_ebrake(ebrake)
			self.draw_imu(accel, gyro)
			self.draw_throttle(throttle)
			self.draw_steering(steering)
			self.root.update()
		except tk.TclError:
			# Window was closed
			pass

	def is_alive(self):
		try:
			self.root.update()
			return True
		except tk.TclError:
			return False


# Example usage with PyBullet simulation
def run_simulation():
	# Initialize display
	display = DataDisplay(max_speed=60.0)

	# Connect to PyBullet
	p.connect(p.GUI)

	# Simulation variables
	sim_time = 0

	try:
		while display.is_alive():
			# Simulate some data (replace with your actual PyBullet data)
			speed = 20 + 15 * math.sin(sim_time * 0.5)
			ebrake = (sim_time % 10) < 2  # E-brake on for 2 seconds every 10

			# Simulate IMU data
			accel = [
				5 * math.sin(sim_time * 0.3),
				3 * math.cos(sim_time * 0.4),
				9.8 + 2 * math.sin(sim_time * 0.2),
			]
			gyro = [
				0.5 * math.sin(sim_time * 0.6),
				0.3 * math.cos(sim_time * 0.7),
				0.2 * math.sin(sim_time * 0.8),
			]

			throttle = 50 + 30 * math.sin(sim_time * 0.4)
			steering = 20 * math.sin(sim_time * 0.3)

			# Update display
			display.update(speed, ebrake, accel, gyro, throttle, steering)

			# PyBullet simulation step
			p.stepSimulation()

			sim_time += 1 / 240
			time.sleep(1 / 240)

	except KeyboardInterrupt:
		print("Simulation stopped")
	finally:
		p.disconnect()


if __name__ == "__main__":
	run_simulation()
