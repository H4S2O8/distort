import imageio
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter

def distort_frame(frame, strength=0.1, mode='random', t=0, total_frames=1):
    """
    Distorts a single frame. Includes a 'bulge' mode for central magnification.
    """
    rows, cols = frame.shape[:2]
    y, x = np.mgrid[0:rows, 0:cols]
    center_x, center_y = cols / 2, rows / 2

    if mode == 'random':
        rand_x = np.random.rand(rows, cols) * strength * rows - strength * rows / 2
        rand_y = np.random.rand(rows, cols) * strength * cols - strength * cols / 2
        dx, dy = rand_x, rand_y

    elif mode == 'swirl':
        radius = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        angle = np.arctan2(y - center_y, x - center_x)
        swirl_amount = strength * np.exp(-radius / (0.2 * max(rows, cols)))
        new_angle = angle + swirl_amount * 2 * np.pi
        dx = (radius * np.cos(new_angle) - x + center_x)
        dy = (radius * np.sin(new_angle) - y + center_y)

    elif mode == 'pinch':
        distance = np.sqrt((x - center_x)**2 + (y-center_y)**2)
        max_distance = min(center_x, center_y)
        pinch_factor = 1- (distance / max_distance) * strength
        dx = (x - center_x) * pinch_factor
        dy = (y- center_y) * pinch_factor

    elif mode == 'wave':
        wave_freq = 5
        wave_amplitude = strength * rows * 0.2
        dx = np.zeros_like(x, dtype=float)
        dy = wave_amplitude * np.sin(2 * np.pi * x / cols * wave_freq + 2 * np.pi * t / total_frames)

    elif mode == 'example2':
        # ... (Your existing 'example2' code here) ...
        y_from_center = (y - center_y) / center_y
        stretch_factor = 1 + strength * (np.exp(y_from_center * 2) - 1)
        dy = (y - center_y) * (stretch_factor - 1)
        compress_factor = 1 / stretch_factor
        dx = (x - center_x) * (compress_factor - 1) * 0.8
        eye_region_top = int(center_y - 0.3 * rows)
        eye_region_bottom = int(center_y + 0.1 * rows)
        eye_mask = (y >= eye_region_top) & (y <= eye_region_bottom)
        dy[eye_mask] *= 0.2
        mouth_region_top = int(center_y + 0.1 * rows)
        mouth_region_bottom = int(center_y + 0.4 * rows)
        mouth_mask = (y >= mouth_region_top) & (y <= mouth_region_bottom)
        dy[mouth_mask] *= 0.6
        body_top = int(center_y + 0.4 * rows)
        dy[y > body_top] -= strength * rows * 0.5
        dx += strength * cols * 0.05

    elif mode == 'bulge':  # Central Magnification - CORRECTED
        # 1. Calculate the radius (distance from the center) IN PIXELS.
        radius = np.sqrt((x - center_x)**2 + (y - center_y)**2)

        # 2. Define a maximum radius for the effect, IN PIXELS.
        max_radius = min(center_x, center_y) * 0.7  # Adjust this (e.g., 0.5, 0.9)

        # 3. Create a 'bulge_factor' that smoothly decreases from the center.
        #    - Use a Gaussian-like function.
        #    - 'sigma' controls the width, IN PIXELS.
        sigma = max_radius * 0.4  # Adjust for wider/narrower bulge (e.g., 0.3, 0.5)

        # --- CORRECTED Calculation ---
        bulge_factor = strength * np.exp(-(radius**2) / (2 * sigma**2))

        # 4. Limit the effect to within max_radius.  ESSENTIAL for the bulge.
        bulge_factor[radius > max_radius] = 0

        # 5. Calculate displacements. Push pixels *away* from center.
        dx = (x - center_x) * bulge_factor
        dy = (y - center_y) * bulge_factor


    else:
        dx, dy = 0, 0

    # Gaussian smoothing (important for a smooth bulge)
    dx = gaussian_filter(dx, sigma=rows * 0.05)
    dy = gaussian_filter(dy, sigma=cols * 0.05)

    new_x = np.clip(x + dx, 0, cols - 1).astype(np.float32)
    new_y = np.clip(y + dy, 0, rows - 1).astype(np.float32)

    if len(frame.shape) == 3:
        distorted_frame = np.zeros_like(frame)
        for i in range(frame.shape[2]):
            distorted_frame[:, :, i] = map_coordinates(frame[:, :, i], [new_y, new_x], order=1)
    else:
        distorted_frame = map_coordinates(frame, [new_y, new_x], order=1)

    return distorted_frame.astype(frame.dtype)
    

def distort_gif(input_path, output_path, strength=0.1, mode='random', default_duration=0.1, animate=False):
    """Distorts a GIF (now with looping)."""
    try:
        reader = imageio.get_reader(input_path)
        metadata = reader.get_meta_data()

        if 'duration' in metadata:
            duration = metadata['duration'] / 1000.0
            if duration <= 0:
                duration = default_duration
        elif 'fps' in metadata:
            fps = metadata['fps']
            duration = 1.0 / fps if fps > 0 else default_duration
        else:
            duration = default_duration

        num_frames = 0
        for _ in reader:
            num_frames += 1

        frames = []
        for i, frame in enumerate(reader):
            t = i if animate else 0
            distorted = distort_frame(frame, strength, mode, t, num_frames)
            frames.append(distorted)

        imageio.mimsave(output_path, frames, duration=duration, loop=0)  # loop=0 for infinite loop
        print(f"Distorted GIF saved to {output_path}")

    except FileNotFoundError:
        print(f"Error: File not found: {input_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example Usage (Bulge)
distort_gif('input.gif', 'output_bulge.gif', strength=1, mode='bulge')

# Example Usage (Fisheye)
# distort_gif('input.gif', 'output_fisheye.gif', strength=0.7, mode='fisheye')

# Example Usage (Use 'example2')
# distort_gif('input.gif', 'output_example2.gif', strength=0.15, mode='example2') # Increased strength

# --- Example Usages ---
# 1. Reduced strength random distortion:
# distort_gif('input.gif', 'output_random.gif', strength=0.3, mode='random')

# 2. Swirling distortion:
# distort_gif('input.gif', 'output_swirl.gif', strength=0.3, mode='swirl')

# 3. Pinch distortion:
# distort_gif('input.gif', 'output_pinch.gif', strength=0.3, mode='pinch')

# 4. Animated wave distortion:
# distort_gif('input.gif', 'output_wave.gif', strength=0.2, mode='wave', animate=True)

# 5.  Distortion similar to the one I showed you:
# distort_gif('input.gif', 'output_example.gif', strength=0.5, mode='example')

# 6. No distortion (for testing):
# distort_gif('input.gif', 'output_none.gif', strength=0, mode='random')