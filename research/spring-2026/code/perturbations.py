import json
import numpy as np

class LinguisticPerturber:
    """Handles text-based perturbations by looking up strings in a central JSON file."""

    def __init__(self, json_path="perturbations.json"):
        self.json_path = json_path
        with open(self.json_path, 'r') as f:
            self.data = json.load(f)

    def get_instruction(self, task_id, pert_category, intensity):
        """
        Retrieves the perturbed instruction.
        Falls back to the baseline string if a specific perturbation doesn't exist.
        """
        task_key = f"task_{task_id}"

        # If task isn't in JSON at all, return a warning string
        if task_key not in self.data:
            return "TASK NOT FOUND IN JSON"

        task_data = self.data[task_key]

        # Return baseline immediately if no perturbation is requested
        if pert_category == "baseline":
            return task_data.get("baseline")

        # Navigate to the specific perturbation category (verbosity, syntax, synonym)
        pert_dict = task_data.get(pert_category, {})
        level_key = f"level_{intensity}"

        # Return the perturbed string, or fallback to baseline if missing
        return pert_dict.get(level_key, task_data.get("baseline"))


class VisualPerturber:
    """Handles pixel-level image perturbations before they are fed to the AI brain."""

    @staticmethod
    def apply_gaussian_noise(image, intensity):
        """Adds static-like noise to the camera feed."""
        # Map intensity levels to Standard Deviation
        std_mapping = {1: 10, 2: 25, 3: 50}
        # Fallback math: 15 std per level for intensities > 3
        std = std_mapping.get(intensity, intensity * 15)

        if std == 0:
            return image

        # Generate random noise array matching the image shape
        noise = np.random.normal(0, std, image.shape)

        # Convert to float to prevent uint8 overflow
        noisy_image = image.astype(np.float32) + noise

        # Clip values to valid pixel range and return to uint8 format
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_brightness(image, intensity):
        """Simulates changes in lighting conditions by adding a scalar offset."""
        # Map intensity levels to pixel offset values
        offset_mapping = {1: 30, 2: 60, 3: 90}
        offset = offset_mapping.get(intensity, 0)

        if offset == 0:
            return image

        bright_image = image.astype(np.float32) + offset
        return np.clip(bright_image, 0, 255).astype(np.uint8)


class SpatialPerturber:
    """Handles X-Y physical displacement of objects inside the MuJoCo simulator."""

    @staticmethod
    def apply_xy_displacement(env, target_keywords, intensity):
        """
        env: The LIBERO OffScreenRenderEnv instance
        target_keywords: list of strings (e.g., ['bowl', 'cream_cheese'])
        """
        # Map intensity to physical meters. Fallback: 1.5cm per level
        radius_mapping = {1: 0.02, 2: 0.05, 3: 0.10}
        radius = radius_mapping.get(intensity, intensity * 0.015)

        if radius == 0:
            return

        # Access the raw MuJoCo physics simulation layer
        sim = env.env.sim

        # Look through every joint in the simulation
        for joint_name in sim.model.joint_names:
            # Objects have "free joints" that end in _joint0
            if joint_name.endswith("_joint0") and any(kw in joint_name for kw in target_keywords):
                
                # Retrieve current physics state [x, y, z, qw, qx, qy, qz]
                qpos = sim.data.get_joint_qpos(joint_name).copy()

                # Pick a random direction (theta) and calculate the X-Y offset
                theta = np.random.uniform(0, 2 * np.pi)
                dx = radius * np.cos(theta)
                dy = radius * np.sin(theta)

                # Apply the displacement (leaving Z-height and rotation alone)
                qpos[0] += dx
                qpos[1] += dy

                # Force the object into its new physical position
                sim.data.set_joint_qpos(joint_name, qpos)

        # Advance the physics engine 1 micro-tick so the environment registers the teleport
        sim.forward()