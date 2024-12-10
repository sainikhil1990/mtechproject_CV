import bpy
import random
import os

# Function to create a simple box
def create_box(location=(0, 0, 0), size=(1, 1, 1)):
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    bpy.ops.transform.resize(value=size)
    return bpy.context.object

# Function to create a conveyor belt (plane)
def create_conveyor_belt(location=(0, 0, -1), size=(10, 10)):
    bpy.ops.mesh.primitive_plane_add(size=size[0], location=location)
    bpy.ops.transform.resize(value=size)
    return bpy.context.object

# Function to setup lighting
def setup_lighting():
    bpy.ops.object.light_add(type='SUN', location=(10, 10, 10))
    light = bpy.context.object
    light.data.energy = 5
    return light

# Function to setup camera
def setup_camera(location=(7, -7, 7)):
    bpy.ops.object.camera_add(location=location)
    bpy.context.object.rotation_euler = (1.1, 0, 0.78)
    bpy.context.scene.camera = bpy.context.object
    return bpy.context.object

# Function to render the scene
def render_scene(filepath):
    bpy.context.scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)

# Directory to save images
output_dir = "/home/eiiv-nn1-l3t04/conv/3d"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Create the scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Add conveyor belt
create_conveyor_belt()

# Add box
box = create_box()

# Setup lighting and camera
setup_lighting()
setup_camera()

# Render images with different box positions and sizes
for i in range(10):  # Number of images to generate
    # Randomize box position and size
    box.location = (random.uniform(-3, 3), random.uniform(-3, 3), 0)
    box.scale = (random.uniform(0.5, 2), random.uniform(0.5, 2), random.uniform(0.5, 2))
    
    # Update shadow settings
    bpy.context.view_layer.update()

    # Render and save image
    render_scene(os.path.join(output_dir, f"box_image_{i:03}.png"))

