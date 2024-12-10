import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def draw_conveyor_belt_with_boxes(stationary_box_position, moving_box_position, moving_circle_position, save_path):
    # Create a blank canvas
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # Draw conveyor belt
    belt_height = 2
    conveyor_belt = plt.Rectangle((0, 2), 20, belt_height, color='gray')
    ax.add_patch(conveyor_belt)

    # Draw stationary box
    stationary_box = plt.Rectangle(stationary_box_position, 1, 1, color='blue')
    ax.add_patch(stationary_box)

    # Draw moving box
    moving_box = plt.Rectangle(moving_box_position, 1, 1, color='red')
    ax.add_patch(moving_box)

    # Draw moving circle
    moving_circle = plt.Circle(moving_circle_position, 0.5, color='red')
    ax.add_patch(moving_circle)

    # Save the image
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

# Generate image with stationary and moving boxes
stationary_box_position = (2.3, 2.5)  # x, y position of the stationary box

moving_box_positions_1 = [(7,2.5), (8, 2.5), (9,2.5),(10,2.5),(11,2.5)]  # x, y positions of the moving box
moving_circle_positions = [(1,2.5),(2, 2.5), (3, 2.5), (4, 2.5), (5,2.5)]

# Generate images for different moving box positions

for i in range(len(moving_box_positions_1)):
    save_path1 = f"image_file/new/conveyor_belt_{i+11}.png"
    draw_conveyor_belt_with_boxes(stationary_box_position, moving_box_positions_1[i],moving_circle_positions[i], save_path1)

print("Images generated successfully.")