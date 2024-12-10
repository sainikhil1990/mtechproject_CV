import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def draw_conveyor_belt_with_boxes( moving_box_position, save_path):
    # Create a blank canvas
    fig, ax = plt.subplots(figsize=(20, 5))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # Draw conveyor belt
    belt_height = 2
    conveyor_belt = plt.Rectangle((0, 2), 20, belt_height, color='gray')
    ax.add_patch(conveyor_belt)

    # Draw moving box
    moving_box = plt.Rectangle(moving_box_position, 0.5, 0.5, color='red')
    ax.add_patch(moving_box)

    # Save the image
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

# Generate image with stationary and moving boxes
stationary_box_position = (2.3, 2.5)  # x, y position of the stationary box
#moving_box_positions = [(0.5,2.5),(1, 2.5), (1.5, 2.5), (2, 2.5), (2.5,2.5),(3,2.5), (3.5, 2.5), (4, 2.5), (4.5, 2.5), (5,2.5),(5.5,2.5)]  # x, y positions of the moving box

# Generate images for different moving box positions
moving_box_positions = np.linspace(0.5,20,40)

for i in moving_box_positions:
    
    save_path = f"image_file/new/box_{i}.png"
    moving_box_position = (i,2.5)
    draw_conveyor_belt_with_boxes(moving_box_position, save_path)
print("Images generated successfully.")

