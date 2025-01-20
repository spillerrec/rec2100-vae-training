import os
from PIL import Image

def create_tiles(input_folder, output_folder, tile_size=(768, 768)):
    """
    Loads all images from a folder and creates 1024x1024 tiles for each image.

    Args:
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder to save the tiles.
        tile_size (tuple): Size of the tiles (width, height). Defaults to (1024, 1024).
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        
        # Check if the file is an image
        try:
            with Image.open(file_path) as img:
                img_name, _ = os.path.splitext(file_name)

                width, height = img.size
                tile_width, tile_height = tile_size

                # Calculate the number of tiles required in each dimension
                x_steps = max(1, (width + tile_width - 1) // tile_width)
                y_steps = max(1, (height + tile_height - 1) // tile_height)

                for i in range(x_steps):
                    for j in range(y_steps):
                        left = i * tile_width
                        upper = j * tile_height

                        # Ensure the tile does not exceed the image dimensions
                        right = min(left + tile_width, width)
                        lower = min(upper + tile_height, height)

                        # Adjust left and upper if the tile is smaller than the specified size
                        if right - left < tile_width:
                            left = max(0, right - tile_width)
                        if lower - upper < tile_height:
                            upper = max(0, lower - tile_height)

                        # Crop the image to create the tile
                        tile = img.crop((left, upper, right, lower))

                        # Scale the tile to the specified size if it's smaller
                        if tile.size != tile_size:
                            tile = tile.resize(tile_size, Image.LANCZOS)

                        # Save the tile with a unique name
                        tile_file_name = f"{img_name}_tile_{i}_{j}.png"
                        tile.save(os.path.join(output_folder, tile_file_name))

                        print(f"Saved: {tile_file_name}")
        except Exception as e:
            print(f"Skipping file {file_name}: {e}")

if __name__ == "__main__":
    input_folder = "L:/waifu_diffusion/anime-tagger/images/PrettyStyle/batch01-dan"  # Replace with your input folder path
    output_folder = "output_768"  # Replace with your output folder path

    create_tiles(input_folder, output_folder)