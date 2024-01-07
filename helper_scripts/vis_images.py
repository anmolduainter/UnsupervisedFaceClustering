from PIL import Image
import os

def create_image_grid(images_folder, output_filename):
    # Get a list of image filenames in the folder
    image_files = [f for f in os.listdir(images_folder) if f.endswith('.jpg') or f.endswith('.png')]

    # Limit to maximum 10 images
    image_files = image_files[:25]

    # Calculate the number of rows and columns for the grid
    rows = int(len(image_files) ** 0.5)
    cols = (len(image_files) + rows - 1) // rows

    # Set the size of each thumbnail (adjust as needed)
    thumbnail_size = 200

    # Create a new blank image for the grid
    grid = Image.new('RGB', (cols * thumbnail_size, rows * thumbnail_size))

    # Paste each thumbnail into the grid
    for i, img_file in enumerate(image_files):
        img = Image.open(os.path.join(images_folder, img_file))
        img.thumbnail((thumbnail_size, thumbnail_size))
        row = i // cols
        col = i % cols
        grid.paste(img, (col * thumbnail_size, row * thumbnail_size))

    # Save the grid image
    grid.save(output_filename)

# Replace 'path_to_images_folder' with the path to your folder containing images
# images_folder = '/home/user/anmol/FaceClust/UnsupervisedFaceClustering/evaluation_results/mobilefacenetv2/kaggle_105_classes/cluster_results/25.0/'
images_folder = '/home/user/anmol/FaceClust/UnsupervisedFaceClustering/evaluation_results/mobilefacenetv2/lfw/cluster_results/26.0/'

# output_filename = './visualize_images/m2_105.jpg'
output_filename = './visualize_images/m2_lfw.jpg'

# Create the image grid
create_image_grid(images_folder, output_filename)
