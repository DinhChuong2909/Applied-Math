from rembg import remove
from PIL import Image

input_path = 'ONC.png'
output_path = 'outputpic.png'
input = Image.open(input_path)
output = remove(input)
output.save(output_path)