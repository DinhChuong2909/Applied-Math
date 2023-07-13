from rembg import remove
from PIL import Image

input_path = 'D:/mel.jpg'
output_path = 'outputpic.png'
input = Image.open(input_path)
output = remove(input)
output.save(output_path)