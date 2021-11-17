from PIL import Image

image = Image.open("./data/image3.jpg")
resized = image.resize((512, 512))
resized.save("./data/image3_512.jpg")
