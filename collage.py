

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os


# Iterate through all the images in the directory
def collage_mask(imgs_path):
    for img_name in os.listdir(imgs_path):

        # If it has mask in the name, skip it
        if 'mask'not in img_name:
            add_square(img_name, imgs_path)

# Iterate through all the images in the directory


def collage_square(imgs_path):
    for img_name in os.listdir(imgs_path):

        # If it has square in the name, skip it
        if 'square'not in img_name:
            add_square(img_name, imgs_path)


# Draw a rectangle where mask overlaps image
def draw_rectangle(img_name, imgs_path):
    temp_dir = ''
    temp_out_dir = 'forms'

    if temp_dir != '':
        temp_out_dir = os.path.join(temp_out_dir, temp_dir)

    # If it has mask in the name, skip it
    if 'mask' in img_name:
        return

    inpainted_img_path = os.path.join(imgs_path, img_name)

    # Separate file path and file name
    inpainted_img_name = os.path.split(inpainted_img_path)[1]

    # separate file name and extension
    inpainted_img_name, img_ext = os.path.splitext(inpainted_img_name)

    img_name_original = inpainted_img_name
    # Remove from file name last suffix divided by '_'
    if imgs_path == 'outputs/inpainted/':
        img_name_original = inpainted_img_name.rsplit('_', 1)[0]

    inpainted_img = Image.open(inpainted_img_path)
    inpainted_img1 = Image.open(inpainted_img_path)

    mask_cell = Image.open(os.path.join(
        'indir/Diverse20Masks/', img_name_original + '_mask' + '.jpeg'))

    inpainted_img = inpainted_img.resize((500, 500))
    mask_cell = mask_cell.resize((500, 500))

    # Call draw Method to add 2D graphics in an image
    I1 = ImageDraw.Draw(inpainted_img)

    # # Make white part of image transparent
    # mask_cell = mask_cell.convert("RGBA")
    # datas = mask_cell.getdata()
    # newData = []

    # for item in datas:
    #     if item[0] == 255 and item[1] == 255 and item[2] == 255:
    #         newData.append((255, 255, 255, 0))
    #     else:
    #         newData.append(item)

    # mask_cell.putdata(newData)

    #   Make image opacity to 0.5, except transparent parts

    mask_cell.putalpha(50)  # Half alpha; alpha argument must be an int
    inpainted_img.paste(mask_cell, (0, 0), mask_cell)


#    # We have as input an image and a mask with the same size
#    # The mask has a white square where the cell is
#    # We want to draw a rectangle around the cell
#     for x in range(1, mask_cell.size[0]):

#         for y in range(1, mask_cell.size[1]):
#             # Get the pixel
#             pixel = mask_cell.getpixel((x, y))
#             # If the pixel is white, we are in the cell
#             if pixel != (255, 255, 255):
#                 # If it's the first pixel, we initialize xmin and ymin
#                 if xmin == 0 and ymin == 0:
#                     xmin = x
#                     ymin = y

#                 # If it's not the first pixel, we update xmax and ymax
#                 else:
#                     xmax = x
#                     ymax = y

#     # Draw the rectangle
#     I1.rectangle(((xmin, ymin), (xmax, ymax)), outline=(255, 0, 0), width=5)
#     inpainted_img.show()

    # Create folder inpaintedGrid if it doesn't exist in outputs directory
    if temp_dir != '':
        temp_out_dir = os.path.join(temp_out_dir, temp_dir)
    output_dir = os.path.join(temp_out_dir, 'original_rectangle')
    os.makedirs(output_dir, exist_ok=True)

    # Make a collage side by side of the inpainted_img and the inpainted_img1
    new = Image.new("RGBA", (1000, 500))
    inpainted_img = inpainted_img.resize((500, 500))
    inpainted_img1 = inpainted_img1.resize((500, 500))

    new.paste(inpainted_img1, (0, 0))
    new.paste(inpainted_img, (500, 0))

    # Save the image
    new.save(os.path.join(
        output_dir, inpainted_img_name + '_rectangle.png'))

    # Bug: the rectangle is not being drawn


def collage_square_inpainted(imgs_path):
    for img_name in os.listdir(imgs_path):
        draw_rectangle(img_name, imgs_path)


def add_text(I1):
    # Custom font style and font size
    # Custom font style and font size
    myFont = ImageFont.truetype('C:\WINDOWS\FONTS\ARIAL.TTF', 65)

    # 1, 50, 32
    # 255, 0, 0
    # Add Text to an image
    I1.text((250, 0), "Original", font=myFont, fill=(1, 50, 32))
    I1.text((750, 0), "Mask", font=myFont, fill=(255, 255, 255))
    I1.text((250, 500), "ASCUS", font=myFont, fill=(1, 50, 32))
    I1.text((750, 500), "ASCH", font=myFont, fill=(1, 50, 32))
    I1.text((250, 1000), "LSIL", font=myFont, fill=(1, 50, 32))
    I1.text((750, 1000), "HSIL", font=myFont, fill=(1, 50, 32))
    I1.text((250, 1500), "CRNM", font=myFont, fill=(1, 50, 32))

    # I1.text((10, 0), "Original", font=myFont, fill=(255, 0, 0))
    # I1.text((510, 0), "Mask", font=myFont, fill=(255, 0, 0))
    # I1.text((10, 500), "ASC-US", font=myFont, fill=(255, 0, 0))
    # I1.text((510, 500), "ASC H", font=myFont, fill=(255, 0, 0))
    # I1.text((10, 1000), "LSIL", font=myFont, fill=(255, 0, 0))
    # I1.text((510, 1000), "HSIL", font=myFont, fill=(255, 0, 0))
    # I1.text((10, 1500), "CRNM", font=myFont, fill=(255, 0, 0))


def edit_image(img_name, imgs_path):
    new = Image.new("RGBA", (1000, 2000))
    temp_dir = 'dif75'
    temp_out_dir = 'outputs'

    if temp_dir != '':
        temp_out_dir = os.path.join(temp_out_dir, temp_dir)
    output_dir = os.path.join(temp_out_dir, 'inpainted')
    os.makedirs(output_dir, exist_ok=True)

    original_img_path = os.path.join(imgs_path, img_name)

    # Separate file path and file name
    img_name = os.path.split(original_img_path)[1]

    # separate file name and extension
    img_name, img_ext = os.path.splitext(img_name)
    mask_cell = Image.open(os.path.join(
        imgs_path, img_name + '_mask' + img_ext))

    # Temporary file name
    img_ext = '.png'

    ascus_cell = Image.open(os.path.join(
        output_dir, img_name + '_ascus0' + img_ext))
    asch_cell = Image.open(os.path.join(
        output_dir, img_name + '_asch0' + img_ext))
    lsil_cell = Image.open(os.path.join(
        output_dir, img_name + '_lsil0' + img_ext))
    hsil_cell = Image.open(os.path.join(
        output_dir, img_name + '_hsil0' + img_ext))
    crnm_cell = Image.open(os.path.join(
        output_dir, img_name + '_crnm0' + img_ext))

    img = Image.open(original_img_path)
    img = img.resize((500, 500))
    mask_cell = mask_cell.resize((500, 500))
    ascus_cell = ascus_cell.resize((500, 500))
    asch_cell = asch_cell.resize((500, 500))
    lsil_cell = lsil_cell.resize((500, 500))
    hsil_cell = hsil_cell.resize((500, 500))
    crnm_cell = crnm_cell.resize((500, 500))

    new.paste(img, (0, 0))
    new.paste(mask_cell, (500, 0))
    new.paste(ascus_cell, (0, 500))
    new.paste(asch_cell, (500, 500))
    new.paste(lsil_cell, (0, 1000))
    new.paste(hsil_cell, (500, 1000))
    new.paste(crnm_cell, (0, 1500))

    # Call draw Method to add 2D graphics in an image
    I1 = ImageDraw.Draw(new)

    add_text(I1)

    # Create folder inpaintedGrid if it doesn't exist in outputs directory
    if temp_dir != '':
        temp_out_dir = os.path.join(temp_out_dir, temp_dir)
    output_dir = os.path.join(temp_out_dir, 'inpaintedGrid')
    os.makedirs(output_dir, exist_ok=True)

    # Save the image
    new.save(os.path.join(output_dir, img_name + '_inpaintedGrid' + img_ext))


def add_square(img_name, imgs_path):
    new = Image.new("RGBA", (1000, 500))
    temp_dir = 'inpainted'
    temp_out_dir = 'outputs'

    if temp_dir != '':
        temp_out_dir = os.path.join(temp_out_dir, temp_dir)

    original_img_path = os.path.join(imgs_path, img_name)

    # Separate file path and file name
    img_name = os.path.split(original_img_path)[1]

    # separate file name and extension
    img_name, img_ext = os.path.splitext(img_name)

    square = Image.open(os.path.join(
        imgs_path, img_name + '_square' + img_ext))

    img = Image.open(original_img_path)
    img = img.resize((500, 500))
    square = square.resize((500, 500))

    new.paste(img, (0, 0))
    new.paste(square, (500, 0))

    # Call draw Method to add 2D graphics in an image
    I1 = ImageDraw.Draw(new)

    # Create folder inpaintedGrid if it doesn't exist in outputs directory
    if temp_dir != '':
        temp_out_dir = os.path.join(temp_out_dir, temp_dir)
    output_dir = os.path.join(temp_out_dir, 'squareGrid')
    os.makedirs(output_dir, exist_ok=True)

    # Save the image
    new.save(os.path.join(output_dir, img_name + '_squareGrid' + '.png'))

# edit_image(img_name='20191024144035191_4.jpeg',
    #    imgs_path='indir/first_20_masks/')


# imgs_path = 'indir/first_20_masks/'
# imgs_path = 'indir/1mask/'
imgs_path = 'indir/Diverse20Masks/'
# imgs_path = 'outputs/inpainted/'
# collage_mask(imgs_path=imgs_path)
# collage_square(imgs_path=imgs_path)
collage_square_inpainted(imgs_path=imgs_path)
