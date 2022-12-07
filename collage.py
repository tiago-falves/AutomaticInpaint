

from PIL import Image, ImageDraw, ImageFont
import os


# Iterate through all the images in the directory
def collage():
    imgs_path = 'indir/first_20_masks/'
    for img_name in os.listdir(imgs_path):

        # If it has mask in the name, skip it
        if 'mask'not in img_name:
            edit_image(img_name, imgs_path)


def add_text(I1):
    # Custom font style and font size
    # Custom font style and font size
    myFont = ImageFont.truetype('C:\WINDOWS\FONTS\ARIAL.TTF', 65)

    #1, 50, 32
    # 255, 0, 0
    # Add Text to an image
    # I1.text((250, 0), "Original", font=myFont, fill=(1, 50, 32))
    # I1.text((750, 0), "Mask", font=myFont, fill=(255, 255, 255))
    # I1.text((250, 500), "ASCUS", font=myFont, fill=(1, 50, 32))
    # I1.text((750, 500), "ASCH", font=myFont, fill=(1, 50, 32))
    # I1.text((250, 1000), "LSIL", font=myFont, fill=(1, 50, 32))
    # I1.text((750, 1000), "HSIL", font=myFont, fill=(1, 50, 32))
    # I1.text((250, 1500), "CRNM", font=myFont, fill=(1, 50, 32))

    I1.text((10, 0), "Original", font=myFont, fill=(255, 0, 0))
    I1.text((510, 0), "Mask", font=myFont, fill=(255, 0, 0))
    I1.text((10, 500), "ASC-US", font=myFont, fill=(255, 0, 0))
    I1.text((510, 500), "ASC H", font=myFont, fill=(255, 0, 0))
    I1.text((10, 1000), "LSIL", font=myFont, fill=(255, 0, 0))
    I1.text((510, 1000), "HSIL", font=myFont, fill=(255, 0, 0))
    I1.text((10, 1500), "CRNM", font=myFont, fill=(255, 0, 0))


def edit_image(img_name, imgs_path):
    new = Image.new("RGBA", (1000, 2000))

    output_dir = os.path.join('outputs', 'inpainted')
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
    output_dir = os.path.join('outputs', 'inpaintedGrid')
    os.makedirs(output_dir, exist_ok=True)

    # Save the image
    new.save(os.path.join(output_dir, img_name + '_inpaintedGrid' + img_ext))


# edit_image(img_name='20191024144035191_4.jpeg',
    #    imgs_path='indir/first_20_masks/')
collage()
