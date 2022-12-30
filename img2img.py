import argparse
import math
import os
import glob
import sys
import traceback

import numpy as np
from PIL import Image, ImageOps, ImageChops
from collections import namedtuple

from modules import devices, sd_samplers
from modules.processing import Processed, StableDiffusionProcessingImg2Img, process_images
from modules.shared import opts, state
import modules.shared as shared
import modules.processing as processing
from modules.ui import plaintext_to_html
import modules.images as images
import modules.scripts
import modules.sd_models as sd_models
from modules.paths import models_path


def process_batch(p, input_dir, output_dir, args):
    processing.fix_seed(p)

    images = shared.listfiles(input_dir)

    print(
        f"Will process {len(images)} images, creating {p.n_iter * p.batch_size} new images for each.")

    save_normally = output_dir == ''

    p.do_not_save_grid = True
    p.do_not_save_samples = not save_normally

    state.job_count = len(images) * p.n_iter

    for i, image in enumerate(images):
        state.job = f"{i+1} out of {len(images)}"
        if state.skipped:
            state.skipped = False

        if state.interrupted:
            break

        img = Image.open(image)
        # Use the EXIF orientation of photos taken by smartphones.
        img = ImageOps.exif_transpose(img)
        p.init_images = [img] * p.batch_size

        proc = modules.scripts.scripts_img2img.run(p, *args)
        if proc is None:
            proc = process_images(p)

        for n, processed_image in enumerate(proc.images):
            filename = os.path.basename(image)

            if n > 0:
                left, right = os.path.splitext(filename)
                filename = f"{left}-{n}{right}"

            if not save_normally:
                os.makedirs(output_dir, exist_ok=True)
                processed_image.save(os.path.join(output_dir, filename))


def img2img(mode: int, prompt: str, negative_prompt: str, prompt_style: str, prompt_style2: str, init_img, init_img_with_mask, init_img_inpaint, init_mask_inpaint, mask_mode, steps: int, sampler_index: int, mask_blur: int, inpainting_fill: int, restore_faces: bool, tiling: bool, n_iter: int, batch_size: int, cfg_scale: float, denoising_strength: float, seed: int, subseed: int, subseed_strength: float, seed_resize_from_h: int, seed_resize_from_w: int, seed_enable_extras: bool, height: int, width: int, resize_mode: int, inpaint_full_res: bool, inpaint_full_res_padding: int, inpainting_mask_invert: int, img2img_batch_input_dir: str, img2img_batch_output_dir: str, *args):
    is_inpaint = mode == 1
    is_batch = mode == 2

    if is_inpaint:
        # Drawn mask
        if mask_mode == 0:
            image = init_img_with_mask['image']
            mask = init_img_with_mask['mask']
            alpha_mask = ImageOps.invert(
                image.split()[-1]).convert('L').point(lambda x: 255 if x > 0 else 0, mode='1')
            mask = ImageChops.lighter(
                alpha_mask, mask.convert('L')).convert('L')
            image = image.convert('RGB')
        # Uploaded mask
        else:
            image = init_img_inpaint
            mask = init_mask_inpaint
    # No mask
    else:
        image = init_img
        mask = None

    # Use the EXIF orientation of photos taken by smartphones.
    if image is not None:
        image = ImageOps.exif_transpose(image)

    assert 0. <= denoising_strength <= 1., 'can only work with strength in [0.0, 1.0]'

    p = StableDiffusionProcessingImg2Img(
        sd_model=shared.sd_model,
        outpath_samples=opts.outdir_samples or opts.outdir_img2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_img2img_grids,
        prompt=prompt,
        negative_prompt=negative_prompt,
        styles=[prompt_style, prompt_style2],
        seed=seed,
        subseed=subseed,
        subseed_strength=subseed_strength,
        seed_resize_from_h=seed_resize_from_h,
        seed_resize_from_w=seed_resize_from_w,
        seed_enable_extras=seed_enable_extras,
        sampler_name=sd_samplers.samplers_for_img2img[sampler_index].name,
        batch_size=batch_size,
        n_iter=n_iter,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        restore_faces=restore_faces,
        tiling=tiling,
        init_images=[image],
        mask=mask,
        mask_blur=mask_blur,
        inpainting_fill=inpainting_fill,
        resize_mode=resize_mode,
        denoising_strength=denoising_strength,
        inpaint_full_res=inpaint_full_res,
        inpaint_full_res_padding=inpaint_full_res_padding,
        inpainting_mask_invert=inpainting_mask_invert,
    )

    p.scripts = modules.scripts.scripts_txt2img
    p.script_args = args

    if shared.cmd_opts.enable_console_prompts:
        print(f"\nimg2img: {prompt}", file=shared.progress_print_out)

    p.extra_generation_params["Mask blur"] = mask_blur

    if is_batch:
        assert not shared.cmd_opts.hide_ui_dir_config, "Launched with --hide-ui-dir-config, batch img2img disabled"

        process_batch(p, img2img_batch_input_dir,
                      img2img_batch_output_dir, args)

        processed = Processed(p, [], p.seed, "")
    else:
        processed = modules.scripts.scripts_img2img.run(p, *args)
        if processed is None:
            processed = process_images(p)

    p.close()

    shared.total_tqdm.clear()

    generation_info_js = processed.js()
    if opts.samples_log_stdout:
        print(generation_info_js)

    if opts.do_not_show_images:
        processed.images = []

    return processed.images
    # return processed.images, generation_info_js, plaintext_to_html(processed.info)


def build_filename(img_path, cell_type_abv, inpainted_img_idx):
    filename = img_path + '_' + cell_type_abv + \
        str(inpainted_img_idx) + '.png'
    return filename


def save_images(images, img_path, save_in_input_dir, cell_type_abv, output_dir):
    '''Saves images for the same batch, appending "_b" and index to the filename.'''

    if(save_in_input_dir):
        img_path = os.path.splitext(img_path)[0]
    else:
        # Create output directory
        temp_output_dir = os.path.join('outputs', output_dir)
        output_dir = os.path.join(temp_output_dir, 'inpainted')
        # output_dir = os.path.join('outputs', 'inpainted')
        os.makedirs(output_dir, exist_ok=True)

        # Creates output directory without extension
        img_path = os.path.join(output_dir, os.path.split(img_path)[1])
        splitted_path = os.path.splitext(img_path)
        img_path = splitted_path[0]

    for inpainted_img_idx in range(len(images)):
        filename = build_filename(
            img_path=img_path, cell_type_abv=cell_type_abv, inpainted_img_idx=inpainted_img_idx)
        images[inpainted_img_idx].save(filename)


def my_load_model(model_name):
    '''Loads model and puts it on the shared variable'''

    CheckpointInfo = namedtuple(
        "CheckpointInfo", ['filename', 'title', 'hash', 'model_name', 'config'])
    model_dir = "Stable-diffusion"
    model_path = os.path.abspath(os.path.join(models_path, model_dir))
    model_path = os.path.join(model_path, model_name)
    checkpoint_path = os.path.abspath('v1-inference.yaml')
    checkpoint_info = CheckpointInfo(
        filename=model_path, title='{model_name} [6227a08c]', hash='6227a08c', model_name='{model_name}', config=checkpoint_path)

    sd_models.load_model(checkpoint_info)


# Variar a:
# Denoising strength, steps, mask blur, cfg scale
def call_inpainting_params(prompt, img_name, img_mask_name, isAnaDataset=True):

    height = 320
    width = 320
    if isAnaDataset:
        height, width = 640, 640

    mode = 1
    prompt = prompt
    negative_prompt = ''
    prompt_style = None
    prompt_style2 = None
    init_img = None
    init_img_with_mask = None
    init_img_inpaint = Image.open(img_name)
    init_mask_inpaint = Image.open(img_mask_name)
    mask_mode = 1
    steps = 100
    sampler_index = 0
    mask_blur = 4
    inpainting_fill = 1
    restore_faces = False
    tiling = False
    n_iter = 1
    batch_size = 1
    cfg_scale = 4
    denoising_strength = 0.75
    seed = -1.0
    subseed = -1.0
    subseed_strength = 0
    seed_resize_from_h = 0
    seed_resize_from_w = 0
    seed_enable_extras = False
    resize_mode = 0
    inpaint_full_res = True
    inpaint_full_res_padding = 32
    inpainting_mask_invert = 0
    img2img_batch_input_dir = ''
    img2img_batch_output_dir = ''

    images = img2img(mode, prompt, negative_prompt, prompt_style, prompt_style2, init_img,
                     init_img_with_mask, init_img_inpaint, init_mask_inpaint, mask_mode, steps,
                     sampler_index, mask_blur, inpainting_fill, restore_faces, tiling, n_iter, batch_size,
                     cfg_scale, denoising_strength, seed, subseed, subseed_strength, seed_resize_from_h,
                     seed_resize_from_w, seed_enable_extras, height, width, resize_mode, inpaint_full_res,
                     inpaint_full_res_padding, inpainting_mask_invert, img2img_batch_input_dir, img2img_batch_output_dir,
                     0, '<ul>\n<li><code>CFG Scale</code> should be 2 or lower.</li>\n</ul>\n', True, True, '', '', True, 50, True, 1, 0, False, 4, 1, '<p style="margin-bottom:0.75em">Recommended settings: Sampling Steps: 80-100, Sampler: Euler a, Denoising strength: 0.8</p>', 128, 8, ['left', 'right', 'up', 'down'], 1, 0.05, 128, 4, 0, ['left', 'right', 'up', 'down'], False, False, False, '', '<p style="margin-bottom:0.75em">Will upscale the image to twice the dimensions; use width and height sliders to set tile size</p>', 64, 0, 1, '', 0, '', True, True, False)

    return images


def prompt_creator(cell_type):

    prompt = '@' + cell_type + ' cell'
    return prompt


def inpaint_multiple(cell_type_abvs, input_folder, output_dir, prompt, control_mask_size):
    '''
    Given a cell type abbreviation, this function will inpaint all images in the folder
    Will create one inpainted image, per cell type abbreviation, per image in the folder
    '''

    input_dir = os.path.join('indir', input_folder)
    is_multiple_model = prompt == ""

    # Parse images and masks
    masks = sorted(glob.glob(os.path.join(input_dir, "*_mask.jpeg")))
    images = [x.replace("_mask.jpeg", ".jpeg") for x in masks]
    print(f"Found {len(masks)} inputs.")

    # Iterate through images and inpaint
    for image_path, mask in zip(images, masks):
        print(f"Processing {image_path} and {mask}")

        # If prompt is empty, create prompt in "@cell_type cell" format
        # Create inpainted image, for each cell type for each image
        if is_multiple_model:
            # Create inpainted image, for each cell type for each image
            for cell_type_abv in cell_type_abvs:
                resized_mask = mask[:]

                if control_mask_size:
                    # Change mask name to add cell_type_abv to the name
                    # These masks are stored in the resized_masks folder
                    resized_mask = resized_mask.replace('.jpeg', f'_{cell_type_abv}0.jpeg')
                    mask_path, mask_name = os.path.split(resized_mask)
                    resized_mask = os.path.join(mask_path, 'resized_masks', mask_name)

                
                prompt = prompt_creator(cell_type_abv)
                images = call_inpainting_params(
                    prompt, image_path, resized_mask, is_multiple_model)

                # Saving for one batch different images
                save_images(images=images, img_path=image_path,
                            save_in_input_dir=False, cell_type_abv=cell_type_abv, output_dir=output_dir)
        else:
            images = call_inpainting_params(prompt, image_path, mask)
            # Separate prompt by spaces
            cell_type_abv = prompt.split(" ")[0]

            # Saving for one batch different images
            save_images(images=images, img_path=image_path,
                        save_in_input_dir=False, cell_type_abv=cell_type_abv, output_dir=output_dir)


def inpaint(model_name, input_folder, output_dir, prompt, control_mask_size):
    '''Inpaints all images in the input folder'''

    # Load model
    my_load_model(model_name)

    # Correct order of cell types
    cell_type_abvs = ['ascus', 'asch', 'lsil', 'hsil', 'crnm']
    inpaint_multiple(cell_type_abvs, input_folder,
                     output_dir, prompt, control_mask_size)


# Parse program arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description='Inpainting for cell images')
    parser.add_argument('--model_name', type=str,
                        help='Model name to use for inpainting')
    parser.add_argument('--input_folder', type=str,
                        help='Folder containing images to inpaint')
    parser.add_argument('--output_dir', type=str,
                        help='Folder to save inpainted images, in the outputs directory')
    parser.add_argument('--prompt', type=str, default="",
                        help='Prompt to use for inpainting')
    # parser.add_argument('--control_mask_size', type=bool, default=False,
    #                     help='If true, will use the mask size to control the size of the inpainted image')
    return parser.parse_args()


def test():
    print("Testing")
    inpaint(model_name='',
            input_folder='640_masks',
            output_dir='640MasksResized',
            prompt="",
            control_mask_size=True)


def vlad_args():
    # Ainda não tem a opçao de controlar o tamanho da mascara
    model_name = '2022-12-20T11-43-15_ASCUS_1_training_images_10000_max_training_steps_ASCUS_token_cell_class_word.ckpt'
    input_folder = 'vlad_w_masks'
    output_dir = 'vlad_w_masks_ASCUS_cell'
    prompt = "ASCUS cell"
    control_mask_size = False
    return model_name, input_folder, output_dir, prompt, control_mask_size 


def vlad_args_multiple():
    # Ainda não tem a opçao de controlar o tamanho da mascara
    model_name = '6000MultiCellAllCellsHandVLAD.ckpt'
    input_folder = 'vlad_w_masks_resized'
    output_dir = 'vlad_w_masks_resized'
    prompt = ""
    control_mask_size = True
    return model_name, input_folder, output_dir, prompt, control_mask_size


def ana_args():
    model_name = '10000MultiCellPersonHandPick.ckpt'
    input_folder = 'Ana_640_patches_w_removed'
    output_dir = 'Ana_640_patches_w_removed_oneCell'
    prompt = ""
    control_mask_size = True
    return model_name, input_folder, output_dir, prompt, control_mask_size


def ana():
    model_name, input_folder, output_dir, prompt, control_mask_size = ana_args()
    inpaint(model_name=model_name,
            input_folder=input_folder,
            output_dir=output_dir,
            prompt=prompt,
            control_mask_size=control_mask_size)
def vlad():
    model_name, input_folder, output_dir, prompt, control_mask_size = vlad_args_multiple()
    inpaint(model_name=model_name,
            input_folder=input_folder,
            output_dir=output_dir,
            prompt=prompt,
            control_mask_size=control_mask_size)


vlad()
# test()
