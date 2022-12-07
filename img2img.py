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


def save_images(images, img_path, save_in_input_dir, cell_type_abv):
    '''Saves images for the same batch, appending "_b" and index to the filename.'''

    if(save_in_input_dir):
        img_path = os.path.splitext(img_path)[0]
    else:
        # Create output directory
        output_dir = os.path.join('outputs', 'inpainted')
        os.makedirs(output_dir, exist_ok=True)

        # Creates output directory without extension
        img_path = os.path.join(output_dir, os.path.split(img_path)[1])
        splitted_path = os.path.splitext(img_path)
        img_path = splitted_path[0]

    for inpainted_img_idx in range(len(images)):
        filename = '{img_path}_{cell_type_abv}{inpainted_img_idx}.png'
        filename = img_path + '_' + cell_type_abv + \
            str(inpainted_img_idx) + '.png'
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


def call_inpainting_params(prompt, img_name, img_mask_name):
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
    steps = 20
    sampler_index = 0
    mask_blur = 4
    inpainting_fill = 1
    restore_faces = False
    tiling = False
    n_iter = 1
    batch_size = 1
    cfg_scale = 7
    denoising_strength = 0.75
    seed = -1.0
    subseed = -1.0
    subseed_strength = 0
    seed_resize_from_h = 0
    seed_resize_from_w = 0
    seed_enable_extras = False
    height = 512
    width = 512
    resize_mode = 0
    inpaint_full_res = False
    inpaint_full_res_padding = 32
    inpainting_mask_invert = 0
    img2img_batch_input_dir = ''
    img2img_batch_output_dir = ''

    # img2img(mode=1,prompt='@crm cell',negative_prompt='',prompt_style=None,prompt_style2=None,init_img=None,init_img_with_mask=None,init_img_inpaint= init_img_inpaint ,init_mask_inpaint= init_mask_inpaint ,mask_mode=1,steps=20,sampler_index=0,mask_blur=4,inpainting_fill=1,restore_faces=False,tiling=False,n_iter=1,batch_size=1,cfg_scale=7,denoising_strength=0.75,seed=-1.0,subseed=-1.0,subseed_strength=0,seed_resize_from_h=0,seed_resize_from_w=0,seed_enable_extras=False,height=512,width=512,resize_mode=0,inpaint_full_res=False,inpaint_full_res_padding=32,inpainting_mask_invert=0,img2img_batch_input_dir='',img2img_batch_output_dir='',args=(0, '<ul>\n<li><code>CFG Scale</code> should be 2 or lower.</li>\n</ul>\n', True, True, '', '', True, 50, True, 1, 0, False, 4, 1,         '<p style="margin-bottom:0.75em">Recommended settings: Sampling Steps: 80-100, Sampler: Euler a, Denoising strength: 0.8</p>', 128, 8, ['left', 'right', 'up', 'down'], 1, 0.05, 128, 4, 0, ['left', 'right', 'up', 'down'], False, False, False, '', '<p style="margin-bottom:0.75em">Will upscale the image to twice the dimensions; use width and height sliders to set tile size</p>', 64, 0, 1, '', 0, '', True, True, False))
    # img2img(1, '@crm cell',
    #         '', None, None, None, None, init_img_inpaint, init_mask_inpaint, 1, 50, 0, 4, 1, False, False, 1, 1, 7, 0.75,
    #         -1.0, -1.0, 0, 0, 0, False, 512, 512, 0, False, 32, 0, '', '',
    #         0, '<ul>\n<li><code>CFG Scale</code> should be 2 or lower.</li>\n</ul>\n', True, True, '', '', True, 50, True, 1, 0, False, 4, 1,
    #         '<p style="margin-bottom:0.75em">Recommended settings: Sampling Steps: 80-100, Sampler: Euler a, Denoising strength: 0.8</p>', 128, 8, ['left', 'right', 'up', 'down'], 1, 0.05, 128, 4, 0, ['left', 'right', 'up', 'down'], False, False, False, '', '<p style="margin-bottom:0.75em">Will upscale the image to twice the dimensions; use width and height sliders to set tile size</p>', 64, 0, 1, '', 0, '', True, True, False)

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


def inpaint_multiple(cell_type_abvs, input_folder):
    '''
    Given a cell type abbreviation, this function will inpaint all images in the folder 
    Will create one inpainted image, per cell type abbreviation, per image in the folder
    '''

    input_dir = os.path.join('indir', input_folder)

    # Parse images and masks
    masks = sorted(glob.glob(os.path.join(input_dir, "*_mask.jpeg")))
    images = [x.replace("_mask.jpeg", ".jpeg") for x in masks]
    print(f"Found {len(masks)} inputs.")

    # Iterate through images and inpaint
    for image_path, mask in zip(images, masks):
        print(f"Processing {image_path} and {mask}")

        # Create inpainted image, for each cell type for each image
        for cell_type_abv in cell_type_abvs:

            # Create inpainting images
            prompt = prompt_creator(cell_type_abv)

            images = call_inpainting_params(
                prompt, image_path, mask)

            # Saving for one batch different images
            save_images(images=images, img_path=image_path,
                        save_in_input_dir=False, cell_type_abv=cell_type_abv)


def inpaint():
    '''Inpaints all images in the input folder'''

    # Load model
    # model_name = '10000MultiCellAllSampRegPerson.ckpt'
    model_name = '10000MultiCellAllTypesHandPick.ckpt'

    my_load_model(model_name)

    # Input folder
    input_folder = 'first_20_masks'

    # Correct order of cell types
    cell_type_abvs = ['ascus', 'asch', 'lsil', 'hsil', 'crnm']
    inpaint_multiple(cell_type_abvs, input_folder)


inpaint()
