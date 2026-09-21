import gzip
import threading
import nibabel as nib
from scipy.ndimage import gaussian_filter
from scipy import ndimage
import os
import numpy as np
import time
from PIL import Image
from utils import *
import tqdm 
import re
from datetime import datetime
from PIL import Image, ImageFont, ImageDraw
import sys
from utils import extract_ct_mask

data_dir = r"D:\Data"

def print_progress(name, percent):
    bar_len = 30
    done = int(bar_len * percent)
    bar = "#" * done + "-" * (bar_len - done)
    sys.stdout.write(f"\r{name[:20]:20} | {bar} | {percent*100:5.1f}%")
    sys.stdout.flush()
        
def with_progress_bar(func):
    def wrapper(*args, **kwargs):
        def progress_callback(current, total):
            if total <= 1: 
                percent = 1
            else:
                percent = current/total
            print_progress(name, percent)
        kwargs['progress_callback'] = progress_callback
        name = func.__name__
        result = func(*args, **kwargs)
        print_progress(name, 1.0)
        print()
        return result
    return wrapper
            
            
@with_progress_bar
def get_air_content_from_niis(main_dir , outname,progress_callback=None, **kwargs):
    mydir = os.path.join(main_dir, outname, 'Results')    
    air_content_dir = os.path.join(mydir, 'Analysis_1')
    alltifs_path = air_content_dir+'/all.tif'
    
    if not os.path.isdir(air_content_dir): os.mkdir(air_content_dir)
    kwargs.setdefault("overwrite", False)
    overwrite = kwargs['overwrite']
    if os.path.exists(alltifs_path) and not overwrite: return 0
    pngs_paths = []
    for i in range(16):
        im_path = os.path.join(mydir, f'CT_{os.path.basename(main_dir)}_R{i}.nii.gz')
        mask_path = os.path.join(mydir, f'CT_{os.path.basename(main_dir)}_R{i}_m.nii.gz')
        out_image_path = os.path.join(air_content_dir, f'R{i}_air_content.png')
        pngs_paths.append(out_image_path)
        get_air_content_image(im_path, mask_path, out_image_path, **kwargs)
        if progress_callback is not None: progress_callback(i+1, 16)
    # combine all figures:
    # ---------------------------------------------------
    template = Image.open(pngs_paths[0]) # template
    width, height = template.size
    template.close()
    imout = Image.new('RGB', (width, height*len(pngs_paths)))
    for i, im_path in enumerate(pngs_paths):
        with Image.open(im_path) as im_i:
            imout.paste(im_i, (0, height*i))
    imout.save(air_content_dir+'/all.tif')
    imout.close()
    return 

@with_progress_bar
def compress_files(main_dir, outname,progress_callback=None, **kwargs):
    mydir = os.path.join(main_dir, outname, 'Results')
    files = [os.path.join(mydir, x) for x in os.listdir(mydir) if x.endswith(".nii")]
    # --- Print initial line even if nothing to do ---
    # Check if the files are zipped 
    threads = []
    for file in files:
        if '.nii.gz' not in file and '.nii' in file and file + '.gz' not in files:
            threads.append(threading.Thread(target=gzip_file, args=(file,)))
    for thread in threads:
        thread.start()
    for j, thread in enumerate(threads):
        if progress_callback is not None: progress_callback(j, len(threads))
        thread.join()
        
    
       
@with_progress_bar       
def segment(main_dir, outname, progress_callback=None,printtolog=False, **kwargs):
    mydir = os.path.join(main_dir, outname, 'Results')
    files = [os.path.join(mydir,x) for x in os.listdir(mydir) if '.nii.gz' in x]
    # Run the segmentation on all the files
    threads = []
    for file in files:
        if '_m.nii.gz' not in file and '_lung.nii.gz' not in file and '.nii' in file and 'sharp' not in file:
            threads.append(threading.Thread(target=extract_ct_mask,args=(file,),kwargs={'printtolog':printtolog, 'lung_at_boundaries':True}))
    for thread in threads:
        thread.start()
    for j, thread in enumerate(threads):
        thread.join()
        if progress_callback is not None: progress_callback(j+1, len(threads))
    save_npy(mydir,run_registered = False, run_jacobian=False)
    raw_gif_fname = os.path.join(mydir, 'raw.gif')
    mask_gif_fname = os.path.join(mydir, 'mask.gif')
    if not os.path.exists(raw_gif_fname):
        m,mnames,a,h = get_files(mydir, mask=True)
        sl = np.argmax(np.sum(m, axis=(0,1,3)))
        images_to_gif(m[:,:,sl,:], mask_gif_fname, mask=True)
        m,mnames,a,h = get_files(mydir, raw=True)
        images_to_gif(m[:,:,sl,:], raw_gif_fname)
        
        
    
# ----------------------------------------------------------------------------------------- 
def gzip_file(file):
    with open(file, 'rb') as src, gzip.open(file + '.gz', 'wb') as dst:
        dst.writelines(src)
    os.remove(file)

def write_nii(filename, image, affine,header):
    img = nib.Nifti1Image(image, affine)
    nib.save(img, filename)
    return filename	
def read_nii(filename):	
    img     = nib.load(filename)
    spacing = np.array(img.header.get_zooms())
    image   = img.get_fdata()
    origin  = [1*x for x in [img.header['qoffset_x'],img.header['qoffset_y'],img.header['qoffset_z']]]
    return image,spacing,img.affine,img.header 

def get_files_expensive(raw_dir, raw=False, mask=False, jacobian=False,warped=False,mask_num=-1,raw_num=-1):   
    if not (raw or mask or jacobian or warped):
        print('you must choose an option!')
        return -1
    # reg_dir = os.path.join(mydir,'Analysis','Registered',subfolder)
    antsfiles = []
    files = []
    im_stack = []
    affine, header = 0,0
    if raw: # Get Raw files
        raw_images = []
        raw_files = [x for x in os.listdir(raw_dir) if '_m' not in x and '_lung' not in x and '.nii' in x]
        num_phases = len(raw_files) # Get the number of phases 
        start = 0
        if raw_num != -1: 
            start = raw_num
            num_phases = raw_num + 1          
        for i in range(start,num_phases):
            x = [os.path.join(raw_dir,x) for x in raw_files if 'R'+ str(i)+'.nii' in x][0]
            files.append(x)
    #             antsfiles.append(ants.image_read(x))
            an_image,s,affine, header  = read_nii(x)
            antsfiles.append(an_image)
    elif mask: # Get mask files
        mask_images = []
        mask_files  = [x for x in os.listdir(raw_dir) if '_m.nii' in x and '_lung' not in x and '.nii' in x]
        num_phases  = len(mask_files) # Get the number of phases 
        start = 0
        if mask_num != -1: 
            start = mask_num
            num_phases = mask_num + 1
        for i in range(start,num_phases):
            x = [os.path.join(raw_dir,x) for x in mask_files if 'R'+ str(i)+'_m.nii' in x][0]
            files.append(x)
            an_image,s,affine, header  = read_nii(x)
            antsfiles.append(an_image)
    #             antsfiles.append(ants.image_read(x))
    elif jacobian or warped: # Jacobian or Warped 
        if len(os.listdir(reg_dir)) == 0: return -1
        # Get fixed image
        reg_folders= [x for x in os.listdir(reg_dir) if '_To_' in x]
        num_phases = len(reg_folders)
        fixed_phase = reg_folders[0].split('_To_')[-1]

        fixed_image_fname = [x for x in os.listdir(raw_dir) if fixed_phase+'.nii' in x][0]
        fixed_image,s,affine,header  = read_nii(raw_dir +'/'+fixed_image_fname)
        im_shape = fixed_image.shape
        # Get the warped image or jacobians
        if jacobian:
            keyword = '_Jacobian'

        else:
            keyword = '_Warped.'

        start = 0
        for i in range(start,num_phases+1	):
            if i == int(fixed_phase[1:]):
                if jacobian: antsfiles.append(np.ones(im_shape))
                if warped  : antsfiles.append(fixed_image) 
                files.append(fixed_phase)

            else:
                reg = [x for x in os.listdir(reg_dir) if 'R'+str(i)+'_To_' in x]
                reg = os.path.join(reg_dir,reg[0])
                jac_warped = [os.path.join(reg,x) for x in os.listdir(reg) if keyword in x][0]        		
                im,s,affine,header = read_nii(jac_warped)
                antsfiles.append(im)
                files.append(jac_warped)
        temp = True

    # combine images into 4D  

    for im in antsfiles:
        try: 
            im_stack.append(im)
        except:
            print('cannot concatenated the Images into 4D array')

    im_stack = np.array(im_stack)
    return im_stack,files, affine, header

def _is_valid_cached_npy(path):
    # A segmentation attempt that fails before any per-phase mask/raw files
    # exist yet (e.g. extract_ct_mask errors out on every file) can still
    # reach this point with an empty file list, producing a 0-size
    # placeholder .npy. Checking only os.path.isfile() - the original guard
    # here - treats that stale empty file as "already computed" and skips
    # ever rebuilding it, even after a later run successfully creates the
    # real per-phase files (confirmed on real data: a dataset stuck with a
    # permanent empty mask.npy this way, crashing every downstream step
    # that assumed a real 4D mask array). mmap_mode avoids loading a large
    # valid array fully into memory just to check it isn't empty.
    if not os.path.isfile(path):
        return False
    try:
        arr = np.load(path, mmap_mode='r')
        return arr.size > 0
    except Exception:
        return False


def save_npy(raw_dir, run_mask=True, run_raw=True, run_registered = True, run_jacobian=True, floats=False):
    #---mask
    if not _is_valid_cached_npy(raw_dir + '/mask.npy') and run_mask:
        mask, mask_names,affine,header = get_files_expensive(raw_dir, mask=1)
        np.save(raw_dir + '/mask.npy', mask.astype(np.uint8))
        np.save(raw_dir + '/mask_names.npy', mask_names)
        np.save(raw_dir + '/affine.npy', affine)
        np.save(raw_dir + '/header.npy', header)
    #---raw
    if not _is_valid_cached_npy(raw_dir + '/raw.npy') and run_raw:
        raw_im_stack, raw_names,affine,header = get_files_expensive(raw_dir, raw=1)
        np.save(raw_dir + '/raw.npy', raw_im_stack.astype(np.int16))
        if floats: np.save(raw_dir + '/raw.npy', raw_im_stack.astype(np.float32))
        np.save(raw_dir + '/raw_names.npy', raw_names)
def get_files(raw_dir, raw=False, mask=False ,mask_num=-1,raw_num=-1, mask_EI=False):
    # init
    if not (raw or mask or jacobian or warped or mask_EI):
        print('you must choose an option!')
        return -1
    if raw:
        if not 'raw.npy' in os.listdir(raw_dir):
            return get_files_expensive(raw_dir, raw=1)
        else:
            m,mnames = np.load(raw_dir+'/raw.npy'),np.load(raw_dir+'/raw_names.npy')
            a ,h    = np.load(raw_dir+'/affine.npy'), np.load(raw_dir+'/header.npy', allow_pickle=True).item()
            return m,mnames,a,h

    if mask:
        if not 'mask.npy' in os.listdir(raw_dir):
            return get_files_expensive(raw_dir, mask=1)
        else:
            m,mnames = np.load(raw_dir+'/mask.npy'),np.load(raw_dir+'/mask_names.npy')
            a     = np.load(raw_dir+'/affine.npy')
            h     = np.load(raw_dir+'/header.npy', allow_pickle=True).item()
            return  m,mnames,a,h

    if mask_EI:
        m = np.load(raw_dir+'/mask.npy')
        lung_sizes = np.sum(m, axis=(1,2,3))
        EI_phase_num = np.argmax(lung_sizes)
        return  m[EI_phase_num]
    
def images_to_gif(imarray,save_name, fps=100, mask=False):
    # Assuming numpy 3D array of multiple 2D slices (1 channel)
    #- from PIL import Image
    # Step 1: Normalize the images to 0-255 
    if mask:
        max_im,min_im = 1,0
    else:
        max_im = 350 #np.max(imarray)
        min_im = -1100#np.min(imarray)
        imarray[imarray>350] = 350
    imarray = ((imarray - min_im) / (max_im - min_im) * 255).astype(np.uint8) 
    # Step 2: Transpost the Image
    try:
        for i in range(imarray.shape[0]): imarray[i] = imarray[i].T
    except:
        pass 
    # Step 3: Convert ndarray to PILLOW
    new_images = []
    for i in range(imarray.shape[0]): new_images.append(Image.fromarray(imarray[i]))
    # Step 4: Save the image
    new_images[0].save(save_name,
               save_all=True, append_images=new_images[1:], duration=fps, loop=0, vmax=78,vmin=1, cmap='gray')
    return 0

        
def save_maps_registrations_core(registrations, jacobians, mask, ei_phase, ee_phase, maps_dir, num_slices=6, dpi=300):
    TV_path  = os.path.join(maps_dir, 'TV.png')
    FV_path  = os.path.join(maps_dir, 'FV.png')
    J_path   = os.path.join(maps_dir, 'J.png')
    modes = [False, True]
    # Compute bounding box once from the mask so all three crops have identical shape
    fx = mask.any(axis=(1, 2))
    fy = mask.any(axis=(0, 2))
    fz = mask.any(axis=(0, 1))
    pad = 4
    def _crop(img):
        out = (img * mask)[fx, :, :][:, fy, :][:, :, fz]
        out = np.pad(out, pad, 'constant', constant_values=0).astype(np.float32)
        out[out == 0] = np.nan
        return out
    ee_masked = _crop(registrations[ee_phase])
    ei_masked = _crop(registrations[ei_phase])
    J_masked  = _crop(jacobians[ee_phase])
    for smooth in modes:
        ee_slices = get_x_slices_from_image(ee_masked, number_of_slices = num_slices, 
                                 coronal_axis=1, transpose=False, airways=False, smooth=smooth)
        ei_slices = get_x_slices_from_image(ei_masked, number_of_slices = num_slices, 
                                 coronal_axis=1, transpose=False, airways=False, smooth=smooth)
        J_slices  = get_x_slices_from_image(J_masked, number_of_slices = num_slices, 
                                 coronal_axis=1, transpose=False, airways=False, smooth=smooth)
        # Compute metrics 
        FV =  1- np.divide(ee_slices*J_slices, ei_slices, where = ei_slices!=0)
        J  =  1-J_slices
        TV = ei_slices/-1000 - ee_slices/-1000 
        # Save 
        if smooth:
            save_image_MI_style(FV, 0, 1, FV_path.replace(".png", "_smoothed.png"), dpi)
            save_image_MI_style(J , 0, 1,  J_path.replace(".png", "_smoothed.png"), dpi)
            save_image_MI_style(TV, 0, 0.5, TV_path.replace(".png", "_smoothed.png"), dpi)
        else:
            save_image_MI_style(FV, 0, 1, FV_path, dpi)
            save_image_MI_style(J , 0, 1, J_path, dpi)
            save_image_MI_style(TV, 0, 0.5, TV_path, dpi)
    return

@with_progress_bar
def get_maps(main_dir , outname, progress_callback=None, **kwargs):
    # Define system paths
    mydir = os.path.join(main_dir, outname, 'Results')
    raw_path   = os.path.join(mydir, 'raw.npy')
    mask_path   = os.path.join(mydir, 'mask.npy')
    if not os.path.isfile(raw_path) or not os.path.isfile(mask_path):
        print("Couldn't generate maps because the raw.npy or mask.npy files do not exist. Run segmentation code first!")
        return
    if progress_callback is not None: progress_callback(0, 4) # Milestone 0/4
    registration_path = os.path.join(mydir, 'Registration', 'registered.npy') 
    jacobian_path     = os.path.join(mydir, 'Registration', 'jacobian.npy') 
    maps_dir = os.path.join(mydir   , 'Maps')
    FRC_path = os.path.join(maps_dir, 'FRC.png')
    TLC_path = os.path.join(maps_dir, 'TLC.png')
    TV_path  = os.path.join(maps_dir, 'TV.png')
    if not os.path.isdir(maps_dir): os.mkdir(maps_dir)
    # Defaults 
    kwargs.setdefault("overwrite", False)
    overwrite = kwargs['overwrite']
    # Check whether to load the masks 
    exist_frc = os.path.exists(FRC_path)
    exist_reg = os.path.exists(registration_path)
    exist_jac = os.path.exists(jacobian_path)
    exist_tv = os.path.exists(TV_path)
    
    if ((not exist_frc) or (exist_frc and overwrite)
        or ( (not exist_tv) and (exist_reg and exist_jac) )
        or ( (exist_tv and overwrite) and (exist_reg and exist_jac)) ):
        masks = np.load(mask_path)
        sizes = np.zeros(16)
        for i in range(16): sizes[i] = masks[i].sum()
        ei_index, ee_index = np.argmax(sizes), np.argmin(sizes)
        if progress_callback is not None: progress_callback(1, 4) # Milestone 1/4    
        
    if (not exist_frc) or (exist_frc and overwrite):
        # 1) Get Raw Images
        imgs  = np.load(raw_path)
        for i in range(imgs.shape[0]): imgs[i] =  gaussian_filter(imgs[i], 1)
        image_array_to_png(imgs[ee_index]/-1000, masks[ee_index], FRC_path, vmin=0, vmax=1) # FRC
        image_array_to_png(imgs[ei_index]/-1000, masks[ei_index], TLC_path, vmin=0, vmax=1) # TLC
        if progress_callback is not None: progress_callback(2, 4) # Milestone 2/4    
    # 4) Registration stuff
    if (( (not exist_tv) and (exist_reg and exist_jac) )
        or ((exist_tv and overwrite) and (exist_reg and exist_jac))):
            jac  = np.load(jacobian_path)
            reg =  np.load(registration_path)
            for i in range(jac.shape[0]): jac[i] =  gaussian_filter(jac[i], 1)
            save_maps_registrations_core(reg, jac, masks[ei_index], ei_index, ee_index, maps_dir)
    save_combined_maps_figure(maps_dir)
    if progress_callback is not None: progress_callback(4, 4) # Done
    return

def save_combined_maps_figure(maps_dir, out_name="maps.png", label_pad_px=300, bg=(0, 0, 0)):
    order = ["FRC", "TLC", "TV", "FV", "J"]
    smooth = ['TV']
    items = [] 
    for name in order:
        p = os.path.join(maps_dir, f"{name}.png") if name not in smooth else os.path.join(maps_dir, f"{name}_smoothed.png")
        if os.path.exists(p):
            img = Image.open(p).convert("RGBA")
            items.append((name, img))

    if not items: return 
    # Make all images the same width (pad, don't rescale)
    max_w = max(img.size[0] for _, img in items)
    padded = []
    for name, img in items:
        w, h = img.size
        if w < max_w:
            canvas = Image.new("RGBA", (max_w, h), bg + (255,))
            canvas.paste(img, (0, 0), img)
            img = canvas
        padded.append((name, img))

    total_h = sum(img.size[1] for _, img in padded)
    out_w = label_pad_px + max_w

    out = Image.new("RGBA", (out_w, total_h), bg + (255,))
    draw = ImageDraw.Draw(out)

    # Font (tries a common font; falls back safely)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 120)
    except Exception:
        font = ImageFont.load_default()

    y = 0
    for name, img in padded:
        h = img.size[1]
        out.paste(img, (label_pad_px, y), img)

        # vertically center label
        bbox = draw.textbbox((0, 0), name, font=font)
        text_h = bbox[3] - bbox[1]
        draw.text((label_pad_px - 10, y + (h - text_h) // 2),
                  name, fill=(255, 255, 255, 255), font=font, anchor="ra")
        y += h

    out_path = os.path.join(maps_dir, out_name)
    out.convert("RGB").save(out_path)  # saves as standard RGB PNG
    return 
    
# Group Images:
@with_progress_bar   
def analysis_over_time(rat_id, **kwargs):
    global data_dir 
    main_dir = os.path.join(data_dir, rat_id)
    analysis_dir_all = os.path.join(data_dir, rat_id, 'Analyzed over time')
    if not os.path.isdir(analysis_dir_all): os.mkdir(analysis_dir_all)
    ei_out_path = os.path.join(analysis_dir_all, 'ei.png')
    ee_out_path = os.path.join(analysis_dir_all, 'ee.png')

    eism_out_path = os.path.join(analysis_dir_all, 'ei_smoothed.png')
    eesm_out_path = os.path.join(analysis_dir_all, 'ee_smoothed.png')

    dates = sorted([x for x in os.listdir(main_dir) if re.match('^\d{4}-\d{2}-\d{2}_\d{2}h\d{2}$', x)])  # 2025-11-07_15h35
    dates_analyzed, ei_png_paths, ee_png_paths, eism_png_paths, eesm_png_paths = [], [], [], [], []
    for date in dates:
        # Check if optimized file exists 
        recon_dir = os.path.join(main_dir, date, 'Optimized_Reconstruction')
        if not os.path.isdir(recon_dir): continue
        # check if analysis_1 exists, if not then do it!
        analysis_dir = os.path.join(recon_dir, 'Results', 'Analysis_1')
        if not os.path.isdir(analysis_dir): continue
        # Get EE and EI images 
        ei = os.path.join(analysis_dir, 'R0_air_content.png')
        eism = os.path.join(analysis_dir, 'R0_air_content_smoothed.png')
        ee = os.path.join(analysis_dir, 'R7_air_content.png')
        eesm = os.path.join(analysis_dir, 'R7_air_content_smoothed.png')
        dates_analyzed.append(datetime.strptime(date, "%Y-%m-%d_%Hh%M")) 
        ei_png_paths.append(ei)
        ee_png_paths.append(ee)
        eesm_png_paths.append(eesm)
        eism_png_paths.append(eism)
        
    # Group them
    if len(dates_analyzed) == 0: 
        return None
    
    # Compute elapsed days relative to first scan
    t0 = datetime.strptime(dates[0], "%Y-%m-%d_%Hh%M")
    elapsed_days = [(dt - t0).days for dt in dates_analyzed]
    
    template = Image.open(ei_png_paths[0]) # template
    width, height = template.size
    template.close()
    
    # Prepare font and text width
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=120)
    except:
        print("Could not load Arial — using default font.")
        font = ImageFont.load_default()
    
    # Use a dummy image to measure max text width
    dummy = Image.new('RGB', (20, 20))
    draw_dummy = ImageDraw.Draw(dummy)
    labels = [f"Day {d}" for d in elapsed_days]
    text_widths = [draw_dummy.textbbox((0, 0), lbl, font=font)[2] for lbl in labels]
    max_text_width = max(text_widths) + 10  # + padding
    new_width = max_text_width + width

    # New canvas width = label area + original image width
    imout_ei = Image.new('RGB', (new_width, height * len(dates_analyzed)), color=(0, 0, 0))
    imout_ee = Image.new('RGB', (new_width, height * len(dates_analyzed)), color=(0, 0, 0))
    imout_eism = Image.new('RGB', (new_width, height * len(dates_analyzed)), color=(0, 0, 0))
    imout_eesm = Image.new('RGB', (new_width, height * len(dates_analyzed)), color=(0, 0, 0))

    draw_ei = ImageDraw.Draw(imout_ei)
    draw_ee = ImageDraw.Draw(imout_ee)
    draw_eism = ImageDraw.Draw(imout_eism)
    draw_eesm = ImageDraw.Draw(imout_eesm)

    for i, (date_i, ei_i, ee_i, eism_i, eesm_i, label) in enumerate(zip(dates_analyzed, ei_png_paths, ee_png_paths, 
                                                                        eism_png_paths, eesm_png_paths, labels)):
         # Vertical offset for this row
        y0 = height * i
        # Center text vertically in the row
        bbox = draw_ei.textbbox((0, 0), label, font=font)
        text_h = bbox[3] - bbox[1]
        text_y = y0 + (height - text_h) // 2
        # Draw label on both EI and EE images
        draw_ei.text((5, text_y), label, font=font, fill=(255, 255, 255))
        draw_ee.text((5, text_y), label, font=font, fill=(255, 255, 255))
        draw_eesm.text((5, text_y), label, font=font, fill=(255, 255, 255))
        draw_eism.text((5, text_y), label, font=font, fill=(255, 255, 255))
        
        with Image.open(ei_i) as im_i:
            imout_ei.paste(im_i, (max_text_width, y0))
        with Image.open(ee_i) as im_i:
            imout_ee.paste(im_i, (max_text_width, y0))
        with Image.open(eesm_i) as im_i:
            imout_eesm.paste(im_i, (max_text_width, y0))
        with Image.open(eism_i) as im_i:
            imout_eism.paste(im_i, (max_text_width, y0))
    imout_ei.save(ei_out_path) ; imout_ei.close()
    imout_ee.save(ee_out_path) ; imout_ee.close()
    imout_eesm.save(eesm_out_path) ; imout_ee.close()
    imout_eism.save(eism_out_path) ; imout_ee.close()