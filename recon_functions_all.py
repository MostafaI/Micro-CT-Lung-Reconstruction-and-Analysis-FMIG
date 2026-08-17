import os 
import numpy as np 
import nibabel as nib 
import pandas as pd 
from matplotlib import pyplot as plt 
from PIL import Image
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, medfilt
import shutil
import time
import imageio.v2 as imageio
import pathlib 
import gzip
from scipy.fft import fft, ifft, fftshift
from scipy.optimize import curve_fit
import warnings
from scipy import interpolate
from tqdm import tqdm
from numba import cuda
from sklearn.metrics import r2_score
from scipy.interpolate import interp1d
import matplotlib as mpl
from sklearn.cluster import KMeans
from scipy.spatial import distance
from collections import deque
import nibabel as nib
from recon_functions import *
import seaborn as sns
import random
from tqdm import trange
import imageio as iio

from Analysis import correct_cropped_images, crop_to_mask

def save_result_images(adir, vmin = -1500,vmax = 350, aslice = 453, transpose=True, origin='upper'):
    images   = [x for x in os.listdir(adir) if '.nii' in x and 'mask' not in x]
    outdir   = os.path.join(adir, 'images')
    gif_name = os.path.join(adir, 'images','s.gif')
    if not os.path.isdir(outdir): os.mkdir(outdir)
    imarray = []
    fps = 100
    for i in range(len(images)):
        im = [x for x in os.listdir(adir) if '_R'+str(i)+'.nii' in x][0]
        img = nib.load(os.path.join(adir,im)).get_fdata()
        imi = img[:,aslice,:]
        if transpose: imi = imi.T
        plt.imsave(os.path.join(outdir, im.replace('.nii','.png')),
                   imi, cmap='gray',vmin=vmin,vmax=vmax, origin=origin)
        imarray.append(Image.open(os.path.join(outdir, im.replace('.nii','.png')))) # img.shape[1]//2
        # plt.imshow(img[:,img.shape[1]//2,:], cmap='gray',vmin=-1500,vmax=5000)
    imarray[0].save(gif_name, save_all=True, append_images=imarray[1:], 
                    duration=fps, loop=0, vmax=vmax,vmin=vmin, cmap='gray') 
    pngs = [os.path.join(outdir,x) for x in os.listdir(outdir) if '.png' in x]
    imgs = []
    for i in range(len(pngs)):
        im = [os.path.join(outdir,x) for x in os.listdir(outdir) if '_R'+str(i)+'.png' in x][0]
        imgs.append(Image.open(im))
    im = Image.open(pngs[0])
    imout = Image.new('RGB', (im.width *4, im.height*4))
    imout.paste(imgs[0], (im.width*0, im.height*0))
    imout.paste(imgs[1], (im.width*1, im.height*0))
    imout.paste(imgs[2], (im.width*2, im.height*0))
    imout.paste(imgs[3], (im.width*3, im.height*0))
    imout.paste(imgs[4], (im.width*0, im.height*1))
    imout.paste(imgs[5], (im.width*1, im.height*1))
    imout.paste(imgs[6], (im.width*2, im.height*1))
    imout.paste(imgs[7], (im.width*3, im.height*1))
    imout.paste(imgs[8], (im.width*0, im.height*2))
    imout.paste(imgs[9], (im.width*1, im.height*2))
    imout.paste(imgs[10], (im.width*2, im.height*2))
    imout.paste(imgs[11], (im.width*3, im.height*2))
    imout.paste(imgs[12], (im.width*0, im.height*3))
    imout.paste(imgs[13], (im.width*1, im.height*3))
    imout.paste(imgs[14], (im.width*2, im.height*3))
    imout.paste(imgs[15], (im.width*3, im.height*3))

    imout.save(outdir+'/all.tif')
    imout.close()
    for im in imgs: im.close()
    print('Done')
    
    
def correct_mCT_file(filename, new_file_name):
    file = open(filename); lines= file.readlines() ; file.close()
    new_lines = []
    new_file_name
    for line in lines:
        if 'gating_proj_per_angle' in line: 
            new_lines.append('gating_proj_per_angle = 1\n')
        else:
            new_lines.append(line)
    new_file = open(new_file_name, 'w')
    for line in new_lines: new_file.write(line)
    new_file.close()
    return 0
def collect_results(main_dir,mname='Mostafa', bm3d = False):
    output_dir = os.path.join(main_dir, mname)
    if bm3d:  output_dir += '-BM3D'
    results_dir = os.path.join(output_dir, 'Results')
    if not os.path.isdir(results_dir): os.mkdir(results_dir)
    phases = [os.path.join(output_dir,x) for x in os.listdir(output_dir) if 'Phase' in x]
    try:
        for i in range(len(phases)):
            phase_result_dir = os.path.join(output_dir,'Phase_'+str(i),'Results')
            file = [x for x in os.listdir(phase_result_dir) if '.nii' in x][0]
            shutil.copy(os.path.join(phase_result_dir, file), 
                        os.path.join(results_dir, file.replace('.nii','_R'+str(i)+'.nii')))
    except Exception as e:
        print(f'Failed at {i}, {e}') 
    return 0
def save_projection_im(indeces,source,d, main_corr_dir, template_image):
    if len(indeces) == 1: # just one projection for that given angle
        if not os.path.isfile(d): shutil.copy(source,d)
    else: # More than 1 projection is found 
        im = imageio.imread(main_corr_dir    +'/proj_000_0_000' +"{:05d}".format(indeces[0]) +'.tif').astype(np.float64) / len(indeces)
        for i in range(1,len(indeces)):
            im += (imageio.imread(main_corr_dir    +'/proj_000_0_000' +"{:05d}".format(indeces[i]) +'.tif').astype(np.float64) / len(indeces))
        im_out = Image.fromarray(im.astype(np.uint16))
        im_out.info, im_out.format = template_image.info , template_image.format
        im_out.save(d, dpi= template_image.info['dpi'])
    return 0
def average_projections(alist,main_corr_dir):
    out  = []
    for aa in alist:
        imgs = []
        for i in range(len(aa)):
                imgs.append(imageio.imread(main_corr_dir+'/proj_000_0_000' 
                                           +"{:05d}".format(aa[i]) +'.tif').astype(np.float64))
        imgs = np.array(imgs)
        out.append(np.mean(imgs,0).astype(np.float64))
    return out
def get_imout(indeces,main_corr_dir):
    out = average_projections([indeces],main_corr_dir)
    out2 = []
    for o in out: 
        if str(type(o))=="<class 'numpy.ndarray'>": out2.append(o)
    out2 = np.array(out2)
    imout = np.mean(out2,0)
    return imout

def get_imgs_from_lists(big_list,main_corr_dir):
    imgs = []
    for aa in big_list:
        for i in range(len(aa)):
                imgs.append(imageio.imread(main_corr_dir+'/proj_000_0_000' 
                                           +"{:05d}".format(aa[i]) +'.tif').astype(np.float64))
    return imgs
def get_unique_angles(lines):
    new_lines = []
    t_phases = lines[:,5]
    for tp in np.unique(t_phases):
        same_phase = lines[lines[:,5] == tp]
        angles =  []
        for line in same_phase: # Don't repeat the angles
            if line[2] not in angles:
                new_lines.append(line)
                angles.append(line[2])
    return np.array(new_lines)

def get_guessed_image_new(lines, angle, phase, method, folder, return_all=False):
    angles = lines[:,2]
    unique_angles = np.unique(angles)
    # labs    = lines[:,4]
    pbs    = lines[:,5]
    mins   = lines[:,6]
    maxs   = lines[:,7]
    ss     = lines[:,8]
    amplitude_min_for_this_phase = mins[pbs==phase][0]
    amplitude_max_for_this_phase = maxs[pbs==phase][0]
    # Get Amplitude bin
    # interval = amplitude_max_for_this_phase - amplitude_min_for_this_phase
    # bin_edges = [-np.inf, amplitude_min_for_this_phase-interval,
    #              amplitude_min_for_this_phase, amplitude_max_for_this_phase,
    #              amplitude_max_for_this_phase+ interval, np.inf]
    # labs = np.digitize(ss, bin_edges) - 3 # ignore, lab-1, lab, lab+1, ignore (1,2,3,4,5)
    num_bins  = 10
    bin_edges = np.linspace(ss.min(), ss.max(), num_bins + 1)
    # Use np.digitize to assign each element in 'data' to a bin
    labs = np.digitize(ss, bin_edges)
    final_phases = {}
    for tp in np.unique(pbs):
        best_i_phase = -1 
        best_i_count = 0
        for ip in np.unique(labs): 
            c = len(lines[(pbs == tp) & (labs == ip)])
            if c > best_i_count:
                best_i_phase= ip
                best_i_count= c
        final_phases[tp] = best_i_phase    
    # find next and prev angles
    i = np.where(unique_angles == angle)[0]
    angle_next = unique_angles[i+1] if angle < 358.5 else unique_angles[0]
    angle_prev = unique_angles[i-1]
    angle_next2 = unique_angles[i+2] if i+2 < len(unique_angles) else unique_angles[i+2-len(unique_angles)]
    angle_prev2 =unique_angles[i-2]
    # find next and prev phase
    k = phase
    amp_bin = final_phases[k]
    num_phases = max(pbs)+1
    k_next = k+1 if k < num_phases-1 else 0
    k_prev = k-1 if k !=0 else num_phases-1
    # --- Phase is the cluster bin
    # Same Angle 
    indeces_same_angle_same_phase = np.where((pbs == k) & (angles == angle))[0]
    indeces_same_angle_same_amplitude  = np.where((pbs != k) & (labs == amp_bin) & (angles == angle))[0]
    indeces_same_angle_next_phase      = np.where((pbs == k_next) & (angles == angle))[0]
    indeces_same_angleـprev_phase      = np.where((pbs == k_prev) & (angles == angle))[0]
    indeces_same_angle_next_lab        = np.where((labs == amp_bin+1) & (angles == angle))[0]
    indeces_same_angle_prev_lab        = np.where((labs == amp_bin-1) & (angles == angle))[0]
    # neighboring angles
    indeces_next_angle_same_phase      =  np.where((pbs == k) & (angles == angle_next))[0]
    indeces_prev_angle_same_phase      =  np.where((pbs == k) & (angles == angle_prev))[0]
    indeces_next_angle_same_amplitude_only   =  np.where((pbs != k) & (labs == amp_bin) & (angles == angle_next))[0]
    indeces_prev_angle_same_amplitude_only   =  np.where((pbs != k) & (labs == amp_bin) & (angles == angle_prev))[0]  
    indeces_next_angle_next_phase            = np.where((pbs == k_next) & (angles == angle_next))[0]
    indeces_next_angle_prev_phase            = np.where((pbs == k_prev) & (angles == angle_next))[0]
    indeces_prev_angle_next_phase            = np.where((pbs == k_next) & (angles == angle_prev))[0]
    indeces_prev_angle_prev_phase            = np.where((pbs == k_prev) & (angles == angle_prev))[0]    
    indeces_next_angle_next_lab              =  np.where((labs == amp_bin+1) & (angles == angle_next))[0]
    indeces_next_angle_prev_lab              =  np.where((labs ==amp_bin-1) & (angles == angle_next))[0]
    indeces_prev_angle_next_lab              =  np.where((labs == amp_bin+1) & (angles == angle_prev))[0]
    indeces_prev_angle_prev_lab              =  np.where((labs == amp_bin-1) & (angles == angle_prev))[0]
    
    # neighboring +2-2, +2-1  
    indeces_next2_angle_same_phase_amplitude  = np.where((pbs == k) & (labs == amp_bin) & (angles == angle_next2))[0]
    indeces_prev2_angle_same_phase_amplitude  = np.where((pbs == k) & (labs == amp_bin) & (angles == angle_prev2))[0]
    
    # Guess Methods now
    im1 = None
    im1s = []
    if method == 1: # PSA-SA ---> Same angle, same amplitude
        for p in indeces_same_angle_same_amplitude:
            im1s.append(get_image(folder, slice_num=p, normalize=False).astype(np.float64))
    elif method == 2: # PNA-SA ---> same angle, nearest amplitude
        for p in indeces_same_angle_next_lab:
            im1s.append(get_image(folder, slice_num=p, normalize=False).astype(np.float64))
        for p in indeces_same_angle_prev_lab:
            im1s.append(get_image(folder, slice_num=p, normalize=False).astype(np.float64))
    elif method == 3: # PNC-SA ----> same angle, nearest phase
        for p in indeces_same_angle_next_phase:
            im1s.append(get_image(folder, slice_num=p, normalize=False).astype(np.float64))
        for p in indeces_same_angleـprev_phase:
            im1s.append(get_image(folder, slice_num=p, normalize=False).astype(np.float64))
    elif method == 4: # ABI-SA ----->  interpolation on the same angle , on amplitude
        if len(indeces_same_angle_next_lab) !=0 and len(indeces_same_angle_prev_lab)!=0:
            for pn in indeces_same_angle_next_lab:
                for pr in indeces_same_angle_prev_lab:
                    im_next_phase = get_image(folder, slice_num=pn, normalize=False).astype(np.float64)
                    im_prev_phase = get_image(folder, slice_num=pr, normalize=False).astype(np.float64)
                    im1 = im_next_phase/2 + im_prev_phase /2
                    im1s.append(im1)
    elif method == 5: # CBI-SA ----> interpolation on the same angle , on phase
        if len(indeces_same_angle_next_phase) !=0 and len(indeces_same_angleـprev_phase)!=0: 
            for pn in indeces_same_angle_next_phase:
                for pr in indeces_same_angleـprev_phase:
                    im_next_phase = get_image(folder, pn, normalize=False).astype(np.float64)
                    im_prev_phase = get_image(folder, pr, normalize=False).astype(np.float64)
                    im1 = im_next_phase/2 + im_prev_phase /2
                    im1s.append(im1)
    # ------------------------------------ Neighboring Angles --------------------------
    elif method == 6: #PSC-NA ------>  neighboring angle, same phase 
        for p in indeces_next_angle_same_phase:
            im1s.append(get_image(folder, slice_num=p, normalize=False).astype(np.float64))
        for p in indeces_prev_angle_same_phase:
            im1s.append(get_image(folder, slice_num=p, normalize=False).astype(np.float64))
    elif method == 7: #PSA-NA ------->  neighboring angle, same amplitude 
        for p in indeces_next_angle_same_amplitude_only:
            im1s.append(get_image(folder, slice_num=p, normalize=False).astype(np.float64))                
        for p in indeces_prev_angle_same_amplitude_only:
            im1s.append(get_image(folder, slice_num=p, normalize=False).astype(np.float64)) 
    elif method == 8: #PNC-NA ---------> neighboring angle, nearest phase 
        for p in indeces_next_angle_next_phase:
            im1s.append(get_image(folder, slice_num=p, normalize=False).astype(np.float64)) 
        for p in indeces_next_angle_prev_phase:
            im1s.append(get_image(folder, slice_num=p, normalize=False).astype(np.float64))
        for p in indeces_prev_angle_next_phase:
            im1s.append(get_image(folder, slice_num=p, normalize=False).astype(np.float64))
        for p in indeces_prev_angle_prev_phase:
            im1s.append(get_image(folder, slice_num=p, normalize=False).astype(np.float64))            
    elif method == 9: # PNA-NA-----> neighboring angle, nearest amplitude
        for p in indeces_next_angle_next_lab:
            im1s.append(get_image(folder, slice_num=p, normalize=False).astype(np.float64))  
        for p in indeces_next_angle_prev_lab:
            im1s.append(get_image(folder, slice_num=p, normalize=False).astype(np.float64))  
        for p in indeces_prev_angle_next_lab:
            im1s.append(get_image(folder, slice_num=p, normalize=False).astype(np.float64))  
        for p in indeces_prev_angle_prev_lab:
            im1s.append(get_image(folder, slice_num=p, normalize=False).astype(np.float64))  
    elif method == 10:  # CBI-NA --------> average next and prev angle -- phase
        if len(indeces_next_angle_same_phase) !=0 and len(indeces_prev_angle_same_phase)!=0: 
            for pn in indeces_next_angle_same_phase:
                for pr in indeces_prev_angle_same_phase: 
                    im_next_angle = get_image(folder, pn , normalize=False).astype(np.float64)
                    im_prev_angle = get_image(folder, pr , normalize=False).astype(np.float64)
                    im1 = im_next_angle/2 + im_prev_angle/2
                    im1s.append(im1)
    elif method == 11:  # ABI-NA -------> average next and prev angle same amplitude only
        if len(indeces_next_angle_same_amplitude_only) !=0 and len(indeces_prev_angle_same_amplitude_only)!=0:
            for pn in indeces_next_angle_same_amplitude_only:
                for pr in indeces_prev_angle_same_amplitude_only: 
                    im_next_angle = get_image(folder, pn , normalize=False).astype(np.float64)
                    im_prev_angle = get_image(folder, pr , normalize=False).astype(np.float64)
                    im1 = im_next_angle/2 + im_prev_angle/2
                    im1s.append(im1)
    elif method == 12:  # NCBI-NA----->  Interpolation on neighbor based on Phase
        if len(indeces_next_angle_next_phase) !=0 and len(indeces_next_angle_prev_phase)!=0:
            for pn in indeces_next_angle_next_phase:
                for pr in indeces_next_angle_prev_phase: 
                    im_next_angle = get_image(folder, pn , normalize=False).astype(np.float64)
                    im_prev_angle = get_image(folder, pr , normalize=False).astype(np.float64)
                    im1 = im_next_angle/2 + im_prev_angle/2
                    im1s.append(im1)
        if len(indeces_prev_angle_next_phase) !=0 and len(indeces_prev_angle_prev_phase)!=0: 
            for pn in indeces_prev_angle_next_phase:
                for pr in indeces_prev_angle_prev_phase: 
                    im_next_angle = get_image(folder, pn , normalize=False).astype(np.float64)
                    im_prev_angle = get_image(folder, pr , normalize=False).astype(np.float64)
                    im1 = im_next_angle/2 + im_prev_angle/2
                    im1s.append(im1)        
    elif method == 13:  # NABI-NA-------> Interpolation on neighbor based on amplitude
        if len(indeces_next_angle_next_lab) !=0 and len(indeces_next_angle_prev_lab)!=0:
            for pn in indeces_next_angle_next_lab:
                for pr in indeces_next_angle_prev_lab: 
                    im_next_angle = get_image(folder, pn , normalize=False).astype(np.float64)
                    im_prev_angle = get_image(folder, pr , normalize=False).astype(np.float64)
                    im1 = im_next_angle/2 + im_prev_angle/2
                    im1s.append(im1)
        if len(indeces_prev_angle_next_lab) !=0 and len(indeces_prev_angle_prev_lab)!=0: 
            for pn in indeces_prev_angle_next_lab:
                for pr in indeces_prev_angle_prev_lab: 
                    im_next_angle = get_image(folder, pn , normalize=False).astype(np.float64)
                    im_prev_angle = get_image(folder, pr , normalize=False).astype(np.float64)
                    im1 = im_next_angle/2 + im_prev_angle/2
                    im1s.append(im1)  
    else:
        print('You must choose a valid method')
        return im1s
    return im1s


def create_milab_structure(main_dir, output_dir='', milab_files_dir='', intensity_phases=10,
                           time_phases=16, delete_corr_dir=False, spring=False,s=None,t=None,a=None, 
                           only_phase_binning=False, optim=False, optim_plus=False,optim_SNR=False, 
                           outname='Mostafa', bm3d=False, bin_phase=False, fancy=False,
                          phase = -1, impute=False, identifier='',percent=100):
    starting_time = time.time()
    if output_dir == '': # On micro-CT Computer 
        # Create subfolders 
        output_dir = os.path.join(main_dir, outname)
        if bm3d: output_dir+= '-BM3D'
        if not os.path.isdir(output_dir): os.mkdir(output_dir)
        images_folder = os.path.join(main_dir, 'ct-data', 'corr')
        if bm3d: images_folder += '_denbm3d'
        proj_file     = os.path.join(main_dir, 'ct-data', 'proj_000_0_log.csv')
        calibration_dir = os.path.join(main_dir,'ct-data', 'calibration')
        other_files     = [ x for x in os.listdir(main_dir) if os.path.isfile(os.path.join(main_dir,x))]
    else: # on my Linux
        images_folder = main_dir
        proj_file = os.path.join(images_folder, 'proj_000_0_log.csv')
        calibration_dir = os.path.join(milab_files_dir, 'calibration')
        other_files     = [ x for x in os.listdir(milab_files_dir) if 'calibration' not in x]
        
    # Get the signal 
    if s is None:
        s,t,a = get_signal_from_image(images_folder,  subtract_baseline = 1,spring=spring)
    # Get Bins
    df, real_time, real_amps, br = new_binning(s,t,a, time_phases, intensity_phases,only_phase_binning=only_phase_binning)
    np.save(os.path.join(output_dir, 'real_time.npy'), real_time)
    np.save(os.path.join(output_dir, 'real_amps.npy'), real_amps)
    np.save(os.path.join(output_dir, 'br.npy'), [br])
    
    lines,new_lines, phases = bin_position_and_time(s,t,a, proj_file, df)
    # Do the work
    filtered_lines = get_unique_angles(new_lines)     
    # return lines,new_lines,phases,filtered_lines
    if phase == -1: phs = phases.keys()
    else: phs = [phase]
    for tp in phs:
        phase_dir = os.path.join(output_dir, 'Phase_'+ str(tp) + identifier)
        if not os.path.isdir(phase_dir): os.mkdir(phase_dir)
        ct_dir = os.path.join(phase_dir, 'ct-data')
        if not os.path.isdir(ct_dir): os.mkdir(ct_dir)
        corr_dir = os.path.join(ct_dir, 'corr')
        if not os.path.isdir(corr_dir): os.mkdir(corr_dir)
        # Copy files
        if not os.path.isdir(os.path.join(ct_dir, 'calibration')):
            shutil.copytree(calibration_dir , os.path.join(ct_dir, 'calibration'))
        for file in other_files: 
            if not os.path.isfile(os.path.join(phase_dir, file)):
                if '.mCT' in file: 
                    correct_mCT_file(os.path.join(main_dir,file), os.path.join(phase_dir, file))
                else:
                    shutil.copy(os.path.join(main_dir,file), os.path.join(phase_dir, file))

        same_phase = filtered_lines[filtered_lines[:,5] == tp][:,:4]
        same_phase_modefied = np.copy(same_phase)
        same_phase_modefied[:,1] = np.arange(len(same_phase_modefied)).astype(np.uint16)

        if only_phase_binning:
            populate_corr_dir(main_corr_dir = images_folder, 
                          lines=lines.copy(), 
                          time_phase_bin = tp,
                          current_corr_dir = corr_dir)
        elif fancy:
            populate_corr_dir_new(main_corr_dir=images_folder,
                                  lines = lines.copy(),
                                  time_phase_bin=tp, 
                                  final_phases = phases,
                                  current_corr_dir =corr_dir, 
                                  df=df,impute=impute)#,percent=percent
        elif optim:
            l_optim = get_optim(s,t,a,lines,phases)
            populate_corr_dir(main_corr_dir = images_folder, 
                          lines=l_optim, 
                          time_phase_bin = tp,
                          current_corr_dir = corr_dir)
        elif optim_plus: # Fill missing data with 1 projection
            optimized_populate_corr_dir(main_corr_dir=images_folder,
                                        lines = lines.copy(),
                                        time_phase_bin=tp, 
                                        final_phases = phases,
                                        current_corr_dir =corr_dir, 
                                        df=df)
        elif optim_SNR: # increase SNR of everything
            populate_corr_dir_new_SNR(main_corr_dir=images_folder,
                                        lines = lines.copy(),
                                        time_phase_bin=tp, 
                                        final_phases = phases,
                                        current_corr_dir =corr_dir, 
                                        df=df)
        else:
            # Create the proj file 
            proj_file_old = open(os.path.join(ct_dir, 'proj_000_0_log_old.csv'), 'w')
            for line in same_phase: proj_file_old.write(','.join(line.astype(str)) + '\n')
            proj_file_old.close() 
            proj_file_new = open(os.path.join(ct_dir, 'proj_000_0_log.csv'), 'w')
            for line in same_phase_modefied: proj_file_new.write(','.join(line.astype(str)) + '\n')
            proj_file_new.close() 
            # Populate the corr folder
            for old_num, new_num in zip(same_phase[:,1].astype(np.uint16), same_phase_modefied[:,1].astype(np.uint16)):
                source = images_folder+'/proj_000_0_000' +"{:05d}".format(old_num) +'.tif'
                d      = corr_dir +'/proj_000_0_000' +"{:05d}".format(new_num) +'.tif'
                if not os.path.isfile(d):
                    shutil.copy(source,d)
        if delete_corr_dir: shutil. rmtree(corr_dir)
    print('Took', round((time.time() -starting_time )/60,2), 'minutes.')
    return 0


def populate_corr_dir(main_corr_dir, lines, time_phase_bin,current_corr_dir):
    angles = lines[:,2]
    unique_angles = np.unique(angles)
    
    same_phase = lines[lines[:,5] == time_phase_bin] # Same time phase 
    current_angles = np.unique(same_phase[:,2])
    new_num = -1
    len_proj_ids = 0
    template_image = Image.open(os.path.join(main_corr_dir,    'proj_000_0_00000000.tif'))
    missing_projections_count = 0
    for i in range(len(unique_angles)):
        angle = unique_angles[i]
        same_phase_angle = same_phase[same_phase[:,2] == angle]
        projections_ids  = same_phase_angle[:,1].astype(np.int16)
        if len(projections_ids) ==0: 
            missing_projections_count +=1
            continue
        new_num += 1
        d      = os.path.join(current_corr_dir, './proj_000_0_000' +"{:05d}".format(new_num) +'.tif')
        source = os.path.join(main_corr_dir,    'proj_000_0_000' +"{:05d}".format(projections_ids[0]) +'.tif')
        if not os.path.isfile(d): shutil.copy(source,d)
        # if len(projections_ids) == 1: # just one projection for that given angle
        if not os.path.isfile(d): shutil.copy(source,d)
        # else:
        #     im = imageio.imread(main_corr_dir    +'/proj_000_0_000' +"{:05d}".format(projections_ids[0]) +'.tif') / len(projections_ids)
        #     for i in range(1,len(projections_ids)):
        #         im += (imageio.imread(main_corr_dir    +'/proj_000_0_000' +"{:05d}".format(projections_ids[i]) +'.tif') / len(projections_ids))
        #     im_out = Image.fromarray(im.astype(np.uint16))
        #     im_out.info, im_out.format = template_image.info , template_image.format
        #     im_out.save(d, dpi= template_image.info['dpi'])
    # Get projection file 
    lines_out = []
    for angle in np.unique(same_phase[:,2]): lines_out.append(same_phase[same_phase[:,2] == angle][0])
    lines_out = np.array(lines_out)
    lines_out[:,1] = np.arange(len(lines_out)).astype(np.int16)
    # Write the projection data 
    proj_file_new = open(os.path.join(os.path.dirname(current_corr_dir), 'proj_000_0_log.csv'), 'w')
    for line in lines_out: proj_file_new.write(','.join(line.astype(str)) + '\n')
    proj_file_new.close()
    print('Time phase =',time_phase_bin, '--Missing projections =',missing_projections_count/360*100)
    return 0


def get_optim(s,t,a,lines,phases, return_missing_projections = False):
    lines = lines.astype(np.float32)
    lines_modified = np.empty((1,lines.shape[1]))
    missing_projections = {}
    missing_projections_2 = {}
    for tp in phases.keys(): 
        missing_projections[tp]= 0
        missing_projections_2[tp]=0
    for angle in np.unique(a):
        for tp in phases.keys():
            ip = phases[tp]
            same_angle = lines[lines[:,2]==angle]
            same_phase_angle = lines[(lines[:,2]==angle)&(lines[:,5]==tp)&(lines[:,4]==ip)]
            same_lab_angle   = lines[(lines[:,2]==angle)&(lines[:,4]==ip)]
            if len(same_phase_angle) == 0: 
                if len(same_angle[same_angle[:,4] == ip]) ==0 :
                    missing_projections_2[tp] += 1
                else:
                    same_lab_angle[:,5] = tp
                    lines_modified = np.vstack((lines_modified, same_lab_angle))
                missing_projections[tp] += 1 #print(angle, tp,ip, len(same_phase_angle))
            else:
                lines_modified = np.vstack((lines_modified, same_phase_angle))
    if return_missing_projections: return lines_modified, missing_projections
    return lines_modified

def optimized_populate_corr_dir(main_corr_dir, lines, time_phase_bin, final_phases, 
                                current_corr_dir, df,snr_all=False):
    # Get 1 projection per angle 
    # Get needed variables
    angles = lines[:,2]
    pbs    = lines[:,5]
    unique_angles = np.unique(angles)
    # Start here
    missing_count, missing_count_before_guessing = 0, 0
    new_num = -1
    all_nums, all_angles,new_lines = [],[],[]
    for i in range(len(unique_angles)):
        angle = unique_angles[i]
        indeces_same_phase_angle  = np.where((pbs == time_phase_bin) & (angles == angle))[0]
        template_image = Image.open(os.path.join(main_corr_dir,    'proj_000_0_00000000.tif'))
        # Run
        imout = 0
        if len(indeces_same_phase_angle) != 0: # Found the projection
            imout = get_imout(indeces_same_phase_angle, main_corr_dir)
        else: # Missing Guess it now! 
            missing_count_before_guessing +=1
            ranked_methods = [1,10,11,4,2,5,7,3,6,9,13,8,12]
            for method in ranked_methods:
                imout = get_guessed_image_new(lines, angle, time_phase_bin, method,main_corr_dir)
                if len(imout)!= 0: 
                    imout =imout[0]
                    break
        # -------------------------------- Final -----------------------------
        if type(imout)==int or imout is None:
            missing_count += 1
        else:
            new_num += 1
            new_lines.append(np.zeros(lines[0].shape))
            d = os.path.join(current_corr_dir, './proj_000_0_000' +"{:05d}".format(new_num)+'.tif')
            im_out = Image.fromarray(imout.astype(np.uint16))
            im_out.info, im_out.format = template_image.info , template_image.format
            im_out.save(d, dpi= template_image.info['dpi'])
            all_nums.append(new_num), all_angles.append(angle)
    new_lines = np.array(new_lines)
    new_lines[:,1] = np.array(all_nums)
    new_lines[:,2] = np.array(all_angles)
    proj_file_new = open(os.path.join(os.path.dirname(current_corr_dir), 'proj_000_0_log.csv'), 'w')
    for line in new_lines: proj_file_new.write(','.join(line.astype(str)) + '\n')
    proj_file_new.close()
    print('Time phase =',time_phase_bin, '--Missing projections =',missing_count, missing_count_before_guessing)
    return 0

def optimized_populate_corr_dir_SNR(main_corr_dir, lines, time_phase_bin, final_phases, 
                                    current_corr_dir, df,snr_all=False):
    # Get needed variables
    labs,pbs = [],[]
    for x in df.LAB: labs += x.tolist()
    for x in df.PB: pbs += x.tolist()
    labs, pbs = np.array(labs), np.array(pbs)
    angles = lines[:,2]
    unique_angles = np.unique(angles)
    # Start here
    missing_count = 0
    new_num = -1
    all_nums, all_angles,new_lines = [],[],[]
    for i in range(len(unique_angles)):
        angle = unique_angles[i]
        k_next = time_phase_bin+1 if time_phase_bin < 15 else 0
        k_prev = time_phase_bin-1 if time_phase_bin !=0 else 15
        # find next and prev angles 
        angle_next  = unique_angles[i+1] if angle < 358.5 else unique_angles[0]
        angle_prev  = unique_angles[i-1] 
        angle_next2 = unique_angles[i+2] if i+2 < len(unique_angles) else unique_angles[i+2-len(unique_angles)]
        angle_prev2 =unique_angles[i-2]  
        k = time_phase_bin
        # Get projection numbers                                        
        indeces_same_phase_amplitude_angle  = np.where((pbs == time_phase_bin)  & (labs == final_phases[time_phase_bin]) & (angles == angle))[0]
        # # Same Angl
        # ------------------------------
        # Same Angle
        indeces_same_angle_same_amplitude  = np.where((pbs != k) & (labs == final_phases[k]) & (angles == angle))[0]
        indeces_same_angle_next_phase      = np.where((pbs == k_next) & (angles == angle))[0]
        indeces_same_angleـprev_phase      = np.where((pbs == k_prev) & (angles == angle))[0]
        indeces_same_angle_next_lab        = np.where((labs == final_phases[k]+1) & (angles == angle))[0]
        indeces_same_angle_prev_lab        = np.where((labs == final_phases[k]-1) & (angles == angle))[0]
        # neighboring angles
        indeces_next_angle_same_phase_amplitude  = np.where((pbs == k) & (labs == final_phases[k]) & (angles == angle_next))[0]
        indeces_next_angle_same_amplitude_only   =  np.where((pbs != k) & (labs == final_phases[k]) & (angles == angle_next))[0]
        indeces_next_angle_same_phase_only       =  np.where((pbs == k) & (labs != final_phases[k]) & (angles == angle_next))[0]  
        indeces_next_angle_next_phase            = np.where((pbs == k_next) & (angles == angle_next))[0]
        indeces_next_angle_prev_phase            = np.where((pbs == k_prev) & (angles == angle_next))[0]
        indeces_next_angle_next_lab              =  np.where((labs == final_phases[k]+1) & (angles == angle_next))[0]
        indeces_next_angle_prev_lab              =  np.where((labs == final_phases[k]-1) & (angles == angle_next))[0]
        indeces_prev_angle_same_phase_amplitude  = np.where((pbs == k) & (labs == final_phases[k]) & (angles == angle_prev))[0]
        indeces_prev_angle_same_amplitude_only   =  np.where((pbs != k) & (labs == final_phases[k]) & (angles == angle_prev))[0]         
        indeces_prev_angle_same_phase_only       =  np.where((pbs == k) & (labs != final_phases[k]) & (angles == angle_prev))[0]
        indeces_prev_angle_next_phase            = np.where((pbs == k_next) & (angles == angle_prev))[0]
        indeces_prev_angle_prev_phase            = np.where((pbs == k_prev) & (angles == angle_prev))[0]
        indeces_prev_angle_next_lab              =  np.where((labs == final_phases[k]+1) & (angles == angle_prev))[0]
        indeces_prev_angle_prev_lab              =  np.where((labs == final_phases[k]-1) & (angles == angle_prev))[0]
        # neighboring +2-2, +2-1  
        indeces_next2_angle_same_phase_amplitude  = np.where((pbs == k) & (labs == final_phases[k]) & (angles == angle_next2))[0]
        indeces_prev2_angle_same_phase_amplitude  = np.where((pbs == k) & (labs == final_phases[k]) & (angles == angle_prev2))[0]
        # ----------------------------------
        
        # Get template image
        template_image = Image.open(os.path.join(main_corr_dir,    'proj_000_0_00000000.tif'))
        # Run
        imout = 0
        imgs = []
        
        # Get projections 
        imgs = get_imgs_from_lists([indeces_same_phase_amplitude_angle,
                                   indeces_same_angle_same_amplitude
                                   ],
                                   main_corr_dir) 
        # Method 5 
        if len(indeces_same_angle_next_phase) !=0 and len(indeces_same_angleـprev_phase)!=0: 
            im_next_phase = get_imout(indeces_same_angle_next_phase,main_corr_dir)
            im_prev_phase = get_imout(indeces_same_angleـprev_phase,main_corr_dir) 
            imout = im_next_phase/2 + im_prev_phase /2
            imgs.append(imout)
        # Method 13
        elif len(indeces_next_angle_same_phase_amplitude) !=0 and len(indeces_prev_angle_same_phase_amplitude)!=0:
            im_next_angle = get_imout(indeces_next_angle_same_phase_amplitude,main_corr_dir)
            im_prev_angle = get_imout(indeces_prev_angle_same_phase_amplitude,main_corr_dir)
            imout = im_next_angle/2 + im_prev_angle/2
            imgs.append(imout)
        # Method 4
        elif len(indeces_same_angle_next_lab) !=0 and len(indeces_same_angle_prev_lab)!=0: 
            im_next_phase = get_imout(indeces_same_angle_next_lab,main_corr_dir)
            im_prev_phase = get_imout(indeces_same_angle_prev_lab,main_corr_dir)
            imout = im_next_phase/2 + im_prev_phase /2
            imgs.append(imout)
        # -------------------------------- Guess missing projection -----------------------------
        if len(imgs)==0:
            # Method 2
            if len(indeces_same_angle_next_lab) != 0:
                imout  = get_imout(indeces_same_angle_next_lab,main_corr_dir);imgs.append(imout)
            elif len(indeces_same_angle_prev_lab) !=0:
                imout  = get_imout(indeces_same_angle_prev_lab,main_corr_dir);imgs.append(imout)
            # Method 15
            elif len(indeces_next_angle_same_phase_only) !=0 and len(indeces_prev_angle_same_phase_only)!=0: 
                im_next_angle = get_imout(indeces_next_angle_same_phase_only,main_corr_dir)
                im_prev_angle = get_imout(indeces_prev_angle_same_phase_only,main_corr_dir)
                imout = im_next_angle/2 + im_prev_angle/2;imgs.append(imout)
            # Method 3 
            elif len(indeces_same_angle_next_phase) != 0:
                imout  = get_imout(indeces_same_angle_next_phase, main_corr_dir);imgs.append(imout)
            elif len(indeces_same_angleـprev_phase) !=0:
                imout  = get_imout(indeces_same_angleـprev_phase, main_corr_dir);imgs.append(imout)
            # Method 6
            elif len(indeces_next_angle_same_phase_amplitude) !=0: 
                imout = get_imout(indeces_next_angle_same_phase_amplitude,main_corr_dir);imgs.append(imout)
            elif len(indeces_prev_angle_same_phase_amplitude) !=0: 
                imout = get_imout(indeces_prev_angle_same_phase_amplitude,main_corr_dir);imgs.append(imout)
            # Method 19
            elif (len(indeces_next_angle_next_phase) !=0 and len(indeces_next_angle_prev_phase)!=0 and
                    len(indeces_prev_angle_next_phase) !=0 and len(indeces_prev_angle_prev_phase)!=0):
                im_next_phase = get_imout(indeces_next_angle_next_phase, main_corr_dir)
                im_prev_phase = get_imout(indeces_next_angle_prev_phase, main_corr_dir)
                im1_1 = im_next_phase/2 + im_prev_phase /2
                im_next_phase = get_imout(indeces_prev_angle_next_phase, main_corr_dir)
                im_prev_phase = get_imout(indeces_prev_angle_prev_phase, main_corr_dir)
                im1_2 = im_next_phase/2 + im_prev_phase /2
                imout = im1_1/2 + im1_2 /2;imgs.append(imout)
            # Method 21
            elif len(indeces_next_angle_same_phase_amplitude) !=0 and len(indeces_prev2_angle_same_phase_amplitude)!=0:
                im_next_prev_1 = get_imout(indeces_next_angle_same_phase_amplitude,main_corr_dir)
                im_next_prev_2 = get_imout(indeces_prev2_angle_same_phase_amplitude,main_corr_dir)
                imout = im_next_prev_1/2 + im_next_prev_2/2;imgs.append(imout)
            elif len(indeces_prev_angle_same_phase_amplitude) !=0 and len(indeces_next2_angle_same_phase_amplitude)!=0:
                im_next_prev_1 = get_imout(indeces_prev_angle_same_phase_amplitude,main_corr_dir)
                im_next_prev_2 = get_imout(indeces_next2_angle_same_phase_amplitude,main_corr_dir)
                imout = im_next_prev_1/2 + im_next_prev_2/2;imgs.append(imout)
            # Method 1
            elif len(indeces_same_angle_same_amplitude) != 0:
                imout = get_imout(indeces_same_angle_same_amplitude, main_corr_dir);imgs.append(imout)
            # Method 18
            elif (len(indeces_next_angle_next_lab) !=0 and len(indeces_next_angle_prev_lab)!=0 and
                len(indeces_prev_angle_next_lab) !=0 and len(indeces_prev_angle_prev_lab)!=0) : 
                im_next_phase = get_imout(indeces_next_angle_next_lab,main_corr_dir)
                im_prev_phase = get_imout(indeces_next_angle_prev_lab,main_corr_dir)
                im1_1 = im_next_phase/2 + im_prev_phase /2
                im_next_phase = get_imout(indeces_prev_angle_next_lab,main_corr_dir)
                im_prev_phase = get_imout(indeces_prev_angle_prev_lab,main_corr_dir)
                im1_2 = im_next_phase/2 + im_prev_phase /2
                imout = im1_1/2 + im1_2 /2;imgs.append(imout)
            # Method 16
            elif (len(indeces_next_angle_next_lab) !=0) and (len(indeces_prev_angle_next_lab)!=0):
                im_next_angle = get_imout(indeces_next_angle_next_lab, main_corr_dir)
                im_prev_angle = get_imout(indeces_prev_angle_next_lab, main_corr_dir)
                imout = im_next_angle/2 + im_prev_angle/2;imgs.append(imout)
            elif (len(indeces_next_angle_prev_lab) !=0) and (len(indeces_prev_angle_prev_lab)!=0):
                im_next_angle = get_imout(indeces_next_angle_prev_lab, main_corr_dir)
                im_prev_angle = get_imout(indeces_prev_angle_prev_lab, main_corr_dir)
                imout = im_next_angle/2 + im_prev_angle/2;imgs.append(imout)
            # Method 8
            elif len(indeces_next_angle_same_phase_only) !=0: 
                imout = get_imout(indeces_next_angle_same_phase_only, main_corr_dir);imgs.append(imout)
            elif len(indeces_prev_angle_same_phase_only) !=0: 
                imout = get_imout(indeces_prev_angle_same_phase_only, main_corr_dir);imgs.append(imout)
            # Method 17
            elif (len(indeces_next_angle_next_phase) !=0) and (len(indeces_prev_angle_next_phase)!=0):
                im_next_angle = get_imout(indeces_next_angle_next_phase, main_corr_dir)
                im_prev_angle = get_imout(indeces_prev_angle_next_phase, main_corr_dir)
                imout = im_next_angle/2 + im_prev_angle/2;imgs.append(imout)
            elif (len(indeces_next_angle_prev_phase) !=0) and (len(indeces_prev_angle_prev_phase)!=0):
                im_next_angle = get_imout(indeces_next_angle_prev_phase, main_corr_dir)
                im_prev_angle = get_imout(indeces_prev_angle_prev_phase, main_corr_dir)
                imout = im_next_angle/2 + im_prev_angle/2;imgs.append(imout)
            # Method 20
            elif len(indeces_next2_angle_same_phase_amplitude) !=0 and len(indeces_prev2_angle_same_phase_amplitude)!=0: 
                im_next2_angle = get_imout(indeces_next2_angle_same_phase_amplitude, main_corr_dir)
                im_prev2_angle = get_imout(indeces_prev2_angle_same_phase_amplitude, main_corr_dir)
                imout = im_next2_angle/2 + im_prev2_angle/2;imgs.append(imout)
            # Method 14
            elif len(indeces_next_angle_same_amplitude_only) !=0 and len(indeces_prev_angle_same_amplitude_only)!=0: 
                im_next_angle = get_imout(indeces_next_angle_same_amplitude_only,main_corr_dir)
                im_prev_angle = get_imout(indeces_prev_angle_same_amplitude_only,main_corr_dir)
                imout = im_next_angle/2 + im_prev_angle/2;imgs.append(imout)
            # Method 9
            elif len(indeces_next_angle_next_lab)   != 0:
                imout  = get_imout(indeces_next_angle_next_lab, main_corr_dir);imgs.append(imout)
            elif len(indeces_next_angle_prev_lab) !=0:
                imout  = get_imout(indeces_next_angle_prev_lab, main_corr_dir);imgs.append(imout)
            elif len(indeces_prev_angle_next_lab) !=0:
                imout  = get_imout(indeces_prev_angle_next_lab, main_corr_dir);imgs.append(imout)
            elif len(indeces_prev_angle_prev_lab) !=0:
                imout  = get_imout(indeces_prev_angle_prev_lab, main_corr_dir);imgs.append(imout)
            # Method 12
            elif len(indeces_next_angle_next_phase) !=0 and len(indeces_next_angle_prev_phase)!=0: 
                im_next_phase = get_imout(indeces_next_angle_next_phase, main_corr_dir)
                im_prev_phase = get_imout(indeces_next_angle_prev_phase, main_corr_dir)
                imout = im_next_phase/2 + im_prev_phase /2;imgs.append(imout)
            elif len(indeces_prev_angle_next_phase) !=0 and len(indeces_prev_angle_prev_phase)!=0: 
                im_next_phase = get_imout(indeces_prev_angle_next_phase, main_corr_dir)
                im_prev_phase = get_imout(indeces_prev_angle_prev_phase, main_corr_dir)
                imout = im_next_phase/2 + im_prev_phase /2;imgs.append(imout)
        # -------------------------------- Final -----------------------------        
        if len(imgs)==0:
                missing_count += 1
        else:
            imout = np.array(imgs)
            imout = np.mean(imout,0)
            new_num += 1
            new_lines.append(np.zeros(lines[0].shape))
            d = os.path.join(current_corr_dir, './proj_000_0_000' +"{:05d}".format(new_num)+'.tif')
            im_out = Image.fromarray(imout.astype(np.uint16))
            im_out.info, im_out.format = template_image.info , template_image.format
            im_out.save(d, dpi= template_image.info['dpi'])
            all_nums.append(new_num), all_angles.append(angle)
                    
    new_lines = np.array(new_lines)
    new_lines[:,1] = np.array(all_nums)
    new_lines[:,2] = np.array(all_angles)
    
    proj_file_new = open(os.path.join(os.path.dirname(current_corr_dir), 'proj_000_0_log.csv'), 'w')
    for line in new_lines: proj_file_new.write(','.join(line.astype(str)) + '\n')
    proj_file_new.close()
    
    print('Time phase =',time_phase_bin, '--Missing projections =',missing_count)
    return 0

def populate_corr_dir_new(main_corr_dir, lines, time_phase_bin, final_phases, 
                      current_corr_dir, df, snr_all = False, impute = False):
    # Get 1 projection per angle 
    # Get needed variables
    angles = lines[:,2]
    pbs    = lines[:,5]
    unique_angles = np.unique(angles)
    # Start here
    missing_count, missing_after_imputation = 0, 0
    new_num = -1
    all_nums, all_angles,new_lines = [],[],[]
    last_method = 0
    for i in range(len(unique_angles)):
        angle = unique_angles[i]
        indeces_same_phase_angle  = np.where((pbs == time_phase_bin) & (angles == angle))[0]
        if len(indeces_same_phase_angle) !=0: # Found a projection
            # print(len(indeces_same_phase_angle))
            new_num += 1
            d = os.path.join(current_corr_dir, './proj_000_0_000' +"{:05d}".format(new_num) +'.tif')
            source = os.path.join(main_corr_dir,    'proj_000_0_000' +"{:05d}".format(indeces_same_phase_angle[0]) +'.tif')
            if not os.path.isfile(d): shutil.copy(source,d)
            all_nums.append(new_num); all_angles.append(angle); new_lines.append(np.zeros(lines[0].shape))
        elif impute: # Missing
            missing_count +=1
            template_image = Image.open(os.path.join(main_corr_dir,    'proj_000_0_00000000.tif'))
            ranked_methods = [1,10,11,4,2,5,7,3,6,9,13,8,12]
            m_track = 0
            for method in ranked_methods:
                imout = get_guessed_image_new(lines, angle, time_phase_bin, method, main_corr_dir)
                m_track += 1
                if m_track > last_method: last_method = m_track
                if len(imout)!= 0: break
            if len(imout) == 0: 
                missing_after_imputation +=1
            else: # found a good guess 
                imout = imout[0]
                new_num += 1
                d = os.path.join(current_corr_dir, './proj_000_0_000' +"{:05d}".format(new_num)+'.tif')
                im_out = Image.fromarray(imout.astype(np.uint16))
                im_out.info, im_out.format = template_image.info , template_image.format
                im_out.save(d, dpi= template_image.info['dpi'])
                all_nums.append(new_num); all_angles.append(angle); new_lines.append(np.zeros(lines[0].shape))
            template_image.close()
        else:
            missing_count +=1 
    # -------------------------- Final --------------------------------
    new_lines = np.array(new_lines)
    new_lines[:,1] = np.array(all_nums)
    new_lines[:,2] = np.array(all_angles)
    proj_file_new = open(os.path.join(os.path.dirname(current_corr_dir), 'proj_000_0_log.csv'), 'w')
    for line in new_lines: proj_file_new.write(','.join(line.astype(str)) + '\n')
    proj_file_new.close()
    print('Time phase:',time_phase_bin, '--Missing:',missing_count,'--Missing after guess:', missing_after_imputation, 'last_method:', last_method)
    # print('\t Percent:',missing_count/360 * 100)
    return 0

def populate_corr_dir_new_SNR(main_corr_dir, lines, time_phase_bin, final_phases, 
                      current_corr_dir, df):
    # Get 1 projection per angle 
    # Get needed variables
    angles = lines[:,2]
    pbs    = lines[:,5]
    unique_angles = np.unique(angles)
    # Start here
    missing_count, missing_after_imputation = 0, 0
    new_num = -1
    all_nums, all_angles,new_lines = [],[],[]
    template_image = Image.open(os.path.join(main_corr_dir,    'proj_000_0_00000000.tif'))
    for i in range(len(unique_angles)):
        angle = unique_angles[i]
        indeces_same_phase_angle  = np.where((pbs == time_phase_bin) & (angles == angle))[0]
        imgs = get_imgs_from_lists([indeces_same_phase_angle], main_corr_dir)
        if len(imgs) < 4:
            ranked_methods = [1,10,11,4,2,5,7,3,6,9,13,8,12]
            for method in ranked_methods:
                imout = get_guessed_image_new(lines, angle, time_phase_bin, method, main_corr_dir)
                imgs+=imout
                if len(imgs) > 4: break
        imout = np.array(imgs)[:4]
        # print(imout.shape)
        imout = np.mean(imout,0)
        new_num += 1
        d = os.path.join(current_corr_dir, './proj_000_0_000' +"{:05d}".format(new_num)+'.tif')
        im_out = Image.fromarray(imout.astype(np.uint16))
        im_out.info, im_out.format = template_image.info , template_image.format
        im_out.save(d, dpi= template_image.info['dpi'])
        all_nums.append(new_num); all_angles.append(angle); new_lines.append(np.zeros(lines[0].shape))
    template_image.close() 
    # -------------------------- Final --------------------------------
    new_lines = np.array(new_lines)
    new_lines[:,1] = np.array(all_nums)
    new_lines[:,2] = np.array(all_angles)
    proj_file_new = open(os.path.join(os.path.dirname(current_corr_dir), 'proj_000_0_log.csv'), 'w')
    for line in new_lines: proj_file_new.write(','.join(line.astype(str)) + '\n')
    proj_file_new.close()
    print('Time phase:',time_phase_bin, '--Missing:',missing_count,
          '--Missing after guess:', missing_after_imputation, '\t Percent:',missing_count/360 * 100)
    return 0

def remove_random(main_dir, percent, rename=False):
    images_folder = os.path.join(main_dir, 'ct-data', 'corr')
    proj_file     = os.path.join(main_dir, 'ct-data', 'proj_000_0_log.csv')
    images_folder_new = os.path.join(main_dir, 'ct-data', 'corr_new')
    proj_file_new     = os.path.join(main_dir, 'ct-data', 'proj_000_0_log_new.csv')
    f = open(proj_file, 'r')
    lines = np.array(f.readlines()); f.close()
    lines = np.array([x.rstrip().split(',') for x in lines])
    random_integers = sorted(random.sample(range(len(lines)), int(percent/100*360)))
    new_lines = lines[random_integers,:]
    old_name = [int(float(x)) for x in new_lines[:,1]]
    new_lines[:,1] = np.arange(new_lines.shape[0])
    new_name = [int(float(x)) for x in new_lines[:,1]]
    f = open(proj_file_new, 'w')
    for line in new_lines: f.write(','.join(line.astype(str)) + '\n')
    f.close()
    # Copy corr
    if not os.path.isdir(images_folder_new): os.mkdir(images_folder_new)
    for x,y in zip(old_name, new_name):
        d = os.path.join(images_folder_new, './proj_000_0_000' +"{:05d}".format(y) +'.tif')
        source = os.path.join(images_folder,    'proj_000_0_000' +"{:05d}".format(x) +'.tif')
        if not os.path.isfile(d): shutil.copy(source,d)
    # Overwrite !!!! 
    if rename:
        os.rename(images_folder, images_folder+'--original')
        os.rename(images_folder_new, images_folder)
        os.rename(proj_file, proj_file+'--original')
        os.rename(proj_file_new, proj_file)
    return 

def populate_corr_dir_new_missing(main_corr_dir, lines, time_phase_bin, final_phases, 
                      current_corr_dir, df, snr_all = False, impute = False, percent=100):
    # Get 1 projection per angle 
    # Get needed variables
    angles = lines[:,2]
    pbs    = lines[:,5]
    unique_angles = np.unique(angles)
    # Start here
    missing_count, missing_after_imputation = 0, 0
    new_num = -1
    all_nums, all_angles,new_lines = [],[],[]
    
    existing_angle = []
    for i in range(len(unique_angles)):
        angle = unique_angles[i]
        indeces_same_phase_angle  = np.where((pbs == time_phase_bin) & (angles == angle))[0]
        if len(indeces_same_phase_angle) ==0: continue
        existing_angle.append(i)
    existing_angle= np.array(existing_angle)
    print('found',len(existing_angle), 'projections')
    
    random_integers = sorted(random.sample(range(len(existing_angle)), int(percent/100*360)))
    print('will keep',len(random_integers), 'projections')
    
    ints_to_keep = existing_angle[random_integers]
    
    for i in range(len(unique_angles)):
        angle = unique_angles[i]
        indeces_same_phase_angle  = np.where((pbs == time_phase_bin) & (angles == angle))[0]
        if i not in ints_to_keep: indeces_same_phase_angle=[]
        if len(indeces_same_phase_angle) !=0: # Found a projection
            new_num += 1
            d = os.path.join(current_corr_dir, './proj_000_0_000' +"{:05d}".format(new_num) +'.tif')
            source = os.path.join(main_corr_dir,    'proj_000_0_000' +"{:05d}".format(indeces_same_phase_angle[0]) +'.tif')
            if not os.path.isfile(d): shutil.copy(source,d)
            all_nums.append(new_num); all_angles.append(angle); new_lines.append(np.zeros(lines[0].shape))
        elif impute: # Missing
            missing_count +=1
            template_image = Image.open(os.path.join(main_corr_dir,    'proj_000_0_00000000.tif'))
            ranked_methods = [1,10,11,4,2,5,7,3,6,9,13,8,12]
            for method in ranked_methods:
                imout = get_guessed_image_new(lines, angle, time_phase_bin, method, main_corr_dir)
                if len(imout)!= 0: break
            if len(imout) == 0: 
                missing_after_imputation +=1
            else: # found a good guess 
                imout = imout[0]
                new_num += 1
                d = os.path.join(current_corr_dir, './proj_000_0_000' +"{:05d}".format(new_num)+'.tif')
                im_out = Image.fromarray(imout.astype(np.uint16))
                im_out.info, im_out.format = template_image.info , template_image.format
                im_out.save(d, dpi= template_image.info['dpi'])
                all_nums.append(new_num); all_angles.append(angle); new_lines.append(np.zeros(lines[0].shape))
        else:
            missing_count +=1 
    # -------------------------- Final --------------------------------
    new_lines = np.array(new_lines)
    new_lines[:,1] = np.array(all_nums)
    new_lines[:,2] = np.array(all_angles)
    proj_file_new = open(os.path.join(os.path.dirname(current_corr_dir), 'proj_000_0_log.csv'), 'w')
    for line in new_lines: proj_file_new.write(','.join(line.astype(str)) + '\n')
    proj_file_new.close()
    print('Time phase =',time_phase_bin, '--Missing projections =',missing_count, missing_after_imputation)
    return 0


def correct_corr_dir(main_corr_dir, out_dir='',pixels=1):
    template_image = Image.open(os.path.join(main_corr_dir,    'proj_000_0_00000000.tif'))
    projections = [x for x in os.listdir(main_corr_dir) if '.tif' in x] 
    count = 0
    for i in trange(len(projections)):
        im_name = 'proj_000_0_000' +"{:05d}".format(i) +'.tif'
        im = iio.v3.imread(os.path.join(main_corr_dir,im_name))
        if out_dir == '':
            outname = os.path.join(main_corr_dir,im_name)
        else:
            outname = os.path.join(out_dir,im_name)
        # Correct
        
        if len(im[im==0]) != 0:
            # print('projection',i,'has',len(im[im<2000]), 'dead pixels')
            count +=1
            xs,ys = np.where(im == 0)
            for x,y in zip(xs,ys):
                # if x == 0: continue
                xmin = x-pixels if x > 0 else 0
                ymin = y-pixels if y > 0 else 0
                if im[x,y] != 0 : print(x,y, im[x,y])
                temp = im[xmin:x+pixels+1,ymin:y+pixels+1]
                im[x,y] = temp[temp != im[x,y]].mean()
            # Save
            im_out = Image.fromarray(im.astype(np.uint16))
            im_out.info, im_out.format = template_image.info , template_image.format
            im_out.save(outname, dpi= template_image.info['dpi'])
            im_out.close()
        elif not os.path.isfile(outname): 
            shutil.copy(os.path.join(main_corr_dir,im_name),
                        outname)
            
    template_image.close()
    print(count)
    
def make_gif(main_dir, outname, aslice, origin='lower'):
    adir = os.path.join(main_dir, outname, 'Results')
    # adir = adir.replace('Results', 'Mostafa\Results')
    images   = [x for x in os.listdir(adir) if '.nii' in x and 'mask' not in x]
    outdir   = os.path.join(adir, 'images')
    gif_name = os.path.join(adir, 'images','video.gif')
    if not os.path.isdir(outdir): os.mkdir(outdir)
    imarray = []
    fps = 100
    # vmin,vmax = -1500,1000
    vmin,vmax = -1150,350
    for i in range(len(images)):
        im = [x for x in os.listdir(adir) if '_R'+str(i)+'.nii' in x][0]
        img = nib.load(os.path.join(adir,im)).get_fdata()
        plt.imsave(os.path.join(outdir, im.replace('.nii','.png')),
                   img[:,aslice,:].T, cmap='gray',vmin=vmin,vmax=vmax,origin=origin)
        # plt.imsave(os.path.join(outdir, im.replace('.nii','.png')),
        #            img[:,aslice,:].T, cmap='gray',vmin=vmin,vmax=vmax)
        imarray.append(Image.open(os.path.join(outdir, im.replace('.nii','.png')))) # img.shape[1]//2
        # plt.imshow(img[:,img.shape[1]//2,:], cmap='gray',vmin=-1500,vmax=5000)
    imarray[0].save(gif_name, save_all=True, append_images=imarray[1:], 
                    duration=fps, loop=0, vmax=vmax,vmin=vmin, cmap='gray') 

    pngs = [os.path.join(outdir,x) for x in os.listdir(outdir) if '.png' in x]
    imgs = []
    for i in range(len(pngs)):
        im = [os.path.join(outdir,x) for x in os.listdir(outdir) if '_R'+str(i)+'.png' in x][0]
        imgs.append(Image.open(im))
    im = Image.open(pngs[0])
    imout = Image.new('RGB', (im.width *4, im.height*4))
    imout.paste(imgs[0], (im.width*0, im.height*0))
    imout.paste(imgs[1], (im.width*1, im.height*0))
    imout.paste(imgs[2], (im.width*2, im.height*0))
    imout.paste(imgs[3], (im.width*3, im.height*0))
    imout.paste(imgs[4], (im.width*0, im.height*1))
    imout.paste(imgs[5], (im.width*1, im.height*1))
    imout.paste(imgs[6], (im.width*2, im.height*1))
    imout.paste(imgs[7], (im.width*3, im.height*1))
    imout.paste(imgs[8], (im.width*0, im.height*2))
    imout.paste(imgs[9], (im.width*1, im.height*2))
    imout.paste(imgs[10], (im.width*2, im.height*2))
    imout.paste(imgs[11], (im.width*3, im.height*2))
    imout.paste(imgs[12], (im.width*0, im.height*3))
    imout.paste(imgs[13], (im.width*1, im.height*3))
    imout.paste(imgs[14], (im.width*2, im.height*3))
    imout.paste(imgs[15], (im.width*3, im.height*3))

    imout.save(outdir+'/all.tif')
    imout.close()
    for im in imgs: im.close()
    print('Done')
    
def make_gif_cropped(main_dir, outname, origin='lower', vmin = -1150, vmax=350, frame= -1):
    # Initialize
    adir = os.path.join(main_dir, outname, 'Results')    
    outdir   = os.path.join(adir, 'images_cropped')
    gif_name = os.path.join(adir, 'images_cropped','video.gif')
    if not os.path.isdir(outdir): os.mkdir(outdir)
    mask_fname = os.path.join(adir, 'mask.npy')
    if not os.path.exists(mask_fname): 
        print('Failed! There are no segmentation files!')
        return -1
    # Load and Crop Data
    mask = np.load(os.path.join(main_dir, outname,'Results', 'mask.npy'))
    raw  = np.load(os.path.join(main_dir, outname,'Results', 'raw.npy'))
    ei_phase = np.sum(mask,axis=(1,2,3)).argmax()
    cropped_images, cropped_masked_images = [], []
    for i in range(raw.shape[0]):  
        im1, im2= crop_to_mask(raw[i] * mask[ei_phase],padding=5, im2=raw[i])
        cropped_images.append(im2)
        cropped_masked_images.append(im1)
    cropped_images = correct_cropped_images(cropped_images)
    cropped_masked_images = correct_cropped_images(cropped_masked_images) 
    cropped_masked_images_bool = cropped_masked_images.astype(bool) 
    axial_slices = np.zeros((cropped_masked_images_bool.shape[3])) 
    for i in range(cropped_masked_images_bool.shape[3]):axial_slices[i] = np.sum(cropped_masked_images_bool[0,:,:,i])
    indeces_for_small_axial = np.where(axial_slices < 300)[0]
    slice_axial = indeces_for_small_axial[np.diff(indeces_for_small_axial).argmax()+1]
    # Extra Crop
    cropped_masked_images = cropped_masked_images[:,:,:,:slice_axial+30]
    cropped_images = cropped_images[:,:,:,:slice_axial+30]
    # select Slice
    if frame == -1: 
        # sls =[]
        # for i in range(cropped_masked_images.shape[2]): sls.append(np.sum(cropped_masked_images[:,:,i].astype(bool)))
        # aslice = np.argmax(sls)
        num = cropped_masked_images.shape[2]
        save_gif(int(num*0.3), raw, outdir, vmin, vmax, origin, 
                 os.path.join(adir, 'images_cropped','video_anterior.gif'),
                 cropped_images)
        save_gif(int(num*0.5), raw, outdir, vmin, vmax, origin, 
                 os.path.join(adir, 'images_cropped','video_mid.gif'),
                 cropped_images)
        save_gif(int(num*0.7), raw, outdir, vmin, vmax, origin, 
                 os.path.join(adir, 'images_cropped','video_posterior.gif'),
                 cropped_images)
    else:
        aslice = frame
        gif_name = os.path.join(adir, 'images_cropped','video.gif')
        save_gif(aslice, raw, outdir, vmin, vmax, origin, gif_name, cropped_images)
    print('Done')
    
def save_gif(aslice, raw, outdir, vmin, vmax, origin, gif_name, cropped_images):
    # Save
    imarray = []
    fps = 100
    for i in range(raw.shape[0]):
        plt.imsave(os.path.join(outdir, 'R_'+str(i)+'.png'),
                   cropped_images[i,:,aslice].T, cmap='gray',vmin=vmin,vmax=vmax,origin=origin)
        imarray.append(Image.open(os.path.join(outdir, 'R_'+str(i)+'.png')))
    imarray[0].save(gif_name, save_all=True, append_images=imarray[1:], 
                    duration=fps, loop=0, vmax=vmax,vmin=vmin, cmap='gray')
    # ---------------------------------------------------
    pngs = [os.path.join(outdir,x) for x in os.listdir(outdir) if '.png' in x]
    imgs = []
    for i in range(len(pngs)):
        im = [os.path.join(outdir,x) for x in os.listdir(outdir) if 'R_'+str(i)+'.png' in x][0]
        imgs.append(Image.open(im))
    im = Image.open(pngs[0])
    imout = Image.new('RGB', (im.width *4, im.height*4))
    imout.paste(imgs[0], (im.width*0, im.height*0))
    imout.paste(imgs[1], (im.width*1, im.height*0))
    imout.paste(imgs[2], (im.width*2, im.height*0))
    imout.paste(imgs[3], (im.width*3, im.height*0))
    imout.paste(imgs[4], (im.width*0, im.height*1))
    imout.paste(imgs[5], (im.width*1, im.height*1))
    imout.paste(imgs[6], (im.width*2, im.height*1))
    imout.paste(imgs[7], (im.width*3, im.height*1))
    imout.paste(imgs[8], (im.width*0, im.height*2))
    imout.paste(imgs[9], (im.width*1, im.height*2))
    imout.paste(imgs[10], (im.width*2, im.height*2))
    imout.paste(imgs[11], (im.width*3, im.height*2))
    imout.paste(imgs[12], (im.width*0, im.height*3))
    imout.paste(imgs[13], (im.width*1, im.height*3))
    imout.paste(imgs[14], (im.width*2, im.height*3))
    imout.paste(imgs[15], (im.width*3, im.height*3))
    imout.save(outdir+'/all.tif')
    imout.close()
    for im in imgs: im.close()