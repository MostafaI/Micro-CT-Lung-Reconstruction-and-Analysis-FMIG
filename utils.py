import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import os
from scipy.ndimage import gaussian_filter
from scipy import ndimage
import time
import warnings
import pydicom
from scipy.stats import linregress
import seaborn as sns
from dicom2nifti.convert_dicom import dicom_series_to_nifti
import pandas as pd 

# from lungmask import LMInferer
# import SimpleITK as sitk


def extract_ct_mask(
    file,
    threshold=-200,
    indentifier="",
    maskdir=None,
    lungdir=None,
    save_files=True,
    lung_at_boundaries=False,
    printtolog=True):
    def touch_boundary(image):
        x, y, z = np.array(image.shape) - 1
        faces = []
        faces.append(np.sum(image[0, :, :]))
        faces.append(np.sum(image[x, :, :]))
        faces.append(np.sum(image[:, 0, :]))
        faces.append(np.sum(image[:, y, :]))
        faces.append(np.sum(image[:, :, 0]))
        faces.append(np.sum(image[:, :, z]))
        faces.remove(max(faces))  # Allow 1 side to have values
        faces.remove(max(faces))  # Allow 2 sides to have values
        if np.sum(faces) == 0:
            return False
        else:
            return True

    if type(file) == str:
        mask_name = file.replace(".nii", "_m.nii")
        lung_name = file.split(".")[0] + indentifier + "_lung.nii.gz"
        if maskdir != None:
            mask_name = os.path.join(maskdir, mask_name.split("/")[-1])
        if lungdir != None:
            lung_name = os.path.join(lungdir, lung_name.split("/")[-1])
        if os.path.isfile(mask_name):
            if printtolog:
                print("Skipping segmentation for " + file.split("/")[-1])
                print("segmentation exists!")
                print(mask_name)
            return 0

    if printtolog:
        print("Segmenting " + file.split("/")[-1])
    start = time.time()

    if type(file) == str:
        img = nib.load(file)
        im = img.get_fdata()
    else:
        im = file
    # Gaussian filter the image
    im = gaussian_filter(im, sigma=3)
    some_size = im.shape[0] * im.shape[1] * im.shape[2] / 20
    mask = im < threshold
    labeled_image, num_labels = ndimage.label(mask)
    labeled_image[labeled_image == labeled_image[0, 0, 0]] = 0  # Set background to 0
    label_sizes = np.bincount(labeled_image.ravel())  # count label sizes
    label_sizes[0] = 0
    count = 0
    if printtolog:
        print("Segmenting file")
    while True:
        count += 1
        lung_label = np.argmax(label_sizes)
        mask = labeled_image == lung_label
        if lung_at_boundaries or not touch_boundary(
            mask
        ):  # and mask.sum() > some_size:
            break
        else:
            label_sizes[lung_label] = 0  # reset this index to 0
        if count == 10:  # Avoid infinite loop
            if printtolog:
                print("Have a problem with this segmentation!")
                raise Exception("Have a problem with this segmentation!")
            return -1
    # mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_closing(mask).astype(np.uint8)
    if printtolog:
        print("Done")
    if save_files == False:
        return mask
    # mask = ndimage.binary_fill_holes(mask,structure=np.ones((1,4,2))).astype(np.uint8)
    nii_mask = nib.Nifti1Image(mask, img.affine)
    nii_mask.to_filename(mask_name)
    # Get Lungs
    # lung = im * mask
    # lung = nib.Nifti1Image(lung.astype(np.int16), img.affine)
    # lung.to_filename(lung_name)
    if printtolog:
        print(
            "Segmentation for "
            + file.split("/")[-1]
            + " finished in "
            + str(int(time.time() - start))
            + " seconds"
        )
    return 0


# In[11]:


def mostafa_average(image, axis):
    im_zero = image.copy()
    im_zero[np.isnan(im_zero)] = 0
    my_ones = np.ones(im_zero.shape)
    my_ones[im_zero == 0] = 0
    r = np.nansum(im_zero, axis=axis)
    s = np.nansum(my_ones, axis=axis) + 1e-12
    ret = np.divide(r, s).astype(np.float32)
    if np.any(np.isnan(image)):
        ret[ret == 0] = np.nan
    return ret  # , where= (s!=0))


def get_x_slices_from_image(
    im, number_of_slices=6, coronal_axis=0, transpose=False, airways=False, smooth=False
):
    # Assumuing coronal slice at the middle
    # Assuming 3D im
    if coronal_axis not in [0, 1, 2]:
        print("coronal_axis is wrong! Expected 0,1,2, but got", coronal_axis)
        return -1
    num_slices = im.shape[coronal_axis]
    cluster_size = int(num_slices / number_of_slices)
    image = []
    if airways:
        x = np.nanmean(im, coronal_axis)
        mask = x != 0
        x[x == 0] = np.nan
        x = filter_nan_gaussian_conserving(x, 1) * mask
        image.append(x)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for i in range(number_of_slices):
                s = i * cluster_size
                e = (i + 1) * cluster_size
                center = s + (e - s) // 2
                if smooth:
                    if i == number_of_slices:
                        if coronal_axis == 0:
                            slices = im[s:,:, :]
                        elif coronal_axis == 1:
                            slices = im[:,s:,:]
                        elif coronal_axis == 2:
                            slices = im[:,:, s:]
                    else:
                        if coronal_axis == 0:
                            slices = im[s:e,:, :]
                        elif coronal_axis == 1:
                            slices = im[:,s:e,:]
                        elif coronal_axis == 2:
                            slices = im[:,:, s:e]
                    image.append(mostafa_average(slices,axis=coronal_axis))

                else:
                    if coronal_axis == 0:
                        slices = im[center, :, :]
                    elif coronal_axis == 1:
                        slices = im[:, center, :]
                    elif coronal_axis == 2:
                        slices = im[:, :, center]
                    image.append(slices)
    if transpose:
        image2 = []
        for im in image:
            image2.append(im.T)
        image = image2
    image = np.array(image)
    return image

def get_cropped_CT_images(
    main_dir, subdir, resample_size=4, jacobian=False, use_voxels=False, raw=False
):
    if jacobian:
        reg = np.load(
            os.path.join(main_dir, "Analysis", "Registered", subdir, "jacobian.npy")
        )
    elif raw:
        reg = mask = np.load(os.path.join(main_dir, "Raw", "CT", subdir, "raw.npy"))
    else:
        reg = np.load(
            os.path.join(main_dir, "Analysis", "Registered", subdir, "registered.npy")
        )

    mask = np.load(os.path.join(main_dir, "Raw", "CT", subdir, "mask.npy"))[0]

    # Resample first
    if resample_size == 1:
        lung = reg.astype(np.float32)
    else:
        #  one of 0 (linear), 1 (nearest neighbor), 2 (gaussian), 3 (windowed sinc), 4 (bspline)
        lung_resampled = []
        mask_ants = ants.from_numpy(mask.astype(np.float32))
        mask = ants.resample_image(
            mask_ants,
            (resample_size, resample_size, resample_size),
            use_voxels=use_voxels,
            interp_type=1,
        ).numpy()
        for i in range(16):
            lung_ants = ants.from_numpy(reg[i].astype(np.float32))
            lung_resampled.append(
                ants.resample_image(
                    lung_ants,
                    (resample_size, resample_size, resample_size),
                    use_voxels=use_voxels,
                    interp_type=2,
                ).numpy()
            )
        lung_resampled = np.array(lung_resampled)
        lung = lung_resampled.astype(np.float32)

    # Crop to lungs
    lungs = []
    for i in range(lung.shape[0]):
        lungi = crop_to_mask(mask * lung[i], padding=1)
        if len(lungi) == 2:
            lungi = lungi[0]
        lungs.append(lungi)
    lungs = correct_cropped_images(lungs)

    # Axes: 1-saggital, 2-coronal, 3-axial
    # # Smooth each slices
    # imgs = np.zeros(lungs.shape)
    # for i in range(lungs.shape[0]):
    #     for j in range(lungs.shape[1]):
    #         imgs[i,j] = gf(lungs[i,j],1)
    # lungs = imgs

    # # Get Air in lungs
    # if not jacobian:
    #     lungs[lungs ==0] = np.nan
    #     lungs /= -1000
    #     lungs[lungs < 0.1] = np.nan

    # air = lungs.copy()
    # air[np.isnan(air)]=0
    # s = air.sum(axis=(1,2,3))
    # air_modefied = np.roll(air,-s.argmax(), axis=0)
    return lungs


def crop_to_mask(img, padding=4, nan_zeros=False, im2=None):
    dim = len(img.shape)
    if im2 is not None: 
        img_not_cropped = im2.copy()
    else:
        img_not_cropped = img.copy()
    if dim == 3:
        filled_axis = img.any(axis=(0, 1))
        img = img[:, :, filled_axis]
        img_not_cropped = img_not_cropped[:, :, filled_axis]
        filled_axis = img.any(axis=(0, 2))
        img = img[:, filled_axis, :]
        img_not_cropped = img_not_cropped[:, filled_axis, :]
        filled_axis = img.any(axis=(1, 2))
        img = img[filled_axis, :, :]
        img_not_cropped = img_not_cropped[filled_axis, :, :]
    elif dim == 2:
        filled_axis = img.any(axis=0)
        img = img[:, filled_axis]
        img_not_cropped = img_not_cropped[:, filled_axis]
        filled_axis = img.any(axis=1)
        img = img[filled_axis,]
        img_not_cropped = img_not_cropped[filled_axis,]
    else:
        raise Exception("The image is not 2D or 3D!")
    img = np.pad(img, padding, "constant", constant_values=(0))
    img_not_cropped = np.pad(img_not_cropped, padding, "constant", constant_values=(0))
    if nan_zeros: 
        img = img.astype(np.float32)
        img[img == 0] = np.nan
    if im2 is None: return img
    return img, img_not_cropped


def correct_cropped_images(im):
    # Correct cropped Images
    minx, miny, minz = 9999999, 9999999, 9999999
    for i in range(len(im)):
        if minx > im[i].shape[0]:
            minx = im[i].shape[0]
        if miny > im[i].shape[1]:
            miny = im[i].shape[1]
        if minz > im[i].shape[2]:
            minz = im[i].shape[2]
    imout = []
    for i in range(len(im)):
        imout.append(im[i][:minx, :miny, :minz])
    return np.array(imout)


def save_image(im, imname, vmin, vmax, cmap="jet", title=""):
    s = 2
    num_slices = len(im)
    num_cols = 6
    num_rows = int(np.ceil(num_slices / num_cols))
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(num_cols * s, num_rows * s), dpi=400, facecolor="k"
    )
    cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])  # Adjust the position as needed
    axes = axes.flatten()
    for i in range(num_slices):
        img = axes[i].imshow(im[i].T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        axes[i].axis("off")
    for j in range(num_slices, len(axes)):
        axes[j].axis("off")
    # Add a single colorbar for the whole figure
    cbar = fig.colorbar(img, cax=cbar_ax)
    cbar.set_ticks([vmin, (vmin + vmax) / 2, vmax])
    cbar.ax.yaxis.set_tick_params(color="white")  # Change tick color
    plt.setp(
        plt.getp(cbar.ax, "yticklabels"), color="white", fontsize=20
    )  # Change tick label color
    fig.suptitle(title, color="white", fontsize=24, weight="bold")
    plt.savefig(imname, bbox_inches="tight", dpi=400)
    plt.close()

def get_inter_heterogeneity(im, 
                            plot=False, 
                            slice_thickness = 0.1,
                            percent_to_ignore = 5,
                            step_size = 1,
                           savename = ''):
    # slice_thickness in cm
    im[np.isnan(im)] = 0
    im = crop_to_mask(im,0)
    im[im==0]=np.nan
    
    num_coronal_slices=im.shape[1]
    slices_to_ignore = int(num_coronal_slices * percent_to_ignore/100)
    slices = []
    # Define the step size (in this case, 10)
    slice_thickness *= step_size
    for i in range(slices_to_ignore, num_coronal_slices - slices_to_ignore, step_size):
        slices.append(np.nanmean(im[:, i:i + step_size]))
    y = slices
    x = np.arange(len(y))
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    if plot or savename !=0:
        sns.set(style="whitegrid", font_scale=1.5)
        plt.figure(figsize=(12, 4))  # Adjust the figure size as needed
        x_range = np.linspace(min(x), max(x), 100)
        y_line = slope * x_range + intercept
        plt.plot(x, y, marker='o', linestyle='', label='Data', markersize=5, color='b')
        plt.plot(x_range, y_line, label=f'Slope = {round(slope/slice_thickness, 4)} /cm', color='r', linewidth=2)
        plt.xlabel('Coronal Slices')
        plt.ylabel('Average value per slice')
        plt.legend()
        plt.title('Interregional Heterogeneity')
        if savename != '':
            plt.savefig(savename, dpi=300, bbox_inches='tight')
            plt.close()
    return slope/slice_thickness

def get_intra_heterogeneity(im, 
                            percent_to_ignore = 5, 
                            step_size = 1, 
                            plot=False,
                           savename = '', 
                           robust=True):
    im[np.isnan(im)] = 0
    im = crop_to_mask(im,0)
    im[im==0] = np.nan
    global_mean = np.nanmean(im)
    num_coronal_slices=im.shape[1]
    slices_to_ignore = int(num_coronal_slices * percent_to_ignore/100)
    per_slice_heterogeneities = []
    for i in range(slices_to_ignore, num_coronal_slices - slices_to_ignore, step_size):
        per_slice_heterogeneities.append(
            np.nanstd(im[:, i:i + step_size]) / abs(np.nanmean(im[:, i:i + step_size])))
        
    intra = np.mean(per_slice_heterogeneities)
    if robust: 
        intra = np.median(per_slice_heterogeneities) 
        
    if plot or savename !='':
        y = per_slice_heterogeneities
        x = np.arange(len(y))
        sns.set(style="whitegrid", font_scale=1.5)
        plt.figure(figsize=(12, 4))  # Adjust the figure size as needed
        x_range = np.linspace(min(x), max(x), 100)
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        y_line = slope * x_range + intercept
        plt.plot(x, y, marker='o', linestyle='', label='Data', markersize=5, color='b')
        plt.plot(x_range, y_line, color='r', linewidth=2)
        plt.xlabel('Coronal Slices')
        plt.ylabel('Average std per slice')
        plt.legend()
        plt.title('Intraregional Heterogeneity')
        if savename != '':
            plt.savefig(savename, dpi=300, bbox_inches='tight')
            plt.close()
    return intra


def save_image_MI_style(im, vmin, vmax, out_path, dpi=300):
    num_slices = im.shape[0]
    fig, axes = plt.subplots(
        1, num_slices, figsize=(16, 4), dpi=400, facecolor="k"
    )
    cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
    for i, ax in enumerate(axes):
        img = ax.imshow(
            im[i].T,
            origin="lower",
            cmap="jet",
            vmin=vmin,
            vmax=vmax,
        )
        ax.axis("off")
    cbar = fig.colorbar(img, cax=cbar_ax)
    cbar.set_ticks([vmin, (vmax-vmin)/2+vmin, vmax])
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.get_yticklabels(), color="white", fontsize=20)
    plt.savefig(out_path, bbox_inches="tight", dpi=dpi, facecolor="black")
    plt.close(fig)
    return 
    
def image_array_to_png(
    image: np.ndarray,
    mask: np.ndarray,
    out_image: str,
    dpi=300,
    overwrite=True,
    num_slices=6,
    vmin = 0,
    vmax = 1
):
    if os.path.exists(out_image) and not overwrite:
        return 0

    # --- core computation ---
    masked_image = image * mask
    cropped_masked_image = crop_to_mask(masked_image, padding=4)
    cropped_masked_image[cropped_masked_image == 0] = np.nan

    modes = [
        (False, out_image),
        (True, out_image.replace(".png", "_smoothed.png")),
    ]

    for smooth, out_path in modes:
        im = get_x_slices_from_image(
            cropped_masked_image,
            number_of_slices=num_slices,
            coronal_axis=1,
            transpose=False,
            airways=False,
            smooth=smooth,
        )
        save_image_MI_style(im, vmin, vmax, out_path, dpi)
    return
    
        
def get_air_content_image(image_nifti_file_path, mask_nifti_file_path, out_image, dpi=300, **kwargs):
    kwargs.setdefault("overwrite", False)
    overwrite = kwargs['overwrite']
    if os.path.exists(out_image) and not overwrite: return 0
    try:
        # Loading the images
        image = nib.load(image_nifti_file_path).get_fdata()
        mask = nib.load(mask_nifti_file_path).get_fdata()         
        masked_image = image * mask
        cropped_masked_image = crop_to_mask(masked_image, padding=4)
        # Calculate air contenct
        air_content = cropped_masked_image / -1000
        air_content[air_content == 0] = np.nan
        num_slices = 6
        modes = [(False, out_image), (True, out_image.replace('.png', '_smoothed.png'))] 
        for smooth, out_path in modes:
            im = get_x_slices_from_image(
                air_content,
                number_of_slices=num_slices,
                coronal_axis=1,
                transpose=False,
                airways=False,
                smooth = smooth
            )
            s = 1
            vmin, vmax = 0, 1
            fig, axes = plt.subplots(
                1, num_slices, figsize=(16 * s, 4 * s), dpi=400, facecolor="k"
            )
            cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])  # Adjust the position as needed
            for i, ax in enumerate(axes):
                img = ax.imshow(im[i].T, origin="lower", cmap="jet", vmin=vmin, vmax=vmax)
                ax.axis("off")
            # Add a single colorbar for the whole figure
            cbar = fig.colorbar(img, cax=cbar_ax)
            cbar.set_ticks([vmin, (vmin + vmax) / 2, vmax])
            cbar.ax.yaxis.set_tick_params(color="white")  # Change tick color
            plt.setp(plt.getp(cbar.ax, "yticklabels"), color="white", fontsize=20)  # \
            parent_dir = os.path.dirname(image_nifti_file_path)
            plt.savefig(out_path, bbox_inches="tight", dpi=dpi, facecolor='black')
            plt.close()
    except Exception as e:
        print(f'Error: {e}')
    
def run(dicom_dir):
    start = time.time()
    parent_dir = os.path.dirname(dicom_dir)
    image_name = os.path.basename(dicom_dir)
    out_dir = os.path.join(parent_dir, image_name+'__Analyzed') 
    if not os.path.isdir(out_dir): os.mkdir(out_dir)
    image_nifti_file_path = os.path.join(out_dir, image_name + ".nii.gz")
    mask_nifti_file_path  = image_nifti_file_path.replace(".nii", "_m.nii")
    out_image = os.path.join(out_dir, image_name + "__air_contect.png")
    out_image_inter = os.path.join(out_dir, image_name + "__inter.png")
    out_image_intra = os.path.join(out_dir, image_name + "__intra.png")
    out_csv = os.path.join(out_dir, image_name + ".csv")
    
    #1) ------------Convert DICOM to NIFTI-------------------
    if not os.path.isfile(image_nifti_file_path):
        print("Converting Dicom to NIFT..")
        dicom_series_to_nifti(dicom_dir, image_nifti_file_path)
    else:
        print("NIFTI files are already there..skip..")
        
    #2) ------------Segmenting the lungs-------------------
    if not os.path.isfile(mask_nifti_file_path):
        print("Segmenting the lungs..")
        extract_ct_mask(image_nifti_file_path, printtolog=False)
    else:
        print("Segmentation mask is already there..skip..")

    #3) ------------Getting the Air content image-------------------
    if not os.path.isfile(out_image):
        print("Getting the air content map..")
        get_air_content_image(image_nifti_file_path, mask_nifti_file_path, out_image)
    else:
        print("The air content image is already there..skip..")

    #4) ------------Getting the Heterogenities-------------------
    if not os.path.isfile(out_csv):
        print("Getting the heterogeneity maps..")
        run_heterogeneity_analysis(image_nifti_file_path, 
                                                mask_nifti_file_path,
                                                out_image_inter,
                                                out_image_intra,
                                                out_csv)  
    else:
        print("The heterogeneity csv file is already there..skip..")
        
    print(
        "Program is complete, finished in " + str(int(time.time() - start)) + " seconds"
    )

def run_heterogeneity_analysis(image_nifti_file_path, 
                               mask_nifti_file_path,
                               out_image_inter,
                               out_image_intra,
                               out_csv
                              ):
    # Loading the images
    imagename = os.path.basename(out_csv).replace('.csv','')
    image = nib.load(image_nifti_file_path).get_fdata()
    mask = nib.load(mask_nifti_file_path).get_fdata()         
    voxel_sizes = nib.load(image_nifti_file_path).header.get_zooms()
    masked_image = image * mask
    cropped_masked_image = crop_to_mask(masked_image, padding=4, im2=None)
    
    inter = get_inter_heterogeneity(cropped_masked_image, 
                                    plot=False, 
                                    slice_thickness = voxel_sizes[1],
                                    percent_to_ignore = 10,
                                    step_size = 5,
                                    savename = out_image_inter)

    intra = get_intra_heterogeneity(cropped_masked_image, 
                            percent_to_ignore = 10, 
                            step_size = 5, 
                            plot=False,
                           savename = out_image_intra)

    df = pd.DataFrame({
        'Name': ['', 'Inter-heterogeneity', 'Intra-heterogeneity'],
        'Value': [imagename, inter, intra]
    })
    df.to_csv(out_csv, index=False, header=False)
    return inter, intra
    
if __name__ == "__main__":
    dicom_dir = "/Users/mostafaismail/Documents/Carly/S6060_MapleSyrup/S6020"
    dicom_dir = input("Dicom Path: ").strip()
    run(dicom_dir)
