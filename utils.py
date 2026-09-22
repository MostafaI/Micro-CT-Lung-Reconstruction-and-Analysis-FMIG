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


def _lung_mask_largest(im, threshold, lung_at_boundaries, printtolog):
    """Original behaviour: after zeroing the component that owns the [0,0,0]
    corner, take the largest remaining air component (optionally skipping any
    that reach the frame boundary). Kept for algo='largest'."""
    def touch_boundary(image):
        x, y, z = np.array(image.shape) - 1
        faces = [np.sum(image[0, :, :]), np.sum(image[x, :, :]),
                 np.sum(image[:, 0, :]), np.sum(image[:, y, :]),
                 np.sum(image[:, :, 0]), np.sum(image[:, :, z])]
        faces.remove(max(faces))  # allow 1 side to have values
        faces.remove(max(faces))  # allow 2 sides to have values
        return np.sum(faces) != 0

    mask = im < threshold
    labeled_image, _ = ndimage.label(mask)
    labeled_image[labeled_image == labeled_image[0, 0, 0]] = 0  # drop exterior air
    label_sizes = np.bincount(labeled_image.ravel())
    label_sizes[0] = 0
    count = 0
    while True:
        count += 1
        lung_label = np.argmax(label_sizes)
        mask = labeled_image == lung_label
        if lung_at_boundaries or not touch_boundary(mask):
            break
        label_sizes[lung_label] = 0
        if count == 10:  # avoid infinite loop
            if printtolog:
                print("Have a problem with this segmentation!")
                raise Exception("Have a problem with this segmentation!")
            return None
    return ndimage.binary_closing(mask).astype(np.uint8)


def _lung_mask_shape(im, threshold, zooms, printtolog=False, fill_holes=True):
    """Select lung air by physical size / shape / position instead of raw size.

    `im` is the already-smoothed volume in HU; `zooms` is the (a0, a1, a2)
    voxel size in mm, so every gate below is in millimetres / millilitres and
    the same numbers work at 60 um and at 200 um.

    Strategy: drop air that reaches >= 3 frame faces (the exterior / cylindrical
    FOV rind and its streak-artifact tendrils), then among what is left keep the
    component(s) that look like a lung - a few mL, reasonably compact (a real rat
    lung fills ~1/4 of its bounding box; streak soup fills ~1/15), not spanning
    the whole frame, roughly centred in the two in-plane axes.
    """
    shape = np.array(im.shape)
    frame_mm = shape * np.asarray(zooms, float)
    voxvol_ml = float(np.prod(zooms)) / 1000.0            # mm^3 -> mL
    scan_ax = int(np.argmin(shape))                       # fewest samples = stack axis
    trans_ax = [a for a in range(3) if a != scan_ax]      # the two in-plane axes
    finite = np.isfinite(im)

    def interior_labels(air_bool):
        """Label the air, then zero every component that reaches >= 3 frame
        faces (exterior air / cylindrical-FOV rind). Survivor labels are left
        as-is - numbering is gappy but bincount / find_objects handle that -
        so this costs one label pass, not two."""
        lab, n = ndimage.label(air_bool)
        if n == 0:
            return lab, []
        drop = np.zeros(n + 1, bool)
        survivors = []
        for i, sl in enumerate(ndimage.find_objects(lab), start=1):
            if sl is None:
                continue
            faces = sum(int(s.start == 0) + int(s.stop == d) for s, d in zip(sl, shape))
            if faces >= 3:
                drop[i] = True
            else:
                survivors.append(i)
        if drop.any():
            lab[drop[lab]] = 0                    # one O(n) pass via lookup table
        return lab, survivors

    def pick(air_bool, min_ml, max_ml, min_ext_mm, min_bbox_fill, max_ext_frac, clo, chi):
        lab, survivors = interior_labels(air_bool)
        if not survivors:
            return None, [], []
        sizes = np.bincount(lab.ravel())
        objs = ndimage.find_objects(lab)
        # a lung is always one of the largest air components once the exterior
        # is gone; only score the biggest handful (bounds work + the log).
        survivors.sort(key=lambda i: sizes[i], reverse=True)
        kept, rej = [], []
        for i in survivors[:12]:
            sl = objs[i - 1]
            cnt = int(sizes[i])
            vol_ml = cnt * voxvol_ml
            ext_mm = np.array([(s.stop - s.start) * z for s, z in zip(sl, zooms)])
            ext_frac = ext_mm / frame_mm
            bbox_vox = float(np.prod([s.stop - s.start for s in sl]))
            bbox_fill = cnt / max(bbox_vox, 1.0)
            faces = sum(int(s.start == 0) + int(s.stop == d) for s, d in zip(sl, shape))
            cen = np.array([(s.start + s.stop) / 2.0 / d for s, d in zip(sl, shape)])
            tmin = float(min(ext_mm[a] for a in trans_ax))
            rec = (i, round(float(vol_ml), 2), tuple(round(float(e), 1) for e in ext_mm),
                   round(float(bbox_fill), 3), faces, tuple(round(float(c), 2) for c in cen))
            ok = (min_ml <= vol_ml <= max_ml
                  and tmin >= min_ext_mm
                  and bbox_fill >= min_bbox_fill
                  and float(ext_frac.max()) <= max_ext_frac
                  and faces <= 2
                  and all(clo <= cen[a] <= chi for a in trans_ax))
            (kept if ok else rej).append(rec)
        if not kept:
            return None, kept, rej
        keep_ids = np.array([rec[0] for rec in kept])
        return np.isin(lab, keep_ids), kept, rej

    # HU ladder: a fully inflated lung is mostly air (< -200 HU) but a lung at
    # end-expiration sits far denser (-400..-700 HU central), so -200 finds only
    # crumbs. Start at the caller's threshold and raise it only if nothing
    # lung-shaped turns up, so inflated phases keep the full -200 mask.
    ladder = [threshold] + [t for t in (-300, -350, -400, -450) if t < threshold]
    lung = kept = rej = None
    stage = None
    for thr in ladder:  # 1) strict gates
        lung, kept, rej = pick(finite & (im < thr), 1.5, 15.0, 12.0, 0.14, 0.85, 0.18, 0.82)
        if lung is not None:
            stage = f"shape@{thr}"
            break
    if lung is None:  # 2) relaxed gates, same ladder
        for thr in ladder:
            lung, kept, rej = pick(finite & (im < thr), 0.8, 15.0, 9.0, 0.11, 0.90, 0.12, 0.88)
            if lung is not None:
                stage = f"shape-relaxed@{thr}"
                break
    if lung is None:  # 3) largest interior blob at the most permissive threshold
        lab, survivors = interior_labels(finite & (im < ladder[-1]))
        if survivors:
            sizes = np.bincount(lab.ravel())
            lung = lab == max(survivors, key=lambda i: sizes[i])
            stage = f"largest-interior-air@{ladder[-1]}"
    if lung is None:  # 4) last resort: largest air not owning the [0,0,0] corner
        lab, n = ndimage.label(finite & (im < threshold))
        if n:
            lab[lab == lab[0, 0, 0]] = 0
            s = np.bincount(lab.ravel()); s[0] = 0
            lung = lab == int(np.argmax(s))
        else:
            lung = np.zeros(im.shape, bool)
        stage = "largest-air-fallback"

    # closing + hole-fill only inside the lung's bounding box (+2 vox), so these
    # stay cheap on a 1e9-voxel 60 um volume instead of scanning the whole thing.
    if lung.any():
        loc = ndimage.find_objects(lung.view(np.uint8))[0]
        box = tuple(slice(max(0, s.start - 2), min(int(d), s.stop + 2))
                    for s, d in zip(loc, shape))
        sub = ndimage.binary_closing(lung[box])
        if fill_holes:
            sub = ndimage.binary_fill_holes(sub)
        lung[box] = sub

    if printtolog:
        rshort = rej if len(rej) <= 8 else rej[:8] + [f"...+{len(rej) - 8} more"]
        print(f"  lung selection [{stage}]  kept={kept}  rejected={rshort}")
    return lung.astype(np.uint8)


def extract_ct_mask(
    file,
    threshold=-200,
    indentifier="",
    maskdir=None,
    lungdir=None,
    save_files=True,
    lung_at_boundaries=False,
    printtolog=True,
    algo="shape",
    zooms=None):
    name = os.path.basename(file) if isinstance(file, str) else "<array>"
    if isinstance(file, str):
        mask_name = file.replace(".nii", "_m.nii")
        lung_name = file.split(".")[0] + indentifier + "_lung.nii.gz"
        if maskdir != None:
            mask_name = os.path.join(maskdir, mask_name.split("/")[-1])
        if lungdir != None:
            lung_name = os.path.join(lungdir, lung_name.split("/")[-1])
        if os.path.isfile(mask_name):
            if printtolog:
                print("Skipping segmentation for " + name)
                print("segmentation exists!")
                print(mask_name)
            return 0

    if printtolog:
        print("Segmenting " + name)
    start = time.time()

    if type(file) == str:
        img = nib.load(file)
        im = img.get_fdata()
        if zooms is None:
            zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    else:
        im = file
        img = None
    if zooms is None:
        zooms = (1.0, 1.0, 1.0)  # array input with no header: assume 1 mm isotropic

    # Gaussian filter the image
    im = gaussian_filter(im, sigma=3)

    if algo == "largest":
        mask = _lung_mask_largest(im, threshold, lung_at_boundaries, printtolog)
    else:
        mask = _lung_mask_shape(im, threshold, zooms, printtolog=printtolog)
    if mask is None:
        return -1
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
            + name
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


def save_image_MI_style(im, vmin, vmax, out_path, dpi=300, cmap="jet"):
    num_slices = im.shape[0]
    fig, axes = plt.subplots(
        1, num_slices, figsize=(16, 4), dpi=400, facecolor="k"
    )
    cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
    for i, ax in enumerate(axes):
        img = ax.imshow(
            im[i].T,
            origin="lower",
            cmap=cmap,
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
