import struct
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
import pathlib 
import gzip
from scipy.fft import fft, ifft, fftshift
from scipy.optimize import curve_fit
import warnings
from scipy import interpolate
from sklearn.metrics import r2_score
from scipy.interpolate import interp1d
import matplotlib as mpl
from sklearn.cluster import KMeans
from scipy.spatial import distance
from tqdm import tqdm
from collections import deque
from scipy.signal import butter, filtfilt
from scipy.signal import find_peaks_cwt
from concurrent.futures import ThreadPoolExecutor


class DegenerateTimingError(Exception):
    """Raised by check_timing when the scanner's own per-projection
    timestamps are degenerate (e.g. every row logged as the -99999 sentinel)
    and use_fallback wasn't explicitly set - a real hardware/logging fault,
    not something to silently paper over. Callers (pipeline_driver.run())
    should surface this to the user and only retry with use_fallback=True
    once they've explicitly agreed to substitute the known-timing template."""
    pass


LEADING_TIMESTAMP_SENTINEL = -99999.0


def _backup_proj_log(logpath):
    """Renames logpath to its '_original.csv' sibling, unless that backup
    already exists (e.g. the leading-sentinel fix below already made one
    this same call) - reusing that same backup instead of trying to rename
    onto an already-taken filename, which would raise on Windows."""
    backup_path = logpath.replace('.csv', '_original.csv')
    if not os.path.isfile(backup_path):
        os.rename(logpath, backup_path)
    else:
        os.remove(logpath)


def _fix_leading_timestamp_sentinel(logpath, lines):
    """Sometimes only the VERY FIRST projection's timestamp comes back as
    the scanner's -99999 sentinel while every other row is a real, valid
    timestamp (seen on real data - H05 2026-09-23_15h20: 1 bad row out of
    11,520). check_timing's whole-log degenerate check below doesn't catch
    this - thousands of otherwise-good unique timestamps easily clear its
    "len(unique) > 10" bar - and even if it did, its fix (substituting a
    completely unrelated generic timing template for the whole session)
    would be a wildly disproportionate cure for one bad row.

    This isn't cosmetic: fancy_binning.interpolate_signal() uses
    self.t[0] as one endpoint of the entire breathing-signal interpolation
    grid (new_time = np.linspace(self.t[0], self.t[-1], ...)) - a -99999
    first timestamp corrupts that whole grid (spanning ~-99999 to +a few
    hundred instead of ~19 to +a few hundred), which corrupts breathing-rate
    estimation and phase gating for the entire session, not just one frame.

    Fix: extrapolate the missing first timestamp backward from the steady
    cadence of the next two real points - lapsed_time = timing[2] -
    timing[1], replacement = timing[1] - lapsed_time. Returns the
    (possibly unchanged) lines array; always applied automatically (unlike
    the whole-log fallback below, no use_fallback opt-in needed) since it
    only repairs one clearly-invalid value from its own session's real
    neighboring data, rather than discarding real data for a foreign
    template.
    """
    timing = lines[:, 3]
    if len(timing) <= 2 or timing[0] != LEADING_TIMESTAMP_SENTINEL:
        return lines
    if timing[1] == LEADING_TIMESTAMP_SENTINEL or timing[2] == LEADING_TIMESTAMP_SENTINEL:
        return lines  # more than just the leading row is bad - let the whole-log check below handle it

    lapsed_time = timing[2] - timing[1]
    fixed_first = timing[1] - lapsed_time
    print(f"\t\tFirst projection timestamp was the {LEADING_TIMESTAMP_SENTINEL} sentinel - replacing it with "
          f"{fixed_first:.4f} (extrapolated backward from timepoints 1-2's spacing of {lapsed_time:.4f}s).")
    lines = lines.copy()
    lines[0, 3] = fixed_first
    _backup_proj_log(logpath)
    with open(logpath, 'w') as proj_file_new:
        for line in lines:
            proj_file_new.write(','.join(line.astype(str)) + '\n')
    return lines


def check_timing(main_dir, use_fallback=False):
    """Returns True if the fallback timing template was actually substituted
    in for this session, False if this session's own timestamps were fine
    (or too malformed to even check) - callers use this to decide whether
    this specific session's output belongs in the normal or "_fallback"
    output folder, so a batch run doesn't mislabel sessions that never
    needed the fallback in the first place. (A leading-sentinel repair - see
    _fix_leading_timestamp_sentinel - does NOT count as "fallback used": it
    keeps this session's own real timing, so still returns False.)"""
    logpath = os.path.join(main_dir,'ct-data', 'proj_000_0_log.csv')
    # check timing
    file = open(logpath); lines= file.readlines() ; file.close()
    alist = []
    for line in lines: alist.append(np.array(line.rstrip().split(',')).astype(np.float64))
    lines = np.array(alist)
    if lines.shape[1] < 4:
        print("\t\tProblem with timeStamps in the proj_log.csv file")
        return False

    lines = _fix_leading_timestamp_sentinel(logpath, lines)

    timing = lines[:, 3]
    if len(np.unique(timing)) > 10:
        print('All good!')
        return False

    if not use_fallback:
        raise DegenerateTimingError(
            f"{logpath} has degenerate per-projection timestamps "
            f"({len(np.unique(timing))} unique value(s) across {len(timing)} rows) - "
            "the scanner failed to log real timing for this acquisition."
        )

    # Fix Timing
    times = np.load('./sample_times_32p_55kv_37ma_20ms.npy')
    lines[:,3] = times
    _backup_proj_log(logpath)
    proj_file_new = open(logpath, 'w')
    for line in lines: proj_file_new.write(','.join(line.astype(str)) + '\n')
    proj_file_new.close()
    print('Created a new timing from save time samples')
    return True

def _simple_tiff_layout(fh):
    """Parses just enough of a TIFF header to find the pixel data of an
    uncompressed, single-strip, single-channel 16-bit image. Returns
    (data_offset, width, height, numpy_dtype) or None for anything else
    (the caller then falls back to PIL). Much cheaper than PIL's header
    parse, which holds the GIL and capped threaded throughput."""
    head = fh.read(8)
    if len(head) < 8 or head[:2] not in (b'II', b'MM'):
        return None
    bo = '<' if head[:2] == b'II' else '>'
    if struct.unpack(bo + 'H', head[2:4])[0] != 42:
        return None  # not classic TIFF (e.g. BigTIFF)
    ifd = struct.unpack(bo + 'I', head[4:8])[0]
    fh.seek(ifd)
    raw = fh.read(2)
    if len(raw) < 2:
        return None
    n = struct.unpack(bo + 'H', raw)[0]
    entries = fh.read(12 * n)
    if len(entries) < 12 * n:
        return None
    tags = {}
    for k in range(n):
        tag, typ, count = struct.unpack(bo + 'HHI', entries[12 * k:12 * k + 8])
        if typ == 3 and count == 1:
            val = struct.unpack(bo + 'H', entries[12 * k + 8:12 * k + 10])[0]
        elif typ == 4 and count == 1:
            val = struct.unpack(bo + 'I', entries[12 * k + 8:12 * k + 12])[0]
        else:
            val = None  # multi-valued (e.g. several strips) - not our simple case
        tags[tag] = (count, val)
    def one(tag, default=None):
        count, val = tags.get(tag, (1, default))
        return val if count == 1 else None
    width, height = one(256), one(257)
    if (width is None or height is None or one(258) != 16 or one(259, 1) != 1
            or one(277, 1) != 1 or one(339, 1) != 1 or one(273) is None):
        return None
    return one(273), width, height, np.dtype(bo + 'u2')


def _raw_tiff_rows(image_path, row_slice):
    """Reads only rows `row_slice` of an uncompressed, single-strip 16-bit
    grayscale TIFF straight from disk, skipping PIL's full-frame decode.
    Returns None when the file isn't in that simple layout (the caller then
    falls back to PIL)."""
    with open(image_path, 'rb') as fh:
        layout = _simple_tiff_layout(fh)
        if layout is None:
            return None
        offset, width, height, dtype = layout
        start, stop, step = row_slice.indices(height)
        if step != 1 or stop <= start:
            return None
        fh.seek(offset + start * width * 2)
        rows = np.fromfile(fh, dtype=dtype, count=(stop - start) * width)
    if rows.size != (stop - start) * width:
        return None  # truncated file - let PIL report it properly
    return rows.reshape(stop - start, width)


def _threshold_lut(threshold):
    """lut[v] == (v / 65535 if v / 65535 <= threshold else 0.0), computed with
    the exact same float64 ops as the per-pixel version, so lut[raw] gives
    bit-identical values while doing a single gather instead of a divide, a
    compare and a masked write on every pixel of every projection."""
    lut = np.arange(65536, dtype=np.uint16) / 65535
    lut[lut > threshold] = 0
    return lut


def get_signal_from_image(folder, array=[], xlimits=[300, 500],
                          ylimits=[0, -1], subtract_baseline=False, spring=False, bm3d=False,
                          rabbit=False, threshold=0.8, n_workers=64):
    # n_workers: reads are I/O-bound once PIL decoding is skipped; on this
    # RAID, cold-cache reads kept speeding up to 64 threads (38.9s at 16 -> 26.5s
    # at 64 for 11,520 projections) with no cost when files are already cached.
    lut = _threshold_lut(threshold)

    def _read_one_projection(args):
        image_path, xlimits, ylimits, threshold = args
        # Fast path: read only the needed rows from disk and map them through
        # the threshold lookup table - bit-identical to the PIL path below.
        rows = _raw_tiff_rows(image_path, slice(xlimits[0], xlimits[1]))
        if rows is not None:
            return float(lut[rows[:, ylimits[0]:ylimits[1]]].mean())
        im = np.asarray(Image.open(image_path)).astype(np.uint16)
        # crop to the region actually used by the mean BEFORE normalizing/thresholding,
        # instead of doing that elementwise work over the whole (mostly-unused) frame
        sub = im[xlimits[0]:xlimits[1], ylimits[0]:ylimits[1]] / 65535
        sub[sub > threshold] = 0
        return float(sub.mean())
    range_limits = [0, 0]

    if array != []:
        range_limits = array
    else:
        range_limits[1] = len([x for x in os.listdir(folder) if ".tif" in x])
    if spring:
        xlimits = [100, 500]

    indices = range(range_limits[0], range_limits[1])
    tasks = [
        (folder + '/proj_000_0_000' + "{:05d}".format(i) + '.tif', xlimits, ylimits, threshold)
        for i in indices
    ]

    # the loop is I/O-bound (opening/decoding thousands of individual small TIFFs), so
    # a thread pool overlaps that I/O latency across workers instead of waiting on one
    # file at a time -- see Develope/benchmark_to_be_optimized.py for the measured speedup
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        msum = list(tqdm(
            executor.map(_read_one_projection, tasks),
            total=len(tasks),
            desc="Projections: ",
        ))
    msum = np.array(msum)

    # Get the angles and times
    proj_file = os.path.join(folder, 'proj_000_0_log.csv')
    if not os.path.exists(proj_file): proj_file = os.path.join(pathlib.Path(folder).parent.absolute(), 'proj_000_0_log.csv')
    file = open(proj_file) ; lines= file.readlines() ; file.close()
    alist = []
    for line in lines: alist.append(np.array(line.rstrip().split(',')).astype(np.float32))
    lines = np.array(alist)[range_limits[0]:range_limits[1], :]
    angles = lines[:,2]
    times  = lines[:,3]
    # Subtract baseline
    if subtract_baseline:
        for angle in np.unique(angles):
            msum[angles == angle] -= msum[angles == angle].min()
    msum = msum * 1000
    return msum, times, angles
# def get_signal_from_image(folder,  array=[], xlimits=[300,500], 
#                           ylimits=[0,-1],subtract_baseline = False, spring=False, bm3d = False, 
#                           rabbit=False, threshold = 0.8):
#     msum,baseline = [], []
#     range_limits = [0,0]
        
#     if array != []:
#         range_limits = array
#     else:
#         range_limits[1] = len([x for x in os.listdir(folder) if ".tif" in x])
#     if spring:
#         xlimits = [100,500]
#     for i in tqdm(range(range_limits[0], range_limits[1]), desc="Projections: "):
#     # for i in range(range_limits[0], range_limits[1]):
#         image = folder+'/proj_000_0_000' +"{:05d}".format(i) +'.tif'
#         im = np.asarray(Image.open(image)).astype(np.uint16) / 65535
#         im[im>threshold] = 0
#         msum.append(np.mean(im[xlimits[0]:xlimits[1],ylimits[0]:ylimits[1]]))
        
#     msum = np.array(msum)
    
#     # Get the angles and times
#     proj_file = os.path.join(folder, 'proj_000_0_log.csv')
#     if not os.path.exists(proj_file): proj_file = os.path.join(pathlib.Path(folder).parent.absolute(), 'proj_000_0_log.csv')
#     file = open(proj_file) ; lines= file.readlines() ; file.close() 
#     alist = []
#     for line in lines: alist.append(np.array(line.rstrip().split(',')).astype(np.float32))
#     lines = np.array(alist)[range_limits[0]:range_limits[1], :]
#     angles = lines[:,2]
#     times  = lines[:,3]
#     # Subtract baseline
#     if subtract_baseline: 
#         for angle in np.unique(angles):
#             msum[angles == angle] -= msum[angles == angle].min()
#             # plt.scatter(times[angles == angle], msum[angles == angle])
#         # Change range of msum
#         # msum = (msum - msum.min()) / ( msum.max()-msum.min() ) * 1000
#     msum = msum * 1000
#     return msum, times, angles

def get_image(folder, slice_num=-1, array=[], normalize=True, threshold=0.8):
    if slice_num == -1 and array == []: 
        print('You must specify slice_num or array')
        return -1
    elif array ==[]:
        image = folder+'/proj_000_0_000' +"{:05d}".format(slice_num) +'.tif'
        if normalize:
            im = np.asarray(Image.open(image)).astype(np.uint16) / 65535
            im[im>threshold]=0
        else:
            im = np.asarray(Image.open(image)).astype(np.uint16)
        return im
    elif slice_num == -1:
        imgs = []
        for i in range(array[0], array[1]+1):
            image = folder+'/proj_000_0_000' +"{:05d}".format(i) +'.tif'
            im = np.asarray(Image.open(image)).astype(np.uint16)/65535
            im[im>threshold]=0
            imgs.append(im)
        return imgs
    
def moving_average(x, w=3, mode = 'same'):
    return np.convolve(x, np.ones(w), mode) / w

def plot_signal(folder, xlimits=[200,500],ylimitsl=[450,600],angle=90, rolling_minimum = False,
                ylimitsr=[600,850], subtract_baseline=False, num_slices=32, normalize=False):

    array=[32*angle,32*angle + num_slices]
    fig, axs = plt.subplots(len(os.listdir(folder)), 1)
    for i, afolder in enumerate(os.listdir(folder)):
        fpath = os.path.join(folder,afolder)
        sigl, tl, baseline = get_signal_from_image(fpath, array = array, rolling_minimum = rolling_minimum,
                               xlimits=[200,500],ylimits=ylimitsl, subtract_baseline=subtract_baseline)
        sigr, tr, baseline = get_signal_from_image(fpath, array = array,rolling_minimum = rolling_minimum,
                               xlimits=[200,500],ylimits=ylimitsr, subtract_baseline=subtract_baseline)
       
        if normalize:
            axs[i].scatter(tl,[x/y for x,y in zip(sigl, sigr)])
        else:
            axs[i].scatter(tl,sigl) , axs[i].scatter(tr,sigr) 
        axs[i].set_title(afolder)
        axs[i].set_ylim([0.97,1.1])
    return 0

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



    
def run_time_binning(s, t, a, n_phases, only_phase_binning=False, plot_path=None):
    """The single breathing-phase (time) binning used by BOTH the
    reconstruction (new_binning -> create_milab_structure) and the app's
    clustering_result.png (pipeline_driver.step_4). Same class, same
    settings, same (deterministic) clustering - so the plot shows exactly
    the labels that become the Phase_<n> folders. If plot_path is given,
    the clustering scatter is also saved there.
    Returns (real_time, real_amps, signal_normalized, labels, breathing_rate)."""
    b = fancy_binning(a, s, t, only_phase_binning=only_phase_binning)
    b.nbins = n_phases
    br = b.get_approximate_breathing_rate()
    if plot_path:
        plt.figure(figsize=(7, 4))
    # return_centroid:  return rep_times, rep_values, self.normalize_signal(), labels
    real_time, real_amps, signal_normalized, bins = b.binning(plot=bool(plot_path), return_centroid=True)
    if plot_path:
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.2])
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
    return real_time, real_amps, signal_normalized, bins, br


def new_binning(s,t,a, PB_num, LAB_num,only_phase_binning=False):
    real_time, real_amps, signal_normalized, bins, br = run_time_binning(
        s, t, a, PB_num, only_phase_binning=only_phase_binning)
    lab = np.digitize(signal_normalized, np.linspace(signal_normalized.min(), signal_normalized.max(), LAB_num + 1), right=False)
    pb = bins
    min_amp, max_amp = np.zeros(pb.shape), np.zeros(pb.shape)
    phases = np.unique(pb)
    for p in phases:
        min_amp[pb == p] = signal_normalized[pb==p].min()
        max_amp[pb == p] = signal_normalized[pb==p].max()
    
    angles = np.unique(a)
    s_label= {}
    for i in range(len(angles)):
        angle = angles[i]
        s_label[str(angle)] = {}
        s_label[str(angle)]['LAB'] = lab[a == angle]
        s_label[str(angle)]['PB']  = pb[a == angle]
        s_label[str(angle)]['min_s']  = min_amp[a == angle]
        s_label[str(angle)]['max_s']  = max_amp[a == angle]
        s_label[str(angle)]['s']  = signal_normalized[a == angle]
    df = pd.DataFrame.from_dict(s_label,orient='index')
    return df , real_time, real_amps, br

def bin_position_and_time(signal, times, angles, proj_file, df):
    # First read proj_file
    f = open(proj_file) ; lines= f.readlines() ; f.close()
    alist = []
    for line in lines: alist.append(np.array(line.rstrip().split(',')).astype(np.float64))
    lines = np.array(alist)
    # ------------- Binning ----------------
    lab,pb,mins,maxs,ss = [],[],[],[],[]
    for x in df.LAB: lab += x.tolist()
    for x in df.PB: pb += x.tolist()
    for x in df.min_s: mins += x.tolist()
    for x in df.max_s: maxs += x.tolist()
    for x in df.s: ss += x.tolist()
    
    lab,pb,mins,maxs,ss = np.array(lab), np.array(pb) ,np.array(mins), np.array(maxs), np.array(ss)
    
    lines = np.concatenate((lines,np.reshape(lab, (len(lab), 1))), axis=1) # append phase bins 4th column
    lines = np.concatenate((lines,np.reshape(pb, (len(pb), 1))), axis=1)  # append amplitude bins 5th column
    lines = np.concatenate((lines,np.reshape(mins, (len(mins), 1))), axis=1)  # append amplitude bins 5th column
    lines = np.concatenate((lines,np.reshape(maxs, (len(maxs), 1))), axis=1)  # append amplitude bins 5th column
    lines = np.concatenate((lines,np.reshape(ss, (len(ss), 1))), axis=1)  # append amplitude bins 5th column
    # Final phases 
    final_phases = {}
    for tp in np.unique(pb):
        best_i_phase = -1 
        best_i_count = 0
        for ip in np.unique(lab): 
            c = len(lines[(pb == tp) & (lab == ip)])
            if c > best_i_count:
                best_i_phase= ip
                best_i_count= c
        final_phases[tp] = best_i_phase    
    # Final Lines 
    new_lines = np.empty((1,lines.shape[1]))
    for t in final_phases.keys():
        new_lines = np.vstack((new_lines,   lines[ (pb == t) & (lab == final_phases[t])]))
    
    return lines, new_lines, final_phases


class fancy_binning():
    def __init__(self, a, s, t, bins=16, debug=False, flip_signal=False, 
                 only_phase_binning=False, max_bpm= 200):
        self.a = a 
        self.s = s
        self.t = t
        self.prominence = 0.2
        self.s_normalized = []
        self.nbins = bins
        self.debug = debug
        self.rep_time = []
        self.rep_signal = []
        self.max_freq = max_bpm # bpm
        self.min_freq = 10 # bpm
        self.flip_signal = flip_signal
        self.only_phase_binning = only_phase_binning
        self.TR = np.median(np.diff(t))
        print(self.TR, 1/ self.TR)
        # self.get_approximate_breathing_rate()
        
    def assign_to_nearest_center(self, x_point, y_point, rep_times, rep_values):
        distances = np.sqrt((rep_times - x_point)**2 + (rep_values - y_point)**2)
        return np.argmin(distances)
    
    def butter_lowpass_filter(self, data, order=4):
        b, a = butter(order, self.max_freq/60, btype='low', analog=False, fs= 1/ self.TR)
        y = filtfilt(b, a, data)
        return y
    
    def get_max_per_anlge(self,plot=False, f = 0.01):
        maxh = []
        for angle in np.unique(self.a): maxh.append(self.s[self.a==angle].max())
        maxh = np.array(maxh)
        # remove outliers
        maxh[maxh > np.percentile(maxh, 95) * 1.2] = 0
        # maxh[maxh < np.percentile(maxh, 30) * 0.8]= np.percentile(maxh, 5)
        if plot or self.debug:
            # plt.scatter(a,s, label='Projections')
            plt.plot(maxh, c='r', label= 'Max(projections)')
        max_filtered = self.butter_lowpass_filter(maxh, order = 3)
        if plot:
            plt.plot(max_filtered, c = 'k', lw=2, label='fitted')
            plt.xlabel('Angle')
            plt.ylabel('signal')
            plt.title('Max signal per angle')
            plt.legend()
        # plt.savefig('./max_sig_per_angle.tif', dpi=300)
        return np.array(max_filtered)
    
    def normalize_signal(self, plot= False):
        # Find max signal per angle and normalize the signal
        max_per_angle = self.get_max_per_anlge(f = 0.02)
        s_normalized = self.s.copy()
        for i, angle in enumerate(np.unique(self.a)): s_normalized[self.a==angle] /= max_per_angle[i]
        # Spikes correct
        s_normalized[s_normalized>1.2] = 1.2
        if plot:
            plt.plot(self.a, s_normalized, label='Projections')
            plt.axhline(y=1, c = 'k', lw=2, label='fitted')
            plt.xlabel('Angle')
            plt.ylabel('signal')
            plt.title('Normalized projections')
            # plt.savefig('./normalized_sig_per_angle.tif', dpi=300)
        if self.flip_signal: s_normalized = 1 - s_normalized
        self.s_normalized = s_normalized
        return s_normalized
    
    def interpolate_signal(self):
        s_normalized = self.normalize_signal()
        interpolator = interp1d(self.t, s_normalized, kind='quadratic') # linear
        new_time = np.linspace(self.t[0], self.t[-1] , num=len(self.t)*3) # *2
        s_normalized_interpolated = interpolator(new_time)
        s_normalized_interpolated[s_normalized_interpolated<0]=0
        return new_time, s_normalized_interpolated
    
    def find_individual_periods(self, plot=False, normalized=True):
        # Interpolate the points to find individual periods
        s_normalized = self.normalize_signal()
        new_time, s_normalized_interpolated = self.interpolate_signal()
        peaks = self.get_peaks()
        # Find individual peaks
        separate_periods_s, separate_periods_t = [], []
        # Find first period
        first_period = new_time[peaks[1]] - new_time[peaks[0]]
        time_of_first_period = self.t[self.t < new_time[peaks[0]]]
        time_of_first_period = time_of_first_period - new_time[peaks[0]] + first_period
        separate_periods_s.append(s_normalized[self.t < new_time[peaks[0]]])
        if normalized: time_of_first_period /= first_period
        separate_periods_t.append(time_of_first_period)
        for i in range(len(peaks)-1):
            criteria = (self.t >= new_time[peaks[i]] ) & (self.t < new_time[peaks[i+1]])
            separate_periods_s.append(s_normalized[criteria])
            real_time_here = self.t[criteria]-new_time[peaks[i]]
            period_here = new_time[peaks[i+1]] - new_time[peaks[i]]
            if normalized: real_time_here /= period_here
            separate_periods_t.append(real_time_here)
        # # Find Last period
        last_period = new_time[peaks[-1]] - new_time[peaks[-2]]
        time_of_last_period = self.t[self.t > new_time[peaks[-1]]] - new_time[peaks[-1]]
        separate_periods_s.append(s_normalized[self.t > new_time[peaks[-1]]])
        if normalized: time_of_last_period /= last_period
        separate_periods_t.append(time_of_last_period)

        # Find individual Breathing cycles
        start_periods_to_plot=0
        end_periods_to_plot=44000
        X_periods, X_times = [], []
        for ss,tt in zip(separate_periods_s[start_periods_to_plot:end_periods_to_plot],
                         separate_periods_t[start_periods_to_plot:end_periods_to_plot]):
            X_times += list(tt); X_periods += list(ss)
            if plot: plt.scatter(tt,ss, alpha=0.7)
        X = np.vstack([X_periods, X_times]).T
        # Get a representative breathing cycle 
        max_time = max([x for sub in separate_periods_t for x in sub])
        nt_bins = self.nbins
        rep_signal,rep_time = {}, {}
        ti, te = 0, max_time/ nt_bins
        for i in range(nt_bins):
            rep_signal[str(i)] = []
            rep_time[str(i)] = (ti + te) / 2
            for period_s, period_t in zip(separate_periods_s, separate_periods_t):
                criteria = (period_t >= ti) & (period_t < te)
                rep_signal[str(i)] += list(period_s[criteria])
            ti = te
            te += (max_time/ nt_bins)
        for i in range(nt_bins): rep_signal[str(i)] = np.mean(rep_signal[str(i)])
        if plot:
            plt.plot(rep_time.values() , rep_signal.values(), linewidth=3, color='k', label='Representative Signal')
            # for i in range(10):  plt.axhspan(i/10,(i+1)/10, color=colors[i])
            plt.xlabel('Time (s)')
            plt.ylabel('Signal')
            plt.legend()
        self.rep_time = rep_time
        self.rep_signal = rep_signal
        return X
    
        
    def plot_all_periods(self):
        a,s,t= self.a, self.s, self.t
        # # Save all periods
        # sf = fourier_smooth(s2, 0.05)
        fig = plt.figure(figsize=(10,10), dpi=300)
        s_normalized = self.normalize_signal()
        new_time, s_normalized_interpolated = self.interpolate_signal()
        sf = self.butter_lowpass_filter(s_normalized_interpolated, order = 3)
        sf[sf>1.25] = 1.25
        peaks = self.get_peaks()
        ti, delay = 0, 20
        for i in range(30):
            plt.subplot(10,3,i+1)
            te = ti+delay
            t_here = (self.t>=ti) & (self.t<te)
            new_t_here = (new_time>ti) & (new_time<te)
            plt.plot(self.t[t_here],s_normalized[t_here])
            plt.plot(new_time[new_t_here],sf[new_t_here])
            for p in peaks: 
                if (new_time[p] >= ti) & (new_time[p] <= te): plt.axvline(x=new_time[p], c='r')
            plt.xlim([ti,te])
            ti += delay
            # plt.xlim([120,140])
        # Add a global x-axis title
        fig.supxlabel('Time (s)')
        plt.tight_layout()
        plt.savefig('./all_periods.tif', dpi=300)
    
    def get_rep_signal(self, x,y):
        nt_bins = self.nbins
        rep_signal, rep_time = [], []
        ti, te = 0, max(x) / nt_bins
        for i in range(nt_bins):
            rep_time.append((ti + te) / 2)
            rep_signal.append(np.median(y[(x >= ti) & (x < te)]))
            ti = te
            te += (max(x) / nt_bins)
        return np.array(rep_time), np.array(rep_signal)

    
    def binning(self, plot=False, return_signal=False, return_centroid=False, withline=True):
        # x,y = self.pre_binning(plot=False)
        # X = np.vstack([y, x]).T
        X = self.find_individual_periods()
        x,y = X[:, 1], X[:, 0]
        rep_x, rep_y = self.get_rep_signal(x,y)
        if self.only_phase_binning: 
            print('Binning based on phase..')
            bin_edges = np.linspace(0, 1, num=self.nbins+1)
            labels = np.digitize(x, bin_edges)
            labels[labels==0] = 1
            labels[labels>self.nbins]=self.nbins
            labels -= 1
            unique_bins = np.unique(labels)
            print(len(unique_bins))
            rep_times , rep_values = [], []
            for bin_num in unique_bins:
                rep_times.append(np.mean(x[labels == bin_num]))
                rep_values.append(np.mean(y[labels == bin_num]))
            rep_times , rep_values = np.array(rep_times) , np.array(rep_values)
        else:
            rep_values  = np.insert(rep_y,0, y[x < 0.01].mean())
            rep_values  = np.insert(rep_values,len(rep_values), y[x > 0.99].mean())
            rep_times  = np.insert(rep_x,0, 0)
            rep_times  = np.insert(rep_times,len(rep_times), 1) 
            interpolator = interp1d(rep_times, rep_values, kind='cubic') # linear
            new_rep_times = np.linspace(rep_times[0], rep_times[-1] , num=200)
            new_rep_values = interpolator(new_rep_times)
            line = np.concatenate([[new_rep_times], [new_rep_values]],axis=0).T
            # Define the number of clusters and initialize KMeans
            kmeans = KMeansWithLineCentroids(n_clusters=self.nbins, 
                                             line = line, 
                                             with_line = withline,
                                             max_iters=1000)
            # Fit KMeans with line-constrained centroids
            kmeans.fit(X)
            labels     = kmeans.predict(X)
            # Already in phase order (row k = phase k). If phase 0's centroid
            # sits just before the period end, express its time as negative
            # so rep_times stays increasing AND aligned with the labels.
            centroids = kmeans.centroids.copy()
            if centroids[0, 1] > kmeans.period / 2:
                centroids[0, 1] -= kmeans.period
            rep_times, rep_values = centroids[:,1], centroids[:,0]
        if plot:
            colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']
            colors = colors * 2
            # Visualize the data points and cluster centers
            # plt.figure(figsize=(8, 6))
            # for i in range(16):  plt.axvspan(i/16 * X[:,1].max(),(i+1)/16 * X[:,1].max(), color=colors[i], alpha=0.2)
            plt.scatter(X[:, 1], X[:, 0], c=labels, cmap='tab20', vmin=0, vmax=self.nbins - 1)
            plt.scatter(rep_times, rep_values, marker='x', s=200, c='red', label='Bin Centers')
            plt.title('Binning Based on K-Means Clustering')
            plt.plot(rep_times , rep_values, linewidth=3, color='k', label='Breathing Signal')
            plt.xlabel('Time (s)')
            plt.ylabel('Signal')
            plt.legend()
        if return_centroid:  return rep_times, rep_values, self.normalize_signal(), labels
        if return_signal: return rep_times, self.normalize_signal(), labels
        return labels
    
    def get_approximate_breathing_rate(self):
        max_breathing_rate = self.max_freq
        min_breathing_rate = self.min_freq
        new_time, signal = self.interpolate_signal()
        sampling_period_in_seconds = np.diff(new_time).mean()
        # Min and max breathing rates are 6 and 120bpm
        sampling_freq = 1/ sampling_period_in_seconds
        freq_per_fft_point = sampling_freq / len(signal)
        freq = np.arange(len(signal)) * freq_per_fft_point
        # print(60/ (len(signal)*0.05 * freq_per_fft_point))
        min_breathing_rate_hz, max_breathing_rate_hz = min_breathing_rate/60, max_breathing_rate/60
        sigfft = np.abs(np.fft.fft(signal)) 
        sigfft[freq<min_breathing_rate_hz] = 0
        sigfft[freq>max_breathing_rate_hz] = 0
        BR = freq[np.argmax(sigfft)] * 60 # bpm
        print(f'Most Representative Breathing Rate: {BR:.2f} BPM')
        return BR
        
    def get_peaks(self):
        tt = self.t
        s_normalized = self.normalize_signal()
        new_time, s_normalized_interpolated = self.interpolate_signal()
        new_peaks = []
        running_periods = []
        peaks_certain, peaks_probable = self.get_certain_and_probable_peaks(new_time, s_normalized_interpolated)
        peaks = peaks_certain.copy()
        # Correct first three periods
        new_peaks.append(peaks[0]) # no matter what, take first detected peak
        for i in range(10): running_periods.append(new_time[peaks[i+1]] - new_time[peaks[i]])
        average_period_first_10_peaks = np.median(running_periods)
        for i in range(1,5):
            current_period = new_time[peaks[i]] - new_time[peaks[i-1]]
            if current_period > 2.4 * average_period_first_10_peaks: # 3 periods
                r =  (peaks[i]-peaks[i-1])//3
                new_peaks.append(peaks[i-1]+r)
                new_peaks.append(peaks[i-1]+r*2)
                new_peaks.append(peaks[i])
            if current_period > 1.4 * average_period_first_10_peaks: # 2 periods
                new_peaks.append(peaks[i-1] + (peaks[i]-peaks[i-1])//2)
                new_peaks.append(peaks[i])
            else: # Good period
                new_peaks.append(peaks[i])
        # Correct rest of the peaks
        running_periods = deque(maxlen=5)
        for i in range(1,4): running_periods.append(new_time[new_peaks[i]] - new_time[new_peaks[i-1]])
        # check first period
        if (new_time[new_peaks[0]] - tt[0]) > 1.4 * np.median(running_periods):
            new_peaks = list(np.insert(new_peaks,0, new_peaks[0] -np.median(np.diff(new_peaks))))
        for i in range(5, len(peaks)-1):
            current_period = new_time[peaks[i]] - new_time[peaks[i-1]]
            if current_period > 2.4 * np.mean(running_periods): # 3 periods
                # Check if there's a probable peaks here, if not then divid time by 3 periods
                pp = [x for x in peaks_probable if (x> peaks[i-1]) * (x< peaks[i])]
                if len(pp) == 0:
                    r =  (peaks[i]-peaks[i-1])//3
                    new_peaks.append(peaks[i-1]+r)
                    new_peaks.append(peaks[i-1]+r*2)
                    new_peaks.append(peaks[i])
                    running_periods.append(current_period/3)
                elif s_normalized_interpolated[pp[0]] > 1.25:# Check if very high peak probably wrong
                    r =  (peaks[i]-peaks[i-1])//3
                    new_peaks.append(peaks[i-1]+r)
                    new_peaks.append(peaks[i-1]+r*2)
                    new_peaks.append(peaks[i])
                    running_periods.append(current_period/3)
                elif len(pp) == 2: # Check if two probable peaks exist
                    r =  (peaks[i]-peaks[i-1])//3
                    new_peaks.append(peaks[i-1]+r)
                    new_peaks.append(peaks[i-1]+r*2)
                    new_peaks.append(peaks[i])
                    running_periods.append(current_period/3)
                else: # Probably just 2 periods not 3, add the probable peak
                    new_peaks.append(pp[0])
                    new_peaks.append(peaks[i])
                    running_periods.append(peaks[i]-pp[0])
            elif current_period > 1.4 * np.mean(running_periods): # 2 periods
                new_peaks.append(peaks[i-1] + (peaks[i]-peaks[i-1])//2)
                new_peaks.append(peaks[i])
                running_periods.append(current_period/2)
            # elif current_period < 0.7 * np.mean(running_periods): # small period
            #     pass
            else: # Good period
                new_peaks.append(peaks[i])
                running_periods.append(current_period)
        new_peaks.append(peaks[-1])
        new_peaks = np.array(new_peaks)
        # return find_peaks_cwt(s_normalized_interpolated, widths=np.arange(1, width))
        return new_peaks
    
    def plot_breathing(self):
        new_peaks = self.get_peaks()
        new_time, s_normalized_interpolated = self.interpolate_signal()
        periods = 60/np.diff(new_time[new_peaks])
        # Smooth out the periods using a moving average
        window_size = 5  # Adjust the window size as needed
        smooth_periods = np.convolve(periods, np.ones(window_size) / window_size, mode='same')
        smooth_periods[:5] = periods[:5]
        smooth_periods[-5:] = periods[-5:]
        plt.plot(new_time[new_peaks][:-1],periods, label = 'signal')
        plt.plot(new_time[new_peaks][:-1],smooth_periods, label='Smoothed signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Breathing Rate (bpm)')
        plt.title('Breathing rate over time (bpm)')
        return new_time[new_peaks][:-1],periods, smooth_periods
        
    
    def get_certain_and_probable_peaks(self, new_time, s_normalized_interpolated):
        tt = self.t
        # sf = self.fourier_smooth(s_normalized_interpolated, 0.2) # 0.1
        sf = self.butter_lowpass_filter(s_normalized_interpolated, order = 3)
        min_period = 60 / self.max_freq
        peak_distance= min_period / np.diff(new_time).mean()
        # print('p:', min_period, peak_distance)
        sf[sf>1.25] = 1.25
        # peaks, _ = find_peaks(sf, prominence=0.4, distance = peak_distance)
        peaks, _ = find_peaks(sf, height=0.4, distance = peak_distance)
        # peaks = find_peaks_cwt(s_normalized_interpolated, widths=np.arange(1, width))
        peaks_certain = []
        peaks_certain.append(peaks[0])
        threshold = np.median(np.diff(self.t)) * 5 # miss 10 samples 
        for i in range(1, len(peaks)):
            p = peaks[i]
            n = len(tt[(tt < new_time[p] + threshold) & (tt > new_time[p] - threshold)])
            if n ==0: continue
            peaks_certain.append(p)
        peaks_certain = np.array(peaks_certain)
        # Get probable peaks 
        peaks, _ = find_peaks(sf, prominence=(0.2), distance = peak_distance)
        peaks_probable = []
        peaks_probable.append(peaks[0])
        threshold = np.median(np.diff(tt)) * 10 # miss 20 samples 
        for i in range(1, len(peaks)):
            p = peaks[i]
            n = len(tt[(tt < new_time[p] + threshold) & (tt > new_time[p] - threshold)])
            if n ==0: continue
            peaks_probable.append(p)
        peaks_probable = np.array(peaks_probable)
        return peaks_certain, peaks_probable


class KMeansWithLineCentroids:
    """K-means on (signal, normalized time) points with centroids kept on the
    representative breathing line. Column 1 of data/centroids is time in
    [0, period).

    Periodicity: only phase 0 - the cluster whose centroid is CURRENTLY
    circularly closest to t=0 - spans the period boundary, so only it uses
    wrapped (time +/- period) distances and a circular time mean. Every
    other cluster, including the last phase, uses plain distances and a
    plain mean, so no other phase can pick up points from the far side of
    the cycle.

    After fit(), centroids are ordered phase 0 first, then the rest by
    time, so predict() returns labels where 0 is the phase at the start of
    the cycle and n_clusters-1 the latest.
    """
    def __init__(self, n_clusters, line, max_iters=100,with_line=True, period=1):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.line = line  # The line defined by x and y coordinates
        self.with_line = with_line
        self.period = period

    def _boundary_clusters(self, centroids):
        """[index of phase 0]: the cluster whose centroid time is circularly
        closest to t=0 (its centroid may sit just after 0 or just before
        the period). The only cluster allowed to wrap."""
        t = np.mod(centroids[:, 1], self.period)
        circ = np.minimum(t, self.period - t)
        return [int(np.argmin(circ))]

    def _phase_order(self, centroids):
        """Cluster indices in phase order: phase 0 first, then the rest by time."""
        p0 = self._boundary_clusters(centroids)[0]
        rest = [int(i) for i in np.argsort(centroids[:, 1], kind='stable') if i != p0]
        return [p0] + rest

    def periodic_centroids(self, data, labels, boundary):
        T = self.period
        clusters = []
        for i in range(self.n_clusters):
            set_cluster_i = data[labels == i]
            cluster_y = set_cluster_i[:,0].mean()
            x = set_cluster_i[:,1]
            if i not in boundary:
                clusters.append([cluster_y, x.mean()])
                continue
            left = x[x < T/2]
            right = x[x >= T/2]
            count_left, count_right = len(left) , len(right)
            mean_left, mean_right = 0,0
            if count_left !=0: mean_left = np.mean(left)
            if count_right !=0: mean_right = np.mean(right)
            if np.abs(mean_right - mean_left) < T/2:
                cluster_x = (mean_left*count_left+ mean_right*count_right) / (count_right+count_left)
            else:
                cluster_x = (mean_left*count_left+ mean_right*count_right+ count_left*T ) / (count_right+count_left)
                cluster_x = cluster_x % T
            clusters.append([cluster_y, cluster_x])
        clusters = np.array(clusters)
        return clusters

    def _distances(self, data, boundary):
        """Point-to-centroid distances; for phase 0 only, the minimum over
        the raw and the time-wrapped (+/- period) point."""
        distances = np.linalg.norm(data[:, np.newaxis] - self.centroids, axis=2)
        for shift in (-self.period, self.period):
            shifted = data.copy()
            shifted[:, 1] += shift
            d = np.linalg.norm(shifted[:, np.newaxis] - self.centroids[boundary], axis=2)
            distances[:, boundary] = np.minimum(distances[:, boundary], d)
        return distances

    def fit(self, data):
        self.data = data
        n_samples, n_features = data.shape

        line_x = self.line[:, 1]
        line_y = self.line[:, 0]

        # Calculate the spacing between the points
        equidistant_indices = [int(i) for i in np.linspace(0, len(line_x)-1, self.n_clusters)]

        # Initialize centroids on the predefined line
        centroids_x = line_x[equidistant_indices]
        centroids_y = line_y[equidistant_indices]
        self.centroids = np.column_stack((centroids_x, centroids_y))
        self.n_inter = 0

        for _ in range(self.max_iters):
            self.n_inter += 1
            boundary = self._boundary_clusters(self.centroids)
            labels = np.argmin(self._distances(data, boundary), axis=1)
            new_centroids = self.periodic_centroids(data, labels, boundary)
            if self.with_line: new_centroids = self.project_centroids_onto_line(new_centroids)
            # Check for convergence using a tolerance level
            if np.allclose(new_centroids, self.centroids):
                break
            self.centroids = new_centroids
        # Relabel: label 0 = phase 0 (the wrapping cluster at the cycle start),
        # then the remaining phases in time order.
        self.centroids = self.centroids[self._phase_order(self.centroids)]

    def project_centroids_onto_line(self, centroids):
        # Initialize an array to store the projected centroids
        projected_centroids = np.zeros(centroids.shape)

        # Iterate through centroids and project them onto the line
        for i, centroid in enumerate(centroids):
            # Calculate the Euclidean distances between the centroid and all points on the line
            distances = np.linalg.norm(self.line[:, [1, 0]] - centroid, axis=1)
            # Find the index of the point with the smallest distance
            closest_index = np.argmin(distances)
            # Get the point on the line closest to the centroid
            closest_point = self.line[:, [1, 0]][closest_index]
            projected_centroids[i] = closest_point

        return projected_centroids

    def predict(self, data):
        boundary = self._boundary_clusters(self.centroids)
        labels = np.argmin(self._distances(data, boundary), axis=1)
            
        # distances = np.linalg.norm(data[:, np.newaxis] - self.centroids, axis=2)
        # labels = np.argmin(distances, axis=1)
        return labels
    
class KMeansWithLineCentroids_old:
    def __init__(self, n_clusters, line, max_iters=100,with_line=True):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.line = line  # The line defined by x and y coordinates
        self.with_line = with_line

    def fit(self, data):
        self.data = data

        n_samples, n_features = data.shape

        line_x = self.line[:, 1]
        line_y = self.line[:, 0]

        # Calculate the spacing between the points
        equidistant_indices = [int(i) for i in np.linspace(0, len(line_x)-1, self.n_clusters)]

        # Initialize centroids on the predefined line
        centroids_x = line_x[equidistant_indices]
        centroids_y = line_y[equidistant_indices]
        self.centroids = np.column_stack((centroids_x, centroids_y))
        self.n_inter = 0

        for _ in range(self.max_iters):
            self.n_inter += 1
            # Assign each data point to the nearest centroid
            distances = np.linalg.norm(data[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)

            # Update centroids as the mean of the assigned data points
            new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(self.n_clusters)])
            if self.with_line: new_centroids = self.project_centroids_onto_line(new_centroids)

            # Check for convergence using a tolerance level
            if np.allclose(new_centroids, self.centroids):
                break
            self.centroids = new_centroids

    def project_centroids_onto_line(self, centroids):
        # Initialize an array to store the projected centroids
        projected_centroids = np.zeros(centroids.shape)

        # Iterate through centroids and project them onto the line
        for i, centroid in enumerate(centroids):
            # Calculate the Euclidean distances between the centroid and all points on the line
            distances = np.linalg.norm(self.line[:, [1, 0]] - centroid, axis=1)
            # Find the index of the point with the smallest distance
            closest_index = np.argmin(distances)
            # Get the point on the line closest to the centroid
            closest_point = self.line[:, [1, 0]][closest_index]
            projected_centroids[i] = closest_point

        return projected_centroids

    def predict(self, data):
        distances = np.linalg.norm(data[:, np.newaxis] - self.centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        return labels