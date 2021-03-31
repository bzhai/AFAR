"""
Step counts implementation in python from "The Verisense Step Count algorithm"
more details please see:
https://github.com/ShimmerEngineering/Verisense-Toolbox/tree/master/Verisense_step_algorithm
"""

import numpy as np
import pyreadr
from collections import OrderedDict
import pandas as pd


def find_peak(acc, peak_win_len=3):
    """
    :param acc: accelerometer raw data
    :param peak_win_len: window len to detect peak
    :return: peak_info: a matrix with shape: (len(acc)/peak_win_len, 5)
            # peak_info[,0] - peak location
            # peak_info[,1] - acc magnitude
            # peak_info[,2] - periodicity (samples)
            # peak_info[,3] - similarity
            # peak_info[,4] - continuity
    """
    half_k = np.round(peak_win_len / 2).astype(np.int)
    segments = np.floor(acc.shape[0] / peak_win_len).astype(np.int)
    peak_info = np.empty((segments, 5))
    peak_info[:] = np.inf
    for i in np.arange(segments):
        start_idx = i * peak_win_len
        end_idx = start_idx + peak_win_len - 1
        tmp_loc_a = np.argmax(acc[start_idx:end_idx+1])
        tmp_loc_b = i * peak_win_len + tmp_loc_a
        start_idx_ctr = tmp_loc_b - half_k
        if start_idx_ctr < 0:
            start_idx_ctr = 0
        end_idx_ctr = tmp_loc_b + half_k
        if end_idx_ctr > len(acc):
            end_idx_ctr = len(acc)
        check_loc = np.argmax(acc[start_idx_ctr:end_idx_ctr+1])
        if check_loc == half_k:
            peak_info[i, 0] = tmp_loc_b
            peak_info[i, 1] = np.max(acc[start_idx:end_idx+1])
    peak_info = peak_info[~np.in1d(peak_info[:, 0], np.inf)]
    return peak_info


def filter_magnitude(peak_info, mag_thres=1.2):
    peak_info = peak_info[peak_info[:, 1] > mag_thres]
    return peak_info


def calc_periodicity(peak_info, period_min=5, period_max=15):
    """
    calculate the period
    :param peak_info:
    :param period_min:
    :param period_max:
    :return:
    """
    num_peaks = peak_info.shape[0]
    # calculate periodicity
    peak_info[:num_peaks-1, 2] = np.diff(peak_info[:, 0])
    peak_info = peak_info[peak_info[:, 2] > period_min]
    peak_info = peak_info[peak_info[:, 2] < period_max]
    return peak_info


def clac_similarity(peak_info, sim_thres=-0.5):
    """
    # calculate similarity for all peaks
    :param peak_info: the step peak table, the similarity will be calculated based on column 2 and store the results in
    column 3. We calculate the difference between every two peaks and consider it as the similarity.
    :param sim_thres: the threshold used to cut off the difference(a.k.a similarity)
    :return: a 5D matrix contains filtered similarity data
    """

    num_peaks = len(peak_info[:, 1])
    peak_info[:(num_peaks-2), 3] = -np.abs(peak_info[:, 1][2:] - peak_info[:, 1][:-2])
    peak_info = peak_info[peak_info[:, 3] > sim_thres]
    peak_info = peak_info[~np.in1d(peak_info[:, 3], np.inf)]
    # num_peaks = len(peak_info[:,1])
    return peak_info


def filter_by_continue_threshold_variance_threshold(peak_info, acc, cont_win_size=3, cont_thres=4, var_thres=0.001):
    """
    Calculate the continuity by a given window length, then calculate the variance and filter the data by
    a given threshold
    :param peak_info: a 5D matrix
    :param cont_win_size: continue window len
    :param cont_thres: continue threshold
    :param var_thres: variance threshold
    :param fs: frequency of accelerometer data
    :return: all_steps: step count list
    """
    end_for = len(peak_info[:,2])-1
    for i in np.arange(cont_thres-1, end_for):
        v_count = 0
        for x in np.arange(1, cont_thres+1):
            if np.var(acc[int(peak_info[i-x+1, 0]):int(peak_info[i-x+2, 0]+1)], ddof=1) > var_thres:
                v_count = v_count + 1
        if v_count >= cont_win_size:
            peak_info[i, 4] = 1
        else:
            peak_info[i, 4] = 0
    peak_info = peak_info[peak_info[:, 4] == 1, 0]
    return peak_info


def counts_peaks(peak_info, acc, fs=15):
    """
    count the peaks from
    :param peak_info:
    :param fs:
    :return:
    """
    peak_info_count = np.ceil(peak_info/fs).astype(int).tolist()
    peak_info_count_dict = dict((x, peak_info_count.count(x)) for x in set(peak_info_count))
    all_steps = pd.Series(np.arange(np.floor(acc.shape[0]/fs).astype(int)))
    all_steps = all_steps.map(peak_info_count_dict).fillna(0)
    return all_steps


def step_counts_per_sec(raw_acc, peak_win_len=3, period_min=5, period_max=15, fs=15, mag_thres=1.2,
                        cont_win_size=3, cont_thres = 4, var_thres = 0.001):
    """
    # peak_info[,0] - peak location
    # peak_info[,1] - acc magnitude
    # peak_info[,2] - periodicity (samples)
    # peak_info[,3] - similarity
    # peak_info[,4] - continuity
    :param raw_acc:  raw accelerometer data consists of x,y,z readings at 15 hz sampling rate shape: [num_sample, 3]
    :param peak_win_len: window length of peak detection algorithms
    :param period_min: minimum period number
    :param period_max: maximum period number
    :param fs: sampling frequency
    :param mag_thres: magnitude threshold for vector magnitude sqrt(x^2+y^2+z^2)
    :param cont_win_size:  window length for calculating continuity
    :param cont_thres: continuity threshold
    :param var_thres: variance threshold
    :return: a pandas series with steps counted for every second
    """
    acc = np.sqrt(np.power(raw_acc[:,0],2) + np.power(raw_acc[:,1],2) + np.power(raw_acc[:,2],2))
    peak_data = find_peak(acc, peak_win_len)
    peak_data = filter_magnitude(peak_data, mag_thres)
    peak_data = calc_periodicity(peak_data, period_min, period_max)
    peak_data = clac_similarity(peak_data, sim_thres=-0.5)
    peak_data = filter_by_continue_threshold_variance_threshold(peak_data, acc, cont_win_size, cont_thres, var_thres)
    peak_data = counts_peaks(peak_data, acc, fs)
    return peak_data

raw_acc = pyreadr.read_r('acc_xyz_for_step_counts.rds')
raw_ac = raw_acc[None]
raw_acc = raw_acc.values
print(f"Total steps are:{step_counts_per_sec(raw_acc)}")

print("all done")

