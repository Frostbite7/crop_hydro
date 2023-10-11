import numpy as np
import pandas as pd


def load_swc_ne3(aflux_file, start, end):
    dfa = pd.read_csv(aflux_file, skiprows=2)
    h_number = 4
    v_number = 4
    swc_h = []
    for point in range(h_number):
        swc_v_ = []
        for depth in range(v_number):
            swc_ = dfa[(dfa.TIMESTAMP_START >= int(
                '{:4d}{:02d}{:02d}0000'.format(start.year, start.month, start.day))) & (dfa.TIMESTAMP_START < int(
                '{:4d}{:02d}{:02d}0000'.format(end.year, end.month, end.day)))].loc[:, 'SWC_PI_F_{}_{}_{}'.format(
                point + 1, depth + 1, 1)]
            # print('Gaps! SWC_H{}V{}R1:'.format(point + 1, depth + 1), np.sum(swc_ == -9999))
            # swc_v_.append(swc_.replace(-9999, np.nan).interpolate().values / 100)
            swc_v_.append(swc_)
        swc_h.append(swc_v_)
    # swc_h_ = np.array(swc_h)
    swc_h = np.ma.masked_values(swc_h, -9999) / 100
    swc_v = np.ma.mean(swc_h, 0)
    swc_v = np.maximum(swc_v, 0.02)
    # print('swc_v:', swc_v)

    return swc_h, swc_v


def load_swc_ne2(aflux_file, start, end):
    dfa = pd.read_csv(aflux_file, skiprows=2)
    h_number = 3
    v_number = 4
    swc_h = []
    for point in range(h_number):
        swc_v_ = []
        for depth in range(v_number):
            swc_ = dfa[(dfa.TIMESTAMP_START >= int(
                '{:4d}{:02d}{:02d}0000'.format(start.year, start.month, start.day))) & (dfa.TIMESTAMP_START < int(
                '{:4d}{:02d}{:02d}0000'.format(end.year, end.month, end.day)))].loc[:, 'SWC_PI_F_{}_{}_{}'.format(
                point + 1, depth + 1, 1)]
            # print('Gaps! SWC_H{}V{}R1:'.format(point + 1, depth + 1), np.sum(swc_ == -9999))
            # swc_v_.append(swc_.replace(-9999, np.nan).interpolate().values / 100)
            swc_v_.append(swc_)
        swc_h.append(swc_v_)
    # swc_h_ = np.array(swc_h)
    swc_h = np.ma.masked_values(swc_h, -9999) / 100
    swc_v = np.ma.mean(swc_h, 0)
    swc_v = np.maximum(swc_v, 0.02)
    # print('swc_v:', swc_v)

    return swc_h, swc_v


def load_swc_bo1(aflux_file, start, end):
    dfa = pd.read_csv(aflux_file, skiprows=2)
    h_number = 1
    v_number = 2
    swc_h = []
    for point in range(h_number):
        swc_v_ = []
        for depth in range(v_number):
            swc_ = dfa[(dfa.TIMESTAMP_START >= int(
                '{:4d}{:02d}{:02d}0000'.format(start.year, start.month, start.day))) & (dfa.TIMESTAMP_START < int(
                '{:4d}{:02d}{:02d}0000'.format(end.year, end.month, end.day)))].loc[:, 'SWC_{}'.format(depth + 1)]
            # print('Gaps! SWC_H{}V{}R1:'.format(point + 1, depth + 1), np.sum(swc_ == -9999))
            # swc_v_.append(swc_.replace(-9999, np.nan).interpolate().values / 100)
            swc_v_.append(swc_)
        swc_h.append(swc_v_)
    # swc_h_ = np.array(swc_h)
    swc_h = np.ma.masked_values(swc_h, -9999) / 100
    swc_v = np.ma.mean(swc_h, 0)
    swc_v = np.maximum(swc_v, 0.02)
    # print('swc_v:', swc_v)

    return swc_h, swc_v


def read_standalone(standalone_path, stress_model, start, end, sim_start, hourly):
    interval = 24 * hourly + 48 * (1 - hourly)
    sel_1 = (start - sim_start).days * interval
    sel_2 = (end - sim_start).days * interval

    results = np.load(standalone_path, allow_pickle=True)
    transpiration = results[1, sel_1:sel_2].astype(np.float64)
    soil_evaporation = results[2, sel_1:sel_2].astype(np.float64)
    canopy_evaporation = results[3, sel_1:sel_2].astype(np.float64)
    ET = transpiration + soil_evaporation + canopy_evaporation
    rssun = results[4, sel_1:sel_2].astype(np.float64)
    rssha = results[5, sel_1:sel_2].astype(np.float64)
    psn = results[6, sel_1:sel_2].astype(np.float64)
    fsun = results[7, sel_1:sel_2].astype(np.float64)
    fsh = results[8, sel_1:sel_2].astype(np.float64)
    sav = results[9, sel_1:sel_2].astype(np.float64)
    sag = results[10, sel_1:sel_2].astype(np.float64)
    fsa = results[11, sel_1:sel_2].astype(np.float64)
    fsr = results[12, sel_1:sel_2].astype(np.float64)
    fira = results[13, sel_1:sel_2].astype(np.float64)
    apar = results[14, sel_1:sel_2].astype(np.float64)
    parsun = results[15, sel_1:sel_2].astype(np.float64)
    parsha = results[16, sel_1:sel_2].astype(np.float64)
    lai = results[17, sel_1:sel_2].astype(np.float64)
    tr_p = results[18, sel_1:sel_2].astype(np.float64)

    if stress_model == 'sdb':
        tr_s = results[19, sel_1:sel_2].astype(np.float64)
        return transpiration, apar, tr_p, tr_s, ET, psn
    else:
        return transpiration, apar, tr_p, ET, psn, lai
