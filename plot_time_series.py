import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from glob import glob

from noah_energy.phm.flux_subroutines import ESAT
from utility_modules import read_standalone, load_swc_ne3, load_swc_ne2, load_swc_ne1, load_swc_bo1, load_swc_br1, load_swc_br3, \
    load_swc_ib1, load_swc_ro1
from run_case import flux_file_suffix_dict, aflux_file_suffix_dict


# this script plots the time series of model run in different time scales
# plot time: maize 7/1-9/1, soy 7/15 - 9/1


def main():
    # site info and times
    site = 'US-Ne3'
    year = 2005
    sim_start = datetime(year, 7, 1)
    save_fig = 1
    test_suffix = ''
    suffix_2 = ''
    CAP_beta = 1
    if CAP_beta == 1:
        suffix_2 = '_capped'
    if site == 'US-Ne3' or site == 'US-Ne2' or site == 'US-Ne1':
        obs_hourly = 1
    else:
        obs_hourly = 0
    model_hourly = 1

    # figs
    case_path = glob('/Users/yangyicge/Desktop/watercon/crop_hydro_case/case_run/{}/{}_{}*/'.format(site, site, year))[0]
    crop = case_path.split('/')[-2].split('_')[-1]
    fig_path = '/Users/yangyicge/Desktop/watercon/crop_hydro_case/fig/bysite/{}/{}_{}_{}/'.format(site, site, year, crop)
    fig_name2 = 'time_series_daily' + test_suffix + '.pdf'
    fig_name3 = 'time_series_midday' + test_suffix + '.pdf'
    fig_name7 = 'time_series_daily_gpp' + test_suffix + '.pdf'
    fig_name8 = 'time_series_midday_gpp' + test_suffix + '.pdf'
    fig_name6 = 'time_series_midday_forcing' + test_suffix + '.pdf'
    fig_name9 = 'time_series_daily_paper' + test_suffix + '.pdf'
    fig_name10 = 'time_series_midday_paper' + test_suffix + '.pdf'
    fig_name11 = 'time_series_midday_gpp_paper' + test_suffix + '.pdf'
    # print('fig_path: ', fig_path)
    # print('case_path: ', case_path)
    # print(case_path.split('/'))
    if crop == 'maize':  # check for crops: maize 7/1-8/15, soy 7/15 - 9/1
        start = datetime(year, 7, 1)
        end = datetime(year, 8, 15)
    else:
        start = datetime(year, 7, 15)
        end = datetime(year, 9, 1)
    print('start: ', start)
    print('end: ', end)

    # file path
    flux_path = '/Users/yangyicge/Desktop/watercon/flux/'
    flux_file_suffix = flux_file_suffix_dict[site]
    aflux_file_suffix = aflux_file_suffix_dict[site]
    flux_file = flux_path + 'fluxnet/' + flux_file_suffix
    aflux_file = flux_path + 'ameriflux/' + aflux_file_suffix
    case_name_1 = 'nostress'
    standalone_path_1 = case_path + 'standalone_flux_nostress{}.npy'.format('')
    case_name_2 = 'beta'
    # suffix_2 = ''
    standalone_path_2 = case_path + 'standalone_flux_beta{}{}.npy'.format(suffix_2, '')
    case_name_3 = 'phm'
    standalone_path_3 = case_path + 'standalone_flux_phm{}.npy'.format('')

    # load standalone model run
    print('reading model data...')

    transpiration_1, apar_1, tr_p_1, ET_1, gpp_1, lai_1 = read_standalone(standalone_path_1, case_name_1, start, end, sim_start,
                                                                          model_hourly)
    # transpiration_2, apar_2, tr_p_2, ET_2, gpp_2, lai_2 = read_standalone(standalone_path_2, case_name_2, start, end, sim_start,
    #                                                                       model_hourly)
    transpiration_2, apar_2, tr_p_2, ET_2, gpp_2 = np.zeros(transpiration_1.shape), np.zeros(apar_1.shape), np.zeros(
        tr_p_1.shape), np.zeros(ET_1.shape), np.zeros(gpp_1.shape)
    transpiration_3, apar_3, tr_p_3, ET_3, gpp_3, lai_3 = read_standalone(standalone_path_3, case_name_3, start, end, sim_start,
                                                                          model_hourly)
    # transpiration_3, apar_3, tr_p_3, ET_3, gpp_3 = np.zeros(transpiration_1.shape), np.zeros(apar_1.shape), np.zeros(
    #     tr_p_1.shape), np.zeros(ET_1.shape), np.zeros(gpp_1.shape)

    # load flux
    print('reading obs. data...')
    if len(flux_file_suffix) > 0:
        print('reading obs data from fluxnet...')
        df = pd.read_csv(flux_file)
        ET_flux = np.ma.masked_values(df[(df.TIMESTAMP_START >= int(
            '{:4d}{:02d}{:02d}0000'.format(start.year, start.month, start.day))) & (df.TIMESTAMP_START < int(
            '{:4d}{:02d}{:02d}0000'.format(end.year, end.month, end.day)))].loc[:, 'LE_F_MDS'].values, -9999)
        prec_flux = np.ma.masked_values(df[(df.TIMESTAMP_START >= int(
            '{:4d}{:02d}{:02d}0000'.format(start.year, start.month, start.day))) & (df.TIMESTAMP_START < int(
            '{:4d}{:02d}{:02d}0000'.format(end.year, end.month, end.day)))].loc[:, 'P_F'].values, -9999)
        temp_flux = np.ma.masked_values(df[(df.TIMESTAMP_START >= int(
            '{:4d}{:02d}{:02d}0000'.format(start.year, start.month, start.day))) & (df.TIMESTAMP_START < int(
            '{:4d}{:02d}{:02d}0000'.format(end.year, end.month, end.day)))].loc[:, 'TA_F'].values, -9999)
        rh_flux = np.ma.masked_values(df[(df.TIMESTAMP_START >= int(
            '{:4d}{:02d}{:02d}0000'.format(start.year, start.month, start.day))) & (df.TIMESTAMP_START < int(
            '{:4d}{:02d}{:02d}0000'.format(end.year, end.month, end.day)))].loc[:, 'RH'].values, -9999)
        swin_flux = np.ma.masked_values(df[(df.TIMESTAMP_START >= int(
            '{:4d}{:02d}{:02d}0000'.format(start.year, start.month, start.day))) & (df.TIMESTAMP_START < int(
            '{:4d}{:02d}{:02d}0000'.format(end.year, end.month, end.day)))].loc[:, 'SW_IN_F'].values, -9999)
        gpp_flux = np.ma.masked_values(df[(df.TIMESTAMP_START >= int(
            '{:4d}{:02d}{:02d}0000'.format(start.year, start.month, start.day))) & (df.TIMESTAMP_START < int(
            '{:4d}{:02d}{:02d}0000'.format(end.year, end.month, end.day)))].loc[:, 'GPP_NT_VUT_REF'].values, -9999)
    else:
        print('reading obs data from ameriflux...')
        df = pd.read_csv(aflux_file, skiprows=2)
        ET_flux = np.ma.masked_values(df[(df.TIMESTAMP_START >= int(
            '{:4d}{:02d}{:02d}0000'.format(start.year, start.month, start.day))) & (df.TIMESTAMP_START < int(
            '{:4d}{:02d}{:02d}0000'.format(end.year, end.month, end.day)))].loc[:, 'LE'].values, -9999)
        prec_flux = np.ma.masked_values(df[(df.TIMESTAMP_START >= int(
            '{:4d}{:02d}{:02d}0000'.format(start.year, start.month, start.day))) & (df.TIMESTAMP_START < int(
            '{:4d}{:02d}{:02d}0000'.format(end.year, end.month, end.day)))].loc[:, 'P'].values, -9999)
        temp_flux = np.ma.masked_values(df[(df.TIMESTAMP_START >= int(
            '{:4d}{:02d}{:02d}0000'.format(start.year, start.month, start.day))) & (df.TIMESTAMP_START < int(
            '{:4d}{:02d}{:02d}0000'.format(end.year, end.month, end.day)))].loc[:, 'TA'].values, -9999)
        rh_flux = np.ma.masked_values(df[(df.TIMESTAMP_START >= int(
            '{:4d}{:02d}{:02d}0000'.format(start.year, start.month, start.day))) & (df.TIMESTAMP_START < int(
            '{:4d}{:02d}{:02d}0000'.format(end.year, end.month, end.day)))].loc[:, 'RH'].values, -9999)
        swin_flux = np.ma.masked_values(df[(df.TIMESTAMP_START >= int(
            '{:4d}{:02d}{:02d}0000'.format(start.year, start.month, start.day))) & (df.TIMESTAMP_START < int(
            '{:4d}{:02d}{:02d}0000'.format(end.year, end.month, end.day)))].loc[:, 'SW_IN'].values, -9999)
        gpp_flux = np.zeros(len(ET_flux))

    # load SWC
    swc = 'holder'
    if site == 'US-Ne3':
        swc_h, swc_v = load_swc_ne3(aflux_file, start, end)
        swc = swc_v[1]
        swc_depth = 0.25
    elif site == 'US-Ne2':
        swc_h, swc_v = load_swc_ne2(aflux_file, start, end)
        swc = swc_v[1]
        swc_depth = 0.25
    elif site == 'US-Ne1':
        swc_h, swc_v = load_swc_ne1(aflux_file, start, end)
        swc = swc_v[1]
        swc_depth = 0.25
    elif site == 'US-Bo1':
        swc_h, swc_v = load_swc_bo1(aflux_file, start, end)
        swc = swc_v[1]
        swc_depth = 0.2
    elif site == 'US-Br1':
        swc_h, swc_v = load_swc_br1(aflux_file, start, end)
        swc = swc_v[0]
        swc_depth = 0.05
    elif site == 'US-Br3':
        swc_h, swc_v = load_swc_br3(aflux_file, start, end)
        swc = swc_v[0]
        swc_depth = 0.05
    elif site == 'US-IB1':
        swc_h, swc_v = load_swc_ib1(aflux_file, start, end)
        swc = swc_v[2]
        swc_depth = 0.25
    elif site == 'US-Ro1':
        swc_h, swc_v = load_swc_ro1(aflux_file, start, end)
        swc = swc_v[0]
        swc_depth = 0.1

    # convert all to hourly
    print('converting to hourly...')
    if not obs_hourly:
        ET_flux = np.ma.mean(np.ma.reshape(ET_flux, (-1, 2)), 1)
        prec_flux = np.ma.mean(np.ma.reshape(prec_flux, (-1, 2)), 1)
        temp_flux = np.ma.mean(np.ma.reshape(temp_flux, (-1, 2)), 1)
        rh_flux = np.ma.mean(np.ma.reshape(rh_flux, (-1, 2)), 1)
        swin_flux = np.ma.mean(np.ma.reshape(swin_flux, (-1, 2)), 1)
        gpp_flux = np.ma.mean(np.ma.reshape(gpp_flux, (-1, 2)), 1)
        swc = np.ma.mean(np.ma.reshape(swc, (-1, 2)), 1)
    if not model_hourly:
        transpiration_1 = np.mean(np.reshape(transpiration_1, (-1, 2)), 1)
        apar_1 = np.mean(np.reshape(apar_1, (-1, 2)), 1)
        tr_p_1 = np.mean(np.reshape(tr_p_1, (-1, 2)), 1)
        ET_1 = np.mean(np.reshape(ET_1, (-1, 2)), 1)
        gpp_1 = np.mean(np.reshape(gpp_1, (-1, 2)), 1)
        lai_1 = np.mean(np.reshape(lai_1, (-1, 2)), 1)
        transpiration_2 = np.mean(np.reshape(transpiration_2, (-1, 2)), 1)
        apar_2 = np.mean(np.reshape(apar_2, (-1, 2)), 1)
        tr_p_2 = np.mean(np.reshape(tr_p_2, (-1, 2)), 1)
        ET_2 = np.mean(np.reshape(ET_2, (-1, 2)), 1)
        gpp_2 = np.mean(np.reshape(gpp_2, (-1, 2)), 1)
        transpiration_3 = np.mean(np.reshape(transpiration_3, (-1, 2)), 1)
        apar_3 = np.mean(np.reshape(apar_3, (-1, 2)), 1)
        tr_p_3 = np.mean(np.reshape(tr_p_3, (-1, 2)), 1)
        ET_3 = np.mean(np.reshape(ET_3, (-1, 2)), 1)
        gpp_3 = np.mean(np.reshape(gpp_3, (-1, 2)), 1)

    # get vpd
    ESW, ESI, DESW, DESI = ESAT(temp_flux)
    vpd_flux = (100 - rh_flux) / 100 * ESW

    # mask data based on swc mask
    transpiration_1 = np.ma.array(transpiration_1, mask=swc.mask)
    apar_1 = np.ma.array(apar_1, mask=swc.mask)
    tr_p_1 = np.ma.array(tr_p_1, mask=swc.mask)
    ET_1 = np.ma.array(ET_1, mask=swc.mask)
    gpp_1 = np.ma.array(gpp_1, mask=swc.mask)
    transpiration_2 = np.ma.array(transpiration_2, mask=swc.mask)
    apar_2 = np.ma.array(apar_2, mask=swc.mask)
    tr_p_2 = np.ma.array(tr_p_2, mask=swc.mask)
    ET_2 = np.ma.array(ET_2, mask=swc.mask)
    gpp_2 = np.ma.array(gpp_2, mask=swc.mask)
    transpiration_3 = np.ma.array(transpiration_3, mask=swc.mask)
    apar_3 = np.ma.array(apar_3, mask=swc.mask)
    tr_p_3 = np.ma.array(tr_p_3, mask=swc.mask)
    ET_3 = np.ma.array(ET_3, mask=swc.mask)
    gpp_3 = np.ma.array(gpp_3, mask=swc.mask)

    # data processing
    print('processing data...')
    mean_length = 24
    mean_length_model = 24
    transpiration_1_d = np.mean(np.reshape(transpiration_1, (-1, mean_length_model)), 1)
    transpiration_2_d = np.mean(np.reshape(transpiration_2, (-1, mean_length_model)), 1)
    transpiration_3_d = np.mean(np.reshape(transpiration_3, (-1, mean_length_model)), 1)
    ET_1_d = np.mean(np.reshape(ET_1, (-1, mean_length_model)), 1)
    ET_2_d = np.mean(np.reshape(ET_2, (-1, mean_length_model)), 1)
    ET_3_d = np.mean(np.reshape(ET_3, (-1, mean_length_model)), 1)
    tr_p_1_d = np.mean(np.reshape(tr_p_1, (-1, mean_length_model)), 1)
    tr_p_2_d = np.mean(np.reshape(tr_p_2, (-1, mean_length_model)), 1)
    tr_p_3_d = np.mean(np.reshape(tr_p_3, (-1, mean_length_model)), 1)
    gpp_1_d = np.mean(np.reshape(gpp_1, (-1, mean_length_model)), 1)
    gpp_2_d = np.mean(np.reshape(gpp_2, (-1, mean_length_model)), 1)
    gpp_3_d = np.mean(np.reshape(gpp_3, (-1, mean_length_model)), 1)
    ET_flux_d = np.mean(np.reshape(ET_flux, (-1, mean_length)), 1)
    prec_flux_d = np.sum(np.reshape(prec_flux, (-1, mean_length)), 1)
    swc_d = np.mean(np.reshape(swc, (-1, mean_length)), 1)
    swin_flux_d = np.mean(np.reshape(swin_flux, (-1, mean_length)), 1)
    temp_flux_d = np.mean(np.reshape(temp_flux, (-1, mean_length)), 1)
    rh_flux_d = np.mean(np.reshape(rh_flux, (-1, mean_length)), 1)
    vpd_flux_d = np.mean(np.reshape(vpd_flux, (-1, mean_length)), 1)
    apar_1_d = np.mean(np.reshape(apar_1, (-1, mean_length_model)), 1)
    gpp_flux_d = np.mean(np.reshape(gpp_flux, (-1, mean_length)), 1)
    lai_1_d = np.mean(np.reshape(lai_1, (-1, mean_length_model)), 1)

    transpiration_1_md = transpiration_1[mean_length_model // 2::mean_length_model]
    transpiration_2_md = transpiration_2[mean_length_model // 2::mean_length_model]
    transpiration_3_md = transpiration_3[mean_length_model // 2::mean_length_model]
    ET_1_md = ET_1[mean_length_model // 2::mean_length_model]
    ET_2_md = ET_2[mean_length_model // 2::mean_length_model]
    ET_3_md = ET_3[mean_length_model // 2::mean_length_model]
    tr_p_1_md = tr_p_1[mean_length_model // 2::mean_length_model]
    tr_p_2_md = tr_p_2[mean_length_model // 2::mean_length_model]
    tr_p_3_md = tr_p_3[mean_length_model // 2::mean_length_model]
    gpp_1_md = gpp_1[mean_length_model // 2::mean_length_model]
    gpp_2_md = gpp_2[mean_length_model // 2::mean_length_model]
    gpp_3_md = gpp_3[mean_length_model // 2::mean_length_model]
    ET_flux_md = ET_flux[mean_length // 2::mean_length]
    swc_md = swc[mean_length // 2::mean_length]
    swin_flux_md = swin_flux[mean_length // 2::mean_length]
    temp_flux_md = temp_flux[mean_length // 2::mean_length]
    rh_flux_md = rh_flux[mean_length // 2::mean_length]
    vpd_flux_md = vpd_flux[mean_length // 2::mean_length]
    apar_1_md = apar_1[mean_length_model // 2::mean_length_model]
    gpp_flux_md = gpp_flux[mean_length // 2::mean_length]
    lai_1_md = lai_1[mean_length_model // 2::mean_length_model]

    # calculate metrics
    print('calculating metrics...')
    r2_1 = np.corrcoef(ET_flux_md, ET_1_md)[0, 1] ** 2
    r2_2 = np.corrcoef(ET_flux_md, ET_2_md)[0, 1] ** 2
    r2_3 = np.corrcoef(ET_flux_md, ET_3_md)[0, 1] ** 2
    print('r2_1: ', r2_1)
    print('r2_2: ', r2_2)
    print('r2_3: ', r2_3)

    # plot
    print('plotting...')
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    times = pd.date_range(start, end, freq='1H')[:-1]
    times_d = pd.date_range(start, end, freq='D')[:-1]
    et_range = [-5, 1000]
    swc_range = [0.1, 0.6]
    prec_range = [0, 50]
    et_range_d = [-5, 500]
    vpd_range = (0, 4000)
    vpd_range_d = (0, 2000)
    rad_range = (0, 1200)
    rad_range_d = (0, 400)
    gpp_range = [0, 120]
    gpp_range_d = [0, 50]

    # plot hourly
    fig, ax = plt.subplots(5, 1, figsize=(14, 15))
    plot_time_series(fig, ax, times, times_d, tr_p_1, tr_p_2, tr_p_3, transpiration_1, transpiration_2, transpiration_3, ET_1, ET_2,
                     ET_3, ET_flux, swc, swc_d, prec_flux_d, swin_flux, apar_1, vpd_flux, swc_range, et_range, prec_range, rad_range,
                     vpd_range, suffix_2)
    fig.suptitle('Response hourly')
    plt.subplots_adjust(top=0.95)

    # plot daily
    fig2, ax2 = plt.subplots(5, 1, figsize=(14, 15))
    plot_time_series(fig2, ax2, times_d, times_d, tr_p_1_d, tr_p_2_d, tr_p_3_d, transpiration_1_d, transpiration_2_d,
                     transpiration_3_d, ET_1_d, ET_2_d, ET_3_d, ET_flux_d, swc_d, swc_d, prec_flux_d, swin_flux_d, apar_1_d, vpd_flux_d,
                     swc_range, et_range_d, prec_range, rad_range_d, vpd_range_d, suffix_2)
    fig2.suptitle('Response daily')
    plt.subplots_adjust(top=0.95)

    fig7, ax7 = plt.subplots(3, 1, figsize=(14, 9))
    plot_time_series_carbon(fig7, ax7, times_d, times_d, gpp_1_d, gpp_2_d, gpp_3_d, gpp_flux_d, swc_d, swc_d, prec_flux_d, swin_flux_d,
                            apar_1_d, vpd_flux_d, swc_range, gpp_range_d, prec_range, rad_range_d, vpd_range_d, suffix_2)
    fig7.suptitle('GPP daily')
    plt.subplots_adjust(top=0.95)

    # plot midday
    fig3, ax3 = plt.subplots(5, 1, figsize=(14, 15))
    plot_time_series(fig3, ax3, times_d, times_d, tr_p_1_md, tr_p_2_md, tr_p_3_md, transpiration_1_md, transpiration_2_md,
                     transpiration_3_md, ET_1_md, ET_2_md, ET_3_md, ET_flux_md, swc_md, swc_d, prec_flux_d, swin_flux_md, apar_1_md,
                     vpd_flux_md, swc_range, et_range, prec_range, rad_range, vpd_range, suffix_2)
    fig3.suptitle('Response midday')
    plt.subplots_adjust(top=0.95)

    fig8, ax8 = plt.subplots(3, 1, figsize=(14, 9))
    plot_time_series_carbon(fig8, ax8, times_d, times_d, gpp_1_md, gpp_2_md, gpp_3_md, gpp_flux_md, swc_md, swc_d, prec_flux_d,
                            swin_flux_md, apar_1_md, vpd_flux_md, swc_range, gpp_range, prec_range, rad_range, vpd_range,
                            suffix_2)
    fig8.suptitle('GPP midday')
    plt.subplots_adjust(top=0.95)

    # plot forcing
    t_range = (-10, 40)
    t_range_d = (-10, 30)
    lai_range = (0, 8)

    # plot hourly
    fig4, ax4 = plt.subplots(4, 1, figsize=(14, 12))
    plot_time_series_force(fig4, ax4, times, times_d, swin_flux, apar_1, temp_flux, rh_flux, vpd_flux, lai_1, prec_flux_d,
                           swc_d, rad_range, t_range, vpd_range, lai_range, prec_range, swc_range)
    fig4.suptitle('Forcing hourly')
    plt.subplots_adjust(top=0.95)

    # plot daily
    fig5, ax5 = plt.subplots(4, 1, figsize=(14, 12))
    plot_time_series_force(fig5, ax5, times_d, times_d, swin_flux_d, apar_1_d, temp_flux_d, rh_flux_d, vpd_flux_d, lai_1_d, prec_flux_d,
                           swc_d, rad_range_d, t_range_d, vpd_range_d, lai_range, prec_range, swc_range)
    fig5.suptitle('Forcing daily')
    plt.subplots_adjust(top=0.95)

    # plot midday
    fig6, ax6 = plt.subplots(4, 1, figsize=(14, 12))
    plot_time_series_force(fig6, ax6, times_d, times_d, swin_flux_md, apar_1_md, temp_flux_md, rh_flux_md, vpd_flux_md, lai_1_md,
                           prec_flux_d, swc_d, rad_range, t_range, vpd_range, lai_range, prec_range, swc_range)
    fig6.suptitle('Forcing midday')
    plt.subplots_adjust(top=0.95)

    # plot paper

    # plot daily
    fig9, ax9 = plt.subplots(2, 1, figsize=(8, 6))
    plot_time_series_paper(fig9, ax9, times_d, times_d, ET_1_d, ET_2_d, ET_3_d, ET_flux_d, swc_d, swc_d, prec_flux_d, vpd_flux_d,
                           swc_range, et_range_d, prec_range, vpd_range_d, swc_depth, r2_1, r2_3)

    # plot midday
    fig10, ax10 = plt.subplots(2, 1, figsize=(8, 6))
    plot_time_series_paper(fig10, ax10, times_d, times_d, ET_1_md, ET_2_md, ET_3_md, ET_flux_md, swc_md, swc_d, prec_flux_d,
                           vpd_flux_md, swc_range, et_range, prec_range, vpd_range, swc_depth, r2_1, r2_3)

    # plot carbon midday
    fig11, ax11 = plt.subplots(2, 1, figsize=(8, 6))
    plot_time_series_carbon_paper(fig11, ax11, times_d, times_d, gpp_1_md, gpp_2_md, gpp_3_md, gpp_flux_md, swc_md, swc_d, prec_flux_d,
                                  vpd_flux_md, swc_range, gpp_range, prec_range, vpd_range, swc_depth)

    if save_fig:
        fig2.savefig(fig_path + fig_name2)
        fig3.savefig(fig_path + fig_name3)
        fig7.savefig(fig_path + fig_name7)
        fig8.savefig(fig_path + fig_name8)
        fig6.savefig(fig_path + fig_name6)
        fig9.savefig(fig_path + fig_name9)
        fig10.savefig(fig_path + fig_name10)
        fig11.savefig(fig_path + fig_name11)


    # plt.show()

    return


def plot_time_series(fig, ax, times, times_d, tr_p_1, tr_p_2, tr_p_3, transpiration_1, transpiration_2, transpiration_3, ET_1, ET_2,
                     ET_3, ET_flux, swc, swc_d, prec_flux_d, swin, apar, vpd, swc_range, et_range, prec_range, rad_range, vpd_range,
                     suffix_2):
    ax[0].plot(times, tr_p_3, label='T NHL', c='b')
    ax[0].plot(times, transpiration_3, label='T PHM', c='c')
    ax[0].plot(times, ET_3, label='ET PHM', c='g')
    ax[0].plot(times, ET_flux, label='ET obs.', c='k')
    ax[0].set_ylabel('PHM T W/m2')
    ax[0].set_ylim(et_range)
    ax[0].legend(loc='upper right')
    ax2 = ax[0].twinx()
    ax2.plot(times, swc, c='tab:brown')
    ax2.set_ylabel('SWC', color='tab:brown')
    ax2.set_ylim(swc_range)
    ax2.tick_params(axis='y', labelcolor='tab:brown')

    ax[1].plot(times, tr_p_2, label='Tww', c='b')
    ax[1].plot(times, transpiration_2, label='T', c='c')
    ax[1].plot(times, ET_2, label='ET', c='g')
    ax[1].plot(times, ET_flux, label='ET obs.', c='k')
    ax[1].set_ylabel('Beta{} T W/m2'.format(suffix_2))
    ax[1].set_ylim(et_range)
    ax[1].legend(loc='upper right')
    ax2 = ax[1].twinx()
    ax2.plot(times, swc, c='tab:brown')
    ax2.set_ylabel('SWC', color='tab:brown')
    ax2.set_ylim(swc_range)
    ax2.tick_params(axis='y', labelcolor='tab:brown')

    ax[2].plot(times, tr_p_1, label='Tww', c='b')
    ax[2].plot(times, transpiration_1, label='T', c='c')
    ax[2].plot(times, ET_1, label='ET', c='g')
    ax[2].plot(times, ET_flux, label='ET obs.', c='k')
    ax[2].set_ylabel('Nostress T W/m2')
    ax[2].set_ylim(et_range)
    ax[2].legend(loc='upper right')
    ax2 = ax[2].twinx()
    ax2.plot(times, swc, c='tab:brown')
    ax2.set_ylabel('SWC', color='tab:brown')
    ax2.set_ylim(swc_range)
    ax2.tick_params(axis='y', labelcolor='tab:brown')

    ax[3].plot(times, apar, c='k', label='APAR')
    ax[3].plot(times, swin, c='k', ls='--', label='SWin')
    ax[3].set_ylabel('radiation W/m2')
    ax[3].set_ylim(rad_range)
    ax[3].legend(loc='upper right')
    ax2 = ax[3].twinx()
    ax2.plot(times, vpd, c='tab:red', label='VPD')
    ax2.set_ylabel('VPD', color='tab:red')
    ax2.set_ylim(vpd_range)
    ax2.tick_params(axis='y', labelcolor='tab:red')

    ax[4].plot(times_d, prec_flux_d, c='k')
    ax[4].set_ylabel('Precipitation mm/d')
    ax[4].set_ylim(prec_range)
    ax2 = ax[4].twinx()
    ax2.plot(times_d, swc_d, c='tab:brown')
    ax2.set_ylabel('SWC', color='tab:brown')
    ax2.set_ylim(swc_range)
    ax2.tick_params(axis='y', labelcolor='tab:brown')

    fig.autofmt_xdate()
    fig.tight_layout()

    return


def plot_time_series_carbon(fig, ax, times, times_d, gpp_1, gpp_2, gpp_3, gpp_flux, swc, swc_d, prec_flux_d, swin, apar, vpd, swc_range,
                            gpp_range, prec_range, rad_range, vpd_range, suffix_2):
    ax[0].plot(times, gpp_1, label='No stress', c='b')
    ax[0].plot(times, gpp_2, label='Beta{}'.format(suffix_2), c='c')
    ax[0].plot(times, gpp_3, label='PHM', c='g')
    ax[0].plot(times, gpp_flux, label='Obs.', c='k')
    ax[0].set_ylabel('GPP umol CO2/m2/s')
    ax[0].set_ylim(gpp_range)
    ax[0].legend(loc='upper right')
    ax2 = ax[0].twinx()
    ax2.plot(times, swc, c='tab:brown')
    ax2.set_ylabel('SWC', color='tab:brown')
    ax2.set_ylim(swc_range)
    ax2.tick_params(axis='y', labelcolor='tab:brown')

    ax[1].plot(times, apar, c='k', label='APAR')
    ax[1].plot(times, swin, c='k', ls='--', label='SWin')
    ax[1].set_ylabel('radiation W/m2')
    ax[1].set_ylim(rad_range)
    ax[1].legend(loc='upper right')
    ax2 = ax[1].twinx()
    ax2.plot(times, vpd, c='tab:red', label='VPD')
    ax2.set_ylabel('VPD', color='tab:red')
    ax2.set_ylim(vpd_range)
    ax2.tick_params(axis='y', labelcolor='tab:red')

    ax[2].plot(times_d, prec_flux_d, c='k')
    ax[2].set_ylabel('Precipitation mm/d')
    ax[2].set_ylim(prec_range)
    ax2 = ax[2].twinx()
    ax2.plot(times_d, swc_d, c='tab:brown')
    ax2.set_ylabel('SWC', color='tab:brown')
    ax2.set_ylim(swc_range)
    ax2.tick_params(axis='y', labelcolor='tab:brown')

    fig.autofmt_xdate()
    fig.tight_layout()

    return


def plot_time_series_force(fig, ax, times, times_d, swin, apar, temp, rh, vpd, lai, prec_flux_d, swc_d, rad_range, t_range,
                           vpd_range, lai_range, prec_range, swc_range):
    ax[0].plot(times, swin, label='SW', c='r')
    ax[0].plot(times, apar, label='APAR1', c='m')
    ax[0].set_ylabel('Radiation W/m2')
    ax[0].set_ylim(rad_range)
    ax[0].legend()

    ax[1].plot(times, temp, c='r')
    ax[1].set_ylabel('Temperature ºC')
    ax[1].set_ylim(t_range)
    ax2 = ax[1].twinx()
    ax2.plot(times, rh, c='tab:brown')
    ax2.set_ylabel('RH', color='tab:brown')
    ax2.set_ylim((0, 100))
    ax2.tick_params(axis='y', labelcolor='tab:brown')

    ax[2].plot(times, vpd, c='r')
    ax[2].set_ylabel('VPD Pa')
    ax[2].set_ylim(vpd_range)
    ax2 = ax[2].twinx()
    ax2.plot(times, lai, c='tab:green')
    ax2.set_ylabel('LAI', color='tab:green')
    ax2.set_ylim(lai_range)
    ax2.tick_params(axis='y', labelcolor='tab:green')

    ax[3].plot(times_d, prec_flux_d, c='k')
    ax[3].set_ylabel('Precipitation mm/d')
    ax[3].set_ylim(prec_range)
    ax2 = ax[3].twinx()
    ax2.plot(times_d, swc_d, c='tab:brown')
    ax2.set_ylabel('SWC', color='tab:brown')
    ax2.set_ylim(swc_range)
    ax2.tick_params(axis='y', labelcolor='tab:brown')

    fig.autofmt_xdate()
    fig.tight_layout()

    return


def plot_time_series_paper(fig, ax, times, times_d, ET_1, ET_2, ET_3, ET_flux, swc, swc_d, prec_flux_d, vpd, swc_range, et_range,
                           prec_range, vpd_range, swc_depth, r2_1, r2_3):
    ax[0].plot(times, ET_1, label='NHL', c='royalblue', lw=1, ls='-')
    ax[0].plot(times, ET_3, label='PHM', c='green', lw=1, ls='-')
    ax[0].plot(times, ET_flux, label='Obs.', c='k', lw=1)
    ax[0].set_ylabel('ET W/m2')
    ax[0].set_ylim(et_range)
    ax[0].legend(loc='upper left')
    ax[0].text(0.69, 0.9, r'$R^2$ NHL: {:.2f}'.format(r2_1), transform=ax[0].transAxes, c='royalblue')
    ax[0].text(0.69, 0.8, r'$R^2$ PHM: {:.2f}'.format(r2_3), transform=ax[0].transAxes, c='green')
    ax2 = ax[0].twinx()
    ax2.plot(times, vpd, c='tab:red', ls='--', lw=1, label='VPD')
    ax2.set_ylabel('VPD Pa', color='tab:red')
    ax2.set_ylim(vpd_range)
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.legend(loc='upper right')

    if not np.sum(~prec_flux_d.mask) == 0:
        ax[1].bar(times_d, prec_flux_d, color='tab:blue', label='Precipitation')
    ax[1].set_ylabel('Precipitation mm/d')
    ax[1].set_ylim(prec_range)
    ax[1].legend(loc='upper left')
    ax2 = ax[1].twinx()
    ax2.plot(times_d, swc_d, c='tab:brown', lw=1, label='SWC at -{}m'.format(swc_depth))
    ax2.set_ylabel('SWC', color='tab:brown')
    ax2.set_ylim(swc_range)
    ax2.tick_params(axis='y', labelcolor='tab:brown')
    ax2.legend(loc='upper right')

    fig.autofmt_xdate()
    fig.tight_layout()

    return


def plot_time_series_carbon_paper(fig, ax, times, times_d, gpp_1, gpp_2, gpp_3, gpp_flux, swc, swc_d, prec_flux_d, vpd, swc_range,
                                  gpp_range,
                                  prec_range, vpd_range, swc_depth):
    ax[0].plot(times, gpp_1, label='NHL', c='royalblue', lw=1, ls='-')
    ax[0].plot(times, gpp_3, label='PHM', c='green', lw=1, ls='-')
    ax[0].plot(times, gpp_flux, label='Obs.', c='k', lw=1)
    ax[0].set_ylabel('GPP umol CO2/m2/s')
    ax[0].set_ylim(gpp_range)
    ax[0].legend(loc='upper left')
    ax2 = ax[0].twinx()
    ax2.plot(times, vpd, c='tab:red', ls='--', lw=1, label='VPD')
    ax2.set_ylabel('VPD Pa', color='tab:red')
    ax2.set_ylim(vpd_range)
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.legend(loc='upper right')

    if not np.sum(~prec_flux_d.mask) == 0:
        ax[1].bar(times_d, prec_flux_d, color='tab:blue', label='Precipitation')
    ax[1].set_ylabel('Precipitation mm/d')
    ax[1].set_ylim(prec_range)
    ax[1].legend(loc='upper left')
    ax2 = ax[1].twinx()
    ax2.plot(times_d, swc_d, c='tab:brown', lw=1, label='SWC at -{}m'.format(swc_depth))
    ax2.set_ylabel('SWC', color='tab:brown')
    ax2.set_ylim(swc_range)
    ax2.tick_params(axis='y', labelcolor='tab:brown')
    ax2.legend(loc='upper right')

    fig.autofmt_xdate()
    fig.tight_layout()

    return


if __name__ == '__main__':
    main()
