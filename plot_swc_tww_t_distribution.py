import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from glob import glob

from utility_modules import read_standalone, load_swc_ne3, load_swc_ne2, load_swc_bo1
from run_case import flux_file_suffix_dict, aflux_file_suffix_dict


# this script plots the binned scatter response of PHM and Beta models
# plot time: maize 7/1-9/1, soy 7/15 - 9/1

def main():
    # set up multiple sites and years
    # sites = ['US-Ne3', 'US-Ne2']
    sites = ['US-Bo1']
    # for maize
    # site_years = {'US-Ne2': [2005, 2007, 2009, 2010, 2011, 2012], 'US-Ne3': [2005, 2007, 2009, 2011],
    #               'US-Bo1': [2001, 2003, 2005, 2007]}
    # for soy
    # site_years = {'US-Ne2': [2004, 2006, 2008], 'US-Ne3': [2004, 2006, 2008, 2010, 2012]}
    # for testing
    site_years = {'US-Bo1': [2007]}
    obs_hourly = 0
    model_hourly = 1
    test_suffix = ''
    save_fig = 0
    fig_prefix = 'NA_'
    suffix_2 = ''
    plot_target_name = 'Beta'
    plot_target_suffix = ''
    CAP_beta = 1
    if CAP_beta == 1:
        suffix_2 = '_capped'
        plot_target_name = 'DHLF'

    # set up figs
    fig_path = '/Users/yangyicge/Desktop/watercon/crop_hydro_case/fig/bysite/aggregate/'
    fig_name1 = fig_prefix + 'swc_tww_distr_obs' + test_suffix + '.pdf'
    fig_name2 = fig_prefix + 'swc_tww_distr_target-ref' + test_suffix + '.pdf'

    # loading data
    ET_flux_l = []
    swc_l = []
    transpiration_1_l = []
    tr_p_1_l = []
    ET_1_l = []
    transpiration_2_l = []
    ET_2_l = []
    transpiration_3_l = []
    ET_3_l = []
    for site in sites:
        for year in site_years[site]:
            print('processing site: {}, year: {}'.format(site, year))
            # site info and times
            # site = 'US-Ne3'
            # year = 2005
            case_path = glob("/Users/yangyicge/Desktop/watercon/crop_hydro_case/case_run/{}/{}_{}*/".format(site, site, year))[0]
            crop = case_path.split('/')[-2].split('_')[-1]
            if crop == 'maize':
                start = datetime(year, 7, 1)
                end = datetime(year, 8, 15)
            else:
                start = datetime(year, 7, 15)
                end = datetime(year, 9, 1)
            sim_start = datetime(year, 7, 1)

            # figs
            fig_path = '/Users/yangyicge/Desktop/watercon/crop_hydro_case/fig/bysite/{}/{}_{}_{}/'.format(site, site, year, crop)
            fig_name1 = 'swc_tww_distr_obs' + test_suffix + '.pdf'
            fig_name2 = 'swc_tww_distr_target-ref' + test_suffix + '.pdf'

            # file path
            flux_path = '/Users/yangyicge/Desktop/watercon/flux/'
            flux_file_suffix = flux_file_suffix_dict[site]
            aflux_file_suffix = aflux_file_suffix_dict[site]
            flux_file = flux_path + 'fluxnet/' + flux_file_suffix
            aflux_file = flux_path + 'ameriflux/' + aflux_file_suffix

            # case info
            case_name_1 = 'nostress'
            standalone_path_1 = case_path + 'standalone_flux_nostress{}.npy'.format(test_suffix)
            case_name_2 = 'beta'
            if site == 'US-Ne2':
                test_suffix = '_kmax4.18'
            standalone_path_2 = case_path + 'standalone_flux_beta{}{}.npy'.format(suffix_2, test_suffix)
            case_name_3 = 'phm'
            if site == 'US-Ne2':
                test_suffix = '_kx1.0'
            standalone_path_3 = case_path + 'standalone_flux_phm{}.npy'.format(test_suffix)
            test_suffix = ''

            # load standalone model run
            # print('reading model data...')
            transpiration_1, apar_1, tr_p_1, ET_1, gpp_1, lai_1 = read_standalone(standalone_path_1, case_name_1, start, end, sim_start,
                                                                                  model_hourly)
            transpiration_2, apar_2, tr_p_2, ET_2, gpp_2, lai_2 = read_standalone(standalone_path_2, case_name_2, start, end, sim_start,
                                                                                  model_hourly)
            # transpiration_2, apar_2, tr_p_2, ET_2, gpp_2 = transpiration_1, apar_1, tr_p_1, ET_1, gpp_1
            transpiration_3, apar_3, tr_p_3, ET_3, gpp_3, lai_3 = read_standalone(standalone_path_3, case_name_3, start, end, sim_start,
                                                                                  model_hourly)
            # transpiration_3, apar_3, tr_p_3, ET_3, gpp_3 = transpiration_1, apar_1, tr_p_1, ET_1, gpp_1
            # print(transpiration_1.shape)

            # load flux
            # print('reading obs. data...')
            if len(flux_file_suffix) > 0:
                df = pd.read_csv(flux_file)
                ET_flux = np.ma.masked_values(df[(df.TIMESTAMP_START >= int(
                    '{:4d}{:02d}{:02d}0000'.format(start.year, start.month, start.day))) & (df.TIMESTAMP_START < int(
                    '{:4d}{:02d}{:02d}0000'.format(end.year, end.month, end.day)))].loc[:, 'LE_F_MDS'].values, -9999)
            else:
                df = pd.read_csv(aflux_file, skiprows=2)
                ET_flux = np.ma.masked_values(df[(df.TIMESTAMP_START >= int(
                    '{:4d}{:02d}{:02d}0000'.format(start.year, start.month, start.day))) & (df.TIMESTAMP_START < int(
                    '{:4d}{:02d}{:02d}0000'.format(end.year, end.month, end.day)))].loc[:, 'LE'].values, -9999)

            # load SWC
            swc = 'holder'
            if site == 'US-Ne3':
                swc_h, swc_v = load_swc_ne3(aflux_file, start, end)
                swc = swc_v[1]
            elif site == 'US-Ne2':
                swc_h, swc_v = load_swc_ne2(aflux_file, start, end)
                swc = swc_v[1]
            elif site == 'US-Bo1':
                swc_h, swc_v = load_swc_bo1(aflux_file, start, end)
                swc = swc_v[1]

            # record data
            ET_flux_l.append(ET_flux)
            swc_l.append(swc)
            transpiration_1_l.append(transpiration_1)
            tr_p_1_l.append(tr_p_1)
            ET_1_l.append(ET_1)
            transpiration_2_l.append(transpiration_2)
            ET_2_l.append(ET_2)
            transpiration_3_l.append(transpiration_3)
            ET_3_l.append(ET_3)

    # convert all to hourly
    print('processing data...')
    print('converting to hourly...')
    ET_flux = swc = transpiration_1 = tr_p_1 = ET_1 = transpiration_2 = ET_2 = transpiration_3 = ET_3 = 'holder'
    if not obs_hourly:
        ET_flux = np.ma.mean(np.ma.reshape(np.ma.array(ET_flux_l), (-1, 2)), 1)
        swc = np.ma.mean(np.ma.reshape(np.ma.array(swc_l), (-1, 2)), 1)
    else:
        ET_flux = np.ma.array(ET_flux_l).flatten()
        swc = np.ma.array(swc_l).flatten()
    if not model_hourly:
        transpiration_1 = np.mean(np.reshape(transpiration_1_l, (-1, 2)), 1)
        tr_p_1 = np.mean(np.reshape(tr_p_1_l, (-1, 2)), 1)
        ET_1 = np.mean(np.reshape(ET_1_l, (-1, 2)), 1)
        transpiration_2 = np.mean(np.reshape(transpiration_2_l, (-1, 2)), 1)
        ET_2 = np.mean(np.reshape(ET_2_l, (-1, 2)), 1)
        transpiration_3 = np.mean(np.reshape(transpiration_3_l, (-1, 2)), 1)
        ET_3 = np.mean(np.reshape(ET_3_l, (-1, 2)), 1)
    else:
        transpiration_1 = np.array(transpiration_1_l).flatten()
        tr_p_1 = np.array(tr_p_1_l).flatten()
        ET_1 = np.array(ET_1_l).flatten()
        transpiration_2 = np.array(transpiration_2_l).flatten()
        ET_2 = np.array(ET_2_l).flatten()
        transpiration_3 = np.array(transpiration_3_l).flatten()
        ET_3 = np.array(ET_3_l).flatten()

    # calculate transpiration difference
    ET_phm_obs = ET_3 - ET_flux
    ET_phm_obs_percent = (ET_3 - ET_flux) / ET_flux
    ET_beta_obs = ET_2 - ET_flux
    ET_beta_obs_percent = (ET_2 - ET_flux) / ET_flux
    ET_ns_obs = ET_1 - ET_flux
    ET_ns_obs_percent = (ET_1 - ET_flux) / ET_flux
    tr_beta_phm = transpiration_2 - transpiration_3
    tr_beta_phm_percent = (transpiration_2 - transpiration_3) / transpiration_3

    # calculate mid-day value
    mean_length = 24
    ET_obs_md = ET_flux[mean_length // 2::mean_length]
    tr_p_1_md = tr_p_1[mean_length // 2::mean_length]
    transpiration_1_md = transpiration_1[mean_length // 2::mean_length]
    transpiration_2_md = transpiration_2[mean_length // 2::mean_length]
    transpiration_3_md = transpiration_3[mean_length // 2::mean_length]
    ET_phm_obs_md = ET_phm_obs[mean_length // 2::mean_length]
    ET_phm_obs_percent_md = ET_phm_obs_percent[mean_length // 2::mean_length]
    ET_beta_obs_md = ET_beta_obs[mean_length // 2::mean_length]
    ET_beta_obs_percent_md = ET_beta_obs_percent[mean_length // 2::mean_length]
    ET_ns_obs_md = ET_ns_obs[mean_length // 2::mean_length]
    ET_ns_obs_percent_md = ET_ns_obs_percent[mean_length // 2::mean_length]
    tr_beta_phm_md = tr_beta_phm[mean_length // 2::mean_length]
    tr_beta_phm_percent_md = tr_beta_phm_percent[mean_length // 2::mean_length]
    swc_md = swc[mean_length // 2::mean_length]

    # prepare colormesh plot
    swc_bins = np.arange(0, 0.6, 0.1)
    tww_bins = np.arange(0, 1200, 200)

    ET_obs_md_bin = np.full((tww_bins.shape[0] - 1, swc_bins.shape[0] - 1), -9999.1)
    transpiration_1_md_bin = np.full((tww_bins.shape[0] - 1, swc_bins.shape[0] - 1), -9999.1)
    transpiration_2_md_bin = np.full((tww_bins.shape[0] - 1, swc_bins.shape[0] - 1), -9999.1)
    transpiration_3_md_bin = np.full((tww_bins.shape[0] - 1, swc_bins.shape[0] - 1), -9999.1)
    ET_phm_obs_md_bin = np.full((tww_bins.shape[0] - 1, swc_bins.shape[0] - 1), -9999.1)
    ET_phm_obs_percent_md_bin = np.full((tww_bins.shape[0] - 1, swc_bins.shape[0] - 1), -9999.1)
    ET_phm_obs_md_sum_bin = np.full((tww_bins.shape[0] - 1, swc_bins.shape[0] - 1), -9999.1)
    ET_phm_obs_percent_md_sum_bin = np.full((tww_bins.shape[0] - 1, swc_bins.shape[0] - 1), -9999.1)
    ET_beta_obs_md_bin = np.full((tww_bins.shape[0] - 1, swc_bins.shape[0] - 1), -9999.1)
    ET_beta_obs_percent_md_bin = np.full((tww_bins.shape[0] - 1, swc_bins.shape[0] - 1), -9999.1)
    ET_beta_obs_md_sum_bin = np.full((tww_bins.shape[0] - 1, swc_bins.shape[0] - 1), -9999.1)
    ET_beta_obs_percent_md_sum_bin = np.full((tww_bins.shape[0] - 1, swc_bins.shape[0] - 1), -9999.1)
    ET_ns_obs_md_bin = np.full((tww_bins.shape[0] - 1, swc_bins.shape[0] - 1), -9999.1)
    ET_ns_obs_percent_md_bin = np.full((tww_bins.shape[0] - 1, swc_bins.shape[0] - 1), -9999.1)
    ET_ns_obs_md_sum_bin = np.full((tww_bins.shape[0] - 1, swc_bins.shape[0] - 1), -9999.1)
    ET_ns_obs_percent_md_sum_bin = np.full((tww_bins.shape[0] - 1, swc_bins.shape[0] - 1), -9999.1)
    tr_beta_phm_md_bin = np.full((tww_bins.shape[0] - 1, swc_bins.shape[0] - 1), -9999.1)
    tr_beta_phm_percent_md_bin = np.full((tww_bins.shape[0] - 1, swc_bins.shape[0] - 1), -9999.1)
    tr_beta_phm_md_sum_bin = np.full((tww_bins.shape[0] - 1, swc_bins.shape[0] - 1), -9999.1)
    tr_beta_phm_percent_md_sum_bin = np.full((tww_bins.shape[0] - 1, swc_bins.shape[0] - 1), -9999.1)
    select_size_bin = np.full((tww_bins.shape[0] - 1, swc_bins.shape[0] - 1), -9999.1)
    swc_scatter_bin = np.full((tww_bins.shape[0] - 1, swc_bins.shape[0] - 1), -9999.1)
    tww_scatter_bin = np.full((tww_bins.shape[0] - 1, swc_bins.shape[0] - 1), -9999.1)
    for i in range(tww_bins.shape[0] - 1):
        for j in range(swc_bins.shape[0] - 1):
            select = np.all([swc_md >= swc_bins[j], swc_md < swc_bins[j + 1], tr_p_1_md >= tww_bins[i], tr_p_1_md < tww_bins[i + 1]], 0)
            ET_obs_md_select = ET_obs_md[select]
            transpiration_1_md_select = transpiration_1_md[select]
            transpiration_2_md_select = transpiration_2_md[select]
            transpiration_3_md_select = transpiration_3_md[select]
            tr_beta_phm_md_select = tr_beta_phm_md[select]
            tr_beta_phm_percent_md_select = tr_beta_phm_percent_md[select]
            ET_phm_obs_md_select = ET_phm_obs_md[select]
            ET_phm_obs_percent_md_select = ET_phm_obs_percent_md[select]
            ET_beta_obs_md_select = ET_beta_obs_md[select]
            ET_beta_obs_percent_md_select = ET_beta_obs_percent_md[select]
            ET_ns_obs_md_select = ET_ns_obs_md[select]
            ET_ns_obs_percent_md_select = ET_ns_obs_percent_md[select]

            swc_scatter_bin[i, j] = (swc_bins[j] + swc_bins[j + 1]) / 2
            tww_scatter_bin[i, j] = (tww_bins[i] + tww_bins[i + 1]) / 2

            if np.sum(select) == 0:
                ET_obs_md_bin[i, j] = np.nan
                transpiration_1_md_bin[i, j] = np.nan
                transpiration_2_md_bin[i, j] = np.nan
                transpiration_3_md_bin[i, j] = np.nan
                ET_phm_obs_md_bin[i, j] = np.nan
                ET_phm_obs_percent_md_bin[i, j] = np.nan
                ET_phm_obs_md_sum_bin[i, j] = np.nan
                ET_phm_obs_percent_md_sum_bin[i, j] = np.nan
                ET_beta_obs_md_bin[i, j] = np.nan
                ET_beta_obs_percent_md_bin[i, j] = np.nan
                ET_beta_obs_md_sum_bin[i, j] = np.nan
                ET_beta_obs_percent_md_sum_bin[i, j] = np.nan
                ET_ns_obs_md_bin[i, j] = np.nan
                ET_ns_obs_percent_md_bin[i, j] = np.nan
                ET_ns_obs_md_sum_bin[i, j] = np.nan
                ET_ns_obs_percent_md_sum_bin[i, j] = np.nan
                tr_beta_phm_md_bin[i, j] = np.nan
                tr_beta_phm_percent_md_bin[i, j] = np.nan
                tr_beta_phm_md_sum_bin[i, j] = np.nan
                tr_beta_phm_percent_md_sum_bin[i, j] = np.nan
                select_size_bin[i, j] = 0
            else:
                ET_obs_md_bin[i, j] = np.mean(ET_obs_md_select)
                transpiration_1_md_bin[i, j] = np.mean(transpiration_1_md_select)
                transpiration_2_md_bin[i, j] = np.mean(transpiration_2_md_select)
                transpiration_3_md_bin[i, j] = np.mean(transpiration_3_md_select)
                ET_phm_obs_md_bin[i, j] = np.mean(ET_phm_obs_md_select)
                ET_phm_obs_percent_md_bin[i, j] = np.mean(ET_phm_obs_percent_md_select)
                ET_phm_obs_md_sum_bin[i, j] = np.sum(np.abs(ET_phm_obs_md_select))
                ET_phm_obs_percent_md_sum_bin[i, j] = np.sum(np.abs(ET_phm_obs_percent_md_select))
                ET_beta_obs_md_bin[i, j] = np.mean(ET_beta_obs_md_select)
                ET_beta_obs_percent_md_bin[i, j] = np.mean(ET_beta_obs_percent_md_select)
                ET_beta_obs_md_sum_bin[i, j] = np.sum(np.abs(ET_beta_obs_md_select))
                ET_beta_obs_percent_md_sum_bin[i, j] = np.sum(np.abs(ET_beta_obs_percent_md_select))
                ET_ns_obs_md_bin[i, j] = np.mean(ET_ns_obs_md_select)
                ET_ns_obs_percent_md_bin[i, j] = np.mean(ET_ns_obs_percent_md_select)
                ET_ns_obs_md_sum_bin[i, j] = np.sum(np.abs(ET_ns_obs_md_select))
                ET_ns_obs_percent_md_sum_bin[i, j] = np.sum(np.abs(ET_ns_obs_percent_md_select))
                tr_beta_phm_md_bin[i, j] = np.mean(tr_beta_phm_md_select)
                tr_beta_phm_percent_md_bin[i, j] = np.mean(tr_beta_phm_percent_md_select)
                tr_beta_phm_md_sum_bin[i, j] = np.sum(np.abs(tr_beta_phm_md_select))
                tr_beta_phm_percent_md_sum_bin[i, j] = np.sum(np.abs(tr_beta_phm_percent_md_select))
                select_size_bin[i, j] = np.sum(select)

    # calculate mean and contribution
    ET_obs_md_mean = np.mean(ET_obs_md)
    tr_1_md_mean = np.mean(transpiration_1_md)
    tr_2_md_mean = np.mean(transpiration_2_md)
    tr_3_md_mean = np.mean(transpiration_3_md)
    ET_phm_obs_md_contribution_bin = np.array(ET_phm_obs_md_sum_bin) / ET_phm_obs_md.size
    ET_phm_obs_percent_md_contribution_bin = np.array(ET_phm_obs_percent_md_sum_bin) / ET_phm_obs_percent_md.size
    ET_phm_obs_md_mean = np.mean(np.abs(ET_phm_obs_md))
    ET_phm_obs_percent_md_mean = np.mean(np.abs(ET_phm_obs_percent_md))
    ET_beta_obs_md_contribution_bin = np.array(ET_beta_obs_md_sum_bin) / ET_beta_obs_md.size
    ET_beta_obs_percent_md_contribution_bin = np.array(ET_beta_obs_percent_md_sum_bin) / ET_beta_obs_percent_md.size
    ET_beta_obs_md_mean = np.mean(np.abs(ET_beta_obs_md))
    ET_beta_obs_percent_md_mean = np.mean(np.abs(ET_beta_obs_percent_md))
    ET_ns_obs_md_contribution_bin = np.array(ET_ns_obs_md_sum_bin) / ET_ns_obs_md.size
    ET_ns_obs_percent_md_contribution_bin = np.array(ET_ns_obs_percent_md_sum_bin) / ET_ns_obs_percent_md.size
    ET_ns_obs_md_mean = np.nanmean(np.abs(ET_ns_obs_md))
    ET_ns_obs_percent_md_mean = np.nanmean(np.abs(ET_ns_obs_percent_md))
    tr_beta_phm_md_contribution_bin = np.array(tr_beta_phm_md_sum_bin) / tr_beta_phm_md.size
    tr_beta_phm_percent_md_contribution_bin = np.array(tr_beta_phm_percent_md_sum_bin) / tr_beta_phm_percent_md.size
    tr_beta_phm_md_mean = np.mean(np.abs(tr_beta_phm_md))
    tr_beta_phm_percent_md_mean = np.mean(np.abs(tr_beta_phm_percent_md))
    print('tr_beta_phm_md.size:', tr_beta_phm_md.size)

    # generate scatter with size and color
    swc_scatter = swc_scatter_bin.flatten()
    tww_scatter = tww_scatter_bin.flatten()
    size_scatter = select_size_bin.flatten() * 15 * 100 / swc_md.size
    ET_obs_md_color = ET_obs_md_bin.flatten()
    # tr_1_md_color = transpiration_1_md_bin.flatten()
    tr_2_md_color = transpiration_2_md_bin.flatten()
    tr_3_md_color = transpiration_3_md_bin.flatten()
    ET_phm_obs_md_color = ET_phm_obs_md_bin.flatten()
    ET_phm_obs_percent_md_color = ET_phm_obs_percent_md_bin.flatten()
    ET_beta_obs_md_color = ET_beta_obs_md_bin.flatten()
    ET_beta_obs_percent_md_color = ET_beta_obs_percent_md_bin.flatten()
    ET_ns_obs_md_color = ET_ns_obs_md_bin.flatten()
    ET_ns_obs_percent_md_color = ET_ns_obs_percent_md_bin.flatten()
    tr_beta_phm_md_color = tr_beta_phm_md_bin.flatten()
    tr_beta_phm_percent_md_color = tr_beta_phm_percent_md_bin.flatten()

    # plot
    print('plotting...')
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    cmap1 = 'viridis'
    cmap2 = 'coolwarm'
    cmap3 = 'RdPu'
    cmap4 = 'YlGnBu'
    et_range = [-5, 1000]
    et_diff_range = [-200, 200]
    et_diff_relative_range = [-0.6, 0.6]
    absolute_contribution_range = [0, 10]
    relative_contribution_range = [0, 0.05]
    tww_lim = [0, 1000]
    swc_lim = [0, 0.5]

    fig1, ax1 = plt.subplots(4, 4, figsize=(20, 14))
    plot_e_obs_size(fig1, ax1, plot_target_name, plot_target_suffix, swc_bins, tww_bins, swc_md, tr_p_1_md, swc_scatter, tww_scatter,
                    size_scatter,
                    cmap1, cmap2, cmap3, cmap4,
                    et_range, et_diff_range, et_diff_relative_range, absolute_contribution_range, relative_contribution_range,
                    ET_obs_md_color, tr_3_md_color, tr_2_md_color,
                    ET_obs_md_mean, tr_3_md_mean, tr_2_md_mean,
                    ET_phm_obs_md_color, ET_phm_obs_percent_md_color, ET_phm_obs_md_contribution_bin,
                    ET_phm_obs_percent_md_contribution_bin,
                    ET_phm_obs_md_mean, ET_phm_obs_percent_md_mean,
                    ET_beta_obs_md_color, ET_beta_obs_percent_md_color, ET_beta_obs_md_contribution_bin,
                    ET_beta_obs_percent_md_contribution_bin,
                    ET_beta_obs_md_mean, ET_beta_obs_percent_md_mean,
                    ET_ns_obs_md_color, ET_ns_obs_percent_md_color, ET_ns_obs_md_contribution_bin,
                    ET_ns_obs_percent_md_contribution_bin,
                    ET_ns_obs_md_mean, ET_ns_obs_percent_md_mean)

    fig2, ax2 = plt.subplots(4, 4, figsize=(20, 14))
    plot_error_size(fig2, ax2, plot_target_name, plot_target_suffix, swc_bins, tww_bins, swc_md, tr_p_1_md, swc_scatter, tww_scatter,
                    size_scatter,
                    cmap1, cmap2, cmap3, cmap4,
                    et_range, et_diff_range, et_diff_relative_range, absolute_contribution_range, relative_contribution_range,
                    ET_obs_md_color, tr_3_md_color, tr_2_md_color,
                    ET_obs_md_mean, tr_3_md_mean, tr_2_md_mean,
                    ET_phm_obs_md_color, ET_phm_obs_percent_md_color, ET_phm_obs_md_contribution_bin,
                    ET_phm_obs_percent_md_contribution_bin,
                    ET_phm_obs_md_mean, ET_phm_obs_percent_md_mean,
                    ET_beta_obs_md_color, ET_beta_obs_percent_md_color, ET_beta_obs_md_contribution_bin,
                    ET_beta_obs_percent_md_contribution_bin,
                    ET_beta_obs_md_mean, ET_beta_obs_percent_md_mean,
                    tr_beta_phm_md_color, tr_beta_phm_percent_md_color, tr_beta_phm_md_contribution_bin,
                    tr_beta_phm_percent_md_contribution_bin,
                    tr_beta_phm_md_mean, tr_beta_phm_percent_md_mean)

    if save_fig:
        fig1.savefig(fig_path + fig_name1)
        fig2.savefig(fig_path + fig_name2)

    plt.show()

    return


def plot_error_size(fig, ax, target_name, suffix, swc_bins, tww_bins, swc_md, tr_p_md, swc_scatter, tww_scatter, size_scatter,
                    cmap1, cmap2, cmap3, cmap4,
                    et_range, et_diff_range, et_diff_relative_range, absolute_contribution_range, relative_contribution_range,
                    et_obs_md_color, tr_ref_md_color, tr_target_md_color,
                    et_obs_md_mean, tr_ref_md_mean, tr_target_md_mean,
                    et_ref_obs_md_color, et_ref_obs_percent_md_color, et_ref_obs_contribution_bin, et_ref_obs_contribution_percent_bin,
                    et_ref_obs_md_mean, et_ref_obs_percent_md_mean,
                    et_target_obs_md_color, et_target_obs_percent_md_color, et_target_obs_contribution_bin,
                    et_target_obs_contribution_percent_bin,
                    et_target_obs_md_mean, et_target_obs_percent_md_mean,
                    tr_target_ref_md_color, tr_target_ref_percent_md_color, tr_target_ref_contribution_bin,
                    tr_target_ref_contribution_percent_bin,
                    tr_target_ref_md_mean, tr_target_ref_percent_md_mean):
    h = ax[0, 0].hist2d(swc_md, tr_p_md, [swc_bins, tww_bins], density=1, cmap=cmap4, cmin=1e-4, vmin=0, vmax=0.08)
    cbar = plt.colorbar(h[3], ax=ax[0, 0])
    cbar.ax.set_ylabel('Density')
    ax[0, 0].set_xlabel('SWC')
    ax[0, 0].set_ylabel('$T_{NHL}$ W/m2')
    ax[0, 0].set_title('SWC $T_{NHL}$ distribution')

    sc = ax[0, 1].scatter(swc_scatter, tww_scatter, s=size_scatter, c=et_obs_md_color, marker='s', cmap=cmap1,
                          vmin=et_range[0], vmax=et_range[1])
    ax[0, 1].set_xlim(0, 0.5)
    ax[0, 1].set_ylim(0, 1000)
    cbar = plt.colorbar(sc, ax=ax[0, 1], extend='both')
    cbar.ax.set_ylabel('ET W/m2')
    ax[0, 1].set_xlabel('SWC')
    ax[0, 1].set_ylabel('$T_{NHL}$ W/m2')
    ax[0, 1].set_title('ET obs (Mean={:.2f})'.format(et_obs_md_mean))

    sc = ax[0, 2].scatter(swc_scatter, tww_scatter, s=size_scatter, c=tr_ref_md_color, marker='s', cmap=cmap1,
                          vmin=et_range[0], vmax=et_range[1])
    ax[0, 2].set_xlim(0, 0.5)
    ax[0, 2].set_ylim(0, 1000)
    cbar = plt.colorbar(sc, ax=ax[0, 2], extend='both')
    cbar.ax.set_ylabel('T W/m2')
    ax[0, 2].set_xlabel('SWC')
    ax[0, 2].set_ylabel('$T_{NHL}$ W/m2')
    ax[0, 2].set_title('T PHM (Mean={:.2f})'.format(tr_ref_md_mean))

    sc = ax[0, 3].scatter(swc_scatter, tww_scatter, s=size_scatter, c=tr_target_md_color, marker='s', cmap=cmap1,
                          vmin=et_range[0], vmax=et_range[1])
    ax[0, 3].set_xlim(0, 0.5)
    ax[0, 3].set_ylim(0, 1000)
    cbar = plt.colorbar(sc, ax=ax[0, 3], extend='both')
    cbar.ax.set_ylabel('T W/m2')
    ax[0, 3].set_xlabel('SWC')
    ax[0, 3].set_ylabel('$T_{NHL}$ W/m2')
    ax[0, 3].set_title('T {}{} (Mean={:.2f})'.format(target_name, suffix, tr_target_md_mean))

    sc = ax[1, 0].scatter(swc_scatter, tww_scatter, s=size_scatter, c=et_ref_obs_md_color, marker='s', cmap=cmap2,
                          vmin=et_diff_range[0], vmax=et_diff_range[1])
    ax[1, 0].set_xlim(0, 0.5)
    ax[1, 0].set_ylim(0, 1000)
    cbar = plt.colorbar(sc, ax=ax[1, 0], extend='both')
    cbar.ax.set_ylabel('ET W/m2')
    ax[1, 0].set_xlabel('SWC')
    ax[1, 0].set_ylabel('$T_{NHL}$ W/m2')
    ax[1, 0].set_title('ET PHM-Obs (MAE={:.2f})'.format(et_ref_obs_md_mean))

    sc = ax[1, 1].scatter(swc_scatter, tww_scatter, s=size_scatter, c=et_ref_obs_percent_md_color, marker='s', cmap=cmap2,
                          vmin=et_diff_relative_range[0], vmax=et_diff_relative_range[1])
    ax[1, 1].set_xlim(0, 0.5)
    ax[1, 1].set_ylim(0, 1000)
    cbar = plt.colorbar(sc, ax=ax[1, 1], extend='both')
    cbar.ax.set_ylabel('-')
    ax[1, 1].set_xlabel('SWC')
    ax[1, 1].set_ylabel('$T_{NHL}$ W/m2')
    ax[1, 1].set_title('ET PHM-Obs relative (MAE={:.2f})'.format(et_ref_obs_percent_md_mean))

    sc = ax[1, 2].pcolormesh(swc_bins, tww_bins, et_ref_obs_contribution_bin, cmap=cmap3, vmin=absolute_contribution_range[0],
                             vmax=absolute_contribution_range[1])
    cbar = plt.colorbar(sc, ax=ax[1, 2], extend='max')
    cbar.ax.set_ylabel('error contribution')
    ax[1, 2].set_xlabel('SWC')
    ax[1, 2].set_ylabel('$T_{NHL}$ W/m2')
    ax[1, 2].set_title('ET PHM-Obs error contribution')

    sc = ax[1, 3].pcolormesh(swc_bins, tww_bins, et_ref_obs_contribution_percent_bin, cmap=cmap3, vmin=relative_contribution_range[0],
                             vmax=relative_contribution_range[1])
    cbar = plt.colorbar(sc, ax=ax[1, 3], extend='max')
    cbar.ax.set_ylabel('relative error contribution')
    ax[1, 3].set_xlabel('SWC')
    ax[1, 3].set_ylabel('$T_{NHL}$ W/m2')
    ax[1, 3].set_title('ET PHM-Obs relative error contribution')

    sc = ax[2, 0].scatter(swc_scatter, tww_scatter, s=size_scatter, c=et_target_obs_md_color, marker='s', cmap=cmap2,
                          vmin=et_diff_range[0], vmax=et_diff_range[1])
    ax[2, 0].set_xlim(0, 0.5)
    ax[2, 0].set_ylim(0, 1000)
    cbar = plt.colorbar(sc, ax=ax[2, 0], extend='both')
    cbar.ax.set_ylabel('ET W/m2')
    ax[2, 0].set_xlabel('SWC')
    ax[2, 0].set_ylabel('$T_{NHL}$ W/m2')
    ax[2, 0].set_title('ET {}{}-Obs (MAE={:.2f})'.format(target_name, suffix, et_target_obs_md_mean))

    sc = ax[2, 1].scatter(swc_scatter, tww_scatter, s=size_scatter, c=et_target_obs_percent_md_color, marker='s', cmap=cmap2,
                          vmin=et_diff_relative_range[0], vmax=et_diff_relative_range[1])
    ax[2, 1].set_xlim(0, 0.5)
    ax[2, 1].set_ylim(0, 1000)
    cbar = plt.colorbar(sc, ax=ax[2, 1], extend='both')
    cbar.ax.set_ylabel('-')
    ax[2, 1].set_xlabel('SWC')
    ax[2, 1].set_ylabel('$T_{NHL}$ W/m2')
    ax[2, 1].set_title('ET {}{}-Obs relative (MAE={:.2f})'.format(target_name, suffix, et_target_obs_percent_md_mean))

    sc = ax[2, 2].pcolormesh(swc_bins, tww_bins, et_target_obs_contribution_bin, cmap=cmap3, vmin=absolute_contribution_range[0],
                             vmax=absolute_contribution_range[1])
    cbar = plt.colorbar(sc, ax=ax[2, 2], extend='max')
    cbar.ax.set_ylabel('error contribution')
    ax[2, 2].set_xlabel('SWC')
    ax[2, 2].set_ylabel('$T_{NHL}$ W/m2')
    ax[2, 2].set_title('ET {}{}-Obs error contribution'.format(target_name, suffix))

    sc = ax[2, 3].pcolormesh(swc_bins, tww_bins, et_target_obs_contribution_percent_bin, cmap=cmap3, vmin=relative_contribution_range[0],
                             vmax=relative_contribution_range[1])
    cbar = plt.colorbar(sc, ax=ax[2, 3], extend='max')
    cbar.ax.set_ylabel('relative error contribution')
    ax[2, 3].set_xlabel('SWC')
    ax[2, 3].set_ylabel('$T_{NHL}$ W/m2')
    ax[2, 3].set_title('ET {}{}-Obs relative error contribution'.format(target_name, suffix))

    sc = ax[3, 0].scatter(swc_scatter, tww_scatter, s=size_scatter, c=tr_target_ref_md_color, marker='s', cmap=cmap2,
                          vmin=et_diff_range[0], vmax=et_diff_range[1])
    ax[3, 0].set_xlim(0, 0.5)
    ax[3, 0].set_ylim(0, 1000)
    cbar = plt.colorbar(sc, ax=ax[3, 0], extend='both')
    cbar.ax.set_ylabel('T W/m2')
    ax[3, 0].set_xlabel('SWC')
    ax[3, 0].set_ylabel('$T_{NHL}$ W/m2')
    ax[3, 0].set_title('T {}{}-PHM (MAE={:.2f})'.format(target_name, suffix, tr_target_ref_md_mean))

    sc = ax[3, 1].scatter(swc_scatter, tww_scatter, s=size_scatter, c=tr_target_ref_percent_md_color, marker='s', cmap=cmap2,
                          vmin=et_diff_relative_range[0], vmax=et_diff_relative_range[1])
    ax[3, 1].set_xlim(0, 0.5)
    ax[3, 1].set_ylim(0, 1000)
    cbar = plt.colorbar(sc, ax=ax[3, 1], extend='both')
    cbar.ax.set_ylabel('-')
    ax[3, 1].set_xlabel('SWC')
    ax[3, 1].set_ylabel('$T_{NHL}$ W/m2')
    ax[3, 1].set_title('T {}{}-PHM relative (MAE={:.2f})'.format(target_name, suffix, tr_target_ref_percent_md_mean))

    sc = ax[3, 2].pcolormesh(swc_bins, tww_bins, tr_target_ref_contribution_bin, cmap=cmap3, vmin=absolute_contribution_range[0],
                             vmax=absolute_contribution_range[1])
    cbar = plt.colorbar(sc, ax=ax[3, 2], extend='max')
    cbar.ax.set_ylabel('error contribution')
    ax[3, 2].set_xlabel('SWC')
    ax[3, 2].set_ylabel('$T_{NHL}$ W/m2')
    ax[3, 2].set_title('T {}{}-PHM error contribution'.format(target_name, suffix))

    sc = ax[3, 3].pcolormesh(swc_bins, tww_bins, tr_target_ref_contribution_percent_bin, cmap=cmap3, vmin=relative_contribution_range[0],
                             vmax=relative_contribution_range[1])
    cbar = plt.colorbar(sc, ax=ax[3, 3], extend='max')
    cbar.ax.set_ylabel('relative error contribution')
    ax[3, 3].set_xlabel('SWC')
    ax[3, 3].set_ylabel('$T_{NHL}$ W/m2')
    ax[3, 3].set_title('T {}{}-PHM relative error contribution'.format(target_name, suffix))

    fig.tight_layout()

    return


def plot_e_obs_size(fig, ax, target_name, suffix, swc_bins, tww_bins, swc_md, tr_p_md, swc_scatter, tww_scatter, size_scatter,
                    cmap1, cmap2, cmap3, cmap4,
                    et_range, et_diff_range, et_diff_relative_range, absolute_contribution_range, relative_contribution_range,
                    et_obs_md_color, tr_ref_md_color, tr_target_md_color,
                    et_obs_md_mean, tr_ref_md_mean, tr_target_md_mean,
                    et_ref_obs_md_color, et_ref_obs_percent_md_color, et_ref_obs_contribution_bin, et_ref_obs_contribution_percent_bin,
                    et_ref_obs_md_mean, et_ref_obs_percent_md_mean,
                    et_target_obs_md_color, et_target_obs_percent_md_color, et_target_obs_contribution_bin,
                    et_target_obs_contribution_percent_bin,
                    et_target_obs_md_mean, et_target_obs_percent_md_mean,
                    et_ns_obs_md_color, et_ns_obs_percent_md_color, et_ns_obs_contribution_bin,
                    et_ns_obs_contribution_percent_bin,
                    et_ns_obs_md_mean, et_ns_obs_percent_md_mean):
    h = ax[0, 0].hist2d(swc_md, tr_p_md, [swc_bins, tww_bins], density=1, cmap=cmap4, cmin=1e-4, vmin=0, vmax=0.08)
    cbar = plt.colorbar(h[3], ax=ax[0, 0])
    cbar.ax.set_ylabel('Density')
    ax[0, 0].set_xlabel('SWC')
    ax[0, 0].set_ylabel('$T_{NHL}$ W/m2')
    ax[0, 0].set_title('SWC Tww distribution')

    sc = ax[0, 1].scatter(swc_scatter, tww_scatter, s=size_scatter, c=et_obs_md_color, marker='s', cmap=cmap1,
                          vmin=et_range[0], vmax=et_range[1])
    ax[0, 1].set_xlim(0, 0.5)
    ax[0, 1].set_ylim(0, 1000)
    cbar = plt.colorbar(sc, ax=ax[0, 1], extend='both')
    cbar.ax.set_ylabel('ET W/m2')
    ax[0, 1].set_xlabel('SWC')
    ax[0, 1].set_ylabel('$T_{NHL}$ W/m2')
    ax[0, 1].set_title('ET Obs (Mean={:.2f})'.format(et_obs_md_mean))

    sc = ax[0, 2].scatter(swc_scatter, tww_scatter, s=size_scatter, c=tr_ref_md_color, marker='s', cmap=cmap1,
                          vmin=et_range[0], vmax=et_range[1])
    ax[0, 2].set_xlim(0, 0.5)
    ax[0, 2].set_ylim(0, 1000)
    cbar = plt.colorbar(sc, ax=ax[0, 2], extend='both')
    cbar.ax.set_ylabel('T W/m2')
    ax[0, 2].set_xlabel('SWC')
    ax[0, 2].set_ylabel('$T_{NHL}$ W/m2')
    ax[0, 2].set_title('T PHM (Mean={:.2f})'.format(tr_ref_md_mean))

    sc = ax[0, 3].scatter(swc_scatter, tww_scatter, s=size_scatter, c=tr_target_md_color, marker='s', cmap=cmap1,
                          vmin=et_range[0], vmax=et_range[1])
    ax[0, 3].set_xlim(0, 0.5)
    ax[0, 3].set_ylim(0, 1000)
    cbar = plt.colorbar(sc, ax=ax[0, 3], extend='both')
    cbar.ax.set_ylabel('T W/m2')
    ax[0, 3].set_xlabel('SWC')
    ax[0, 3].set_ylabel('$T_{NHL}$ W/m2')
    ax[0, 3].set_title('T {}{} (Mean={:.2f})'.format(target_name, suffix, tr_target_md_mean))

    sc = ax[1, 0].scatter(swc_scatter, tww_scatter, s=size_scatter, c=et_ref_obs_md_color, marker='s', cmap=cmap2,
                          vmin=et_diff_range[0], vmax=et_diff_range[1])
    ax[1, 0].set_xlim(0, 0.5)
    ax[1, 0].set_ylim(0, 1000)
    cbar = plt.colorbar(sc, ax=ax[1, 0], extend='both')
    cbar.ax.set_ylabel('ET W/m2')
    ax[1, 0].set_xlabel('SWC')
    ax[1, 0].set_ylabel('$T_{NHL}$ W/m2')
    ax[1, 0].set_title('ET PHM-Obs (MAE={:.2f})'.format(et_ref_obs_md_mean))

    sc = ax[1, 1].scatter(swc_scatter, tww_scatter, s=size_scatter, c=et_ref_obs_percent_md_color, marker='s', cmap=cmap2,
                          vmin=et_diff_relative_range[0], vmax=et_diff_relative_range[1])
    ax[1, 1].set_xlim(0, 0.5)
    ax[1, 1].set_ylim(0, 1000)
    cbar = plt.colorbar(sc, ax=ax[1, 1], extend='both')
    cbar.ax.set_ylabel('-')
    ax[1, 1].set_xlabel('SWC')
    ax[1, 1].set_ylabel('$T_{NHL}$ W/m2')
    ax[1, 1].set_title('ET PHM-Obs relative (MAE={:.2f})'.format(et_ref_obs_percent_md_mean))

    sc = ax[1, 2].pcolormesh(swc_bins, tww_bins, et_ref_obs_contribution_bin, cmap=cmap3, vmin=absolute_contribution_range[0],
                             vmax=absolute_contribution_range[1])
    cbar = plt.colorbar(sc, ax=ax[1, 2], extend='max')
    cbar.ax.set_ylabel('error contribution')
    ax[1, 2].set_xlabel('SWC')
    ax[1, 2].set_ylabel('$T_{NHL}$ W/m2')
    ax[1, 2].set_title('ET PHM-Obs error contribution')

    sc = ax[1, 3].pcolormesh(swc_bins, tww_bins, et_ref_obs_contribution_percent_bin, cmap=cmap3, vmin=relative_contribution_range[0],
                             vmax=relative_contribution_range[1])
    cbar = plt.colorbar(sc, ax=ax[1, 3], extend='max')
    cbar.ax.set_ylabel('relative error contribution')
    ax[1, 3].set_xlabel('SWC')
    ax[1, 3].set_ylabel('$T_{NHL}$ W/m2')
    ax[1, 3].set_title('ET PHM-Obs relative error contribution')

    sc = ax[2, 0].scatter(swc_scatter, tww_scatter, s=size_scatter, c=et_target_obs_md_color, marker='s', cmap=cmap2,
                          vmin=et_diff_range[0], vmax=et_diff_range[1])
    ax[2, 0].set_xlim(0, 0.5)
    ax[2, 0].set_ylim(0, 1000)
    cbar = plt.colorbar(sc, ax=ax[2, 0], extend='both')
    cbar.ax.set_ylabel('ET W/m2')
    ax[2, 0].set_xlabel('SWC')
    ax[2, 0].set_ylabel('$T_{NHL}$ W/m2')
    ax[2, 0].set_title('ET {}{}-Obs (MAE={:.2f})'.format(target_name, suffix, et_target_obs_md_mean))

    sc = ax[2, 1].scatter(swc_scatter, tww_scatter, s=size_scatter, c=et_target_obs_percent_md_color, marker='s', cmap=cmap2,
                          vmin=et_diff_relative_range[0], vmax=et_diff_relative_range[1])
    ax[2, 1].set_xlim(0, 0.5)
    ax[2, 1].set_ylim(0, 1000)
    cbar = plt.colorbar(sc, ax=ax[2, 1], extend='both')
    cbar.ax.set_ylabel('-')
    ax[2, 1].set_xlabel('SWC')
    ax[2, 1].set_ylabel('$T_{NHL}$ W/m2')
    ax[2, 1].set_title('ET {}{}-Obs relative (MAE={:.2f})'.format(target_name, suffix, et_target_obs_percent_md_mean))

    sc = ax[2, 2].pcolormesh(swc_bins, tww_bins, et_target_obs_contribution_bin, cmap=cmap3, vmin=absolute_contribution_range[0],
                             vmax=absolute_contribution_range[1])
    cbar = plt.colorbar(sc, ax=ax[2, 2], extend='max')
    cbar.ax.set_ylabel('error contribution')
    ax[2, 2].set_xlabel('SWC')
    ax[2, 2].set_ylabel('$T_{NHL}$ W/m2')
    ax[2, 2].set_title('ET {}{}-Obs error contribution'.format(target_name, suffix))

    sc = ax[2, 3].pcolormesh(swc_bins, tww_bins, et_target_obs_contribution_percent_bin, cmap=cmap3, vmin=relative_contribution_range[0],
                             vmax=relative_contribution_range[1])
    cbar = plt.colorbar(sc, ax=ax[2, 3], extend='max')
    cbar.ax.set_ylabel('relative error contribution')
    ax[2, 3].set_xlabel('SWC')
    ax[2, 3].set_ylabel('$T_{NHL}$ W/m2')
    ax[2, 3].set_title('ET {}{}-Obs relative error contribution'.format(target_name, suffix))

    sc = ax[3, 0].scatter(swc_scatter, tww_scatter, s=size_scatter, c=et_ns_obs_md_color, marker='s', cmap=cmap2,
                          vmin=et_diff_range[0], vmax=et_diff_range[1])
    ax[3, 0].set_xlim(0, 0.5)
    ax[3, 0].set_ylim(0, 1000)
    cbar = plt.colorbar(sc, ax=ax[3, 0], extend='both')
    cbar.ax.set_ylabel('ET W/m2')
    ax[3, 0].set_xlabel('SWC')
    ax[3, 0].set_ylabel('$T_{NHL}$ W/m2')
    ax[3, 0].set_title('ET NHL-Obs (MAE={:.2f})'.format(et_ns_obs_md_mean))

    sc = ax[3, 1].scatter(swc_scatter, tww_scatter, s=size_scatter, c=et_ns_obs_percent_md_color, marker='s', cmap=cmap2,
                          vmin=et_diff_relative_range[0], vmax=et_diff_relative_range[1])
    ax[3, 1].set_xlim(0, 0.5)
    ax[3, 1].set_ylim(0, 1000)
    cbar = plt.colorbar(sc, ax=ax[3, 1], extend='both')
    cbar.ax.set_ylabel('-')
    ax[3, 1].set_xlabel('SWC')
    ax[3, 1].set_ylabel('$T_{NHL}$ W/m2')
    ax[3, 1].set_title('ET NHL-Obs relative (MAE={:.2f})'.format(et_ns_obs_percent_md_mean))

    sc = ax[3, 2].pcolormesh(swc_bins, tww_bins, et_ns_obs_contribution_bin, cmap=cmap3, vmin=absolute_contribution_range[0],
                             vmax=absolute_contribution_range[1])
    cbar = plt.colorbar(sc, ax=ax[3, 2], extend='max')
    cbar.ax.set_ylabel('error contribution')
    ax[3, 2].set_xlabel('SWC')
    ax[3, 2].set_ylabel('$T_{NHL}$ W/m2')
    ax[3, 2].set_title('ET NHL-Obs error contribution')

    sc = ax[3, 3].pcolormesh(swc_bins, tww_bins, et_ns_obs_contribution_percent_bin, cmap=cmap3, vmin=relative_contribution_range[0],
                             vmax=relative_contribution_range[1])
    cbar = plt.colorbar(sc, ax=ax[3, 3], extend='max')
    cbar.ax.set_ylabel('relative error contribution')
    ax[3, 3].set_xlabel('SWC')
    ax[3, 3].set_ylabel('$T_{NHL}$ W/m2')
    ax[3, 3].set_title('ET NHL-Obs relative error contribution')

    fig.tight_layout()

    return


if __name__ == '__main__':
    main()
