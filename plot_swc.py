import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from glob import glob
import os

from utility_modules import aflux_file_suffix_dict, site_year_dict, load_swc_ne3, load_swc_ne2, load_swc_bo1, load_swc_br1, load_swc_br3, \
    load_swc_ib1, load_swc_ro1


# this script inspects the SWC in a flux tower site
# sensor depth for each site: Ne2, Ne3 -0.1, -0.25, -0.5, -1.0

def main():
    # site info
    site = 'US-Ro1'
    obs_hourly = 0
    year = 2011
    start = datetime(year, 7, 1)
    end = datetime(year, 9, 15)
    save_fig = 1

    # fig save path
    try:
        case_path = glob("/Users/yangyicge/Desktop/watercon/crop_hydro_case/case_run/{}/{}_{}*/".format(site, site, start.year))[0]
        crop = case_path.split('/')[-2].split('_')[-1]
    except IndexError:
        print('case not found! Please mannualy specify crop type')
        crop = 'maize'
    fig_path = '/Users/yangyicge/Desktop/watercon/crop_hydro_case/fig/bysite/{}/{}_{}_{}/'.format(site, site, start.year, crop)

    # read SWC data
    flux_path = '/Users/yangyicge/Desktop/watercon/flux/'
    aflux_file = flux_path + 'ameriflux/' + aflux_file_suffix_dict[site]

    swc_h = swc_v = 'holder'
    if site == 'US-Ne3':
        swc_h, swc_v = load_swc_ne3(aflux_file, start, end)
    elif site == 'US-Ne2':
        swc_h, swc_v = load_swc_ne2(aflux_file, start, end)
    elif site == 'US-Bo1':
        swc_h, swc_v = load_swc_bo1(aflux_file, start, end)
    elif site == 'US-Br1':
        swc_h, swc_v = load_swc_br1(aflux_file, start, end)
    elif site == 'US-Br3':
        swc_h, swc_v = load_swc_br3(aflux_file, start, end)
    elif site == 'US-IB1':
        swc_h, swc_v = load_swc_ib1(aflux_file, start, end)
    elif site == 'US-Ro1':
        swc_h, swc_v = load_swc_ro1(aflux_file, start, end)
    h_number = swc_h.shape[0]
    v_number = swc_h.shape[1]
    print('swc shape:', swc_h.shape)

    # interpolate swc_v to fill missing values
    # swc_v = pd.DataFrame(swc_v).interpolate(axis=1).values

    # read LAI data
    force_start_year = site_year_dict[site][0]
    force_end_year = site_year_dict[site][1]
    full_model_force_path = '/Users/yangyicge/Desktop/watercon/forcing/gapfilled_crop/'
    force_start = datetime(force_start_year, 1, 1)
    lines_to_skip = 71
    slice_start = (start - force_start).days * 24 + lines_to_skip
    slice_end = (end - force_start).days * 24 + lines_to_skip
    full_model_force = open(full_model_force_path + '{}_{}-{}/force.dat'.format(site[3:], force_start_year, force_end_year), 'r')

    lai_force = []
    i = 0
    for line in full_model_force.readlines():
        if slice_start <= i < slice_end:
            lai_force.append(float(line[157:169]))
        i = i + 1

    # plot
    if not obs_hourly:
        times = pd.date_range(start, end, freq='30min')[:-1]
        lai_force = np.repeat(lai_force, 2)
    else:
        times = pd.date_range(start, end, freq='H')[:-1]
    # print('time shape:', times.shape)
    # print(times)

    swc_range = [0, 0.6]
    lai_range = [0, 8]

    # plot individual
    fig, ax = plt.subplots(max(v_number, 2), 1, figsize=(14, 3 * max(v_number, 2)))
    for depth in range(v_number):
        for point in range(h_number):
            ax[depth].plot(times, swc_h[point, depth, :], label='H{}'.format(point + 1))
            ax[depth].set_ylim(swc_range)
            ax[depth].set_ylabel('SWC')
            ax[depth].legend()
            ax[depth].set_title('Depth {}'.format(depth + 1))
    fig.autofmt_xdate()
    fig.tight_layout()

    # plot mean
    fig2, ax2 = plt.subplots(max(v_number, 2), 1, figsize=(14, 3 * max(v_number, 2)))
    for depth in range(v_number):
        ax2[depth].plot(times, swc_v[depth, :])
        ax2[depth].set_ylim(swc_range)
        ax2[depth].set_ylabel('SWC')
        ax2[depth].set_title('Depth {} mean'.format(depth + 1))
    fig2.autofmt_xdate()
    fig2.tight_layout()

    # plot LAI
    fig3, ax3 = plt.subplots(1, 1, figsize=(14, 3))
    ax3.plot(times, lai_force)
    ax3.set_ylabel('LAI')
    ax3.set_ylim(lai_range)
    fig3.tight_layout()
    fig3.autofmt_xdate()

    # plot gaps
    fig1, ax1 = plt.subplots(1, 1, figsize=(14, 3))
    ax1.plot(times, swc_h[0, 0, :].mask)
    ax1.set_ylabel('gaps')
    fig1.tight_layout()
    fig1.autofmt_xdate()

    # save fig
    if save_fig:
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        fig.savefig(fig_path + 'swc_inspect_individual.pdf')
        fig2.savefig(fig_path + 'swc_inspect_mean.pdf')
        fig3.savefig(fig_path + 'lai_inspect.pdf')
        # fig1.savefig(fig_path + 'swc_inspect_gaps.pdf')

    plt.show()


if __name__ == '__main__':
    main()
