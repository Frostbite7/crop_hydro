from datetime import datetime
from noah_energy.crop_hydro.snap_run_model import run_model
from utility_modules import coord_dict, site_year_dict, flux_file_suffix_dict, aflux_file_suffix_dict


# case info
# Sim period: 7/1 - 9/15


def main():
    # sim config
    site = 'US-IB1'
    year = 2007
    crop = 'soy'
    mode = 'phm'
    start = datetime(year, 7, 1)
    end = datetime(year, 9, 15)
    test_suffix = ''
    mode_suffix = ''
    CAP_beta = 0
    if CAP_beta == 1 and mode == 'beta':
        mode_suffix = '_capped'

    # site info
    coordinates = coord_dict[site]
    force_start_year = site_year_dict[site][0]
    force_end_year = site_year_dict[site][1]
    utc_offset = -6
    if site == 'US-Ne3' or site == 'US-Ne2' or site == 'US-Ne1':
        DT = 3600
    else:
        DT = 3600

    # file paths
    parm_path = '/Users/yangyicge/Desktop/watercon/script/noah_energy/parameters/{}/energy_parameter_{}_{}.ini'.format(site, crop, mode)
    flux_file_suffix = flux_file_suffix_dict[site]
    aflux_file_suffix = aflux_file_suffix_dict[site]
    results_path = '/Users/yangyicge/Desktop/watercon/crop_hydro_case/case_run/{}/{}_{}_{}/'.format(site, site, start.year, crop)
    results_suffix = '_' + mode + mode_suffix + test_suffix

    # run case
    run_model(site, coordinates, utc_offset, DT, parm_path, start, end, results_path, results_suffix, flux_file_suffix,
              aflux_file_suffix, force_start_year, force_end_year, mode, DYNAMIC_beta=0, CAP_beta=CAP_beta)

    return


if __name__ == '__main__':
    main()
