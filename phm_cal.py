import os
import sys
import configparser
from datetime import datetime
import numpy as np
import pandas as pd
import spotpy
from mpi4py import MPI

sys.path.append('/u/sciteam/yang7/watercon/script')
# sys.path.append('/Users/yangyicge/Desktop/watercon/script')
from noah_energy.phm.calibration_run_phm import read_forcing
from noah_energy.phm.calibration_run_phm import run_model_time_series


def main():
    # MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()
    print('size:', size, 'rank:', rank, 'name:', name)

    parm_path = 'holder'
    if rank == 0:
        # paths for local machines and server
        # path_pre = '/Users/yangyicge/Desktop/'
        path_pre = '/u/sciteam/yang7/'
        flux_path = path_pre + 'watercon/flux/'
        dbpath_pre = path_pre + 'watercon/noah_energy_case/calibration/'
        parm_path = path_pre + 'watercon/script/noah_energy/calibration/'

        # site and time info
        print('\n------------------------\nsite info and data preparation...')
        site = 'US-Me2'
        LAT = 44.4523
        LON = -121.5574
        time_res = 1800
        utc_offset = -8
        start = datetime(2013, 5, 1)
        end = datetime(2013, 5, 4)
        # dbpath = dbpath_pre + '{}_server_test/'.format(site)
        dbpath = os.getcwd() + '/'
        dbpath_log = dbpath + 'log/'

        # read flux data for forcing
        flux_path_full = flux_path + \
                         'fluxnet/FLX_US-Me2_FLUXNET2015_SUBSET_2002-2014_1-4/FLX_US-Me2_FLUXNET2015_SUBSET_HH_2002-2014_1-4.csv'
        amx_path_full = flux_path + 'ameriflux/AMF_US-Me2_BASE-BADM_15-5/AMF_US-Me2_BASE_HH_15-5.csv'
        flux_forcing = read_forcing(start, end, flux_path_full, amx_path_full)

        # read flux data for observation
        obs_path = flux_path_full
        df = pd.read_csv(obs_path)
        ET_flux = np.ma.masked_values(df[(df.TIMESTAMP_START >= int(
            '{:4d}{:02d}{:02d}0000'.format(start.year, start.month, start.day))) & (df.TIMESTAMP_START < int(
            '{:4d}{:02d}{:02d}0000'.format(end.year, end.month, end.day)))].loc[:, 'LE_F_MDS'].values, -9999)

        # set up spotpy class for calibration
        config_phm = configparser.ConfigParser()
        config_phm.read(parm_path + '3model_fixed_parm.ini')
        spotpy_setup_data = [config_phm, flux_forcing, ET_flux, LAT, LON, utc_offset, time_res, dbpath, dbpath_log, start]
        if not os.path.exists(dbpath_log):
            os.makedirs(dbpath_log)
        print('\n------------------------\nbegin calibration...')
    else:
        spotpy_setup_data = None

    # run calibration
    spotpy_setup_data = comm.bcast(spotpy_setup_data, root=0)
    [config_phm, flux_forcing, ET_flux, LAT, LON, utc_offset, time_res, dbpath, dbpath_log, start] = spotpy_setup_data
    phm_cal = spotpy_setup(config_phm, flux_forcing, ET_flux, LAT, LON, utc_offset, time_res, dbpath_log, rank)
    sampler = spotpy.algorithms.sceua(phm_cal, parallel='mpi', dbname=dbpath + 'phm_cal_sceua_{}'.format(start.year), dbformat='csv',
                                      save_sim=False)
    sampler.sample(10)
    print('rank:', rank, 'calibration done!')

    if rank == 0:
        # get the optimal case
        print('\n------------------------\ncalibration done, running and saving optimal case...')
        loglist = [f for f in os.listdir(dbpath_log) if f.startswith('rmse')]
        record_list = np.concatenate([np.load(dbpath_log + r) for r in loglist])
        rmse_list = record_list[:, 0]
        parms_list = record_list[:, 1:]
        print('rmse_list:', rmse_list)
        print('parms_list:', parms_list)
        opt_number = np.argmin(rmse_list)
        opt_phm_parms = parms_list[opt_number, :]
        print('optimal phm parameters:', opt_phm_parms)

        # run optimal case again and save
        config_opt = configparser.ConfigParser()
        config_opt.read(parm_path + '3model_fixed_parm.ini')
        config_opt = config_parameters(config_opt, opt_phm_parms)

        model_results, swc_results = np.array(run_model_time_series(config_opt, flux_forcing, LAT, LON, utc_offset, time_res))
        np.save(dbpath + 'sceua_opt_phm_parms_{}.npy'.format(start.year), opt_phm_parms)
        np.save(dbpath + 'sceua_opt_phm_results_{}.npy'.format(start.year), model_results)
        np.save(dbpath + 'sceua_opt_phm_results_swc_{}.npy'.format(start.year), swc_results)

    return


class spotpy_setup:
    def __init__(self, config, flux_forcing, ET_flux, LAT, LON, utc_offset, time_res, dbpath_log, rank):
        # parameter distribution
        dk_exp, db, dp, dvmx, dmp = default_parameters()

        self.params = [spotpy.parameter.Uniform('BEXP', 0.8 * db, 1.35 * db, 0.175 * db, db, 0.8 * db, 1.35 * db),
                       spotpy.parameter.Uniform('DKSAT_log10', max(1.2 * dk_exp, -6.2), min(0.8 * dk_exp, -4.4),
                                                -0.2 * dk_exp, dk_exp, max(1.2 * dk_exp, -6.2), min(0.8 * dk_exp, -4.4)),
                       spotpy.parameter.Uniform('PSISAT', max(0.65 * dp, 0.2), min(1.35 * dp, 0.85), 0.2 * dp, dp,
                                                max(0.65 * dp, 0.2), min(1.35 * dp, 0.85)),
                       spotpy.parameter.Uniform('VGKSAT_e5', 0.1, 20, 2, 2, 0.1, 20),
                       spotpy.parameter.Uniform('VGSP50', -1000, -5, 100, -250, -1000, -5),
                       spotpy.parameter.Uniform('VGA2', 1, 15, 3, 4, 1, 15),
                       spotpy.parameter.Uniform('VGTLP', -1000, -5, 100, -100, -800, -5),
                       spotpy.parameter.Uniform('VGA3', 1, 15, 3, 4, 1, 15),
                       spotpy.parameter.Uniform('VCMX25', 0.8 * dvmx, 1.35 * dvmx, 0.175 * dvmx, dvmx, 0.8 * dvmx, 1.35 * dvmx),
                       spotpy.parameter.Uniform('MP', 0.8 * dmp, 1.35 * dmp, 0.175 * dmp, dmp, 0.8 * dmp, 1.35 * dmp),
                       # spotpy.parameter.Uniform('RAI_log10', -1, 2, 0.5, 0.5, -1, 2),
                       ]

        # some configurations for model run
        self.config = config
        self.flux_forcing = flux_forcing
        self.LAT = LAT
        self.LON = LON
        self.utc_offset = utc_offset
        self.time_res = time_res

        # obs and mask
        self.ET_flux_mask = ~ET_flux.mask
        self.ET_flux_masked = ET_flux[self.ET_flux_mask]

        # for recording
        self.record_list = []
        self.rank = rank
        self.dbpath_log = dbpath_log

    def parameters(self):
        return spotpy.parameter.generate(self.params)

    def simulation(self, vector):
        # config parameters
        config = config_parameters(self.config, vector)

        # run model
        model_results, swc_results = run_model_time_series(config, self.flux_forcing, self.LAT, self.LON, self.utc_offset, self.time_res)
        transpiration = model_results[1, :].astype(np.float64)
        soil_evaporation = model_results[2, :].astype(np.float64)
        canopy_evaporation = model_results[3, :].astype(np.float64)
        ET = transpiration + soil_evaporation + canopy_evaporation
        ET_masked = ET[self.ET_flux_mask]

        # log data
        rmse = spotpy.objectivefunctions.rmse(self.ET_flux_masked, ET_masked)
        self.record_list.append(np.concatenate(([rmse], vector.copy())))
        print('rank {} recording rmse {}'.format(self.rank, rmse))
        np.save(self.dbpath_log + 'rmse_{}'.format(self.rank), self.record_list)

        return ET_masked

    def evaluation(self):
        return self.ET_flux_masked

    def objectivefunction(self, simulation, evaluation):
        objectivefunction = spotpy.objectivefunctions.rmse(evaluation, simulation)
        return objectivefunction


def config_parameters(config, vector):
    # read relevant soil parms
    SMCMAX = float(config['soil']['SMCMAX'])

    # config calirbated parameters
    # soil
    BEXP = vector[0]
    DKSAT = 10 ** vector[1]
    PSISAT = vector[2]
    refsmc1 = SMCMAX * (5.79e-9 / DKSAT) ** (1 / (2 * BEXP + 3))
    SMCREF = refsmc1 + 1 / 3 * (SMCMAX - refsmc1)
    SMCWLT = 0.5 * SMCMAX * (200 / PSISAT) ** (-1 / BEXP)
    DWSAT = BEXP * DKSAT * (PSISAT / SMCMAX)

    # plant
    VGKSAT = vector[3] * 1e-5
    VGSP50 = vector[4]
    VGA2 = vector[5]
    VGTLP = vector[6]
    VGA3 = vector[7]
    VCMX25 = vector[8]
    MP = vector[9]
    # RAI = 10 ** vector[10]

    # write to config
    config['soil']['BEXP'] = str(BEXP)
    config['soil']['DKSAT'] = str(DKSAT)
    config['soil']['PSISAT'] = str(PSISAT)
    config['soil']['SMCREF'] = str(SMCREF)
    config['soil']['SMCWLT'] = str(SMCWLT)
    config['soil']['DWSAT'] = str(DWSAT)
    config['phm']['VGKSAT'] = str(VGKSAT)
    config['phm']['VGSP50'] = str(VGSP50)
    config['phm']['VGA2'] = str(VGA2)
    config['phm']['VGTLP'] = str(VGTLP)
    config['phm']['VGA3'] = str(VGA3)
    config['vege_flux']['VCMX25'] = str(VCMX25)
    config['vege_flux']['MP'] = str(MP)
    # config['phm']['RAI'] = str(RAI)

    return config


def default_parameters():
    # default parameters
    DKSAT = 3.42e-5
    BEXP = 2.38
    PSISAT = 0.141
    VCMX25 = 122
    MP = 25

    # convert approriate scales
    DKSAT_log10 = np.log(DKSAT) / np.log(10)

    return DKSAT_log10, BEXP, PSISAT, VCMX25, MP


if __name__ == '__main__':
    main()
