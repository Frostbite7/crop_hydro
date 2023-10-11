import configparser
import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import os

# this script gives the stress-only model response with scatter plot and the line plot (different lines for different Tww)

# define const.
HVAP = 2.51e6


def main():
    # set up and fig names
    site = 'US-Ne3'
    crop = 'soy'
    fig_path = '/Users/yangyicge/Desktop/watercon/crop_hydro_case/fig/stress_only/{}_{}/'.format(site, crop)
    fig_name_pre = 'stress_only_'
    save_fig = 0

    # load basic parameters and config
    parm_path = '/Users/yangyicge/Desktop/watercon/script/noah_energy/parameters/{}/energy_parameter_{}_nostress.ini'.format(site, crop)
    # parm_path = '3model_parm.ini'
    config = configparser.ConfigParser()
    config.read(parm_path)
    DT_unused, NSOIL, ZSOIL, DZSNSO = SOIL_LAYER_CONFIG(config)
    # temporary
    # DZSNSO[2] = 1.1

    # set phm parms
    SPAI = 15.8E-4
    VEGH = 1
    R2SR = 1
    RAI = 1000
    VGKSAT = 1.0e-6
    VGSP50 = -150
    VGA2 = 3
    VGA1 = 1
    VGTLP = -100
    VGA3 = 4
    DROOT = 5e-4
    fRS_str = '0.0, 1.0, 0.0, 0.0'
    fRS = np.array([float(idx) for idx in fRS_str.split(',')])

    # set beta parms
    # water uptake fraction from each layer, currently unsed
    # beta_rootu_fraction = np.sqrt(fRS) / np.array(DZSNSO) / np.sum(np.sqrt(fRS) / np.array(DZSNSO))
    beta_fraction = fRS
    beta_fraction_str = '{}, {}, {}, {}'.format(beta_fraction[0], beta_fraction[1], beta_fraction[2], beta_fraction[3])
    print('beta_fraction:', beta_fraction)
    BETA50 = -50
    BETAEXP = 2.5
    DYNAMIC_beta = 0
    krmax = 3e+4
    # BETA50_tww_ref = 2e5
    CAP_beta = 1
    kmax_adjusted = 4.18
    # krmax = DKSAT * np.sqrt(RAI) / DZSNSO / np.pi * HVAP * 1000
    # kmax = VGKSAT * SPAI / VGA1 / VEGH * HVAP * 1000
    # kmax_adjusted = kmax * ((-VGTLP) / 100) ** (1.2 + (VEGH - 9) / 45) * (1 - (VEGH - 9) / 80)

    # fig names
    suffix_1 = ''
    if DYNAMIC_beta:
        suffix_1 = suffix_1 + '_dynamic'
    if CAP_beta:
        suffix_1 = suffix_1 + '_capped'

    # config parm
    config['phm'] = {'SPAI': SPAI, 'VEGH': VEGH, 'R2SR': R2SR, 'RAI': RAI, 'VGKSAT': VGKSAT, 'VGSP50': VGSP50, 'VGA2': VGA2,
                     'VGA1': VGA1, 'VGTLP': VGTLP, 'VGA3': VGA3, 'DROOT': DROOT, 'fRS': fRS_str}
    config['beta'] = {'BETA50': BETA50, 'BETAEXP': BETAEXP, 'beta_fraction': beta_fraction_str,
                      'kmax_beta_adjusted': kmax_adjusted, 'krmax': krmax}

    # generate forcing data
    FCTR_l = np.arange(0, 810, 100)
    SMC_l = np.concatenate([np.arange(0.05, 0.07, 0.002), np.arange(0.07, 0.3, 0.005), np.arange(0.3, 0.5, 0.01)])
    # PSIS_l = np.arange(-50, 0, 0.25)
    # SMC_l = SMCMAX * (np.minimum(PSIS_l, np.full(PSIS_l.shape, -PSISAT)) / -PSISAT) ** (-1 / BEXP)

    # recording variables
    swc = []
    fctr_p = []
    fctr_phm = []
    fctr_beta = []
    psix = []
    psis = []
    wcnd = []
    kx = []
    psi_l = []
    fs = []

    # run models
    print('running stress-only model...')
    for FCTR in FCTR_l:
        for SMC_ in SMC_l:
            SMC = [SMC_, SMC_, SMC_, SMC_]
            SH2O = SMC

            # run PHM
            FCTR_d_phm, FSTOMATA, SOLPSI, VGPSIS, VGPSIL, KA, SRCD, WCND, FCTR_d_sequence = PHM(config, NSOIL, ZSOIL, DZSNSO, FCTR,
                                                                                                SMC)

            # run BETA
            FCTR_d_beta, BTRAN = BETA(config, NSOIL, ZSOIL, DZSNSO, FCTR, SH2O, DYNAMIC_beta, CAP_beta)

            # inner recording
            swc.append(SMC_)
            fctr_p.append(FCTR)
            fctr_phm.append(FCTR_d_phm)
            fctr_beta.append(FCTR_d_beta)

            psix.append(VGPSIS)
            psis.append(SOLPSI[0])
            wcnd.append(WCND[0])
            kx.append(KA)
            psi_l.append(VGPSIL)
            fs.append(FSTOMATA)

    # processing data
    psis = np.array(psis)
    psix = np.array(psix)
    psil = np.array(psi_l)
    fctr_p = np.array(fctr_p)
    fctr_phm = np.array(fctr_phm)
    fctr_beta = np.array(fctr_beta)
    beta_phm = fctr_beta - fctr_phm

    # fctr_p_select = [300]
    fctr_p_select = [100, 200, 400, 800]
    PSIS_l = psis[fctr_p == 0]
    fctr_phm_lines = []
    fctr_beta_lines = []
    beta_phm_lines = []
    psix_lines = []
    psil_lines = []

    for fctr_p_ in fctr_p_select:
        fctr_phm_lines.append(fctr_phm[fctr_p == fctr_p_])
        fctr_beta_lines.append(fctr_beta[fctr_p == fctr_p_])
        beta_phm_lines.append(beta_phm[fctr_p == fctr_p_])
        psix_lines.append(psix[fctr_p == fctr_p_])
        psil_lines.append(psil[fctr_p == fctr_p_])

    # plot lines
    et_lines_range = [0, 1000]
    et_diff_lines_range = [-200, 200]
    psi_lines_range = [-1000, 0]
    # colors = ['b', 'c', 'r']
    colors = ['b', 'g', 'r', 'm']
    # colors = ['r']

    fig3, ax3 = plt.subplots(3, 4, figsize=(19, 10))
    plot_lines(fig3, ax3, SMC_l, PSIS_l, fctr_p_select, fctr_phm_lines, fctr_beta_lines, beta_phm_lines, psix_lines, psil_lines,
               et_lines_range, et_diff_lines_range, psi_lines_range, 'Beta', suffix_1, colors)

    if save_fig:
        if os.path.exists(fig_path) is False:
            os.makedirs(fig_path)
        fig3.savefig(fig_path + fig_name_pre + 'beta' + suffix_1 + '.pdf')

    plt.show()
    return


def plot_lines(fig, ax, SMC_l, PSIS_l, fctr_p_select, fctr_phm_lines, fctr_reduced_lines, fctr_diff_lines, psix_lines, psil_lines,
               et_lines_range, et_diff_lines_range, psi_lines_range, reduced_model_name, reduced_model_suffix, colors):
    for i in range(len(fctr_p_select)):
        fctr_p_ = fctr_p_select[i]
        ax[0, 0].plot(SMC_l, fctr_phm_lines[i], label=r'$T_{NHL}$' + '={}'.format(fctr_p_), c=colors[i])
        ax[0, 0].plot(SMC_l, np.full(fctr_phm_lines[i].shape, fctr_p_), ls='--', c=colors[i])
        ax[0, 0].set_xlabel('SWC')
        ax[0, 0].set_ylabel('T W/m2')
        ax[0, 0].set_ylim(et_lines_range)
        ax[0, 0].set_title('PHM transpiration response')

        ax[0, 1].plot(SMC_l, fctr_reduced_lines[i], label='Tww={}'.format(fctr_p_), c=colors[i])
        ax[0, 1].set_xlabel('SWC')
        ax[0, 1].set_ylabel('T W/m2')
        ax[0, 1].set_ylim(et_lines_range)
        ax[0, 1].set_title('{}{}'.format(reduced_model_name, reduced_model_suffix))

        ax[0, 2].plot(SMC_l, fctr_diff_lines[i], label='Tww={}'.format(fctr_p_), c=colors[i])
        ax[0, 2].set_xlabel('SWC')
        ax[0, 2].set_ylabel('T W/m2')
        ax[0, 2].set_ylim(et_diff_lines_range)
        ax[0, 2].set_title('{}{}-PHM'.format(reduced_model_name, reduced_model_suffix))

        ax[0, 3].plot(SMC_l, fctr_phm_lines[i], '-', label='Tww={}_PHM'.format(fctr_p_), c=colors[i])
        ax[0, 3].plot(SMC_l, fctr_reduced_lines[i], '--', label='Tww={}_{}'.format(fctr_p_, reduced_model_name), c=colors[i])
        ax[0, 3].set_xlabel('SWC')
        ax[0, 3].set_ylabel('T W/m2')
        ax[0, 3].set_ylim(et_lines_range)
        ax[0, 3].set_title('PHM {}{} overlay'.format(reduced_model_name, reduced_model_suffix))

        ax[1, 0].plot(PSIS_l, fctr_phm_lines[i], label='Tww={}'.format(fctr_p_), c=colors[i])
        ax[1, 0].set_xlabel(r'$\psi_s$ m')
        ax[1, 0].set_ylabel('T W/m2')
        ax[1, 0].set_ylim(et_lines_range)

        ax[1, 1].plot(PSIS_l, fctr_reduced_lines[i], label='Tww={}'.format(fctr_p_), c=colors[i])
        ax[1, 1].set_xlabel(r'$\psi_s$ m')
        ax[1, 1].set_ylabel('T W/m2')
        ax[1, 1].set_ylim(et_lines_range)

        ax[1, 2].plot(PSIS_l, fctr_diff_lines[i], label='Tww={}'.format(fctr_p_), c=colors[i])
        ax[1, 2].set_xlabel(r'$\psi_s$ m')
        ax[1, 2].set_ylabel('T W/m2')
        ax[1, 2].set_ylim(et_diff_lines_range)

        ax[1, 3].plot(PSIS_l, fctr_phm_lines[i], '-', label='Tww={}_PHM'.format(fctr_p_), c=colors[i])
        ax[1, 3].plot(PSIS_l, fctr_reduced_lines[i], '--', label='Tww={}_{}'.format(fctr_p_, reduced_model_name), c=colors[i])
        ax[1, 3].set_xlabel(r'$\psi_s$ m')
        ax[1, 3].set_ylabel('T W/m2')
        ax[1, 3].set_ylim(et_lines_range)

        ax[2, 0].plot(SMC_l, psix_lines[i], label='Tww={}'.format(fctr_p_), c=colors[i])
        ax[2, 0].set_xlabel('SWC')
        ax[2, 0].set_ylabel(r'$\psi_x$ m')
        ax[2, 0].set_ylim(psi_lines_range)

        ax[2, 1].plot(SMC_l, psil_lines[i], label='Tww={}'.format(fctr_p_), c=colors[i])
        ax[2, 1].set_xlabel('SWC')
        ax[2, 1].set_ylabel(r'$\psi_l$ m')
        ax[2, 1].set_ylim(psi_lines_range)

    ax[0, 0].legend(loc='upper left', ncol=2, bbox_to_anchor=(-0, 1.02))
    ax[0, 3].legend(loc='upper left', bbox_to_anchor=(1.1, 0.9))

    fig.tight_layout()


def PHM(config, NSOIL, ZSOIL, DZSNSO, FCTR, SMC):
    # config parameters
    SMCMAX = float(config['soil']['SMCMAX'])
    BEXP = float(config['soil']['BEXP'])
    # DKSAT = float(config['soil']['DKSAT'])
    PSISAT = float(config['soil']['PSISAT'])
    WLTSMC = float(config['soil']['SMCWLT'])

    # R2SR = float(config['phm']['R2SR'])
    RAI = float(config['phm']['RAI'])
    VEGH = float(config['phm']['VEGH'])
    # DROOT = float(config['phm']['DROOT'])
    VGSP50 = float(config['phm']['VGSP50'])
    VGA2 = float(config['phm']['VGA2'])
    VGKSAT = float(config['phm']['VGKSAT'])
    VGA1 = float(config['phm']['VGA1'])
    SPAI = float(config['phm']['SPAI'])
    VGTLP = float(config['phm']['VGTLP'])
    VGA3 = float(config['phm']['VGA3'])
    fRS = np.array([float(idx) for idx in config['phm']['fRS'].split(',')])
    # A_FENG = float(config['phm']['A_FENG'])

    # set iteration
    NITER = 1000
    relax = 0.07

    # convert transpiration to mm/s
    ETRAN = FCTR / HVAP

    # set root profile
    RAI = RAI * fRS

    # soil to root conductance
    WCND = np.zeros(NSOIL)
    SRCD = np.zeros(NSOIL)
    SOLPSI = np.zeros(NSOIL)
    ZMS2G = np.zeros(NSOIL)
    for IZ in range(NSOIL):
        WCND[IZ], WDF = WDFCND1(config, SMC[IZ])
        # SRCD[IZ] = WCND[IZ] * np.sqrt(RAI[IZ]) / (np.pi * DZSNSO[IZ])
        # When allocating RAI to the equivalent layer, normalize the DZSNSO to 1 m
        SRCD[IZ] = WCND[IZ] * np.sqrt(RAI[IZ]) / (np.pi * 1)
        # if IZ == 2:
        #     SRCD[IZ] = WCND[IZ] * np.sqrt(RAI / DROOT / 1.1)
        #     SRCD[IZ] = WCND[IZ] * np.sqrt(RAI) / (np.pi * 1.1)
        SOLPSI[IZ] = -PSISAT * (max(WLTSMC / SMCMAX, min(SMC[IZ], SMCMAX) / SMCMAX)) ** (-BEXP)
        ZMS2G[IZ] = max(0.0, -ZSOIL[IZ] - DZSNSO[IZ] * 0.5)
    SRCDt = np.sum(SRCD)

    # iteration to solve transpiration
    VGPSIS = VGPSIL = FSTOMATA = KA = 'holder'

    ETRAN_d = ETRAN
    ETRAN_d_l = []
    for i in range(NITER):
        ETRAN_d_old = ETRAN_d
        ETRAN_d_l.append(ETRAN_d)

        # soil to root flux, temporary calculation
        QS2R_temp = np.zeros(NSOIL)
        for IZ in range(NSOIL):
            QS2R_temp[IZ] = SRCD[IZ] * (SOLPSI[IZ] - ZMS2G[IZ])
            # QS2R_temp[IZ] = SRCD[IZ] * (SOLPSI[IZ])
        QS2R_temp_sum = np.sum(QS2R_temp)
        VGPSIS = (QS2R_temp_sum - ETRAN_d / 1000) / SRCDt

        # xylem to leaf flux
        KA = VGKSAT / (1 + (VGPSIS / VGSP50) ** VGA2)  # Xu et al.
        # KA = VGKSAT * (1 - 1 / (1 + np.exp(A_FENG * (VGPSIS - VGSP50))))  # Feng et al.
        VGPSIL = VGPSIS - VEGH - ETRAN_d / 1000 * VGA1 * VEGH / max(KA * SPAI, 1e-20)

        # stomatal downregulation factor
        FSTOMATA = 1. / (1. + (VGPSIL / VGTLP) ** VGA3)
        ETRAN_d = ETRAN_d_old + relax * (FSTOMATA * ETRAN - ETRAN_d_old)

        change = (ETRAN_d - ETRAN_d_old) / max(ETRAN_d_old, 1e-6)
        if i > 5 and np.abs(change) < 1e-3:
            # print('itertion converged, iter =', i + 1)
            break

        if iter == NITER:
            print('iteration not converging, change = ', change)

    # convert back to W/m2
    FCTR_d = ETRAN_d * HVAP
    FCTR_d_l = np.array(ETRAN_d_l) * HVAP

    return FCTR_d, FSTOMATA, SOLPSI, VGPSIS, VGPSIL, KA, SRCD, WCND, FCTR_d_l


def PHM_rev(config, NSOIL, ZSOIL, DZSNSO, FCTR, SMC):
    # config parameters
    SMCMAX = float(config['soil']['SMCMAX'])
    BEXP = float(config['soil']['BEXP'])
    # DKSAT = float(config['soil']['DKSAT'])
    PSISAT = float(config['soil']['PSISAT'])
    WLTSMC = float(config['soil']['SMCWLT'])

    # R2SR = float(config['phm']['R2SR'])
    RAI = float(config['phm']['RAI'])
    VEGH = float(config['phm']['VEGH'])
    # DROOT = float(config['phm']['DROOT'])
    VGSP50 = float(config['phm']['VGSP50'])
    VGA2 = float(config['phm']['VGA2'])
    VGKSAT = float(config['phm']['VGKSAT'])
    VGA1 = float(config['phm']['VGA1'])
    SPAI = float(config['phm']['SPAI'])
    VGTLP = float(config['phm']['VGTLP'])
    VGA3 = float(config['phm']['VGA3'])
    fRS = np.array([float(idx) for idx in config['phm']['fRS'].split(',')])
    # A_FENG = float(config['phm']['A_FENG'])

    # set iteration
    NITER = 1000
    relax = 0.07

    # convert transpiration to mm/s
    ETRAN = FCTR / HVAP

    # set root profile
    RAI = RAI * fRS

    # soil to root conductance
    WCND = np.zeros(NSOIL)
    SRCD = np.zeros(NSOIL)
    SOLPSI = np.zeros(NSOIL)
    ZMS2G = np.zeros(NSOIL)
    for IZ in range(NSOIL):
        WCND[IZ], WDF = WDFCND1(config, SMC[IZ])
        # SRCD[IZ] = WCND[IZ] * np.sqrt(RAI[IZ]) / (np.pi * DZSNSO[IZ])
        # When allocating RAI to the equivalent layer, normalize the DZSNSO to 1 m
        SRCD[IZ] = WCND[IZ] * np.sqrt(RAI[IZ]) / (np.pi * 1)
        # if IZ == 2:
        #     SRCD[IZ] = WCND[IZ] * np.sqrt(RAI / DROOT / 1.1)
        #     SRCD[IZ] = WCND[IZ] * np.sqrt(RAI) / (np.pi * 1.1)
        SOLPSI[IZ] = -PSISAT * (max(WLTSMC / SMCMAX, min(SMC[IZ], SMCMAX) / SMCMAX)) ** (-BEXP)
        ZMS2G[IZ] = max(0.0, -ZSOIL[IZ] - DZSNSO[IZ] * 0.5)
    SRCDt = np.sum(SRCD)

    # iteration to solve transpiration
    VGPSIS = VGPSIL = FSTOMATA = KA = 'holder'

    ETRAN_d = ETRAN
    VGPSIL = 0
    RHS = 0
    ETRAN_d_l = []
    for i in range(NITER):
        VGPSIL_pre = VGPSIL
        RHS_pre = RHS
        ETRAN_d_old = ETRAN_d
        ETRAN_d_l.append(ETRAN_d)

        # soil to root flux, temporary calculation
        QS2R_temp = np.zeros(NSOIL)
        for IZ in range(NSOIL):
            QS2R_temp[IZ] = SRCD[IZ] * (SOLPSI[IZ] - ZMS2G[IZ])
            # ignore the gravity term for now
            # QS2R_temp[IZ] = SRCD[IZ] * (SOLPSI[IZ])
        QS2R_temp_sum = np.sum(QS2R_temp)
        VGPSIS = (QS2R_temp_sum - ETRAN_d / 1000) / SRCDt

        # xylem to leaf flux
        KA = VGKSAT / (1 + ((VGPSIS + min(max(VGPSIL, 2 * VGTLP), VGPSIS)) / 2 / VGSP50) ** VGA2)
        # KA = VGKSAT / (1 + ((VGPSIS + VGPSIL) / 2 / VGSP50) ** VGA2)
        # KA = VGKSAT / (1 + (VGPSIS / VGSP50) ** VGA2)  # Xu et al.
        # KA = VGKSAT * (1 - 1 / (1 + np.exp(A_FENG * (VGPSIS - VGSP50))))  # Feng et al.
        VGPSIL = VGPSIS - VEGH - ETRAN_d / 1000 * VGA1 * VEGH / max(KA * SPAI, 1e-20)
        # VGPSIL = max(VGPSIL, -1000)

        # # prevent divergence
        # KA_pseudo = VGKSAT / (1 + ((0 + VGPSIL) / 2 / VGSP50) ** VGA2)
        # RHS = max(KA_pseudo * SPAI, 1e-30) / VGA1 / VEGH * (0 - VGPSIL - VEGH) * HVAP * 1000
        # # print('RHS diff:', RHS - RHS_pre, 'VGPSIL diff:', VGPSIL - VGPSIL_pre)
        # if ((RHS - RHS_pre) * (VGPSIL - VGPSIL_pre)) > 0:
        #     # VGPSIL = max(VGPSIL_pre + np.abs(VGPSIL - VGPSIL_pre), 0)
        #     print('RHS and VGPSIL have the same sign, RHS-RHS_pre = {}, VGPSIL-VGPSIL_pre = {}'.format(RHS - RHS_pre,
        #                                                                                                VGPSIL - VGPSIL_pre))

        # stomatal downregulation factor
        FSTOMATA = 1. / (1. + (VGPSIL / VGTLP) ** VGA3)
        ETRAN_d = ETRAN_d_old + relax * (FSTOMATA * ETRAN - ETRAN_d_old)

        change = (ETRAN_d - ETRAN_d_old) / max(ETRAN_d_old, 1e-6)
        if i > 5 and np.abs(change) < 1e-3:
            # print('itertion converged, iter =', i + 1)
            break

        if iter == NITER:
            print('iteration not converging, change = ', change)

    # convert back to W/m2
    FCTR_d = ETRAN_d * HVAP
    FCTR_d_l = np.array(ETRAN_d_l) * HVAP

    return FCTR_d, FSTOMATA, SOLPSI, VGPSIS, VGPSIL, KA, SRCD, WCND, FCTR_d_l


def BETA(config, NSOIL, ZSOIL, DZSNSO, FCTR, SH2O, DYNAMIC_beta, CAP_beta):
    # constant
    # MPE = 1e-6

    # config parameters
    PSISAT = float(config['soil']['PSISAT'])
    SMCMAX = float(config['soil']['SMCMAX'])
    BEXP = float(config['soil']['BEXP'])
    WLTSMC = float(config['soil']['SMCWLT'])
    BETAEXP = float(config['beta']['BETAEXP'])
    beta_fraction = [float(idx) for idx in config['beta']['beta_fraction'].split(',')]

    if DYNAMIC_beta:
        # old implementation
        # BETA50_tww_ref = float(config['beta']['BETA50_tww_ref'])
        # BETA50 = -(BETA50_tww_ref / FCTR) ** 0.5

        # new implementation
        krmax = float(config['beta']['krmax'])
        kmax_beta_adjusted = float(config['beta']['kmax_beta_adjusted'])
        VEGH = float(config['phm']['VEGH'])
        VGSP50 = float(config['phm']['VGSP50'])
        VGA2 = float(config['phm']['VGA2'])
        VGTLP = float(config['phm']['VGTLP'])
        VGA3 = float(config['phm']['VGA3'])

        BETA50 = dynamic_psis50(FCTR, kmax_beta_adjusted, VEGH, VGTLP, VGA3, VGSP50, VGA2, krmax, 0.7, PSISAT, BEXP)
    else:
        BETA50 = float(config['beta']['BETA50'])

    if CAP_beta:
        # config parms
        kmax_beta_adjusted = float(config['beta']['kmax_beta_adjusted'])
        # calculate cap
        FCTR_cap = PHM_ww_cap(kmax_beta_adjusted, 9, -100, 6, FCTR)
    else:
        FCTR_cap = FCTR

    # old implementation
    # PSI = -PSISAT * (max(WLTSMC, min(SH2O[2], SMCMAX)) / SMCMAX) ** (-BEXP)
    # BTRAN = 1. / (1. + (PSI / BETA50) ** BETAEXP)

    PSI = -PSISAT * (np.maximum(np.full(len(SH2O), WLTSMC), np.minimum(SH2O, np.full(len(SH2O), SMCMAX))) / SMCMAX) ** (-BEXP)
    BTRAN_temp = np.sum(1. / (1. + (PSI / BETA50) ** BETAEXP) * np.array(beta_fraction))
    # print('PSI:', PSI)
    # print('BTRAN:', BTRAN)

    FCTR_d = BTRAN_temp * FCTR_cap
    BTRAN = FCTR_d / FCTR

    return FCTR_d, BTRAN


def PHM_ww_cap(kmax, VEGH, VGTLP, VGA3, FCTR):
    # optimize and result
    res = minimize_scalar(psil_cap_opt, args=(FCTR, VGTLP, VGA3, kmax, VEGH))
    VGPSIL = res.x
    td = 1. / (1. + (VGPSIL / VGTLP) ** VGA3) * FCTR

    return td


def psil_cap_opt(VGPSIL, FCTR, VGTLP, VGA3, kmax, VEGH):
    td = 1. / (1. + (VGPSIL / VGTLP) ** VGA3) * FCTR
    ts = kmax * (-VGPSIL - VEGH)
    result = (td - ts) ** 2

    return result


def dynamic_psis50(FCTR, kmax, VEGH, VGTLP, VGA3, VGSP50, VGA2, krmax, DSOIL, PSISAT, BEXP):
    # calculate well watered cap
    CAP = PHM_ww_cap(kmax, VEGH, VGTLP, VGA3, FCTR) / FCTR
    # print('CAP:', CAP)

    # calculate leaf water potential at 1/2 * cap
    VGPSIL = VGTLP * (2 / CAP - 1) ** (1 / VGA3)
    # print('VGPSIL:', VGPSIL)

    # calculate xylem water potential
    res = minimize_scalar(solve_psix_for_psis50, args=(FCTR, VGPSIL, CAP, kmax, VEGH, VGSP50, VGA2))
    VGPSIS = res.x
    # print('VGPSIS:', VGPSIS)

    # calculate soil water potential
    res1 = minimize_scalar(solve_psis_for_psis50, bounds=(-100, 0), args=(FCTR, CAP, VGPSIS, krmax, DSOIL, PSISAT, BEXP))
    SOLPSI = res1.x
    # print('SOLPSI:', SOLPSI)

    return SOLPSI


def solve_psis_for_psis50(SOLPSI, FCTR, CAP, VGPSIS, krmax, DSOIL, PSISAT, BEXP):
    left = krmax * (SOLPSI - VGPSIS - DSOIL)
    right = 1 / 2 * CAP * FCTR * (SOLPSI / (-PSISAT)) ** (2 + 3 / BEXP)
    result = np.abs(left - right)

    return result


def solve_psix_for_psis50(VGPSIS, FCTR, VGPSIL, CAP, kmax, VEGH, VGSP50, VGA2):
    left = kmax * (VGPSIS - VGPSIL - VEGH)
    right = 1 / 2 * CAP * FCTR * (1 + (VGPSIS / VGSP50) ** VGA2)
    result = (left - right) ** 2

    return result


def WDFCND1(config, SMC):
    # config parameters
    SMCMAX = float(config['soil']['SMCMAX'])
    BEXP = float(config['soil']['BEXP'])
    DWSAT = float(config['soil']['DWSAT'])
    DKSAT = float(config['soil']['DKSAT'])

    # soil water diffusivity
    FACTR = max(0.01, SMC / SMCMAX)
    EXPON = BEXP + 2.0
    WDF = DWSAT * FACTR ** EXPON

    # hydraulic conductivity
    EXPON = 2.0 * BEXP + 3.0
    WCND = DKSAT * FACTR ** EXPON
    # WCND = 0.1 * DKSAT

    return WCND, WDF


def SOIL_LAYER_CONFIG(config):
    DT = float(config['configuration']['DT'])
    NSOIL = int(config['configuration']['NSOIL'])
    ZSOIL = np.array([float(idx) for idx in config['configuration']['ZSOIL'].split(',')])
    DZSNSO = np.array([float(idx) for idx in config['configuration']['DZSNSO'].split(',')])

    return DT, NSOIL, ZSOIL, DZSNSO


if __name__ == '__main__':
    main()
