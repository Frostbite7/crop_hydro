import numpy as np
import matplotlib.pyplot as plt

HVAP = 2.51e6


def main():
    # set up parameters
    SPAI = 15.8E-4
    VEGH = 1
    # R2SR = 1
    # RAI = 210
    VGKSAT = 1.5e-6
    VGSP50 = -100
    VGA2 = 3
    VGA1 = 1
    VGTLP = -300
    VGA3 = 4
    # DROOT = 5e-4

    # set up variables
    TR = 800
    VGPSIS = 0
    VGPSIL_l = np.arange(-1000, 0, 1)
    RHS_l = []
    LHS_l = []
    R_L_l = []

    # begin calculation
    for VGPSIL in VGPSIL_l:
        # KA = VGKSAT / (1 + ((VGPSIS + max(VGPSIL, 2 * VGTLP)) / 2 / VGSP50) ** VGA2)
        # KA = VGKSAT / (1 + ((VGPSIS + VGPSIL) / 2 / VGSP50) ** VGA2)
        KA = VGKSAT / (1 + ((VGPSIS + max(VGPSIL, 3 * VGTLP)) / 2 / VGSP50) ** VGA2)
        KAC = max(KA * SPAI, 1e-20) / VEGH / VGA1
        RHS = KAC * (VGPSIS - VGPSIL - VEGH) * 1000 * HVAP
        FSTOMATA = 1. / (1. + (VGPSIL / VGTLP) ** VGA3)
        LHS = FSTOMATA * TR
        R_L = RHS - LHS

        RHS_l.append(RHS)
        LHS_l.append(LHS)
        R_L_l.append(R_L)

    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    ax[0, 0].plot(VGPSIL_l, RHS_l, label='RHS')
    ax[0, 0].plot(VGPSIL_l, LHS_l, label='LHS')
    # ax[0, 0].plot(VGPSIL_l, R_L_l, label='R-L')
    ax[0, 0].set_xlabel('VGPSIL')
    ax[0, 0].set_ylabel('T w/m2')
    ax[0, 0].set_ylim([-200, 800])
    ax[0, 0].legend()

    ax[0, 1].plot(VGPSIL_l, R_L_l, label='R-L', color='g')
    ax[0, 1].plot(VGPSIL_l, np.zeros_like(VGPSIL_l), color='k')
    ax[0, 1].set_xlabel('VGPSIL')
    ax[0, 1].set_ylabel('T w/m2')
    ax[0, 1].set_ylim([-200, 800])
    ax[0, 1].legend()

    fig.tight_layout()
    plt.show()

    return


if __name__ == '__main__':
    main()
