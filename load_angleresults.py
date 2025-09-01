import os
import numpy as np
from matplotlib import pyplot as plt

# Tests Y axis
# datefile = ['0813', '0818']
# '0821' is the mix of results between '0818' and '0820'

# Tests X axis
datefile = ['0827', '0828']

# Toggle this to also plot angles
plot_angles = True

# Toggle this to plot fits in some plots
plot_fits = True

# Vector to save all results to plot normalized histogram
c_lin01_sum = np.array([])

# Vector to save results for cases 1 and 2, independently to fit in corresponding figures.
l_angle0_sum = np.array([])
c_lin0_sum = np.array([])
l_angle1_sum = np.array([])
c_lin1_sum = np.array([])
dy_sum = np.array([])
c_lin2_sum = np.array([])

for item in datefile:
    filename = f'test_sala_oscura/angle_results_{item}_binary_deg2.csv'

    data_names = np.genfromtxt(filename, skip_header=1, delimiter=',', dtype=None, encoding='utf-8', usecols=range(0,1))
    data_values = np.genfromtxt(filename, skip_header=1, delimiter=',', dtype=float, encoding='utf-8', usecols=range(1,8))

    # Get all columns data
    laser_angle = data_values[:,0]
    grid_angle = data_values[:,1]
    c_quad = data_values[:,2]
    c_lin = data_values[:,3]
    c_off = data_values[:,4]
    angle = data_values[:,5]
    sig_angle = data_values[:,6]
    
    # Indicate where the split in experiments happen
    split1 = 10
    split2 = 18

    #### Test 1: Only changing the Angle of the laser keeping the wiregrid in the same state ####
    l_angle0 = laser_angle[:split1]
    c_quad0 = c_quad[:split1]
    c_lin0 = c_lin[:split1]
    angle0 = angle[:split1]

    # Produce fit lines for both cases, quadratic coefficient and linear coefficient
    l_range0 = np.linspace(l_angle0[0], l_angle0[-1], 51)
    coeffs_quad0, cov =  np.polyfit(l_angle0, c_quad0, deg=1, cov=True)
    coeffs_lin0, cov = np.polyfit(l_angle0, c_lin0, deg=1, cov=True)
    c_quad0_range = np.polyval(coeffs_quad0, l_range0)
    c_lin0_range = np.polyval(coeffs_lin0, l_range0)

    # Save linear data for histogram plots
    l_angle0_sum = np.concatenate([l_angle0_sum, l_angle0])
    c_lin0_sum = np.concatenate([c_lin0_sum, c_lin0])

    fig_n = 0
    plt.figure(fig_n)
    plt.scatter(l_angle0, c_quad0, label=item)
    if plot_fits:
        plt.plot(l_range0, c_quad0_range, '.-',label=f'fitted {item}')
    plt.xlabel('degrees (°)')
    plt.title('Quadratic coefficient (adim) vs Laser angle (deg)')
    plt.legend()

    fig_n += 1 
    plt.figure(fig_n)
    plt.scatter(l_angle0, c_lin0, label=item)
    if plot_fits:
        plt.plot(l_range0, c_lin0_range, '.-',label=f'fitted {item}')
    plt.xlabel('degrees (°)')
    plt.title('Linear coefficient (adim) vs Laser angle (deg)')
    plt.legend()

    if plot_angles:
        fig_n += 1 
        plt.figure(fig_n)
        plt.scatter(l_angle0, angle0, label=item)
        plt.xlabel('degrees (°)')
        plt.ylabel('degrees (°)')
        plt.title('Measured projected angle (deg) vs Laser angle (deg)')
        plt.legend()

    print(f'Angle standard deviation for {item}: {np.std(angle0)} for different laser positions.')


    #### Test 2: Same angle between laser and wiregrid ####
    l_angle1 = laser_angle[split1:split2]
    c_quad1 = c_quad[split1:split2]
    c_lin1 = c_lin[split1:split2]
    angle1 = angle[split1:split2]

    # Produce fit lines for both cases, quadratic coefficient and linear coefficient
    l_range1 = np.linspace(l_angle1[0], l_angle1[-1], 51)
    coeffs_quad1, cov =  np.polyfit(l_angle1, c_quad1, deg=1, cov=True)
    coeffs_lin1, cov = np.polyfit(l_angle1, c_lin1, deg=1, cov=True)
    c_quad1_range = np.polyval(coeffs_quad1, l_range1)
    c_lin1_range = np.polyval(coeffs_lin1, l_range1)

    # Save linear data for histogram plots
    l_angle1_sum = np.concatenate([l_angle1_sum, l_angle1])
    c_lin1_sum = np.concatenate([c_lin1_sum, c_lin1])

    fig_n += 1 
    plt.figure(fig_n)
    plt.scatter(l_angle1, c_quad1, label=item)
    if plot_fits:
        plt.plot(l_range1, c_quad1_range, '.-',label=f'fitted {item}')
    plt.xlabel('degrees (°)')
    plt.title('Quadratic coefficient (adim) vs Wiregrid angle (deg)')
    plt.legend()

    fig_n += 1 
    plt.figure(fig_n)
    plt.scatter(l_angle1, c_lin1, label=item)
    if plot_fits:
        plt.plot(l_range1, c_lin1_range, '.-',label=f'fitted {item}')
    plt.xlabel('degrees (°)')
    plt.title('Linear coefficient (adim) vs Wiregrid angle (deg)')
    plt.legend()

    if plot_angles:
        fig_n += 1 
        plt.figure(fig_n)
        plt.scatter(l_angle1, angle1, label=item)
        plt.xlabel('degrees (°)')
        plt.ylabel('degrees (°)')
        plt.title('Measured projected angle (deg) vs Wiregrid angle (deg)')
        plt.legend()

    print(f'Angle standard deviation for {item}: {np.std(angle1)} for different wall proyections.')

    try:
        #### Test 3: Changing the position of the laser over the wiregrid across the X or Y axis
        delta_y = np.array([-30, -30, -10, -10, 0, 0, 10, 10, 30, 30])
        c_quad2 = c_quad[split2:]
        c_lin2 = c_lin[split2:]
        angle2 = angle[split2:]

        # Produce fit lines for both cases, quadratic coefficient and linear coefficient
        dy_range = np.linspace(delta_y[0], delta_y[-1], 51)
        coeffs_quad2, cov =  np.polyfit(delta_y, c_quad2, deg=1, cov=True)
        coeffs_lin2, cov = np.polyfit(delta_y, c_lin2, deg=1, cov=True)
        c_quad2_range = np.polyval(coeffs_quad2, dy_range)
        c_lin2_range = np.polyval(coeffs_lin2, dy_range)

        # Save linear data for histogram plots
        dy_sum = np.concatenate([dy_sum, delta_y])
        c_lin2_sum = np.concatenate([c_lin2_sum, c_lin2])

        fig_n += 1
        plt.figure(fig_n)
        plt.scatter(delta_y, c_quad2, label=item)
        if plot_fits:
            plt.plot(dy_range, c_quad2_range, '.-',label=f'fitted {item}')
        plt.xlabel('milimeters (mm)')
        plt.title('Quadratic coefficient (adim) vs delta axis (mm)')
        plt.legend()

        fig_n += 1 
        plt.figure(fig_n)
        plt.scatter(delta_y, c_lin2, label=item)
        if plot_fits:
            plt.plot(dy_range, c_lin2_range, '.-',label=f'fitted {item}')
        plt.xlabel('milimeters (mm)')
        plt.title('Linear coefficient (adim) vs delta axis (mm)')
        plt.legend()

        if plot_angles:
            fig_n += 1 
            plt.figure(fig_n)
            plt.scatter(delta_y, angle2, label=item)
            plt.xlabel('milimeters (mm)')
            plt.ylabel('degrees (°)')
            plt.title('Measured projected angle (deg) vs delta axis (mm)')
            plt.legend()

        print(f'Angle standard deviation for {item}: {np.std(angle2)} for different laser offsets in one axis.')

    except:
        print("Test 3 was not done")


    # Make histogram of tests 1 & 2 only
    c_lin01 = c_lin[:split2]
    c_lin01_norm = c_lin01-np.mean(c_lin01)

    fig_n += 1
    plt.figure(fig_n)
    plt.hist(c_lin01, bins=9, label=item)
    plt.title("Histogram of the linear coefficients")
    plt.legend()

    # Append all results into the same vector
    c_lin01_sum = np.concatenate([c_lin01_sum, c_lin01_norm])

    # Make normalized histogram of tests 1 & 2 only
    fig_n += 1
    plt.figure(fig_n)
    plt.hist(c_lin01_norm, bins=9, alpha=.75, label=item)
    plt.title("Histogram of the linear coefficients")

# To the last figure, add the histogram with results of all measurements
plt.hist(c_lin01_sum, bins=9, label='All', zorder=0)
plt.legend()
plt.title("Normalized histogram of the sum of linear coefficients")

# Fit a line for all datasets together and plot it alongside the other two
if plot_fits:
    ### Test 1 fit
    coeffs_lin0sum, cov = np.polyfit(l_angle0_sum, c_lin0_sum, deg=1, cov=True)
    c_lin0sum_range = np.polyval(coeffs_lin0sum, l_range0)

    fig_m = 1
    plt.figure(fig_m)
    plt.plot(l_range0, c_lin0sum_range, '.-',label=f'fitted all')
    plt.legend()

    ### Test 2 fit
    coeffs_lin1sum, cov = np.polyfit(l_angle1_sum, c_lin1_sum, deg=1, cov=True)
    c_lin1sum_range = np.polyval(coeffs_lin1sum, l_range1)

    fig_m = 3
    if plot_angles:
        fig_m += 1
    plt.figure(fig_m)
    plt.plot(l_range1, c_lin1sum_range, '.-',label=f'fitted all')
    plt.legend()

    ### Test 3 fit
    try:
        coeffs_lin2sum, cov = np.polyfit(dy_sum, c_lin2_sum, deg=1, cov=True)
        c_lin2sum_range = np.polyval(coeffs_lin2sum, dy_range)

        fig_m = 5
        if plot_angles:
            fig_m += 2
        plt.figure(fig_m)
        plt.plot(dy_range, c_lin2sum_range, '.-',label=f'fitted all')
        plt.legend()
    except: pass

# Plot all figures
plt.show()
