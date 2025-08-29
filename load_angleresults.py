import os
import numpy as np
from matplotlib import pyplot as plt

# Tests axis Y
# datefile = ['0813', '0818']
# '0821' is the mix of results between '0818' and '0820'

# Tests axis X
datefile = ['0827', '0828']

# Toggle this to also plot angles
plot_angles = False

# Vector to save all results to plot normalized histogram
c_lin_sum = np.empty(2)

for item in datefile:
    filename = f'test_sala_oscura/angle_results_{item}_binary_deg2.csv'

    data_names = np.genfromtxt(filename, skip_header=1, delimiter=',', dtype=None, encoding='utf-8', usecols=range(0,1))
    data_values = np.genfromtxt(filename, skip_header=1, delimiter=',', dtype=float, encoding='utf-8', usecols=range(1,8))

    laser_angle = data_values[:,0]
    grid_angle = data_values[:,1]
    c_quad = data_values[:,2]
    c_lin = data_values[:,3]
    c_off = data_values[:,4]
    angle = data_values[:,5]
    sig_angle = data_values[:,6]
    
    split1 = 10
    split2 = 18

    fig_n = 0

    #### Only changing the Angle of the laser keeping the wiregrid in the same state ####
    plt.figure(fig_n)
    plt.scatter(laser_angle[:split1], c_quad[:split1], label=item)
    plt.xlabel('degrees (°)')
    plt.title('Quadratic coefficient (adim) vs Laser angle (deg)')
    plt.legend()

    fig_n += 1 
    plt.figure(fig_n)
    plt.scatter(laser_angle[:split1], c_lin[:split1], label=item)
    plt.xlabel('degrees (°)')
    plt.title('Linear coefficient (adim) vs Laser angle (deg)')
    plt.legend()

    if plot_angles:
        fig_n += 1 
        plt.figure(fig_n)
        plt.scatter(laser_angle[:split1], angle[:split1], label=item)
        plt.xlabel('degrees (°)')
        plt.ylabel('degrees (°)')
        plt.title('Measured projected angle (deg) vs Laser angle (deg)')
        plt.legend()

    print(f'Angle standard deviation for {item}: {np.std(angle[:split1])} for different laser positions.')


    #### Same angle between laser and wiregrid ####
    fig_n += 1 
    plt.figure(fig_n)
    plt.scatter(laser_angle[split1:split2], c_quad[split1:split2], label=item)
    plt.xlabel('degrees (°)')
    plt.title('Quadratic coefficient (adim) vs Wiregrid angle (deg)')
    plt.legend()

    fig_n += 1 
    plt.figure(fig_n)
    plt.scatter(laser_angle[split1:split2], c_lin[split1:split2], label=item)
    plt.xlabel('degrees (°)')
    plt.title('Linear coefficient (adim) vs Wiregrid angle (deg)')
    plt.legend()

    if plot_angles:
        fig_n += 1 
        plt.figure(fig_n)
        plt.scatter(laser_angle[split1:split2], angle[split1:split2], label=item)
        plt.xlabel('degrees (°)')
        plt.ylabel('degrees (°)')
        plt.title('Measured projected angle (deg) vs Wiregrid angle (deg)')
        plt.legend()

    print(f'Angle standard deviation for {item}: {np.std(angle[split1:split2])} for different wall proyections.')

    try:
        # Changing the position of the laser over the wiregrid across the Y axis
        delta_y = np.array([-30, -30, -10, -10, 0, 0, 10, 10, 30, 30])

        fig_n += 1
        plt.figure(fig_n)
        plt.scatter(delta_y, c_quad[split2:], label=item)
        plt.xlabel('milimeters (mm)')
        plt.title('Quadratic coefficient (adim) vs delta Y (mm)')
        plt.legend()

        fig_n += 1 
        plt.figure(fig_n)
        plt.scatter(delta_y, c_lin[split2:], label=item)
        plt.xlabel('milimeters (mm)')
        plt.title('Linear coefficient (adim) vs delta Y (mm)')
        plt.legend()

        if plot_angles:
            fig_n += 1 
            plt.figure(fig_n)
            plt.scatter(delta_y, angle[split2:], label=item)
            plt.xlabel('milimeters (mm)')
            plt.ylabel('degrees (°)')
            plt.title('Measured projected angle (deg) vs delta Y (mm)')
            plt.legend()

        print(f'Angle standard deviation for {item}: {np.std(angle[split2:])} for different laser offsets in one axis.')
    
    except:
        print("No offset tests were done")
        fig_n -= 1


    # Make histogram of tests 1 & 2 only
    fig_n += 1
    plt.figure(fig_n)
    plt.hist(c_lin[:split2], bins=9, label=item)
    plt.title("Histogram of the linear coefficients")
    plt.legend()

    # Make histogram of tests 1 & 2 only
    fig_n += 1
    plt.figure(fig_n)
    plt.hist(c_lin[:split2]-np.mean(c_lin[:split2]), bins=9, label=item)
    plt.title("Histogram of the linear coefficients")
    plt.legend()

    # Append all results into the same vector
    c_lin_sum = np.concatenate([c_lin_sum, (c_lin[:split2]-np.mean(c_lin[:split2]))])

# Histogram of all results across all measurements
fig_n += 1
plt.figure(fig_n)
plt.hist(c_lin_sum, bins=9)
plt.title("Normalized histogram of the sum of linear coefficients")
plt.show()


# Faltan fits lineales