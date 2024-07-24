import numpy as np
from matplotlib import pyplot as plt

def calculate_gridangle(cntrd, I, cntrd_offset, searchradius, plot, skip, skipstart, skiplength):
# Auxiliary function to calculate the relative angle between the grid and the camera in the frame

    # print(cntrd, I.shape, cntrd_offset, searchradius, plot, skip, skipstart, skiplength)
    loopvar = np.arange(searchradius)
    
    # If skip flag is enabled, the list will skip a certain range, defined by skipstart, skiplength and the offset center
    if skip == 1:
        skipstart = skipstart - cntrd_offset
        loopvar = loopvar[((loopvar <= skipstart) | (loopvar > skipstart + skiplength))]
        
    # Create lists to save values for angle and standard deviation of the angle
    angle_vec = []
    valid_angle_diff = []
    STD_vec = []

    for i in loopvar:
        top = I[int(np.floor(cntrd[0]-cntrd_offset-(i+1))), :]
        bot = I[int(np.floor(cntrd[0]+cntrd_offset+(i+1))), :]
        
        if sum(top) != 0 and sum(bot) != 0:
            top_weights = np.arange(1, 1+top.shape[-1]) * top
            top_cntrd = np.sum(top_weights) / np.sum(top)
            
            bot_weights = np.arange(1, 1+bot.shape[-1]) * bot
            bot_cntrd = np.sum(bot_weights) / np.sum(bot)
            
            angle_top = np.degrees(np.arctan((top_cntrd - cntrd[1])/(cntrd_offset + (i+1))))
            angle_bot = np.degrees(np.arctan((bot_cntrd - cntrd[1])/(cntrd_offset + (i+1))))
            
            angle_diff = angle_top - angle_bot
            radius_vector = cntrd_offset + (i+1)
            
            if angle_diff != 0:
                valid_angle_diff.append(angle_diff)

            angle_vec.append(angle_diff)  
            STD_vec.append(np.std(valid_angle_diff))
    
    # Convert the lists to numpy ndarray format
    angle_vec = np.array(angle_vec)
    STD_vec = np.array(STD_vec)
    
    # Create new lists of angle and standard deviation for plotting, where zero values are replaced by np.nan
    angle_plot = angle_vec.copy()
    angle_plot[angle_plot==0] = np.nan
    
    STD_plot = STD_vec.copy()
    STD_plot[STD_plot==0] = np.nan

    # Calculate the relative angle and standard deviation of the angle
    relative_angle = np.mean(valid_angle_diff)
    STD_final = np.std(valid_angle_diff)
    
    # 
    if plot == 1:
        # Create the x axis for the plot
        actual_radius = np.arange(1+cntrd_offset, 1+cntrd_offset+angle_plot.shape[-1])
        
        # Create subplots
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        
        # First subplot
        ax1.plot(actual_radius, angle_plot, 'r-', linewidth=0.5, label='Relative angle')
        ax1.set_xlabel('Distance from center (pix)')
        ax1.set_ylabel('Relative angle (°)', color='r')
        ax1.axhline(y=relative_angle, linewidth=0.5, linestyle='-', color='b', label='Mean relative angle')
        ax1.axhline(y=relative_angle+0.5, linewidth=0.5, linestyle='--', color='b')
        ax1.axhline(y=relative_angle-0.5, linewidth=0.5, linestyle='--', color='b')

        # Second subplot with its own y-axis
        ax2.plot(actual_radius, STD_plot, 'g-', linewidth=0.5, label='Accumulated Error (STD)')
        ax2.set_ylabel('Error (°)', color='g')
        
        # Combine the legend handles and labels from both axes
        handles, labels = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        all_handles = handles + handles2
        all_labels = labels + labels2

        # Create a single legend, add a gridline, text and the title to the plot
        ax1.legend(all_handles, all_labels, loc='upper right')
        ax1.grid(visible=True, which='both')
        
        plt.annotate(f"Relative angle is: {np.round(relative_angle,3)}° ± {np.round(STD_final,3)}°", xy=(0.02, 0.95), xycoords='axes fraction', textcoords='axes fraction', ha='left')
        plt.title(f"Relative angle vs distance from center")
        plt.show()

    return STD_final, relative_angle