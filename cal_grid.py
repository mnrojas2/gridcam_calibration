import numpy as np
import scipy.io

mat = scipy.io.loadmat('vars.mat')

cntrd = mat['c'][0]
I = mat['BW']
cntrd_offset = mat['cntrd_offset'][0][0]
searchradius = mat['search_radius'][0][0]
verbose = mat['showSTD'][0][0]
skip = mat['skip'][0][0]
skipstart = mat['skipstart'][0][0]
skiplength = mat['skiplength'][0][0]

print(cntrd, I.shape, cntrd_offset, searchradius, verbose, skip, skipstart, skiplength)

loopvar = np.arange(searchradius)
    
if skip == 1:
    skipstart = skipstart - cntrd_offset
    loopvar = loopvar[((loopvar <= skipstart) | (loopvar > skipstart + skiplength))]
    
# print(loopvar.shape, loopvar[245:255])
angle_diff_plot = []
for_STD_angle_diff = []
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
            angle_diff_plot.append(angle_diff)
            for_STD_angle_diff.append(angle_diff)
            
        else:
            angle_diff_plot.append(np.nan)
            
        STD_vec.append(np.std(for_STD_angle_diff))
    
angle_diff_plot = np.array(angle_diff_plot)
STD_vec = np.array(STD_vec)

relangle = np.mean(for_STD_angle_diff)
STD_final = np.std(for_STD_angle_diff)

print(STD_final, relangle)

from calculate_gridangle import calculate_gridangle

print(calculate_gridangle(cntrd, I, cntrd_offset, searchradius, verbose, skip, skipstart, skiplength))
# STD_vec_mean = np.mean((np.array((STD_vec))[STD_vec>0]))
            
    