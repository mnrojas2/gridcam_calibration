Grid-Camera calibration
=======================

Scripts and information for obtaining the angle between the wave grid polarization and the camera set next to it.

Note: You will require a video of a laser passing through the wiregrid, being recorded by the camera used for the photogrammetry section of the project. It must be a video, ideally using the same or similar parameters to record the flights.

1) Run ``python get_frames.py <directory of the video>`` to get all frames of the video.

2) Run ``python calibrate_grid.py <directory of the frames folder> -cb <directory of a previous camera calibration parameters> -fp`` to determine the relative orientation of the wiregrid to the camera.
    
    Note 2: More options are available by checking the help section of the files using ``python <py script file> -h``

    Note 3: Optionally remove some outliers from the list by removing the frames and running the script again to try to improve the standard deviation of the result.