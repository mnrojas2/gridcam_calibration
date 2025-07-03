import cv2
import numpy as np
import argparse
from time import time
from matplotlib import pyplot as plt
from hovercal.library.camera import get_camera_params
from skimage import measure, morphology
from scipy import ndimage
from tqdm import tqdm


def get_frames_from_video(filename, clip=30, mono=False):
    """
    Produce a list of frames based on a video.
        
    Parameters: 
    - filename (str): Directory of the videofile.
    - clip (int): Number of frames to be cut from the start and end of the video.
    - mono (bool): Obtain a single layer of the frame by adding the 3 channels (RGB).
    
    Returns (list): a list of frames
    """
    # Open the video
    vidcap = cv2.VideoCapture(filename)
    if not vidcap.isOpened():
        print("Error: Cannot open video.")
        return []

    total_frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    for frame_idx in tqdm(range(total_frame_count), desc="Loading Images", unit=" frames"):
        frame_exists, curr_frame = vidcap.read()
        if frame_exists:
            if mono:
                curr_frame = np.sum(curr_frame, axis=2)
            frames.append(curr_frame)
        else:
            print(f"Failed to read frame {frame_idx}")
            break

    if len(frames) < 2*clip:
        print("Clip larger than frame length. Setting clip = 0")
        clip = 0

    return frames[clip:-clip]


def get_line_center(img, thresh=254, maxval=1):
    """
    Find the center of the laser trace, by producing a binary image that only keeps the brightest point of the image.
        
    Parameters: 
    - img (numpy.ndarray): Image containing the laser trace.
    - thresh (int): Set threshold to remove every bright point of the image except for the center of the laser trace.
    - maxval (int): Set maximum value in the resulting binary image from thresholding the original. 
    
    Returns (tuple): position of the found center
    """
    # Get a binary image by thresholding it with a top value of brightness (looking to find the traces of the polarized laser)
    _, img_center = cv2.threshold(img, thresh, maxval, cv2.THRESH_BINARY)

    # Find the centroids of the laser traces in the image
    BW_cntrd_labels = measure.label(img_center)
    BW_cntrd_props = sorted(measure.regionprops(BW_cntrd_labels), key=lambda r: r.area, reverse=True)

    # Choose the brightest centroid (the center of the laser trace)
    center = BW_cntrd_props[0].centroid
    return center


def fit_line(img, deg=2, enhance_pow=2, offset=5000, bthr=100, min_size=5):
    """
    Calculate a fit of the laser trace, based on the total number of pixels found by applying 
        an exponential contrast filter, from the center to the corners.
    
    Parameters: 
    - img (numpy.ndarray): Image containing the laser trace, centered in the laser trace.
    - deg (int): Polynomial degree to be fit to the list of points.
    - enhance_pow (int): 
    - offset (int): 
    - bthr (int): Set threshold value to find only the laser trace.
    - min_size (int): Set minimum number of contiguous pixels that must be kept in the binarized image.
    - plot (bool): Enable plotting figures to show binarized images. 
    
    Returns: line coefficients (numpy.ndarray), covariance matrix (numpy.ndarray) and
        binarized image (numpy.ndarray).
    """
    h, w = img.shape
    x = np.linspace(-int(h/2), int(h/2), h)
    img2 = img * (np.abs(x[:,np.newaxis])**enhance_pow + offset)
    img_bgrd = img2[:,:int(w/4)].mean(axis=1)
    img2 -= img_bgrd[:,np.newaxis]
    img_u8 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Threshold the image to find the laser trace
    _, binary = cv2.threshold(img_u8, bthr, 1, cv2.THRESH_BINARY)
    binary = (morphology.remove_small_objects(np.array(binary, dtype=bool), min_size)).astype(int)
    
    # Find the position of each bright pixel.
    xx, yy = np.where(binary)
    
    # Fit the list of points and get the coefficients and covariance matrix.
    coeffs, cov = np.polyfit(xx - h/2, yy - w/2, deg=deg, cov=True)
    return coeffs, cov, binary


def fit_centroids(img, deg=2, center_offset=125, bthr=10, min_size=20):
    """
    Calculate pixel positions of the centroids per each row where the laser exists in the image.
    
    Parameters: 
    - img (numpy.ndarray): Image containing the laser trace, centered in the laser trace.
    - deg (int): Polynomial degree to be fit to the list of points.
    - center_offset (int): Number of elements to skip around the laser center in vertical direction.
    - bthr (int): Set threshold value to find only the laser trace.
    - min_size (int): Set minimum number of contiguous pixels that must be kept in the binarized image.
    - plot (bool): Enable plotting figures to show binarized images. 
    
    Returns: line coefficients (numpy.ndarray), covariance matrix (numpy.ndarray), 
        binarized image (numpy.ndarray) and list of (x,y) elements (numpy.ndarray).
    """
    # Auxiliary function to calculate the relative angle between the grid and the camera in the frame
    h, w = img.shape

    _, binary = cv2.threshold(img, bthr, 1, cv2.THRESH_BINARY)    
    binary = (morphology.remove_small_objects(np.array(binary, dtype=bool), min_size)).astype(int)

    # Get all row centroids and determine the distance between all centroids and the center of the line (cntrd).
    centroids = []
    for i in range(len(binary)):
        if np.sum(binary[i]) != 0:
            com = ndimage.center_of_mass(binary[i])[0]
            centroids.append([i, com])
    centroids = np.array(centroids)

    # Filter all elements near the center (center_offset) and get the maximum distance shared by both tails to have a symmetrical result.
    centroids = centroids[(centroids[:, 0] >= int(h // 2 - np.min(np.abs(centroids[[0, -1], 0] - h // 2))))
                          & (centroids[:, 0] <= int(h // 2 + np.min(np.abs(centroids[[0, -1], 0] - h // 2))))]
    centroids = centroids[(np.abs(centroids[:,0] - h//2) > center_offset)]

    # Define x, y from the filtered list of points
    xx = centroids[:, 0]
    yy = centroids[:, 1]

    # Fit a straight line to the list of points
    coeffs, cov = np.polyfit(xx - h/2, yy - w/2, deg=deg, cov=True)

    return coeffs, cov, binary, centroids


def get_grid_angle(videofile, calibfile, method, plot=False, **kwargs):
    """
    Calculate the relative angle of a laser trace seen by a static camera recording a video.
    
    Parameters: 
    - videofile (str): Name of video file containing the frames..
    - calibfile (int): Camera calibration file containing calibration parameters. 
    - method (str): Method used for fitting. "binary": fits a complete binary image, with enhanced tails.
            "centroids": fits line centers per image row.
    - plot (bool): Enable plotting figures to show binarized images. 
    - kwargs (dict): Dictionary containing optional parameters used in image filtering and other functions.
    
    Returns: line coefficients (numpy.ndarray), covariance matrix (numpy.ndarray), 
        center of the laser trace (tuple), original averaged image (numpy.ndarray),
        binarized image (numpy.ndarray) and list of (x,y) elements (numpy.ndarray/None).
    """
    # Determine global arguments
    mask_width = kwargs.get("mask_width", 150)
    kwargs.get('threshold', 150)
    mask_height = kwargs.get("mask_height", 1000)
    deg = kwargs.get("deg", 2)

    # Get image from video
    frames = get_frames_from_video(videofile, mono=False)
    
    # Merge all frames to get an average of them
    tic = time()
    img0=np.sum(np.mean(frames, axis=(0)), axis=2)
    toc = time()
    print("It took", toc-tic, "seconds")

    # Get height and width of the image
    h, w = img0.shape

    # Undistort image
    cam = get_camera_params(calibfile)
    camera_matrix = cam.cam_matrix()
    dist_coeff = cam.dist_coeff()
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeff, (w, h), 1, (w, h))
    img0 = cv2.undistort(img0, camera_matrix, dist_coeff, None, new_camera_matrix)

    # If the image is not vertical, rotate it
    if w > h:
        img0 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Change this value to increase or decrease the horizontal (vertical) range where the laser trace would be in the image
    center = get_line_center(img0)

    # Print the image indicating the center of the laser
    if plot:
        plt.figure()
        plt.imshow(img0)
        plt.axhline(y=center[0], linestyle='--', linewidth=0.5, color='red')
        plt.axvline(x=center[1], linestyle='--', linewidth=0.5, color='red')
        plt.title("Averaged undistorted image")

    # Create a mask around the centroid found
    img_cut = img0[int(center[0])-mask_height:int(center[0])+mask_height,
                   int(center[1])-mask_width:int(center[1])+mask_width]
    
    # Print the image indicating the center of the laser
    if plot:
        plt.figure()
        plt.imshow(img_cut)
        plt.title("Laser centered image")

    # Select method and determine specific variables for each of them
    if method == "centroids":
        bthr = kwargs.get("bthr", 10)
        min_size = kwargs.get("min_size", 20)
        center_offset = kwargs.get("center_offset", 125)
        coeffs, cov, binary, xy = fit_centroids(img_cut, deg=deg, center_offset=center_offset,
                                        bthr=bthr, min_size=min_size)
    elif method == "binary":
        bthr = kwargs.get("bthr", 100)
        min_size = kwargs.get("min_size", 5)
        offset = kwargs.get("offset", 5000)
        enhance_pow = kwargs.get("enhance_pow", 2)
        coeffs, cov, binary = fit_line(img_cut, deg=deg, enhance_pow=enhance_pow, offset=offset,
                                       bthr=bthr, min_size=min_size)
        xy=None
    else:
        raise ValueError('No available fitting method has been selected')
    
    if plot:
        plt.figure()
        plt.imshow(binary)
        try: 
            plt.plot(xy[:,1], xy[:,0], '.')
        except: pass
        plt.title("Binarized laser trace image")
        plt.show()

    # POSITIVE ANGLE MEANS ROTATING CCW AS SEEN BY THE CAMERA (IMAGE)
    angle = np.arctan(coeffs[-2]) * 180 / np.pi
    # ERROR ESTIMATION
    sig_diag = np.sqrt(np.diag(cov))
    sig_angle = 1 / (1 + coeffs[-2] ** 2) * sig_diag[-2] * 180 / np.pi

    print("The grid angle is %7.4f +/- %6.4f degrees rotating CCW around the line of sight of the camera" % (
    angle, sig_angle))

    return coeffs, cov, center, img0, binary, xy


def plot_line_fit(coeffs, center, img0, alpha=10, beta=0, filename=None):
    """
    Plot the resulting fit line on top of the original image 
    """
    h, w = img0.shape
    x_fit = np.arange(h) - center[0]
    y_fit = np.polyval(coeffs, x_fit)
    plt.matshow(cv2.convertScaleAbs(img0, alpha=alpha, beta=beta))
    plt.plot(y_fit + center[1], x_fit + center[0], color="r", linestyle=":")
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()



if __name__ == '__main__':
    # Initialize parser
    parser = argparse.ArgumentParser(
        description='Obtains the relative angle between the camera and polarized wave grid from a series of pictures.')
    parser.add_argument('videofile', type=str, help='Name of video file containing the frames.')
    parser.add_argument('calibfile', type=str, metavar='file',
                        help='Name of the file containing calibration results (*.txt), '
                             'for point reprojection and/or initial guess during calibration.')
    parser.add_argument('method', choices=['binary', 'centroids'],
                        help='Method used for fitting. "centroids": fits line centers per image row.'
                             '"binary": fits a complete binary image, with enhanced tails.')
    parser.add_argument('-p', '--plot', action='store_true', default=False,
                         help='Show plots for debugging.')
    
    # global arguments
    parser.add_argument('-mw', '--mask_width', type=int, 
                        help="Maximum width of image from center to sides (Default: 150).")
    parser.add_argument('-mh', '--mask_height', type=int, 
                        help="Maximum height of image from center to top and bottom (Default: 1000).")
    parser.add_argument('-deg', '--polynomial_degree', type=int, dest='deg',
                        help="Polynomial degree of the fit (Default: 2).")
    parser.add_argument('-bthr', '--binary_threshold', type=int, dest='bthr',
                        help="Threshold value to filter the laser trace (Default: 100 (bin) & 10 (centroid)).")
    parser.add_argument('-ms', '--min_size', type=int, 
                        help="Minimum area size allowed to stay in binarized image with laser trace (Default: 5 (bin) & 20 (centroid)).")
    
    # binary exclusive arguments
    parser.add_argument('-off', '--offset', type=int, help="(Default: 5000)")
    parser.add_argument('-pow', '--enhance_pow', type=int, help="(Default: 2)")
    
    # centroids exclusive arguments
    parser.add_argument('-coff', '--center_offset', type=int, 
                        help="Number of elements to skip around the laser center in vertical direction. (Default: 125).")

    # Get parse data
    args = parser.parse_args()

    keys = ['mask_width', 'mask_height', 'deg', 'bthr', 'min_size', 'offset', 'enhance_pow', 'center_offset']
    kwargs = {k: getattr(args, k) for k in keys if getattr(args, k) is not None}
    
    # Main
    coeffs, cov, center, img0, binary, xy = get_grid_angle(args.videofile, args.calibfile, args.method, args.plot, **kwargs)
    
    if args.plot:
        plot_line_fit(coeffs, center, img0, alpha=10, beta=0)
        plot_line_fit(coeffs, np.array(binary.shape)/2, binary, alpha=1, beta=0)