import cv2
import glob
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import random
from PIL import Image
import os
import csv
import numpy
import matplotlib
from matplotlib import pyplot, image

# sources from https://github.com/TobiasRoeddiger/GazePointHeatMap
def draw_display(dispsize, imagefile=None, ax=None):
    """Returns a matplotlib.pyplot Figure and its axes, with a size of
    dispsize, a black background colour, and optionally with an image drawn
    onto it
    arguments
    dispsize		-	tuple or list indicating the size of the display,
                    e.g. (1024,768) width, height
    keyword arguments
    imagefile		-	full path to an image file over which the heatmap
                    is to be laid, or None for no image; NOTE: the image
                    may be smaller than the display size, the function
                    assumes that the image was presented at the centre of
                    the display (default = None)
    returns
    fig, ax		-	matplotlib.pyplot Figure and its axes: field of zeros
                    with a size of dispsize, and an image drawn onto it
                    if an imagefile was passed
    """

    # construct screen (black background)
    screen = numpy.zeros((dispsize[1], dispsize[0], 3), dtype='uint8')
    # if an image location has been passed, draw the image
    if imagefile != None:
        # check if the path to the image exists
        if not os.path.isfile(imagefile):
            raise Exception("ERROR in draw_display: imagefile not found at '%s'" % imagefile)
        # load image
        img = image.imread(imagefile)
        img = img[..., :3]

        # width and height of the image
        w, h = len(img[0]), len(img)
        # x and y position of the image on the display
        x = int(dispsize[0] / 2) - int(w / 2)
        y = int(dispsize[1] / 2) - int(h / 2)
        # draw the image on the screen
        screen[y:y + h, x:x + w, :] += img
    # dots per inch
    dpi = 100.0
    # determine the figure size in inches
    figsize = (dispsize[0] / dpi, dispsize[1] / dpi)
    
    # create a figure
    if ax is not None:
        fig =None
        ax.set_axis_off()
    else:
        fig = pyplot.figure(figsize=figsize, dpi=dpi, frameon=False)
        ax = pyplot.Axes(fig, [0, 0, 1, 1])
        ax.set_axis_off()
        fig.add_axes(ax)
        
    # plot display
    ax.axis([0, dispsize[0], dispsize[1], 0])
    ax.imshow(screen) #, origin='lower'

    return fig, ax

def gaussian(x, sx, y=None, sy=None):
    """Returns an array of numpy arrays (a matrix) containing values between
    1 and 0 in a 2D Gaussian distribution
    arguments
    x		-- width in pixels
    sx		-- width standard deviation
    keyword argments
    y		-- height in pixels (default = x)
    sy		-- height standard deviation (default = sx)
    """

    # square Gaussian if only x values are passed
    if y == None:
        y = x
    if sy == None:
        sy = sx
    # centers
    xo = int(x / 2)
    yo = int(y / 2)
    # matrix of zeros
    M = numpy.zeros([y, x], dtype=float)
    # gaussian matrix
    for i in range(x):
        for j in range(y):
            M[j, i] = numpy.exp(
                -1.0 * (((float(i) - xo) ** 2 / (2 * sx * sx)) + ((float(j) - yo) ** 2 / (2 * sy * sy))))

    return M

def draw_heatmap(
    gazepoints,
    dispsize,
    imagefile=None,
    alpha=0.5,
    savefilename=None,
    gaussianwh=200,
    gaussiansd=None,
    ax=None
):
    """Draw human FDM heatmap and blend with original image for consistency."""

    # === Step 1: Create raw heatmap from gaze points === #
    gwh = gaussianwh
    gsdwh = int(gwh / 6) if (gaussiansd is None) else gaussiansd
    gaus = gaussian(gwh, gsdwh)

    strt = int(gwh / 2)
    heatmapsize = dispsize[1] + 2 * strt, dispsize[0] + 2 * strt
    heatmap = np.zeros(heatmapsize, dtype=float)

    for x, y, duration in gazepoints:
        x = strt + x - int(gwh / 2)
        y = strt + y - int(gwh / 2)

        if (0 < x < dispsize[0]) and (0 < y < dispsize[1]):
            heatmap[y:y + gwh, x:x + gwh] += gaus * duration
        else:
            # Handle partially out-of-bound fixations
            hadj = [0, gwh]
            vadj = [0, gwh]
            if x < 0:
                hadj[0] = abs(x)
                x = 0
            elif x > dispsize[0]:
                hadj[1] = gwh - int(x - dispsize[0])
            if y < 0:
                vadj[0] = abs(y)
                y = 0
            elif y > dispsize[1]:
                vadj[1] = gwh - int(y - dispsize[1])
            try:
                heatmap[y:y + vadj[1], x:x + hadj[1]] += gaus[vadj[0]:vadj[1], hadj[0]:hadj[1]] * duration
            except:
                pass

    heatmap = heatmap[strt:dispsize[1] + strt, strt:dispsize[0] + strt]

    # === Step 2: Normalize heatmap === #
    heatmap = np.nan_to_num(heatmap)  # remove NaNs
    if heatmap.max() > 0:
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # === Step 3: Load and resize original image === #
    bg = cv2.imread(str(imagefile))
    bg = cv2.resize(bg, (dispsize[0], dispsize[1]))

    # === Step 4: Blend and display === #
    blended = cv2.addWeighted(heatmap_color, alpha, bg, 1 - alpha, 0)

    if ax is not None:
        ax.set_axis_off()
        ax.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    else:
        plt.figure(figsize=(dispsize[0] / 100, dispsize[1] / 100), dpi=100)
        plt.axis('off')
        plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
        if savefilename:
            plt.savefig(savefilename, bbox_inches='tight', pad_inches=0)
            plt.close()

    return blended


# def draw_heatmap(gazepoints, dispsize, imagefile=None, alpha=0.5, savefilename=None, gaussianwh=200, gaussiansd=None, ax=None):
#     """Draws a heatmap of the provided fixations, optionally drawn over an
#     image, and optionally allocating more weight to fixations with a higher
#     duration.
#     arguments
#     gazepoints		-	a list of gazepoint tuples (x, y)
    
#     dispsize		-	tuple or list indicating the size of the display,
#                     e.g. (1024,768)
#     keyword arguments
#     imagefile		-	full path to an image file over which the heatmap
#                     is to be laid, or None for no image; NOTE: the image
#                     may be smaller than the display size, the function
#                     assumes that the image was presented at the centre of
#                     the display (default = None)
#     alpha		-	float between 0 and 1, indicating the transparancy of
#                     the heatmap, where 0 is completely transparant and 1
#                     is completely untransparant (default = 0.5)
#     savefilename	-	full path to the file in which the heatmap should be
#                     saved, or None to not save the file (default = None)
#     returns
#     fig			-	a matplotlib.pyplot Figure instance, containing the
#                     heatmap
#     """

#     # IMAGE
#     fig, ax = draw_display(dispsize, imagefile=imagefile, ax=ax)

#     # HEATMAP
#     # Gaussian
#     gwh = gaussianwh
#     gsdwh = int(gwh/6) if (gaussiansd is None) else gaussiansd
#     gaus = gaussian(gwh, gsdwh)
#     # matrix of zeroes
#     strt = int(gwh/2)
#     heatmapsize = dispsize[1] + 2 * strt, dispsize[0] + 2 * strt
#     heatmap = numpy.zeros(heatmapsize, dtype=float)
#     # create heatmap
#     for i in range(0, len(gazepoints)):
#         # get x and y coordinates
#         x = strt + gazepoints[i][0] - int(gwh / 2)
#         y = strt + gazepoints[i][1] - int(gwh / 2)
#         # correct Gaussian size if either coordinate falls outside of display boundaries
#         if (not 0 < x < dispsize[0]) or (not 0 < y < dispsize[1]):
#             hadj = [0, gwh];
#             vadj = [0, gwh]
#             if 0 > x:
#                 hadj[0] = abs(x)
#                 x = 0
#             elif dispsize[0] < x:
#                 hadj[1] = gwh - int(x - dispsize[0])
#             if 0 > y:
#                 vadj[0] = abs(y)
#                 y = 0
#             elif dispsize[1] < y:
#                 vadj[1] = gwh - int(y - dispsize[1])
#             # add adjusted Gaussian to the current heatmap
#             try:
#                 heatmap[y:y + vadj[1], x:x + hadj[1]] += gaus[vadj[0]:vadj[1], hadj[0]:hadj[1]] * gazepoints[i][2]
#             except:
#                 # fixation was probably outside of display
#                 pass
#         else:
#             # add Gaussian to the current heatmap
#             heatmap[y:y + gwh, x:x + gwh] += gaus * gazepoints[i][2]
#     # resize heatmap
#     heatmap = heatmap[strt:dispsize[1] + strt, strt:dispsize[0] + strt]
#     # remove zeros
#     lowbound = numpy.mean(heatmap[heatmap > 0])
#     heatmap[heatmap < lowbound] = numpy.NaN
#     # draw heatmap on top of image
#     ax.imshow(heatmap, cmap='jet', alpha=alpha)

#     # save the figure, not visualize it if a file name was provided
#     if savefilename != None:
#         directory = os.path.dirname(savefilename)
#         if not os.path.exists(directory):
#             # Create a new directory because it does not exist 
#             os.makedirs(directory)
#             print(f"The new directory {directory} is created!")
#         plt.savefig(savefilename)
#         plt.close(fig)
#     else:
#         pass
#         # plt.show()

#     return 