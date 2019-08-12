import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np
from PIL import Image

from monopsr.datasets.kitti import calib_utils, obj_utils


# Window sizes
CV2_SIZE_2_COL = (930, 280)
CV2_SIZE_3_COL = (620, 187)
CV2_SIZE_4_COL = (465, 140)


def plots_from_image(img,
                     subplot_rows=1,
                     subplot_cols=1,
                     display=True,
                     fig_size=None):
    """Forms the plot figure and axis for the visualization

    Args:
        img: image to plot
        subplot_rows: number of rows of the subplot grid
        subplot_cols: number of columns of the subplot grid
        display: display the image in non-blocking fashion
        fig_size: (optional) size of the figure
    """

    def set_plot_limits(axes, image):
        # Set the plot limits to the size of the image, y is inverted
        axes.set_xlim(0, image.shape[1])
        axes.set_ylim(image.shape[0], 0)

    if fig_size is None:
        img_shape = np.shape(img)
        fig_height = img_shape[0] / 100 * subplot_rows
        fig_width = img_shape[1] / 100 * subplot_cols
        fig_size = (fig_width, fig_height)

    # Create the figure
    fig, axes = plt.subplots(subplot_rows, subplot_cols, figsize=fig_size, sharex=True)
    fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, hspace=0.0)

    # Plot image
    if subplot_rows == 1 and subplot_cols == 1:
        # Single axis
        axes.imshow(img)
        set_plot_limits(axes, img)
    else:
        # Multiple axes
        for idx in range(axes.size):
            axes[idx].imshow(img)
            set_plot_limits(axes[idx], img)

    if display:
        plt.show(block=False)

    return fig, axes


def plots_from_sample_name(image_dir,
                           sample_name,
                           subplot_rows=1,
                           subplot_cols=1,
                           display=True,
                           fig_size=(15, 9.15)):
    """Forms the plot figure and axis for the visualization

    Args:
        image_dir: directory of image files in the wavedata
        sample_name: sample name of the image file to present
        subplot_rows: number of rows of the subplot grid
        subplot_cols: number of columns of the subplot grid
        display: display the image in non-blocking fashion
        fig_size: (optional) size of the figure
    """
    sample_name = int(sample_name)

    # Grab image data
    img = np.array(Image.open("{}/{:06d}.png".format(image_dir, sample_name)), dtype=np.uint8)

    # Create plot
    fig, axes = plots_from_image(img, subplot_rows, subplot_cols, display, fig_size)

    return fig, axes


def set_plt_titles(axes, titles):
    axes_flat = axes.flatten()
    for axes, title in zip(axes_flat, titles):
        axes.set_title(title)


def move_plt_figure(fig, x, y):
    """Move matplotlib figure to position
    https://stackoverflow.com/a/37999370

    Args:
        fig: Figure handle
        x: Position x
        y: Position y
    """
    plt_backend = matplotlib.get_backend()
    if plt_backend == 'TkAgg':
        fig.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif plt_backend == 'WXAgg':
        fig.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        fig.canvas.manager.window.move(x, y)


def cv2_imshow(window_name, image,
               size_wh=None, row_col=None, location_xy=None):
    """Helper function for specifying window size and location when
        displaying images with cv2

    Args:
        window_name (string): Window title
        image: image to display
        size_wh: resize window
            Recommended sizes for 1920x1080 screen:
                2 col: (930, 280)
                3 col: (620, 187)
                4 col: (465, 140)
        row_col: Row and column to show images like subplots
        location_xy: location of window
    """

    if size_wh is not None:
        cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(window_name, *size_wh)
    else:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)

    if row_col is not None:
        start_x_offset = 60
        start_y_offset = 25
        y_offset = 28

        subplot_row = row_col[0]
        subplot_col = row_col[1]
        location_xy = (start_x_offset + subplot_col * size_wh[0],
                       start_y_offset + subplot_row * size_wh[1] + subplot_row * y_offset)

    if location_xy is not None:
        cv2.moveWindow(window_name, *location_xy)

    cv2.imshow(window_name, image)


def get_point_colours(points, cam_p, image):
    points_in_im = calib_utils.project_pc_to_image(points.T, cam_p)
    points_in_im_rounded = np.round(points_in_im).astype(np.int32)

    point_colours = image[points_in_im_rounded[1], points_in_im_rounded[0]]

    return point_colours


def draw_obj_as_box_2d(ax, obj, color='g', linewidth=2):
    """Draws the 2D boxes given an ObjectLabel object by
     converting to box_2d format then calling draw_box_2d

    Args:
        ax: subplot handle
        obj: ObjectLabel object
        color: color of box

    """
    box_2d = np.asarray((obj.y1, obj.x1, obj.y2, obj.x2))
    draw_box_2d(ax, box_2d, color, linewidth)


def draw_box_2d(ax, box_2d, color='#90EE900', linewidth=2):
    """Draws 2D boxes given coordinates in box_2d format

    Args:
        ax: subplot handle
        box_2d: ndarray containing box coordinates in box_2d format (y1, x1, y2, x2)
        color: color of box
    """
    box_x1 = box_2d[1]
    box_y1 = box_2d[0]
    box_w = box_2d[3] - box_x1
    box_h = box_2d[2] - box_y1

    rect = patches.Rectangle((box_x1, box_y1),
                             box_w, box_h,
                             linewidth=linewidth,
                             edgecolor=color,
                             facecolor='none')
    ax.add_patch(rect)


def draw_obj_as_box_3d(ax, obj, cam_p, show_orientation=True,
                       color_table=None, line_width=3, double_line=True,
                       box_color=None):
    """Draws the projection of object label as a 3D box

    Args:
        ax: subplot handle
        obj: ObjectLabel
        cam_p: camera projection matrix
        show_orientation: optional, draw a line showing orientation
        color_table: optional, a custom table for coloring the boxes,
            should have 4 values to match the 4 truncation values. This color
            scheme is used to display boxes colored based on difficulty.
        line_width: optional, custom line width to draw the box
        double_line: optional, overlays a thinner line inside the box lines
        box_color: optional, use a custom color for box (instead of
            the default color_table
    """

    corners_3d = obj_utils.compute_obj_label_corners_3d(obj)
    corners, face_idx = obj_utils.project_corners_3d_to_image(corners_3d, cam_p)

    # define colors
    if color_table:
        if len(color_table) != 4:
            raise ValueError('Invalid color table length, must be 4')
    else:
        color_table = ["#00cc00", 'y', 'r', 'w']

    trun_style = ['solid', 'dashed']
    trc = int(obj.truncation > 0.1)

    if len(corners) > 0:
        for i in range(4):
            x = np.append(corners[0, face_idx[i, ]],
                          corners[0, face_idx[i, 0]])
            y = np.append(corners[1, face_idx[i, ]],
                          corners[1, face_idx[i, 0]])

            # Draw the boxes
            if box_color is None:
                box_color = color_table[int(obj.occlusion)]

            ax.plot(x, y, linewidth=line_width,
                    color=box_color,
                    linestyle=trun_style[trc])

            # Draw a thinner second line inside
            if double_line:
                ax.plot(x, y, linewidth=line_width / 3.0, color='b')

    if show_orientation:
        # Compute orientation 3D
        orientation = obj_utils.compute_orientation_3d(obj, cam_p)

        if orientation is not None:
            x = np.append(orientation[0, ], orientation[0, ])
            y = np.append(orientation[1, ], orientation[1, ])

            # draw the boxes
            ax.plot(x, y, linewidth=4, color='w')
            ax.plot(x, y, linewidth=2, color='k')
