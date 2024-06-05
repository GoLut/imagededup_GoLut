#this is a snapshot taken from the imagededup library and is used to plot the images in the image directory
#more specifically it lets you manually pick the images that you want to keep and discard

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib import figure
from pathlib import Path, PurePath
from typing import Dict, Union, List
from matplotlib.widgets import Button

import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener
register_heif_opener()


def _formatter(val: Union[int, np.float32]):
    """
    For printing floats only upto 3rd precision. Ints are unchanged.
    """
    if isinstance(val, np.float32):
        return f'{val:.3f}'
    else:
        return val


def _plot_images(
    image_dir: PurePath,
    orig: str,
    image_list: List,
    scores: bool = False,
    outfile: str = None,
) -> None:
    """
    Plotting function for plot_duplicates() defined below.

    Args:
        image_dir: image directory where all files in duplicate_map are present.
        orig: filename for which duplicates are to be plotted.
        image_list: List of duplicate filenames, could also be with scores (filename, score).
        scores: Whether only filenames are present in the image_list or scores as well.
        outfile:  Name of the file to save the plot.
    """

    n_ims = len(image_list) -1 #skip the self image
    ncols = 4  # fixed for a consistent layout
    nrows = int(np.ceil(n_ims / ncols)) + 1
    fig = figure.Figure(figsize=(10, 14))

    #buttorn writes to list
    output_data = []

    # Define the button's on click event
    def keep_figure(event):
        print('keeping the figure')
        output_data.append(2)
        plt.close("all")
    
    def discard_figure(event):
        print('discarding the figure')
        output_data.append(1)
        plt.close("all")

    # Add a subplot for the button
    ax_button_0 = plt.axes([0.7, 0.01, 0.1, 0.075])
    ax_button_1 = plt.axes([0.82, 0.01, 0.1, 0.075])

    button0 = Button(ax_button_0, 'Discard')
    button0.on_clicked(discard_figure)

    button1 = Button(ax_button_1, 'Keep')
    button1.on_clicked(keep_figure)

    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols)
    ax = plt.subplot(
        gs[0, 1:3]
    )  # Always plot the original image in the middle of top row
    ax.imshow(Image.open(image_dir / orig))
    ax.set_title('Original Image: {}'.format(orig))
    ax.axis('off')
    # plt.ion()
    for i in range(0, n_ims): #skip the self image
        row_num = (i // ncols) + 1
        col_num = i % ncols

        ax = plt.subplot(gs[row_num, col_num])
        if scores:
            img = Image.open(image_dir / image_list[i+1][0])
            #give_the_image overlay
            if image_list[i+1][2] == 2:
                overlay = Image.new('RGB', img.size, color='green')
                img = Image.blend(img, overlay, alpha=0.5)
            if image_list[i+1][2] == 1:
                overlay = Image.new('RGB', img.size, color='red')
                img = Image.blend(img, overlay, alpha=0.5)
            ax.imshow(img)
            val = _formatter(image_list[i+1][1])
            title = ' '.join([image_list[i+1][0], f'({val})'])
        else:
            img = Image.open(image_dir / image_list[i+1][0])
            #give_the_image overlay
            if image_list[i+1][2] == 2:
                overlay = Image.new('RGB', img.size, color='green')
                img = Image.blend(img, overlay, alpha=0.5)
            if image_list[i+1][2] == 1:
                overlay = Image.new('RGB', img.size, color='red')
                img = Image.blend(img, overlay, alpha=0.5)
            ax.imshow(img)
            # ax.imshow(Image.open(image_dir / image_list[i]))
            title = image_list[i+1]

        ax.set_title(title, fontsize=6)
        ax.axis('off')
    gs.tight_layout(fig)

    if outfile:
        plt.savefig(outfile)
    plt.show()
    return output_data[0]


def _validate_args(
    image_dir: Union[PurePath, str], duplicate_map: Dict, filename: str
) -> PurePath:
    """Argument validator for plot_duplicates() defined below.
    Return PurePath to the image directory"""

    image_dir = Path(image_dir)
    assert (
        image_dir.is_dir()
    ), 'Provided image directory does not exist! Please provide the image directory where all files are present!'

    if not isinstance(duplicate_map, dict):
        raise ValueError('Please provide a valid Duplicate map!')
    if filename not in duplicate_map.keys():
        raise ValueError(
            'Please provide a valid filename present as a key in the duplicate_map!'
        )
    return image_dir


def filter_linkely_duplicates_manualy(
    image_dir: Union[PurePath, str],
    duplicate_map: Dict,
    filename: str,
    outfile: str = None,
) -> None:
    """
    Given filename for an image, plot duplicates along with the original image using the duplicate map obtained using
    find_duplicates method.

    Args:
        image_dir: image directory where all files in duplicate_map are present.
        duplicate_map: mapping of filename to found duplicates (could be with or without scores).
        filename: Name of the file for which duplicates are to be plotted, must be a key in the duplicate_map.
        dictionary.
        outfile: Optional, name of the file to save the plot. Default is None.

    Example:
    ```
        from imagededup.utils import plot_duplicates
        plot_duplicates(image_dir='path/to/image/directory',
                        duplicate_map=duplicate_map,
                        filename='path/to/image.jpg')
    ```
    """
    # validate args
    image_dir = _validate_args(image_dir=image_dir, duplicate_map=duplicate_map, filename=filename)

    retrieved = duplicate_map[filename]
    assert len(retrieved) != 0, 'Provided filename has no duplicates!'

    # plot
    output = None
    if isinstance(retrieved[0], tuple):
        output = _plot_images(
            image_dir=image_dir,
            orig=filename,
            image_list=retrieved,
            scores=True,
            outfile=outfile,
        )

    else:
        output = _plot_images(
            image_dir=image_dir,
            orig=filename,
            image_list=retrieved,
            scores=False,
            outfile=outfile,
        )
    print("output: ", output)
    return output