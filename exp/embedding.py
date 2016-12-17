# Image embedding by Sean Bell
import math

import numpy as np
from PIL import Image

from pilkit.processors import ResizeCanvas, ResizeToFill, ResizeToFit
from pilkit.utils import open_image
from tsne import bh_sne


def apply_tsne(features):
    print 'Computing embedding using T-SNE...'
    coords = bh_sne(features.astype(np.float64))

    return coords


def embedding_to_image_grid(photos, coords, **kwargs):
    '''Creates an embedding image grid for the specified photos with coords
    coordinates.

    :param photos: A list of photos. These can be filenames or objects which
    can be opened using "open_image"

    :param coords: A list of (x, y) coordinate tuples with the same order as
    the corresponding images in photos.

    :returns: A PIL image containing the embedding image grid.
    '''
    items = [
        (y, x, photo)
        for photo, (x, y) in zip(photos, coords)
    ]

    # rescale and clip to grid
    min_r, max_r = min(t[0] for t in items), max(t[0] for t in items)
    min_c, max_c = min(t[1] for t in items), max(t[1] for t in items)
    scale_r = math.sqrt(len(items)) / (max_r - min_r)
    scale_c = math.sqrt(len(items)) / (max_c - min_c)
    items = [
        (int((t[0] - min_r) * scale_r), int((t[1] - min_c) * scale_c), t[2])
        for t in items
    ]

    return create_image_grid(items, **kwargs)


def create_image_grid(items, thumb_width=500, thumb_height=500, padding=0,
        background=(255, 255, 255), resize_to_fill=False):
    """ Greate a grid of images.

   :param items:  iterable of [(row, col, image), ...]
   """

    # adjust so that the top left is (0, 0)
    min_row = min(t[0] for t in items)
    min_col = min(t[1] for t in items)
    items = [(row - min_row, col - min_col, image) for (row, col, image) in items]

    # find bottom-right
    nrows = 1 + max(t[0] for t in items)
    ncols = 1 + max(t[1] for t in items)

    # output canvas
    cell_width = thumb_width + padding
    cell_height = thumb_height + padding
    size = (ncols * cell_width, nrows * cell_height)
    print 'Creating image grid...'
    print 'nrows:', nrows
    print 'ncols:', ncols
    print 'size:', size
    out = Image.new(mode='RGB', size=size, color=background)

    # splat images
    filled = np.zeros((nrows, ncols), dtype=np.bool)
    for (row, col, image) in items:
        if filled[row, col]:
            continue
        filled[row, col] = True

        try:
            if isinstance(image, basestring):
                thumb = Image.open(image)
            else:
                thumb = open_image(image)
        except Exception as e:
            print e
            continue

        if resize_to_fill:
            thumb = ResizeToFill(thumb_width, thumb_height).process(thumb)
        else:
            thumb = ResizeToFit(thumb_width, thumb_height).process(thumb)
            thumb = ResizeCanvas(thumb_width, thumb_height, color=background).process(thumb)

        x = col * cell_width
        y = row * cell_height
        out.paste(thumb, box=(x, y, x + thumb_width, y + thumb_height))

    return out
