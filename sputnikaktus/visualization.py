from typing import (Tuple, Iterable, Dict, Union)
import sys
import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.pyplot import Subplot

from rasterio.plot import (reshape_as_image)
from skimage import exposure

from sputnikaktus.utils import hist_stretch_image

FigSplt = Tuple[Figure, Subplot]
Legend = Iterable[str]


def show_history(name: str, history: Dict, params: Dict = None) -> None:
    epochs = len(history['loss'])

    fig, ax1 = plt.subplots()
    plt.suptitle('Loss and accuracy for {}'.format(name), y=1)

    if params:
        descr = 'trained on {samples} samples with batch size of {batch_size} for {epochs} epochs'.format(**params)
        ax1.set_title(descr, fontsize=10, y=1)

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.set_xticks(range(epochs))

    legend_handles = []
    
    if 'loss' in history:
        l1 = ax1.plot(history['loss'], color=color, label='Loss')
        legend_handles += l1
    
    if 'val_loss' in history:
        l2 = ax1.plot(history['val_loss'], color=color, linestyle='--', label='Validation loss')
        legend_handles += l2

    ax1.tick_params(axis='y', labelcolor=color)

    # instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()

    color = 'tab:blue'
    # we already handled the x-label with ax1
    ax2.set_ylabel('Accuracy (%)', color=color)
    
    if 'accuracy' in history:
        l3 = ax2.plot(history['accuracy'], color=color, label='Accuracy')
        legend_handles += l3

    if 'val_accuracy' in history:
        l4 = ax2.plot(history['val_accuracy'], color=color, linestyle='--', label='Validation accuracy')
        legend_handles += l4
        
    ax2.tick_params(axis='y', labelcolor=color)

    plt.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.1),
               fancybox=True, shadow=True, ncol=4)

    # otherwise the right y-label is slightly clipped
    fig.tight_layout()

    return plt


def get_fig_splt(figsplt: Union[bool, FigSplt]):
    if figsplt is None:
        fig = plt.figure()
        splt = fig.add_subplot(1, 1, 1)
    elif figsplt == False:
        fig = plt.figure()
        splt = None
    else:
        fig, splt = figsplt

    return fig, splt


def visualize_label(
    img,
    title: str = '',
    filename: str = None,
    legend: Legend = None,
    figsplt: FigSplt = None
):
    """Visualize the lbl prepared from the reference images."""

    values = np.unique(img.ravel())

    fig, splt = get_fig_splt(figsplt)

    splt.set_title(title)
    iplt = plt.imshow(np.squeeze(img))

    if legend:
        assert len(legend) >= len(values), \
            'Legend items should not be less than values'

        # get the colors of the values, according to the colormap used by imshow
        colors = [iplt.cmap(iplt.norm(value)) for value in values]

        # create a patch (proxy artist) for every color
        patches = [mpatches.Patch(color=colors[i], label=j)
                   for i, j in zip(range(len(values)), legend)]

        # put those patched as legend-handles into the legend
        plt.legend(handles=patches, bbox_to_anchor=(
            1.05, 1), loc=2, borderaxespad=0.)


def visualize_image(
    img,
    title: str = '',
    filename: str = None,
    rgb: Tuple[int, int, int] = (0, 1, 2),
    figsplt: FigSplt = None,
    stretch: bool = True,
    show: bool = False,
    as_image: bool = False
):
    """Visualize the satellite image data."""
    assert len(rgb) == 3, 'Exactly three bands should be passed as rgb'
    assert img.dtype in (np.uint8, np.float32, np.float64), 'RGB images can be only certain types in matplotlib'

    if as_image:
        img = reshape_as_image(img)

    if stretch:
        img = hist_stretch_image(img)
    else:
        img = hist_stretch_image(img, clip_extremes=0)


    fig, splt = get_fig_splt(figsplt)

    splt.set_title(title)
    splt.imshow(img)

    if filename:
        fig.savefig(filename)

    if show:
        plt.show()

    return splt


def visualize_pair(
    img_raw,
    img_lbl,
    legend: Legend = None,
    title: str = None,
    title_raw: str = None,
    title_lbl: str = None,
    filename: str = None,
    rgb: Tuple[int, int, int] = (0, 1, 2),
    figsplt: FigSplt = None
):
    fig, splt = get_fig_splt(figsplt)

    splt = fig.add_subplot(1, 2, 1)

    visualize_image(img_raw, title=title_raw, rgb=rgb, figsplt=(fig, splt))

    splt = fig.add_subplot(1, 2, 2)
    visualize_label(img_lbl, title=title_lbl,
                    legend=legend, figsplt=(fig, splt))

    if filename:
        fig.savefig(filename)


def visualize_pairs(
    imgs_raw,
    imgs_lbl,
    legend: Legend,
    title: str = None,
    title_raw: str = None,
    title_lbl: str = None,
    rgb: Tuple[int, int, int] = (0, 1, 2),
    index: int = None
):
    assert isinstance(index, int) and index >= 0

    state = {
        'current': 0,
        'last': 0,
        'total': imgs_raw.shape[0],
        'input': [],
    }

    def press(event):
        number_keys = [str(k) for k in range(0, 9)]
        sys.stdout.flush()
        current = state['current']

        if event.key == 'ctrl+n':
            state['last'] = state['current']
            state['current'] += 1
            state['current'] = state['current'] % state['total']

        if event.key == 'ctrl+p':
            state['last'] = state['current']
            state['current'] -= 1
            state['current'] = state['current'] if state['current'] >= 0 else state['total'] - 1

        if event.key == 'ctrl+l':
            state['last'], state['current'] = state['current'], state['last']

        if event.key == 'ctrl+r':
            state['last'] = state['current']
            number = random.randint(0, state['total'])
            state['current'] = number

        if event.key == 'enter' and len(state['input']) > 0:
            number = ''.join(state['input'])
            state['input'] = []
            state['last'] = state['current']
            state['current'] = int(number) % state['total']

        if event.key in number_keys:
            state['input'].append(event.key)

        if event.key == 'escape':
            plt.close(fig)
            return

        if current != state['current']:
            visualize_pair(imgs_raw[state['current']], imgs_lbl[state['current']], legend=legend,
                           title=title, title_raw=title_raw, title_lbl=title_lbl, figsplt=(fig, splt))
            fig.canvas.draw()

    fig, splt = get_fig_splt(False)
    fig.canvas.mpl_connect('key_press_event', press)

    visualize_pair(imgs_raw[state['current']], imgs_lbl[state['current']], legend=legend,
                   title=title, title_raw=title_raw, title_lbl=title_lbl, figsplt=(fig, splt))

    plt.show()
