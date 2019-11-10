from typing import (Dict)

import numpy as np
import matplotlib.pyplot as plt

def show_history(name: str, history: Dict) -> None:
    epochs = len(history['loss'])
    
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_title('Loss and accuracy for {}'.format(name))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.set_xticks(range(epochs))
    ax1.plot(history['loss'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()  

    color = 'tab:blue'
    # we already handled the x-label with ax1
    ax2.set_ylabel('Accuracy (%)', color=color)
    ax2.plot(history['accuracy'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # otherwise the right y-label is slightly clipped
    fig.tight_layout()

    return plt


def visualize_labels(labels, fig_width=15, fig_height=12):
    """Visualize the labels prepared from the reference images."""
    fig = plt.figure(figsize=(fig_width, fig_height))
    a = fig.add_subplot(1, 1, 1)
    values = np.unique(labels.ravel())
    im = plt.imshow(labels[:, :, 0])
    a.set_title("Labeled image")
    # get the colors of the values, according to the
    # colormap used by imshow
    colors = [im.cmap(im.norm(value)) for value in values]
    # create a patch (proxy artist) for every color
    labels = ["Other", "Informal Settlements"]
    patches = [mpatches.Patch(color=colors[i], label=j)
               for i, j in zip(range(len(values)), labels)]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(
        1.05, 1), loc=2, borderaxespad=0.)


def visualize_data(data, title, fig_width=15, fig_height=12):
    """Visualize the satellite image data."""
    # visualize only RGB bands
    data = data[:, :, 0:-1]
    _ = data[:, :, 0].copy()
    data[:, :, 0] = data[:, :, 2]
    data[:, :, 2] = _
    data = data.astype(np.float)
    # perform stretching for better visualization
    for i in range(data.shape[2]):
        p2, p98 = np.percentile(data[:, :, i], (2, 98))
        data[:, :, i] = exposure.rescale_intensity(data[:, :, i],
                                                   in_range=(p2, p98))
    fig = plt.figure(figsize=(fig_width, fig_height))
    a = fig.add_subplot(1, 1, 1)
    a.set_title(title)
    plt.imshow(data)


history = {'loss': [1.1727547984064361, 0.8204017723048175,
                    0.8189218073715399], 'accuracy': [0.8770104, 0.9415306, 0.9437733]}

show('Ivan', history).show()
