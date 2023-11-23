from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
import numpy as np


def plot_confusion_matrix(
    conf_mat: np.ndarray,
    hide_spines: bool = False,
    hide_ticks: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: Optional[Colormap] = None,
    colorbar: bool = False,
    show_absolute: bool = True,
    show_normed: bool = False,
    norm_colormap: Optional[str] = None,
    class_names: Optional[List[str]] = None,
    figure: Optional[Figure] = None,
    axis: Optional[Axes] = None,
    fontcolor_threshold: float = 0.5,
) -> Tuple[Figure, Axes]:
    """Plot a confusion matrix via matplotlib.

    Parameters
    -----------
    conf_mat : array-like, shape = [n_classes, n_classes]
        Confusion matrix from evaluate.confusion matrix.

    hide_spines : bool (default: False)
        Hides axis spines if True.

    hide_ticks : bool (default: False)
        Hides axis ticks if True

    figsize : tuple (default: (2.5, 2.5))
        Height and width of the figure

    cmap : matplotlib colormap (default: `None`)
        Uses matplotlib.pyplot.cm.Blues if `None`

    colorbar : bool (default: False)
        Shows a colorbar if True

    show_absolute : bool (default: True)
        Shows absolute confusion matrix coefficients if True.
        At least one of  `show_absolute` or `show_normed`
        must be True.

    show_normed : bool (default: False)
        Shows normed confusion matrix coefficients if True.
        The normed confusion matrix coefficients give the
        proportion of training examples per class that are
        assigned the correct label.
        At least one of  `show_absolute` or `show_normed`
        must be True.

    norm_colormap : bool (default: False)
        Matplotlib color normalization object to normalize the
        color scale, e.g., `matplotlib.colors.LogNorm()`.

    class_names : array-like, shape = [n_classes] (default: None)
        List of class names.
        If not `None`, ticks will be set to these values.

    figure : None or Matplotlib figure  (default: None)
        If None will create a new figure.

    axis : None or Matplotlib figure axis (default: None)
        If None will create a new axis.

    fontcolor_threshold : Float (default: 0.5)
        Sets a threshold for choosing black and white font colors
        for the cells. By default all values larger than 0.5 times
        the maximum cell value are converted to white, and everything
        equal or smaller than 0.5 times the maximum cell value are converted
        to black.

    Returns
    -----------
    fig, ax : matplotlib.pyplot subplot objects
        Figure and axis elements of the subplot.

    Examples
    -----------
    For usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/plotting/plot_confusion_matrix/

    """
    if not (show_absolute or show_normed):
        raise AssertionError("Both show_absolute and show_normed are False")
    if class_names is not None and len(class_names) != len(conf_mat):
        raise AssertionError(
            "len(class_names) should be equal to number of" "classes in the dataset"
        )

    total_samples = conf_mat.sum(axis=1)[:, np.newaxis]
    normed_conf_mat = conf_mat.astype("float") / total_samples

    if figure is None and axis is None:
        fig, ax = plt.subplots(figsize=figsize)
    elif axis is None:
        fig = figure
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig, ax = figure, axis

    ax.grid(False)
    if cmap is None:
        cmap = plt.cm.Blues

    if figsize is None:
        figsize = (len(conf_mat) * 1.25, len(conf_mat) * 1.25)

    if show_normed:
        matshow = ax.matshow(normed_conf_mat, cmap=cmap, norm=norm_colormap)
    else:
        matshow = ax.matshow(conf_mat, cmap=cmap, norm=norm_colormap)

    if colorbar:
        fig.colorbar(matshow)

    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            cell_text = ""
            if show_absolute:
                num = conf_mat[i, j].astype(np.int64)
                cell_text += format(num, "d")
                if show_normed:
                    cell_text += "\n" + "("
                    cell_text += format(normed_conf_mat[i, j], ".2f") + ")"
            else:
                cell_text += format(normed_conf_mat[i, j], ".2f")

            if show_normed:
                ax.text(
                    x=j,
                    y=i,
                    s=cell_text,
                    va="center",
                    ha="center",
                    color=(
                        "white"
                        if normed_conf_mat[i, j] > 1 * fontcolor_threshold
                        else "black"
                    ),
                )
            else:
                ax.text(
                    x=j,
                    y=i,
                    s=cell_text,
                    va="center",
                    ha="center",
                    color="white"
                    if conf_mat[i, j] > np.max(conf_mat) * fontcolor_threshold
                    else "black",
                )
    if class_names is not None:
        tick_marks = np.arange(len(class_names))
        plt.xticks(
            tick_marks, class_names, rotation=45, ha="right", rotation_mode="anchor"
        )
        plt.yticks(tick_marks, class_names)

    if hide_spines:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    if hide_ticks:
        ax.axes.get_yaxis().set_ticks([])
        ax.axes.get_xaxis().set_ticks([])

    plt.xlabel("predicted label")
    plt.ylabel("true label")
    return fig, ax


def plot_beautify(
    conf_mat: np.ndarray, class_names: List[str], out_path: Optional[str] = None
) -> None:
    """Plots a confusion matrix with the configured parameters."""
    # binary_25samples = np.array([[23, 23], [2, 298]])

    # binary_36samples = np.array([[28, 50], [8, 271]])

    # binary_combined = np.array([[52, 50], [9, 271]])

    # class_names = ["suspicious", "healthy"]
    plt.rcParams["axes.labelweight"] = "bold"

    SMALLER_SIZE = 18
    SMALL_SIZE = 24
    BIGGER_SIZE = 48
    plt.rc("font", size=BIGGER_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=SMALL_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALLER_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALLER_SIZE)  # fontsize of the tick labels
    # plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    # plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    fig, ax = plot_confusion_matrix(
        conf_mat=conf_mat, class_names=class_names, figsize=(9, 9)
    )
    # plt.xlabel('true label')
    # plt.ylabel('predicted label')
    plt.xticks(ha="center")
    plt.yticks(va="center")
    plt.xticks(rotation=0)
    plt.yticks(rotation=90)
    if out_path is not None:
        plt.savefig(out_path, bbox_inches="tight", dpi=100)
    else:
        plt.tight_layout()
        plt.show()
