import matplotlib.pyplot as plt
from matplotlib.axes import Axes

import numpy as np

# 🎨 Predefined colour palette for plots
COLOUR = [#'#1E90FF', # 0 # Dodgerblue
          '#51ADDA', # 0 # Vivid Sky Blue(LXB)
        # '#FFBF00', # 1 # Amber
        '#E6A817', # 1 # Harvest Gold
        '#FF6347', # 2 # Tomato
        '#00A86B', # 3 # Jade
        '#8A2BE2', # 4 # Blueviolet
        '#FF6FFF', # 5 # Ultra Pink
        '#00CCFF', # 6 # Vivid Sky Blue
        # '#A7FC00', # 7 # Spring Bud
        '#00FF40', # 7 # Erin
        '#FF004F', # 8 # Folly
        '#0063A6', # 9 # Lapis Lazuli
        ]
# Function to get colour by index
def getColour(i):
    return COLOUR[i]


HATCH = [
    "..",
    "//",
    "",
    "",
    "",
    "\\\\",
    "",
    # "oo",
    "",
    "",
    "**",
    # "..",
    # "++",
    # "--",
    # "||",
]

def getHatch(i):
    return HATCH[i % len(HATCH)]

def setMplParam(classNum=10, isVaryLineStyle=True):
    """
    Configure global Matplotlib parameters for consistent and visually clear plots.

    Parameters:
    - classNum (int): Number of classes (colours, markers, styles) to use.
    - isVaryLineStyle (bool): If True, vary line styles across classes; otherwise, use solid lines.

    Effects:
    - Updates Matplotlib's rcParams globally, setting default colours, line styles,
      marker shapes, font sizes, grid settings, and tick formatting.
    """
    # Define effective colors, line styles, and markers based on the class number
    LINE = ['-', '-.', '--', '-.', ':','--','-.','-', ':', '--']
    MARKER = ['.','*', '^', 's', '.', 'p', 'o', 's', '.', 'd']
    COLOUR_EFF = COLOUR[:classNum]
    MARKER_EFF = MARKER[:classNum]

    # Decide whether to vary the line styles or not
    if isVaryLineStyle:
        LINE_EFF = LINE[:classNum]
    else:
        LINE_EFF = ['-'] * classNum  # Use solid lines for all if not varying

    # Set the color cycle for lines including color, line style, and marker
    plt.rcParams['axes.prop_cycle'] = (plt.cycler(color=COLOUR_EFF) +
                                        plt.cycler(linestyle=LINE_EFF) +
                                        plt.cycler(marker=MARKER_EFF))

    # Set default line and marker sizes
    plt.rcParams['lines.markersize'] = 3  # Example size
    plt.rcParams['lines.linewidth'] = 2   # Example width for lines

    # Set label and title sizes
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['axes.titlesize'] = 20

    # Set tick properties
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['ytick.labelsize'] = 20

    # Set legend font size
    plt.rcParams['legend.fontsize'] = 12

    # Enable and configure grid
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.8
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.linewidth'] = 1

    # Set axes line width
    plt.rcParams['axes.linewidth'] = 2

    # Set tick sizes and widths
    plt.rcParams['xtick.major.size'] = 7
    plt.rcParams['xtick.major.width'] = 3
    plt.rcParams['xtick.minor.size'] = 2
    plt.rcParams['xtick.minor.width'] = 2

    plt.rcParams['ytick.major.size'] = 7
    plt.rcParams['ytick.major.width'] = 3
    plt.rcParams['ytick.minor.size'] = 2
    plt.rcParams['ytick.minor.width'] = 2


def getHistoParam(data, Nbins=None, binwidth=None, isDensity=False, isLog=False):
    """Compute histogram parameters with support for both linear and log binning.
    
    Parameters:
    - data: array-like, input data
    - Nbins: int, number of bins (ignored if binwidth is provided)
    - binwidth: float, fixed width of bins (overrides Nbins)
    - isDensity: bool, if True, normalise histogram counts
    - isLog: bool, if True, use logarithmic binning

    Returns:
    - Nbins: int, number of bins used
    - binwidth: float, width of each bin
    - bins: array, bin edges
    - counts: array, histogram counts
    - bin_centers: array, centers of the bins
    
    Notes:
    - Filters out NaN values automatically.
    - Raises ValueError for empty data or invalid log binning.
    
    """
    # Ensure `data` is a NumPy array and filter out NaNs and None values
    data = np.array(data, dtype=float)
    data = data[~np.isnan(data)]  # Remove NaNs

    # 🚨 **Handle Completely Empty Data**
    if data.size == 0:
        raise ValueError("Histogram cannot be computed: No valid data points.")

    data_min, data_max = np.min(data), np.max(data)  # Safe now

    if isLog:
        if data_min <= 0:
            raise ValueError("Logarithmic binning requires positive data values.")

        if Nbins is not None:
            bins = np.logspace(np.log10(data_min), np.log10(data_max), Nbins + 1)
        elif binwidth is not None:
            bins = np.geomspace(data_min, data_max, 
                                int((np.log(data_max) - np.log(data_min)) / np.log(1 + binwidth)) + 1)
        else:
            Nbins = int(np.sqrt(len(data)))
            bins = np.logspace(np.log10(data_min), np.log10(data_max), Nbins + 1)

        binwidth = np.diff(bins)  # Log bins have varying width

    else:
        if Nbins is not None:
            bins = np.linspace(np.floor(data_min), np.ceil(data_max), Nbins + 1)
            binwidth = bins[1] - bins[0]
        elif binwidth is not None:
            bins = np.arange(np.floor(data_min), np.ceil(data_max) + binwidth, binwidth)
            Nbins = len(bins) - 1
        else:
            Nbins = int(np.sqrt(len(data)))
            bins = np.linspace(np.floor(data_min), np.ceil(data_max), Nbins + 1)
            binwidth = bins[1] - bins[0]

    counts, bin_edges = np.histogram(data, bins=bins, density=isDensity)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Linear binning

    return Nbins, binwidth, bins, counts, bin_centers

def enable_histogram_counts(
    fontsize: int = 12,
    rotation: int = 0,
    min_count: float = 1,
    every: int = 1,
    y_offset_frac: float = 0.015,
):
    """
    Enable automatic annotation of histogram counts in Matplotlib.

    Works for both bar-style and step-style histograms by using the
    returned histogram counts and bin edges, rather than relying on
    rectangle patches.

    Parameters
    ----------
    fontsize:
        Annotation font size.
    rotation:
        Text rotation angle.
    min_count:
        Only annotate bins with count >= min_count.
    every:
        Annotate every Nth non-empty bin, useful to reduce clutter.
    y_offset_frac:
        Vertical offset as a fraction of current y-axis maximum.
    """
    if getattr(Axes, "_hist_counts_annotated", False):
        return

    _orig_hist = Axes.hist

    def _hist_with_counts(self, *args, **kwargs):
        # Optional per-call controls
        annotate = kwargs.pop("annotate_counts", True)
        local_fontsize = kwargs.pop("count_fontsize", fontsize)
        local_rotation = kwargs.pop("count_rotation", rotation)
        local_min_count = kwargs.pop("count_min", min_count)
        local_every = kwargs.pop("count_every", every)

        out = _orig_hist(self, *args, **kwargs)

        if not annotate:
            return out

        try:
            n, bins, patches = out
        except Exception:
            return out

        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        # n can be:
        # - shape (Nbins,) for one dataset
        # - shape (Ndatasets, Nbins) for multiple datasets
        n_arr = np.asarray(n, dtype=float)

        if n_arr.ndim == 1:
            count_groups = [n_arr]
        else:
            count_groups = list(n_arr)

        ymax = self.get_ylim()[1]
        y_offset = ymax * y_offset_frac

        for group_idx, counts in enumerate(count_groups):
            nonzero_counter = 0

            for x, count in zip(bin_centers, counts):
                if count < local_min_count or not np.isfinite(count):
                    continue

                if nonzero_counter % local_every != 0:
                    nonzero_counter += 1
                    continue

                if np.isclose(count, round(count)):
                    label = f"{int(round(count))}"
                else:
                    label = f"{count:.1f}"

                self.text(
                    x,
                    count + y_offset,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=local_fontsize,
                    rotation=local_rotation,
                    clip_on=True,
                )

                nonzero_counter += 1

        return out

    Axes.hist = _hist_with_counts
    Axes._hist_counts_annotated = True
    
def annotate_histogram(ax, fontsize=12):
    """Manually annotate bars of the most recent histogram on a given Axes."""
    for container in ax.containers:
        if hasattr(container, "patches"):  # typical for hist containers
            for patch in container.patches:
                h = patch.get_height()
                if h <= 0 or not np.isfinite(h):
                    continue
                x = patch.get_x() + patch.get_width() / 2
                ax.text(x, h, f"{int(h)}", ha="center", va="bottom", fontsize=fontsize)
