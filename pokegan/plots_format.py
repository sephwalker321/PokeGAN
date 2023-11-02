#!/usr/bin/env python
from cycler import cycler
import matplotlib as mpl
import matplotlib.font_manager
mpl.use('Tkagg')
import matplotlib.pyplot as plt

try:
    plt.style.use(["science", "no-latex", "bright"])
except Exception:
    plt.style.use("default")
    print("Using default matplotlib style")
    
def fig_initialize():
    """ 
    Initialize matplotlib figures with custom set up.
    
    Parameters
    ----------

    Returns
    -------

    """
    #Set up tex rendering
    plt.rc('text', usetex=True)
    # plt.rc('text.latex',preamble=[
    # 	r'\usepackage{amsmath}',
    # 	r'\usepackage{amsthm}',
    # 	r'\usepackage{amssymb}',
    # 	r'\usepackage{amsfonts}'
    # 	]
    # )
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = "STIX"
    mpl.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams['axes.facecolor']='white'
    plt.rcParams['figure.facecolor']='white'
    plt.rcParams['savefig.facecolor']='white'

    mpl.rcParams["font.size"] = 11
    mpl.rcParams['lines.linewidth'] = 1
    mpl.rcParams["axes.labelsize"] = 11
    mpl.rcParams["xtick.labelsize"] = 9
    mpl.rcParams["ytick.labelsize"] = 9
    mpl.rcParams["legend.labelspacing"] = 0.5
    plt.rc('legend',**{'fontsize':9})
    plt.rc('legend',**{'frameon':False})

    #Define a custom cycler
    custom_cycler = (cycler(color=['steelblue','maroon','midnightblue','r','cadetblue','orange']) + \
                     cycler(linestyle=['-','-.','--',':','--','-']))

    plt.rc('axes',prop_cycle=custom_cycler)
    return

def set_size(width=None, fraction=1, subplots=(1, 1)):
    """ 
    Set figure dimensions to avoid scaling in LaTeX.
    
    Credit : https://jwalton.info/Embed-Publication-Matplotlib-Latex/

    Parameters
    ----------
    width: float or string
        Document width in points, or string of predined document type
    fraction: float, optional
        Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
        The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
        Dimensions of figure in inches
    """
    if width is None:
        width_pt=392.0
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)