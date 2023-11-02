import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from torch.utils.data import DataLoader
import torchvision.utils as vutils


from pokegan.plots_format import fig_initialize, set_size
fig_initialize()

def plot_batch(cls):    
    """ 
    Plot an example batch of the training images.
    
    Parameters
    ----------
        cls: GAN Class
            The GAN, required the configuration settings
    
    Returns
    -------
    
    """
    real_batch, _, _ = cls.get_batch()
    f, ax = plt.subplots(1,1) 
    f.set_size_inches(5,5)
    ax.axis("off")
    ax.set_title("Training Images")
    ax.imshow(np.transpose(vutils.make_grid(real_batch.to(cls.config.device)[:cls.config.batch_plot_tiling*cls.config.batch_plot_tiling], nrow=cls.config.batch_plot_tiling, padding=2, normalize=True).cpu(),(1,2,0)))
    #plt.savefig(f"{cls.gan_dir}{os.sep}batch_example.{cls.config.imformat}", dpi=cls.config.dpi, format=cls.config.imformat)
    plt.show()
    return

def plot_noise(cls):
    """ 
    Plot the fixed noise used for training visualizations
    
    Parameters
    ----------
        cls: GAN Class
            The GAN, required the configuration settings
    
    Returns
    -------
    
    """
    f, ax = plt.subplots(1,1) 
    f.set_size_inches(5,5)
    ax.axis("off")
    ax.imshow(np.transpose(vutils.make_grid(cls.viz_noise_plot.to(cls.config.device)[:cls.config.batch_plot_tiling*cls.config.batch_plot_tiling], nrow=cls.config.batch_plot_tiling, padding=2, normalize=True).cpu(),(1,2,0)))
    plt.savefig(f"{cls.gan_dir}{os.sep}training_images{os.sep}Noise.{cls.config.imformat}", dpi=cls.config.dpi, format=cls.config.imformat)
    plt.close()
    return

def plot_epoch(cls):
    """ 
    Plot the fixed noise and its output for training visualizations
    
    Parameters
    ----------
        cls: GAN Class
            The GAN, required the configuration settings
    
    Returns
    -------
    
    """
    f, ax = plt.subplots(1,1) 
    f.set_size_inches(5,5)
    ax.axis("off")
    ax.imshow(np.transpose(vutils.make_grid(cls.netG(cls.viz_noise).detach().cpu(), padding=2, normalize=True),(1,2,0)))
    plt.savefig(f"{cls.gan_dir}{os.sep}training_images{os.sep}{cls.epoch:05}.{cls.config.imformat}", dpi=cls.config.dpi, format=cls.config.imformat)
    plt.close()
    return
    
def plot_training(cls):
    """ 
    Plot loss of the generator and discriminator for each epoch in the training.
    
    Parameters
    ----------
        cls: GAN Class
            The GAN, required the configuration settings
    
    Returns
    -------
    
    """
    f, ax = plt.subplots(1,1) 
    f.set_size_inches(set_size(subplots=(1,1), fraction=1))
    ax.plot(cls.training_data["epoch"], cls.training_data["d_loss"], label="discriminator",linestyle="-",alpha=.6)
    ax.plot(cls.training_data["epoch"], cls.training_data["g_loss"], label="generator",linestyle="-",alpha=.6)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    plt.savefig(f"{cls.gan_dir}{os.sep}training_loss.{cls.config.imformat}", dpi=cls.config.dpi, format=cls.config.imformat)
    plt.show()
    return
    
def plot_image(
        cls,
        noises = None,
        fakes = None,
        scores = None,
        nrow = None,
        figure_name = "",
    ):    
    """ 
    Plot a set of example artificial images. Compare the noise, image and scores in three subplots.
    
    Parameters
    ----------
        cls: GAN Class
            The GAN, required the configuration settings
        noises: pytorch tensor
            The input noise in the generator
        fakes: pytorch tensor
            The generated artificial images
        scores: pytorch tensor
            The descriminator scores
        nrow: int
            Number of rows for the image tilings
        figure_name: str
            Figure name to save to file
    
    Returns
    -------
    
    """
    if nrow is None:
        nrow = cls.config.batch_plot_tiling
        
    if noises is None:
        f, (axF, axS) = plt.subplots(1,2)
        
    else:
        f, (axN, axF, axS) = plt.subplots(1,3)
        axN.axis("off")
        axN.set_title("Input") 
        axN.imshow(np.transpose(vutils.make_grid(noises[:nrow*nrow], nrow=nrow, padding=2, normalize=True).cpu(),(1,2,0))) 
        
    f.set_size_inches(set_size(subplots=(1,2), fraction=3))
    
    axF.axis("off")
    axF.set_title("Artificial Images")
    axS.axis("off")
    axS.set_title("Scores")
    
    #Set the colourbar such that real = 1, fake = 0
    if cls.config.fake_label == 1:
        score_matrix = 1-scores[:nrow*nrow].reshape(nrow,nrow,1).numpy()
    elif cls.config.real_label == 1:
        score_matrix = scores[:nrow*nrow].reshape(nrow,nrow,1).numpy()
    
    
    axF.imshow(np.transpose(vutils.make_grid(fakes[:nrow*nrow], nrow=nrow, padding=2, normalize=True).cpu(),(1,2,0)))
    im = axS.imshow(score_matrix, vmin=0, vmax=1, cmap="RdYlGn")
        
    divider = make_axes_locatable(axS)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.set_ylabel(r'$p(\textrm{real})$')
        
    if nrow <= 4:
        for i in range(nrow):
            for j in range(nrow):
                text = axS.text(j, i, f"{score_matrix[i, j][0]:.3f}", ha="center", va="center", color="black")

    plt.savefig(f"{cls.gan_dir}{os.sep}{figure_name}.{cls.config.imformat}", dpi=cls.config.dpi, format=cls.config.imformat)
    plt.show()
    return