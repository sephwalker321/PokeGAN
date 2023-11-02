import os
import glob
import re

import numpy as np

import torch
import torch.utils.data
import torchvision.transforms as transforms

from torchvision.io import read_image, ImageReadMode

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from pokegan.functions import Setup
from pokegan.plots import plot_batch

class CustomImageDataset(Dataset):
    def __init__(
        self,
        config = Setup(),

    ):
        """ 
        A class dataset loader.
        
        Parameters
        ----------
            config: configuration class
                Class of configuration settings used across the module.
                
        TODO Customisable transformations, read_mode and imformat.
        """
        self.imformat = "png"
        
        self.config = config
        
        self.img_dir = f"{self.config.images_base_dir}{os.sep}{self.config.images_dir}"
        self.transform =  transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.RandomRotation(10, expand=True, fill=0),
            transforms.RandomHorizontalFlip(0.5),
            transforms.Resize(self.config.x_dim, antialias=True),
            transforms.Normalize((0.5,), (0.5,)),
        ])        
        self.image_read_mode =  ImageReadMode.RGB_ALPHA
        
        self.load_images(self.glob_dir())
        
    def __len__(self):
        """ 
        Length of the available data
        
        Parameters
        ----------
        
        Returns
        -------
            length: int
                The number of available images
        """
        return len(self.img_labels)

    def __getitem__(self, idx):
        """ 
        Retrieve single image, its label and weight. 
        
        Parameters
        ----------
            idx: int
                Index of item
        Returns
        -------
            image: tensor
                A training image
            label: int
                The image label
            weight: float
                The image weight
        """
        image = self.images[idx]
        label = self.img_labels[idx]
        weight = self.weights[idx]
    
        image = self.remove_alphas(image)
        image = self.transform(image)
        
        return image, label, weight
    
    def remove_alphas(self, image):
        """ 
        Use the alpha channel of the images as a mask then reduce the dimensionality.
        
        Parameters
        ----------
            image: tensor
                A training image
        
        Returns
        -------
            image: tensor
                A trainig image
        """
        mask = image[-1,:,:] == 255
        image = torch.mul(image, mask.unsqueeze(0))
        if self.config.image_channel in [None, 3]:
            return image[:-1,:,:]
        else:
            print(f"Image channel {self.config.image_channel} not implemented.\nPlease use RGB images.")
            return
        
    def glob_dir(self):
        """ 
        Collect together all the training images.
        Weights 1/N_forms of the pokemon. 
        
        Parameters
        ----------
        
        Returns
        -------
            glob_info: dict
                dictionary of image paths, and weights.
        """
        globbed = glob.glob(f"{self.img_dir}{os.sep}*{self.imformat}")
        glob_info = {}
        for im in globbed:            
            id = re.findall(r'\d+', im)
            
            if len(id) == 1:
                id = int(id[0])
            elif len(id) == 0:
                continue
            else:
                id = int(id[0])
            
            if id == 0:
                continue
                
            form = im.split(f"{id}")[-1].split(f".{self.imformat}")[0]
            if id not in glob_info.keys():
                glob_info[id] = {"forms" : []}
            if form is not None:
                glob_info[id]["forms"].append(form)
        for id in glob_info.keys():
            N = len(glob_info[id]["forms"])
            if N > 0:
                glob_info[id]["weight"] = 1/N
            else:
                glob_info[id]["weight"] = 1
        return glob_info
    
    def load_images(self, glob_info):
        """ 
        Load the images into the dataloader class
        
        Parameters
        ----------
            glob_info: dict
                dictionary of image paths, and weights.
        
        Returns
        -------
        
        """
        image_list = []
        weights = []
        for id in glob_info.keys():
            if len(glob_info[id]["forms"])>0:
                for form in glob_info[id]["forms"]:
                    try:
                        im=read_image(f"{self.img_dir}/{id}{form}.{self.imformat}", self.image_read_mode)
                        image_list.append(im)   
                        weights.append(glob_info[id]["weight"])
                    except:
                        pass
            else:
                try:
                    im=read_image(f"{self.img_dir}/{id}.{self.imformat}", self.image_read_mode)
                    image_list.append(im)  
                    weights.append(glob_info[id]["weight"]) 
                except: 
                    pass
            
                
        self.images = image_list
        self.weights = weights
        self.img_labels = np.zeros(len(weights))+self.config.real_label
        return
    
    def plot_batch(self):    
        """ 
        Plot an example batch of the training images.
        """
        plot_batch(self)
        
    def get_batch(self, b_size=None):
        """ 
        Get a batch of training images.
        
        Parameters
        ----------
            b_size: int
                The number of images to generate
        
        Returns
        ----------
        image: tensor
                A training image
            label: int
                The image label
            weight: float
                The image weight
        """
        if b_size == None:
            b_size=self.config.batch_plot_tiling*self.config.batch_plot_tiling
        return next(iter(DataLoader(self, batch_size=b_size, shuffle=True)))