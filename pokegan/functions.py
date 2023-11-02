from pathlib import Path
import os
import yaml

import torch
import torch.backends.cudnn as cudnn

def load_yaml(filename):
    """ 
    Load yaml into python dict
    
    Parameters
    ----------
        filename: str
            absolute path of .yaml file 

    Returns
    -------
        YAML: dict
            Required file
    """
    with open(filename, "r") as stream:
        try:
            YAML = yaml.safe_load(stream)
            return YAML
        except yaml.YAMLError as exc:
            print(exc)
    return


class Setup:
    def __init__(
        self, 
        x_dim = None,
        image_channel = None,
        batch_size = None,
    ):
        """ 
        Class of configuration settings used across the module.
        Default values are set in load_config() using config file.
        
        Parameters
        ----------
            x_dim: int
                The linear dimension of the training images
            image_channel: int
                The number of image colour channels
            batch_size: int
                The batch size for the dataloader
        """
        self.x_dim = x_dim
        self.image_channel = image_channel
        self.batch_size = batch_size
        
        self.set_fixed()
        self.load_config()
        
        self.cuda_activate()
        
        
        torch.manual_seed(self.seed) 
                
    def set_fixed(self):
        """ 
        Set the default fixed configuration parameters.
        
        Parameters
        ----------

        Returns
        -------

        """
        self.working_dir = Path(os.getcwd()).parent
        self.images_base_dir = f"{self.working_dir}"
        return

    def load_config(self):
        """ 
        Set the customisable configuration parameters.
        
        Parameters
        ----------

        Returns
        -------

        """
        yaml_in = load_yaml(f"{self.working_dir}/pokegan/config.yaml")
        
        #Plotting density
        self.dpi               = float(yaml_in["dpi"])
        self.imformat          = yaml_in["imformat"]
        self.batch_plot_tiling = yaml_in["batch_plot_tiling"]

        #SpritesLocation
        self.images_dir      = yaml_in["images_default"]
        
        self.use_cuda        = yaml_in["cuda"]
        
        self.epoch_save_rate = yaml_in["epoch_save_rate"]
        
        if self.batch_size is None:
            self.batch_size      = yaml_in["batch_size"]
        if self.image_channel is None:
            self.image_channel = yaml_in["image_channel"]
        if self.x_dim is None:
            self.x_dim = yaml_in["x_dim"]
        
        self.real_label      = yaml_in["real_label"]
        self.fake_label      = yaml_in["fake_label"]
        self.noisy_label     = yaml_in["noisy_label"]
        self.noisy_label_std = yaml_in["noisy_label_std"]

        self.lr_D = yaml_in["lr_D"]
        self.lr_G = yaml_in["lr_G"]
        self.seed = yaml_in["seed"]        
        return

    def change_dir(
            self,
            images_dir_new=None,
        ):
        """ 
        Change the road and place directory from the default.
        TODO Check directory exists?
        
        Parameters
        ----------
            images_dir_new: str
                folder path containing pokemon images

        Returns
        -------

        """
        if images_dir_new is not None:
            self.images_dir = images_dir_new
        return
    
    def cuda_activate(self):
        """ 
        Check the environment for GPU and initialize.
        
        Parameters
        ----------
        
        Returns
        -------
        
        """
        self.use_cuda = self.use_cuda and torch.cuda.is_available()
        print("PyTorch version: {}".format(torch.__version__))
        if self.use_cuda:
            print("CUDA version: {}\n".format(torch.version.cuda))

        if self.use_cuda:
            torch.cuda.manual_seed(self.seed)
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        cudnn.benchmark = True
        return

    def printout(self):
        """ 
        Print out summary of configuration set up
        TODO Write function
        
        Parameters
        ----------

        Returns
        -------

        """
        return