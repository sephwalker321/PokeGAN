import os
import glob
import numpy as np
import pandas as pd
import pickle
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.optim as optim

from pokegan.functions import Setup            
from pokegan.plots import plot_noise, plot_epoch, plot_training


class Generator(nn.Module):
    def __init__(
            self,
            x_dim = None,
            image_channel = None,
            latent_dim = None,
            g_hidden = None,
        ):
        """ 
        The neural network generator.
    
        Parameters
        ----------
            x_dim: int
                The linear dimension of the training images
            image_channel: int
                The number of image colour channels
            latent_dim: int
                The latent dimension for the input noise. Should be a square number for plots.
            g_hidden: int
                The hidden layers dimension
        
        """
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        self.x_dim = x_dim
        
        poweroftwo = int(x_dim/4) #TODO CHECK THIS x_dim != 64.
        n_layers = int(np.log2(poweroftwo))
        
        self.main = nn.Sequential()
        self.main += self.hidden_layer(
            in_channels=latent_dim,
            out_channels= g_hidden * poweroftwo,
            kernel_size=4,
            stride=1,
            padding=0,
        ) 
        for i in range(n_layers):
            self.main += self.hidden_layer(
                in_channels = g_hidden * poweroftwo,
                out_channels= g_hidden * int(poweroftwo / 2),
                kernel_size=4,
                stride=2,
                padding=1,
            )
            poweroftwo = int(poweroftwo/2)
            
        self.main += nn.Sequential(
            nn.ConvTranspose2d(in_channels=g_hidden, out_channels=image_channel, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
        
    def forward(self, input):
        """ 
        Preform a forward pass for the network
        
        Parameters
        ----------
            input: tensor
                noisy input
        
        Returns
        -------
            output: tensor
                generated artificial image
        
        """
        return self.main(input)
    
    def hidden_layer(
            self,
            in_channels = None,
            out_channels = None,
            kernel_size = None,
            stride = None,
            padding = None,
        ):
        """ 
        A hidden layer unit of the generator consisting of a ConvTranspose2d, BatchNorm2D and ReLU.
        Refer to pytorch documentation for details of the parameters.
        
        Parameters
        ----------
            in_channels: int
            out_channels: int
            kernel_size: int
            stride: int
            padding: int

        Returns
        -------
            hidden_layer: sequential container
                The sequential container of our hidden layer unit
        
        """
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        
    def calc_output(
            self,
            h_in = None,
            kernel_size = None,
            stride = 1,
            padding = 0,
            output_padding = 0,
            dilation = 1,
        ):
        """ 
        Calculate the output dimension for a ConvTranspose2d layer.
        Refer to pytorch documentation for details of the parameters.
        
        Parameters
        ----------
            h_in: int
            kernel_size: int
            stride: int
            padding: int
            output_padding: int
            dilation: int
        
        Returns
        -------
            h_out: int
                The output dimensions of ConvTranspose2d
        
        """
        return (h_in-1)*stride - 2*padding + dilation*(kernel_size-1) + output_padding + 1

  
class Discriminator(nn.Module):
    def __init__(
            self,
            x_dim = None,
            image_channel = None,
            d_hidden = None,
        ):
        """ 
        The neural network discriminator.
    
        Parameters
        ----------
            x_dim: int
                The linear dimension of the training images
            image_channel: int
                The number of image colour channels
            d_hidden: int
                The hidden layers dimension
        """
        super(Discriminator, self).__init__()

        poweroftwo = int(x_dim/4)  #TODO CHECK THIS x_dim != 64.
        n_layers = int(np.log2(poweroftwo))
        poweroftwo = 1
        
        self.main = nn.Sequential()
        self.main += self.hidden_layer(
                in_channels=image_channel,
                out_channels=d_hidden,
                kernel_size=4,
                stride=2,
                padding=1,
                batchnorm = False,
            ) 
        for i in range(n_layers):
            self.main += self.hidden_layer(
                in_channels = d_hidden * poweroftwo,
                out_channels= d_hidden * int(poweroftwo*2),
                kernel_size=4,
                stride=2,
                padding=1,
                batchnorm = True,
            )
            poweroftwo = int(poweroftwo*2)
            
        self.main += nn.Sequential(
            nn.Conv2d(d_hidden*poweroftwo, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        """ 
        Preform a forward pass for the network
        
        Parameters
        ----------
            input: tensor
                An images
        
        Returns
        -------
            output: tensor
                score of likelihood of real
        
        """
        return self.main(input).view(-1, 1).squeeze(1)
    
    def hidden_layer(
            self,
            in_channels = None,
            out_channels = None,
            kernel_size = None,
            stride = 1,
            padding = 0,
            dilation = 1,
            batchnorm = True,
        ):
        """ 
        A hidden layer unit of the discriminator consisting of a Conv2d, BatchNorm2D and ReLU.
        Refer to pytorch documentation for details of the parameters.
        
        Parameters
        ----------
            in_channels: int
            out_channels: int
            kernel_size: int
            stride: int
            padding: int
            dilation: int
            batchnorm: bool
                Preform batch normalisation in this layer unit.
        
        Returns
        -------
            hidden_layer: sequential container
                The sequential container of our hidden layer unit
        
        """
        if batchnorm:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False),
                nn.ReLU(True),
            )

class GAN():
    def __init__(
        self,
        config = Setup(),

        latent_dim = 10*10,
        g_hidden = 64,
        d_hidden = 64,
        criterion=nn.BCELoss(),
        
        generator = None,
        discriminator = None,
    
        gan_name = "default"
    ):
        """ 
        The GAN class network.
        
        Parameters
        ----------
            config: configuration class
                Class of configuration settings used across the module.
            latent_dim: int
                The latent dimension for the input noise. Should be a square number for plots.
            g_hidden: int
                The hidden layers dimension of generator
            d_hidden: int
                The hidden layers dimension of discriminator
            criterion: nn loss
                The training loss function
            generator: generator class
                The neural network generator
            discriminator: discriminator class
                The neural network discriminator
            gan_name: str
                Folder name for the GAN training and results    
        """
        self.config = config
        
        self.latent_dim = latent_dim
        self.g_hidden = g_hidden
        self.d_hidden = d_hidden
        self.criterion = criterion
        
        self.initalise_networks(generator, discriminator)
        self.initalise_noise()
        self.initalise_training_data()
        
        self.initalise_dirs(gan_name)
    
        #Save the class template
        self.save_class()
         
    def weights_init(self, m):
        """ 
        Set the initial weights of the GAN.
        
        Parameters
        ----------
            m: class layer
                Layer in the networks
        Returns
        -------
        
        """
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        return
            
    def initalise_networks(self, generator, discriminator):
        """ 
        Initalise the generator and the discriminator networks.
        
        Parameters
        ----------
            generator: generator class
                The neural network generator
            discriminator: discriminator class
                The neural network discriminator
        
        Returns
        -------
        
        """
        if generator is not None:
            self.netG = generator(
                x_dim = self.config.x_dim,
                image_channel = self.config.image_channel,
                latent_dim = self.latent_dim,
                g_hidden = self.g_hidden,
            ).to(self.config.device)
            self.netG.apply(self.weights_init)
            print(self.netG)
        else:
            print(f"No generator provided")
            return
            
        if discriminator is not None:
            self.netD = discriminator(
                x_dim = self.config.x_dim,
                image_channel = self.config.image_channel,
                d_hidden = self.d_hidden,
            ).to(self.config.device)
            self.netD.apply(self.weights_init)
            print(self.netD)
        else:
            print(f"No discriminator provided")
            return    
        
        # Adam optimizers for both G and D
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.config.lr_D, betas=(0.5, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.config.lr_G, betas=(0.5, 0.999))
        
        # Untrained
        self.epoch = -1 
        return
            
    def test_inout(self):
        """ 
        Print out the dimensions of the networks
        
        Mainly for bug fixing
        
        Parameters
        ----------
        
        Returns
        -------
        
        """
        noise = torch.randn(1, self.latent_dim, 1, 1, device=self.config.device)
        # Generate fake image batch with G
        g_output = self.netG(noise).to(self.config.device)
        print(f"generator dimensions: {g_output.shape}")

        d_output = self.netD(g_output)
        print(f"distriminator dimensions: {d_output.shape}\n")
        return
        
    def initalise_noise(self):
        """ 
        Generate the fixed noise used for training visualizations
        
        Parameters
        ----------
        
        Returns
        -------
        
        """
        # Create batch of latent vectors that visualise the progression of the generator
        self.viz_noise = torch.randn(self.config.batch_plot_tiling*self.config.batch_plot_tiling, self.latent_dim, 1, 1, device=self.config.device)
        self.viz_noise_plot = self.viz_noise.reshape(shape=(self.config.batch_plot_tiling*self.config.batch_plot_tiling, 1,int(np.sqrt(self.latent_dim)),int(np.sqrt(self.latent_dim))))
        return
    
    def initalise_training_data(self):
        """ 
        Generate empty data frame to log training histories
        
        Parameters
        ----------
        
        Returns
        -------
        
        """
        self.training_data = pd.DataFrame(data={
                "epoch": [],
                "time" : [],
                "d_loss": [],
                "g_loss": [],
                "D_x": [],
                "D_G_z1": [],
                "D_G_z2": [],
        })
        return
        
        
    def initalise_dirs(self, gan_name):
        """ 
        Create the directories for the GAN checkpoints and history plots
        
        Parameters
        ----------
            gan_name: str
                Folder name for the GAN training and results    

        Returns
        -------
        
        """
        self.gan_dir = f"{self.config.working_dir}{os.sep}runs{os.sep}{gan_name}"
        globbed = glob.glob(f"{self.gan_dir}*")
        
        self.gan_dir += f"_{int(len(globbed)+1)}" 
        os.mkdir(self.gan_dir)
        os.mkdir(f"{self.gan_dir}{os.sep}training_images")
        os.mkdir(f"{self.gan_dir}{os.sep}networks")
        return
        
    def generate_fake(self, b_size=1):
        """ 
        Generate a artificial image.
        
        Parameters
        ----------
            b_size: int
                The number of images to generate
        
        Returns
        -------
            noise_plot: tensor 
                Images of the input random noise
            fakes: tensor
                Images of the artificial images
        """
        noise = torch.randn(b_size, self.latent_dim, 1, 1, device=self.config.device)
        noise_plot = noise.reshape(shape=(b_size, 1, int(np.sqrt(self.latent_dim)),int(np.sqrt(self.latent_dim)))).detach().to(self.config.device)
        fakes = self.netG(noise).detach().to(self.config.device)
        return noise_plot, fakes

    def test_fake(self, images):
        """ 
        Test the discriminator and produce the scores
        
        Parameters
        ----------
            images: tensor
                Images
        
        Returns
        -------
            scores: tensors
                The scores (fyi be careful of the real / fake label)
        
        """
        return self.netD(images).detach().to(self.config.device)
    
    def plot_noise(self):
        """ 
        Plot the fixed noise used for training visualizations
        """
        plot_noise(self)
        
    def plot_epoch(self):
        """ 
        Plot the fixed noise and its output for training visualizations
        """
        plot_epoch(self)
        
    def plot_training(self):
        """ 
        Plot loss of the generator and discriminator for each epoch in the training.
        """
        plot_training(self)
            
    def train(
        self,
        dataloader,
        N_epochs = 10
    ):
        """ 
        Train the GAN.
        Save out checkpoints and training logs.
        
        Parameters
        ----------
            dataloader: A class dataset loader
                The image dataloader
            N_epochs: int
                The number of epochs to train
        
        Returns
        -------
        
        """
        iters = 0
        start_epoch = max(0, self.epoch)
        
        if start_epoch == 0:
            print("Starting training loop...")
            self.plot_noise()
        else:
            print(f"Continuing training from epoch {start_epoch}")
            
        for epoch in range(start_epoch, N_epochs):
            t1 = time.time()
            
            self.epoch = epoch
            self.plot_epoch()
            for i, data in enumerate(dataloader, 0):
                # Loop over batches of the dataloader
            
                # 1. Update the discriminator with real data
                self.netD.zero_grad()
                real_cpu = data[0].to(self.config.device)
                b_size = real_cpu.size(0)
                
                # Noisy labels
                label_R_noise = np.abs(torch.normal(mean=0.0, std=self.config.noisy_label_std, size=(b_size,)))
                label_F_noise = np.abs(torch.normal(mean=0.0, std=0.1, size=(b_size,)))
                label = torch.full((b_size,), self.config.real_label, dtype=torch.float, device=self.config.device)
                if self.config.noisy_label:
                    if self.config.real_label == 0:
                        label += label_R_noise
                    else:
                        label -= label_R_noise            
                    
                # Forward pass real batch through D
                output = self.netD(real_cpu).view(-1)
                
                # Calculate loss on all-real batch
                self.criterion.weight=data[2].to(self.config.device)
                errD_real = self.criterion(output, label)

                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                # 2. Update the discriminator with fake data
                # Generate batch of latent vectors
                noise = torch.randn(b_size, self.latent_dim, 1, 1, device=self.config.device)
                # Generate fake image batch with G
                fake = self.netG(noise)
                
                # Noisy labels
                label.fill_(self.config.fake_label)
                if self.config.noisy_label:
                    if self.config.fake_label == 0:
                        label += label_F_noise
                    else:
                        label -= label_F_noise
                        
                # Classify all fake batch with D
                output = self.netD(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = self.criterion(output, label)
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Compute error of D as sum over the fake and the real batches
                errD = errD_real + errD_fake
                # Update D
                self.optimizerD.step()

                # 3. Update the generator with fake data
                self.netG.zero_grad()
                label.fill_(self.config.real_label)
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = self.netD(fake).view(-1)
                # Calculate G's loss based on this output
                errG = self.criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                self.optimizerG.step()

                # Output training stats
                if i % 10 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                        % (epoch, N_epochs, i, len(dataloader),
                            errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                    
                iters += 1
                t2 = time.time()
                                 
            dat_epoch =  {
                "epoch": [int(epoch+1)],
                "time" : [t2-t1],
                "d_loss": [errD.item()],
                "g_loss": [errG.item()],
                "D_x": [D_x],
                "D_G_z1": [D_G_z1],
                "D_G_z2": [D_G_z2],
            }

            self.training_data = pd.concat([self.training_data, pd.DataFrame(dat_epoch)], ignore_index=True)
            if epoch % self.config.epoch_save_rate == 0 and epoch > 0:
                torch.save(self.netD.state_dict(), f"{self.gan_dir}{os.sep}networks{os.sep}D_{epoch+1:05}")
                torch.save(self.netG.state_dict(), f"{self.gan_dir}{os.sep}networks{os.sep}G_{epoch+1:05}")
                
            #Save metrics
            self.training_data.to_csv(f"{self.gan_dir}{os.sep}training_data.csv", index=False)
            #Save latest networks
            torch.save(self.netD.state_dict(), f"{self.gan_dir}{os.sep}networks{os.sep}D_latest")
            torch.save(self.netG.state_dict(), f"{self.gan_dir}{os.sep}networks{os.sep}G_latest")
        return
                
                
    def save_class(self):
        """ 
        Save the GAN class to file
        
        Parameters
        ----------
        
        Returns
        -------
        
        """
        with open(f'{self.gan_dir}{os.sep}gan_template.pkl', 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
        return
        
def load_gan(
        working_dir=Setup().working_dir,
        run_name = "default_1",
        load_checkpoint = True
    ):
    """ 
    Load in the GANs from class pickle and checkpoints
    
    Parameters
    ----------
        working_dir: str
            The directory of the GAN project folder
        run_name: str
            Folder name for the GAN training and results    
        load_checkpoint: bool
            Load checkpoints into generator and discriminator
    
    Returns
    -------
        gan: The GAN class network
            The GAN
    """
    gan_dir = f"{working_dir}{os.sep}runs{os.sep}{run_name}"
    if os.path.exists(gan_dir):
        with open(f'{gan_dir}{os.sep}gan_template.pkl', 'rb') as inp:
            gan = pickle.load(inp)
        
        d_bool = os.path.exists(f"{gan_dir}{os.sep}networks{os.sep}D_latest") 
        g_bool = os.path.exists(f"{gan_dir}{os.sep}networks{os.sep}G_latest")
        t_bool = os.path.exists(f"{gan_dir}{os.sep}training_data.csv")
        
        if load_checkpoint and d_bool and g_bool and t_bool:
            gan.netD.load_state_dict(torch.load(f"{gan_dir}{os.sep}networks{os.sep}D_latest"))
            gan.netG.load_state_dict(torch.load(f"{gan_dir}{os.sep}networks{os.sep}G_latest"))
            gan.training_data = pd.read_csv(f"{gan_dir}{os.sep}training_data.csv", )
            gan.epoch = int(gan.training_data["epoch"].values[-1])
                    
        elif load_checkpoint and (not d_bool or not g_bool or not t_bool):
            print(f"Network checkpoints don't exist")
        return gan
        
    else:
        print(f"{gan_dir} does not exist")
        return
                     