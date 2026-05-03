import torch
from torch import nn

class Conv_AE(nn.Module):
    """Convolutional autoencoder for reconstructing same-shaped image inputs.

    Input:
    input_shape [tuple[int, int, int]]: input tensor shape as (channels, height, width).
    latent_dim [int]: size of the encoded latent vector.

    Output:
    Conv_AE [nn.Module]: initialized convolutional autoencoder model.
    """
    def __init__(self, input_shape, latent_dim=64):
        super(Conv_AE, self).__init__()
        
        kernel_size = 3
        stride_1 = 1
        stride_2 = 2
        padding = 1
        
        self.enc_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=kernel_size, stride=stride_1, padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=kernel_size, stride=stride_1, padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel_size, stride=stride_2, padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, stride=stride_2, padding=padding),
            nn.ReLU()
        )
        
        # find dimention after convolution and pooling
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            z_enc = self.enc_conv(dummy)               
            unflatten_dim = z_enc.shape
            flat_dim = z_enc.flatten(1).shape[1]
        
        self.flatten = nn.Flatten()
        
        self.enc_fully_connected = nn.Linear(flat_dim, latent_dim)
        
        self.dec_fully_connected = nn.Linear(latent_dim, flat_dim) 
        
        self.unflatten = nn.Unflatten(1, unflatten_dim[1:])
        
        self.dec_conv = nn.Sequential(

            nn.ConvTranspose2d(64, 64, kernel_size=kernel_size, stride=stride_2, padding=padding, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=kernel_size, stride=stride_2, padding=padding, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=kernel_size, stride=stride_1, padding=padding),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=kernel_size, stride=stride_1, padding=padding),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Encode x through conv/linear layers and reconstruct it.

        Input:
        x [torch.Tensor]: input image tensor with shape (batch_size, channels, height, width).

        Output:
        x_recon [torch.Tensor]: reconstructed image tensor with the decoder output shape.
        z [torch.Tensor]: latent tensor with shape (batch_size, latent_dim).
        """
        
        z_enc = self.enc_conv(x)
        z = self.flatten(z_enc)
        z = self.enc_fully_connected(z)
        
        x_recon = self.dec_fully_connected(z)
        x_recon = self.unflatten(x_recon)
        x_recon = self.dec_conv(x_recon)
        return x_recon, z
    
    


class Conv_AE_diff_targ(nn.Module):
    """Convolutional autoencoder with different input and target output shapes.

    Input:
    input_shape [tuple[int, int, int]]: input tensor shape as (channels, height, width).
    target_shape [tuple[int, int, int]]: target tensor shape as (channels, height, width).
    latent_dim [int]: size of the encoded latent vector.

    Output:
    Conv_AE_diff_targ [nn.Module]: initialized convolutional autoencoder model.
    """
    def __init__(self, input_shape=(1, 104, 52), target_shape=(1, 56, 28), latent_dim=64):
        super().__init__()

        kernel_size = 3
        stride_1 = 1
        stride_2 = 2
        padding = 1

        in_channels = input_shape[0]
        out_channels = target_shape[0]

        self.enc_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=kernel_size, stride=stride_1, padding=padding),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=kernel_size, stride=stride_1, padding=padding),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=kernel_size, stride=stride_2, padding=padding),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=kernel_size, stride=stride_2, padding=padding),
            nn.ReLU()
        )

        # encoder output shape from input
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            z_enc = self.enc_conv(dummy)
            enc_unflatten_shape = z_enc.shape[1:]   # (64, 26, 13)
            enc_flat_dim = z_enc.flatten(1).shape[1]

        # decoder start shape chosen to end at target 56x28
        dec_unflatten_shape = (64, 14, 7)
        dec_flat_dim = 64 * 14 * 7

        self.flatten = nn.Flatten()

        self.enc_fully_connected = nn.Linear(enc_flat_dim, latent_dim)
        self.dec_fully_connected = nn.Linear(latent_dim, dec_flat_dim)

        self.unflatten = nn.Unflatten(1, dec_unflatten_shape)

        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=kernel_size, stride=stride_2, padding=padding, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=kernel_size, stride=stride_2, padding=padding, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=kernel_size, stride=stride_1, padding=padding),
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, kernel_size=kernel_size, stride=stride_1, padding=padding),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Encode x and reconstruct it to the target-shaped decoder output.

        Input:
        x [torch.Tensor]: input image tensor with shape (batch_size, *input_shape).

        Output:
        x_recon [torch.Tensor]: reconstructed tensor with shape (batch_size, *target_shape).
        z [torch.Tensor]: latent tensor with shape (batch_size, latent_dim).
        """
        z_enc = self.enc_conv(x)
        z = self.flatten(z_enc)
        z = self.enc_fully_connected(z)

        x_recon = self.dec_fully_connected(z)
        x_recon = self.unflatten(x_recon)
        x_recon = self.dec_conv(x_recon)

        return x_recon, z
