import torch
from torch import nn


class AutoEncoder_desp(nn.Module):
    """Fully connected autoencoder for compressing descriptor vectors.

    Input:
    input_dim [int]: size of each input vector.
    latent_dim [int]: size of the encoded latent vector.
    output_dim [int | None]: size of each reconstructed vector; defaults to input_dim.
    p [float]: unused dropout parameter kept for compatibility.

    Output:
    AutoEncoder_desp [nn.Module]: initialized encoder-decoder model.
    """
    def __init__(self, input_dim=300, latent_dim=50, output_dim=None, p=0.2):
        super(AutoEncoder_desp, self).__init__()

        if output_dim is None:
            output_dim = input_dim
        
        e_1, e_2 = 2048, 512

        d_1, d_2 = 512, 2048
    
        #-----------------------------------------------------------
        self.encoder_1 = nn.Sequential(
            nn.Linear(input_dim, e_1),
            nn.ReLU()
        )
        
        nn.init.kaiming_uniform_(self.encoder_1[0].weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.encoder_1[0].bias)
        
        #----------------------------------------------------------
        
        self.encoder_2 = nn.Sequential(
            nn.Linear(e_1, e_2),
            nn.ReLU()
        )
        
        nn.init.kaiming_uniform_(self.encoder_2[0].weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.encoder_2[0].bias)
        
        #---------------------------------------------------------------
        
        self.encoder_3 = nn.Sequential(
            nn.Linear(e_2, latent_dim),
            nn.ReLU()
        )
        
        nn.init.kaiming_uniform_(self.encoder_3[0].weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.encoder_3[0].bias)
        
        #---------------------------------------------------------------

        self.decoder_1 = nn.Sequential(
            nn.Linear(latent_dim, d_1),
            nn.ReLU()
        )
        
        nn.init.kaiming_uniform_(self.decoder_1[0].weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.decoder_1[0].bias)
        
        #---------------------------------------------------------------------------
        
        self.decoder_2 = nn.Sequential(
            nn.Linear(d_1, d_2),
            nn.ReLU()
        )
        
        nn.init.kaiming_uniform_(self.decoder_2[0].weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.decoder_2[0].bias)
        
        #---------------------------------------------------------------------------
        
        self.decoder_3 = nn.Sequential(
          nn.Linear(d_2, output_dim),
          nn.Sigmoid()  
        )
        
        nn.init.xavier_uniform_(self.decoder_3[0].weight)
        nn.init.zeros_(self.decoder_3[0].bias)
        
    def forward(self, x):
        """Encode x into a latent vector and decode it back.

        Input:
        x [torch.Tensor]: input tensor with shape (batch_size, input_dim).

        Output:
        x_recon [torch.Tensor]: reconstructed tensor with shape (batch_size, output_dim).
        z [torch.Tensor]: latent tensor with shape (batch_size, latent_dim).
        """
        
        z = self.encoder_1(x)
        z = self.encoder_2(z)
        z = self.encoder_3(z)
        
        x_recon = self.decoder_1(z)
        x_recon = self.decoder_2(x_recon)
        x_recon = self.decoder_3(x_recon)

        return x_recon, z 
