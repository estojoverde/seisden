


import torch
import torch.nn as nn
import torch.nn.functional as F




#NOT WORKING
class FourierLayer(nn.Module):
    """
    A PyTorch layer that applies a 2D real Fourier transform to the input tensor
    and optionally concatenates the original input as additional channels.

    Args:
        freq_pad (int): Number of frequency bins to pad on the right of the FFT components.
                        Defaults to 127.
        b_cat_orig (bool): If True, concatenates the original input channels with the
                           FFT real and imaginary channels. If False, only FFT channels
                           are returned. Defaults to False.
    """
    def __init__(self, freq_pad: int = 127, b_cat_orig: bool = False):
        super(FourierLayer, self).__init__()
        self.freq_pad = freq_pad
        self.b_cat_orig = b_cat_orig

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure height and width are even by reflection padding
        pad_h = x.size(-2) % 2
        pad_w = x.size(-1) % 2
        if pad_h != 0 or pad_w != 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        # Compute 2D real FFT
        x_fft = torch.fft.rfft2(x, dim=(-2, -1))
        real = x_fft.real
        imag = x_fft.imag

        # Pad FFT outputs in the frequency domain to a fixed size
        # Padding on the last dimension (frequency axis)
        real_p = F.pad(real, (0, self.freq_pad), mode='constant', value=0)
        imag_p = F.pad(imag, (0, self.freq_pad), mode='constant', value=0)

        # Concatenate channels: either only FFT or original + FFT
        if self.b_cat_orig:
            out = torch.cat([x, real_p, imag_p], dim=1)
        else:
            out = torch.cat([real_p, imag_p], dim=1)

        return out


# Example usage:
# layer = FourierLayer(freq_pad=127, b_cat_orig=True)\# output = layer(input_tensor)


#NOT WORKING
class InverseFourierLayer(nn.Module):
    """
    A PyTorch layer that applies the inverse 2D real Fourier transform to the input tensor
    (which should contain real and imaginary frequency components, optionally preceded
    by the original spatial channels) and reconstructs the spatial-domain tensor.

    Args:
        freq_pad (int): Number of frequency bins that were padded on the right of the
                        FFT components during the forward Fourier transform. Defaults to 127.
        b_cat_orig (bool): If True, expects the input to include the original spatial channels
                           concatenated before the FFT components, and will return a concatenation
                           of those original channels with the reconstructed tensor. Defaults to False.
    """
    def __init__(self, freq_pad: int = 127, b_cat_orig: bool = False):
        super(InverseFourierLayer, self).__init__()
        self.freq_pad = freq_pad
        self.b_cat_orig = b_cat_orig

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Determine how many channels correspond to original vs FFT components
        total_ch = x.size(1)
        if self.b_cat_orig:
            # Input layout: [orig | real_fft | imag_fft]
            orig_ch = total_ch // 3
            orig = x[:, :orig_ch]
            real = x[:, orig_ch:orig_ch*2]
            imag = x[:, orig_ch*2:orig_ch*3]
        else:
            # Input layout: [real_fft | imag_fft]
            orig = None
            real_ch = total_ch // 2
            real = x[:, :real_ch]
            imag = x[:, real_ch:real_ch*2]

        # Remove frequency padding on the last dimension
        if self.freq_pad > 0:
            real = real[..., :-self.freq_pad]
            imag = imag[..., :-self.freq_pad]

        # Reconstruct complex tensor and apply inverse FFT
        freq_tensor = torch.complex(real, imag)
        recon = torch.fft.irfft2(freq_tensor, dim=(-2, -1))

        # If requested, concatenate original spatial channels back
        if self.b_cat_orig:
            return torch.cat([orig, recon], dim=1)
        else:
            return recon
        
        
