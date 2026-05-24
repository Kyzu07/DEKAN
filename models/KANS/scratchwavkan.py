import torch as th
import numpy as np # Used for np.sqrt in initialization

class NaiveWaveletKANLayer(th.nn.Module):
    """
    A KAN layer using Mexican Hat wavelets as the basis functions.
    This replaces the Fourier (sine/cosine) basis with a set of scaled
    and translated Mexican Hat wavelets, providing a localized and
    oscillatory representation.

    The basis functions are defined by fixed translations (centers) and
    varying scales (dilations), creating a multi-resolution-like effect.
    Specifically, scales are inversely proportional to their index,
    meaning smaller scales (sharper, higher-frequency wavelets) for larger indices.

    Args:
        inputdim (int): Dimension of the input features.
        outdim (int): Dimension of the output features.
        gridsize (int): Number of distinct wavelet basis functions per input dimension.
                        Each function will have a unique, pre-defined scale and translation.
        addbias (bool): Whether to add a bias term.
        smooth_initialization (bool): If True, coefficients corresponding to smaller
                                     (higher-frequency/sharper) wavelets are initialized
                                     with smaller magnitudes, promoting smoother functions initially.
        device (torch.device, optional): The device (cpu or cuda) to place the tensors on.
    """
    def __init__(self, inputdim, outdim, gridsize, addbias=True, smooth_initialization=False, device=None):
        super(NaiveWaveletKANLayer, self).__init__()
        self.gridsize = gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim
        self.device = device if device is not None else th.device('cpu')

        # Define fixed translations (b in (x-b)/a) for the Mexican Hat wavelets.
        # Spreading translations over a reasonable range (e.g., [-5, 5])
        # helps cover typical normalized input values. These are not learned parameters.
        # Shape: (1, 1, 1, gridsize) for broadcasting with input x.
        self.translations = th.nn.Parameter(
            th.linspace(-5.0, 5.0, gridsize, device=self.device).reshape(1, 1, 1, gridsize),
            requires_grad=False
        )

        # Define fixed scales (a in (x-b)/a) for the Mexican Hat wavelets.
        # Smaller scales 'a' correspond to more compressed, higher-frequency wavelets.
        # We make them inversely proportional to (index + 1) to cover a range of frequencies.
        # A `base_scale` of `(max_translation - min_translation) / some_factor` (e.g., 10.0 / 2 = 5.0)
        # provides a good starting point for the broadest wavelet.
        base_scale = 5.0 
        scales_tensor = base_scale / (th.arange(gridsize, device=self.device) + 1.0)
        # Shape: (1, 1, 1, gridsize) for broadcasting.
        self.scales = th.nn.Parameter(
            scales_tensor.reshape(1, 1, 1, gridsize),
            requires_grad=False
        )

        # The `grid_norm_factor` logic. For wavelets, "higher frequency" means smaller scales.
        # Since `self.scales[j]` is inversely proportional to `(j+1)`, a larger `j` indicates
        # a smaller scale (higher effective frequency). Thus, `(th.arange(gridsize) + 1)**2`
        # appropriately attenuates coefficients for higher-frequency wavelets when
        # `smooth_initialization` is True.
        if smooth_initialization:
            grid_norm_factor_val = (th.arange(gridsize, device=self.device) + 1)**2
        else:
            # Make it a vector filled with the same scalar value
            grid_norm_factor_val = th.full((gridsize,), th.sqrt(th.tensor(gridsize, dtype=th.float32, device=self.device)), device=self.device)

        # grid_norm_factor_val = (th.arange(gridsize, device=self.device) + 1)**2 if smooth_initialization else th.sqrt(th.tensor(gridsize, dtype=th.float32, device=self.device))
        
        # Reshape grid_norm_factor to (1, 1, 1, gridsize) for broadcasting during initialization
        grid_norm_factor = grid_norm_factor_val.reshape(1, 1, gridsize)

        # `waveletcoeffs` stores the learnable weights for each Mexican Hat wavelet instance.
        # Shape: (outdim, inputdim, gridsize) - there's one set of `gridsize` wavelets
        # per (input_dim, output_dim) pair.
        # self.waveletcoeffs = th.nn.Parameter(
        #     th.randn(outdim, inputdim, gridsize, device=self.device) / 
        #     (np.sqrt(inputdim) * grid_norm_factor)
        # )
        self.waveletcoeffs = th.nn.Parameter(
            th.randn(outdim, inputdim, gridsize, device=self.device) / 
            (np.sqrt(inputdim) * grid_norm_factor)
        )

        if self.addbias:
            self.bias = th.nn.Parameter(th.zeros(1, outdim, device=self.device))

    # x.shape: ( ..., inputdim )
    # out.shape: ( ..., outdim )
    def forward(self, x):
        xshp = x.shape
        outshape = xshp[0:-1] + (self.outdim,)
        # Flatten leading dimensions into a single batch dimension
        x = th.reshape(x, (-1, self.inputdim)) # Shape: (batch_size_flat, input_dim)

        # Reshape x for broadcasting with `translations` and `scales`:
        # `(batch_size_flat, 1, input_dim, 1)` -> `input_dim` is the dimension we're iterating over for basis functions.
        # The `1` in the second dimension helps broadcasting with `self.waveletcoeffs` later.
        xrshp = th.reshape(x, (x.shape[0], 1, x.shape[1], 1))

        # Calculate the normalized variable u = (x - b) / a
        # `xrshp`: (batch_size_flat, 1, input_dim, 1)
        # `self.translations`: (1, 1, 1, gridsize)
        # `self.scales`: (1, 1, 1, gridsize)
        # Result `u`: (batch_size_flat, 1, input_dim, gridsize)
        u = (xrshp - self.translations) / self.scales

        # Compute the Mexican Hat wavelet values: (1 - u^2) * exp(-0.5 * u^2)
        # Note: We omit the 1/sqrt(a) or 1/a normalization factor often seen in wavelet definitions
        # because its effect can be absorbed by the learnable `waveletcoeffs`.
        wavelet_values = (1 - u**2) * th.exp(-0.5 * u**2)
        
        # The output `y` is computed by summing the contributions of all
        # wavelet basis functions across each input dimension for each output dimension.
        # This is equivalent to an element-wise product followed by summation over
        # `input_dim` and `gridsize`.
        # `wavelet_values`: (batch_size_flat, 1, input_dim, gridsize) -> `b,a,i,g`
        # `self.waveletcoeffs`: (outdim, input_dim, gridsize) -> `o,i,g`
        # Using einsum for clarity and direct matching of required summation:
        # Sums over `input_dim` (`i`) and `gridsize` (`g`).
        # `a` (the dummy `1` dimension in `wavelet_values`) broadcasts.
        # Result `y`: (batch_size_flat, outdim) -> `b,o`
        y = th.einsum("baig,oig->bo", wavelet_values, self.waveletcoeffs)

        if self.addbias:
            y += self.bias
        
        # Reshape the output back to its original leading dimensions plus `outdim`.
        y = th.reshape(y, outshape)
        return y