"""
Common functions that can be used by other python scripts:
1) init_weights
2) get_padding
3) convert_pad_shape
4) intersperse
5) kl_divergence
6) rand_gumbel
7) rand_gumbel_like
8) slice_segments
9) rand_slice_segments
10) get_timing_signal_1d
11) add_timing_signal_1d
12) cat_timing_signal_1d
13) subsequent_mask
14) fused_add_tanh_sigmoid_multiply
15) shift_1d
16) sequence_mask
17) generate_path
18) clip_grad_value
"""


import math
from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


def init_weights(m: nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    """Initialize model weights.

    Args:
        m (nn.Module):
            Pytorch model.
        mean (float):
            Mean value for initializing weights (Default: 0.0).
        std (float):
            Standard deviation for intiializing weights (Default: 0.01).

    Returns:
        None

    """

    # Extract name of model class
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        # Initalize weights with values drawn from normal distribution
        # with specified mean and std
        m.weight.data.normal_(mean, std)

    # return None


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """Calculates padding size requried for convolution.

    Args:
        kernel_size (int):
            Size of convolutional kernel.
        dilation (int):
            Dilation factor for convolution (Default: 1 i.e. no dilation).

    Returns:
        (int):
            Padding size required for both sides of input so that
            output size matches input size.
    """

    return int((kernel_size * dilation - dilation) / 2)


def convert_pad_shape(pad_shape: List[List[int]]) -> List[int]:
    """Convert pad_shape into flattened list.

    Args:
        pad_shape (List[List[int]]):
            List of list containing padding amounts for a particular dimension.

    Returns:
        pad_shape (List[int]):
            Flattened List of integers as 'pad' input parameter
            to torch.nn.functional.pad.
    """

    # Reverse order of elements in list 'pad_shape' because
    # torch.nn.functional.pad starts from last dimension and moving forward
    lst = pad_shape[::-1]

    # Flatten 'reverse' list
    pad_shape = [item for sublist in lst for item in sublist]

    return pad_shape


def intersperse(lst: List[int], item: int):
    """Intersperse or add item in between elements in lst.

    Args:
      lst (List[int]):
        List of integer ids after converting symbols in text strings to
        corresponding integer id.
      item (int):
        Integer id to be interspersed in lst i.e. add item in between
        each integer ids.

    Returns:
      result (List[int]):
        List of interpersed integer id.
    """

    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst

    return result


def kl_divergence(m_p: float, logs_p: float, m_q: float, logs_q: float) -> torch.Tensor:
    """Computes Kullback-Leibler (KL) divergence between
    probability distribution P and Q i.e. KL(P||Q).

    Args:
        m_p (float):
            Mean of probability distribution P.
        logs_p (float):
            Log-variances of probability distribution P.
        m_q (float):
            Mean of probability distribution Q.
        logs_q (float):
            Log-variances of probability distribution Q.

    Returns:
        kl (torch.Tensor):
            KL divergence between probability distribution P and Q.
    """

    # Compute first term of KL divergence
    kl = (logs_q - logs_p) - 0.5

    # Compute second term of KL divergence
    kl += 0.5 * (torch.exp(2.0 * logs_p) + ((m_p - m_q) ** 2)) * torch.exp(-2.0 * logs_q)

    return kl


def rand_gumbel(shape: Tuple[int]) -> torch.Tensor:
    """Sample from the Gumbel distribution, protect from overflows.

    Args:
        shape (Tuple[int]):
            Shape of output tensor.

    Returns:
        (torch.Tensor):
            Torch tensor containing Gumbel-distributed random samples.
    """

    # Generate uniform random samples in range (0.00001, 0.99999)
    uniform_samples = torch.rand(shape) * 0.99998 + 0.00001

    # Generate Gumbel-distributed samples by applying double
    # negative logarithmic operations to scaled uniform samples
    return -torch.log(-torch.log(uniform_samples))


def rand_gumbel_like(x: torch.Tensor) -> torch.Tensor:
    """Generate Gumbel-distributed samples with same shape as x.

    Args:
        x (torch.Tensor):
            Tensor of consideration.

    Returns:
        g (torch.Tensor):
            Generated random samples from Gumbel distribution
            with shape matching 'x'.
    """

    g = rand_gumbel(x.size()).to(dtype=x.dtype, device=x.device)

    return g


# HYL: y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
def slice_segments(x: torch.Tensor, ids_str: torch.Tensor, segment_size: int = 4) -> torch.Tensor:
    """Slice segments of input tensor x along 3rd dimension.

    Args:
        x (torch.Tensor):
            Tensor of consideration. Shape [batch_size, num_mels, max_spec_len].
        ids_str (torch.Tensor):
            Tensor containing indices representing the start of segment.
        segment_size (int):
            Size of slice = hps.train.segment_size // hps.data.hop_length
            (default: 4).

    Returns:
        ret (torch.Tensor):
            Sliced tensor of 'x' tensor.
            Shape [batch_size, num_mels, segment_size]
    """

    # Create tensor with zeros having same shape as first
    # 'segment_size' elements of x in the 3rd dimension
    ret = torch.zeros_like(x[:, :, :segment_size])

    # Iterate through batch
    for i in range(x.size(0)):
        # Starting and ending index for current segment i
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size

        # Assign sliced segment from 'x' to corresponding slice in 'ret'
        ret[i] = x[i, :, idx_str:idx_end]

    return ret


def rand_slice_segments(
    x: torch.Tensor, x_lengths: Optional[int] = None, segment_size: int = 4
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Randomly select segments of given 'x' tensor along time dimension.

    Args:
        x (torch.Tensor):
            Tensor of consideration.
        x_lengths (Optional[int]):
            Number of time steps.
        segment_size (int):
            Size of segments of latent representations to
            extract (default: 4).

    Returns:
        ret (torch.Tensor):
            Sliced tensor of 'x' tensor.
        ids_str (torch.Tensor):
            Tensor containing list of randomly generated starting indices
            within valid ranges for each batch.
    """

    # Get batch size (b), number of channels (d),
    # and total time steps (t) in x
    b, _, t = x.size()
    if x_lengths is None:
        x_lengths = t

    # Calculate maximum starting index (ids_str_max) for each batch
    ids_str_max = x_lengths - segment_size + 1

    # Randomly generate starting indices (ids_str)
    # within valid ranges for each batch
    # pylint: disable=E1101
    ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)

    # Extract segments from x based on random starting indices
    ret = slice_segments(x, ids_str, segment_size)

    return ret, ids_str


def get_timing_signal_1d(
    length: int, channels: int, min_timescale: float = 1.0, max_timescale: float = 1.0e4
) -> torch.Tensor:
    """Generates a timing signal for 1-dimensional sequences.

    Args:
        length (int):
            Number of time steps.
        channels (int):
            Number of channels (components) in timing signal.
        min_timescale (float):
            Fastest repeating pattern to capture in signal.
        max_timescale
            Slowest repeating pattern to capture in signal.

    Returns:
        signal (torch.Tensor):
            Signal that provides positional encoding.
    """

    # Create a 1D tensor 'position' containing values
    # from 0 to 'length-1'
    # pylint: disable=E1101
    position = torch.arange(length, dtype=torch.float)

    # Both sine and cosine componenets are generated for each timescale
    # Therefore number of timescales is set to half of number of channels
    num_timescales = channels // 2

    # Calculate the increment in logarithmic timescale values
    log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / (num_timescales - 1)

    # Calculate inverse timescales using exponential values to cover a
    # range of timescales between minimum and maximum values
    # pylint: disable=E1101
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales, dtype=torch.float) * -log_timescale_increment
    )

    # Create matrix where each row correspond to a different timescale
    scaled_time = position.unsqueeze(0) * inv_timescales.unsqueeze(1)

    # Compute sine and cosine components of the timing signal
    # pylint: disable=E1101
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 0)

    # Pad the signal along channel dimension to match
    # the desired number of channels
    signal = F.pad(signal, [0, 0, 0, channels % 2])

    # Reshape the signal to have shape (1, channels, length)
    signal = signal.view(1, channels, length)

    return signal


def add_timing_signal_1d(x: torch.Tensor, min_timescale: float = 1.0, max_timescale: float = 1.0e4) -> torch.Tensor:
    """Add timing signal to given input tensor.

    Args:
        x (torch.Tensor):
            Tensor in consideration.
        min_timescale (float):
            Fastest repeating pattern to capture in signal.
        max_timescale (float):
            Slowest repeating pattern to capture in signal.

    Returns:
        (torch.Tensor):
            Input tensor with positional encoding.
    """

    # batch size, channels and sequence length
    _, channels, length = x.size()

    # Generate the timing signal for the given
    # sequence length and number of channels
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)

    # Add timing signal to input tensor
    return x + signal.to(dtype=x.dtype, device=x.device)


def cat_timing_signal_1d(
    x: torch.Tensor,
    min_timescale: float = 1.0,
    max_timescale: float = 1.0e4,
    axis: int = 1,
) -> torch.Tensor:
    """Concatenate timing signal to given input tensor.

    Args:
        x (torch.Tensor):
            Tensor in consideration.
        min_timescale (float):
            Fastest repeating pattern to capture in signal.
        max_timescale (float):
            Slowest repeating pattern to capture in signal.
        axis (int):
            Axis to concatenate (Default: 1 i.e. column-wise).

    Returns:
        (torch.Tensor):
            Input tensor with positional encoding.
    """

    # batch size, channels and sequence length
    _, channels, length = x.size()

    # Generate the timing signal for the given
    # sequence length and number of channels
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)

    # Concatenate timing signal to input tensor
    # pylint: disable=E1101
    return torch.cat([x, signal.to(dtype=x.dtype, device=x.device)], axis)


def subsequent_mask(length) -> torch.Tensor:
    """Generate mask to prevent attending to future positions.

    Args:
        length (int):
            Sequence length of signal.

    Returns:
        mask (torch.Tensor):
            Generated mask tensor.
    """

    # Create mask by zeroing out upper-right portion of square matrix
    # with dimensions: length x length. Unsqueeze operations to make
    # mask tensor compatible with batch dimension
    # pylint: disable=E1101
    mask = torch.tril(torch.ones(length, length)).unsqueeze(0).unsqueeze(0)

    return mask


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(
    input_a: torch.Tensor, input_b: torch.Tensor, n_channels: torch.Tensor
) -> torch.Tensor:
    """Combines several operations efficiently using element-wise addition,
    hyperbolic tangent, sigmoid, and element-wise multiplication.

    Args:
        input_a (torch.Tensor):
            Input tensor.
        input_b (torch.Tensor):
            Another input tensor.
        n_channels (torch.Tensor):
            Tensor with a single integer value.

    Returns:
        acts (torch.Tensor):
            Final tensor output after multiple operations.
    """

    # Capture number of channels
    n_channels_int = n_channels[0]

    # Element-wise addition
    in_act = input_a + input_b

    # Apply hyberbolic tangent activation
    # pylint: disable=E1101
    t_act = torch.tanh(in_act[:, :n_channels_int, :])

    # Apply sigmoid activation
    # pylint: disable=E1101
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])

    # Element-wise multiplication between tanh and sigmoid activation
    acts = t_act * s_act

    return acts


def shift_1d(x: torch.Tensor) -> torch.Tensor:
    """Shift elements of 1-dimensional tensor to left by one position.

    Args:
        x (torch.Tensor):
            1-dimensional tensor to be adjusted.

    Returns:
        x (torch.Tensor):
            Adjusted tensor with elements shifted to left by one position.
    """

    # Removes last element from padded tensor, effectively shifting all
    # elements to the left.
    x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [1, 0]]))[:, :, :-1]

    return x


def sequence_mask(length: torch.Tensor, max_length: Optional[int] = None) -> torch.Tensor:
    """Generate binary mask for sequences based on their lengths.

    Args:
        length (torch.Tensor):
            1-dimensional tensor in consideration.
        max_length (Optional[int]):
            User-specified max length of tensor.

    Returns:
        (torch.Tensor):
    """

    # Set max_length to be the maximum length of
    # tensor if not provided
    if max_length is None:
        max_length = length.max()

    # Create 1D tensor containing values from 0 to 'max_length-1'
    # pylint: disable=E1101
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)

    # Create binary mask by comparing each element in 1D tensor
    # x with actual lengths of sequence. 'True' if corresponding
    # position is within length of sequence.
    return x.unsqueeze(0) < length.unsqueeze(1)


def generate_path(duration: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Generate path given signal duration and mask.

    Args:
        duration (torch.Tensor):
            Tensor of shape [b, 1, t_x] where b and t_x represent batch size
            and time dimension respectively.
        mask (torch.Tensor)
            Tensor of shape [b, 1, t_y, t_x] where b, t_y and t_x represent
            batch size, and time dimensions along 2 different axis respectively.

    Returns:
        path (torch.Tensor):
            Path that is sequentially aligned.

    """

    # Extract batch size and time dimensions from mask
    b, _, t_y, t_x = mask.shape

    # Compute cumulative sum of duration tensor along last dimension
    # pylint: disable=E1101
    cum_duration = torch.cumsum(duration, -1)

    # Flatten cum_duration tensor by concatenating elements
    # along batch and time dimension
    cum_duration_flat = cum_duration.view(b * t_x)

    # Apply binary mask to cumulative duration to generate path
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)

    # Reshape tensor to shape [b, t_x, t_y]
    path = path.view(b, t_x, t_y)

    # Shift elements of each row by one position to the left
    path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]

    # Reshape path to [b, 1, t_y, t_x] to make it compatible
    # with shape of mask tensor before element-wise multiplication
    path = path.unsqueeze(1).transpose(2, 3) * mask

    return path


def clip_grad_value_(parameters: torch.Tensor, clip_value: float, norm_type: int = 2) -> float:
    """Compute gradient norm and optionally, clip gradients
    of a list of parameters.

    Args:
        parameters (torch.Tensor)
            Tensor containing list of parameters.
        clip_value (float):
            Maximum allwoed absolute value for gradients.
        norm_type (int):
            Type of norm used for computing gradient norm
            (Default: 2 i.e. L2 norm).

    Returns:
        total_norm (float):
            Norm of aggregated gradients after applying gradient clipping
    """

    # Convert parameters into a list of single tensor
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    # Retain tensors whose gradients are not None
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)

    if clip_value is not None:
        clip_value = float(clip_value)

    total_norm = 0
    for param in parameters:
        # Compute gradient norm using specified norm_type
        param_norm = param.grad.data.norm(norm_type)

        # Append computed parameter norm raised
        # to power of 'norm_type'
        total_norm += param_norm.item() ** norm_type

        # Clip gradient data to range [-clip_value, clip_value]
        # if clip_value provided
        if clip_value is not None:
            param.grad.data.clamp_(min=-clip_value, max=clip_value)

    # Norm of aggregated gradients after applying gradient clipping
    total_norm = total_norm ** (1.0 / norm_type)

    return total_norm
