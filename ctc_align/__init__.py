"""
Author: Huahuan Zheng (maxwellzh@outlook.com)

CUDA binding implementation of CTC alignment.

P.S.
CTC alignment means removing all consecutive symbols and zeros (blank symbols)
    from the source sequences. e.g.
    [0, 1, 1, 0, 1, 2, 2, 0, 0, 3] -> [1, 1, 2, 3]
"""
import ctc_align._C as core
import torch
from pkg_resources import get_distribution
from typing import *

__version__ = get_distribution('ctc_align').version


def align_(src_indices: Union[torch.IntTensor, torch.LongTensor], src_lens: Optional[torch.IntTensor] = None) -> Tuple[Union[torch.IntTensor, torch.LongTensor], torch.IntTensor]:
    """Conduct CTC alignment in-place.
    
    Argument:
        src_indices (torch.int | torch.long): (N, T), the source indices. assume 0 as <blk>
        src_lens    (optional, torch.int)   : (N, ), lengths of each sequence, if None, assume all equal to T.
    
    Return:
        (indices, lens)
        indices : (N, T), same dtype and shape as `src_indices`, indeed the reference of `src_indices`, since this is in-place alignment.
        lens    : (N, ), indicate the length of aligned sequences.
    """
    assert isinstance(src_indices, torch.Tensor)
    assert src_indices.is_cuda, "expect indices to be CUDA tensor."
    assert src_indices.dim(
    ) == 2, f"expect 2-dim indices, instead {src_indices.dim()}"
    assert src_indices.dtype in (
        torch.int, torch.long), f"expect indices to be one of [torch.int, torch.long], instead {src_indices.dtype}"

    if src_lens is None:
        src_lens = src_indices.shape[1] * torch.ones(
            src_indices.shape[0], device=src_indices.device, dtype=torch.int)
    else:
        src_lens = src_lens.to(dtype=torch.int32, device=src_indices.device)

    return core.align_(src_indices, src_lens)


def align(src_indices: Union[torch.IntTensor, torch.LongTensor], src_lens: Optional[torch.IntTensor] = None) -> Tuple[Union[torch.IntTensor, torch.LongTensor], torch.IntTensor]:
    """Out-of-place version of `align_`"""
    return align_(src_indices.clone(), src_lens)


__all__ = [align, align_]
