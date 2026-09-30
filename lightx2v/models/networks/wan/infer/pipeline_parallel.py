"""P2P activation + pre_infer metadata for Wan layer pipeline parallel."""

from __future__ import annotations

import torch
import torch.distributed as dist

from lightx2v.models.networks.wan.infer.module_io import GridOutput, WanPreInferModuleOutput

_PP_TAG_META = 101
_PP_TAG_ACTIVATION = 100
_PP_TAG_NOISE = 102


def _global_rank(group, group_rank: int) -> int:
    return dist.get_global_rank(group, group_rank)


def _p2p_send(tensor: torch.Tensor, dst_group_rank: int, group, tag: int) -> None:
    dist.send(tensor, dst=_global_rank(group, dst_group_rank), group=group, tag=tag)


def _p2p_recv(tensor: torch.Tensor, src_group_rank: int, group, tag: int) -> None:
    dist.recv(tensor, src=_global_rank(group, src_group_rank), group=group, tag=tag)


def _dtype_from_meta(meta: torch.Tensor, device: torch.device) -> torch.dtype:
    if int(meta[0].item()):
        return torch.float32
    if int(meta[1].item()):
        return torch.bfloat16
    if int(meta[2].item()):
        return torch.int64
    # Torch RoPE transports complex tensors; preserving dtype is required
    # to keep NCCL send/receive byte counts identical.
    extra = int(meta[3].item())
    if extra:
        return {1: torch.int32, 2: torch.float64, 3: torch.complex64, 4: torch.complex128}[extra]
    return torch.float16


def _encode_dtype(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    flags = torch.zeros(4, dtype=torch.int64, device=device)
    if dtype == torch.float32:
        flags[0] = 1
    elif dtype == torch.bfloat16:
        flags[1] = 1
    elif dtype == torch.int64:
        flags[2] = 1
    elif dtype in (torch.int32, torch.float64, torch.complex64, torch.complex128):
        flags[3] = {torch.int32: 1, torch.float64: 2, torch.complex64: 3, torch.complex128: 4}[dtype]
    elif dtype != torch.float16:
        raise TypeError(f"Unsupported PP transport dtype: {dtype}")
    return flags


def _send_tensor(t: torch.Tensor, dst: int, group, tag: int) -> None:
    if t.device.type != "cuda":
        t = t.cuda(torch.cuda.current_device())
    shape = torch.zeros(8, dtype=torch.int64, device=t.device)
    shape[0] = t.ndim
    for i, s in enumerate(t.shape):
        shape[i + 1] = s
    meta = _encode_dtype(t.dtype, t.device)
    _p2p_send(shape, dst, group, tag)
    _p2p_send(meta, dst, group, tag)
    if t.numel() > 0:
        _p2p_send(t.contiguous(), dst, group, tag)


def _recv_tensor(src: int, group, tag: int, device: torch.device) -> torch.Tensor:
    shape_buf = torch.zeros(8, dtype=torch.int64, device=device)
    _p2p_recv(shape_buf, src, group, tag)
    ndim = int(shape_buf[0].item())
    shape = tuple(int(shape_buf[i + 1].item()) for i in range(ndim))
    meta = torch.zeros(4, dtype=torch.int64, device=device)
    _p2p_recv(meta, src, group, tag)
    dtype = _dtype_from_meta(meta, device)
    out = torch.empty(shape, device=device, dtype=dtype)
    if out.numel() > 0:
        _p2p_recv(out, src, group, tag)
    return out


def send_pre_metadata(pre: WanPreInferModuleOutput, dst: int, group, tag: int = _PP_TAG_META) -> None:
    for t in (pre.context, pre.embed, pre.embed0, pre.grid_sizes.tensor):
        _send_tensor(t, dst, group, tag)
    # SF (freqs/seq_lens present) uses freqs-based RoPE; skip large cos_sin on the wire.
    is_sf = getattr(pre, "freqs", None) is not None or getattr(pre, "seq_lens", None) is not None
    if (not is_sf) and pre.cos_sin is not None:
        _send_tensor(pre.cos_sin, dst, group, tag)
    else:
        _send_tensor(torch.empty(0, device=pre.x.device), dst, group, tag)
    seq_lens = getattr(pre, "seq_lens", None)
    _send_tensor(seq_lens if seq_lens is not None else torch.empty(0, device=pre.x.device), dst, group, tag)
    _send_tensor(torch.empty(0, device=pre.x.device), dst, group, tag)
    dev = pre.x.device if pre.x.is_cuda else torch.device(f"cuda:{torch.cuda.current_device()}")
    flags = torch.tensor(
        [int(getattr(pre, "valid_token_len", 0) or 0), int(getattr(pre, "valid_latent_num", 0) or 0)],
        device=dev,
        dtype=torch.int64,
    )
    _p2p_send(flags, dst, group, tag)


def recv_pre_metadata(src: int, group, device: torch.device, tag: int = _PP_TAG_META) -> WanPreInferModuleOutput:
    context = _recv_tensor(src, group, tag, device)
    embed = _recv_tensor(src, group, tag, device)
    embed0 = _recv_tensor(src, group, tag, device)
    grid_t = _recv_tensor(src, group, tag, device)
    cos_sin_t = _recv_tensor(src, group, tag, device)
    seq_lens_t = _recv_tensor(src, group, tag, device)
    freqs_t = _recv_tensor(src, group, tag, device)
    flags = torch.zeros(2, dtype=torch.int64, device=device)
    _p2p_recv(flags, src, group, tag)
    cos_sin = cos_sin_t if cos_sin_t.numel() > 0 else None
    seq_lens = seq_lens_t if seq_lens_t.numel() > 0 else None
    freqs = freqs_t if freqs_t.numel() > 0 else None
    grid_list = grid_t.reshape(-1).tolist()
    grid_tuple = tuple(int(x) for x in grid_list[-3:])
    return WanPreInferModuleOutput(
        embed=embed,
        grid_sizes=GridOutput(tensor=grid_t, tuple=grid_tuple),
        x=torch.empty(0, device=device),
        embed0=embed0,
        context=context,
        cos_sin=cos_sin,
        seq_lens=seq_lens,
        freqs=freqs,
        valid_token_len=int(flags[0].item()),
        valid_latent_num=int(flags[1].item()),
    )


def send_activation(x: torch.Tensor, dst: int, group, tag: int = _PP_TAG_ACTIVATION) -> None:
    _send_tensor(x, dst, group, tag)


def recv_activation(src: int, group, device: torch.device, tag: int = _PP_TAG_ACTIVATION) -> torch.Tensor:
    return _recv_tensor(src, group, tag, device)


def send_noise_pred(noise_pred: torch.Tensor, dst: int, group, tag: int = _PP_TAG_NOISE) -> None:
    _send_tensor(noise_pred, dst, group, tag)


def recv_noise_pred(src: int, group, device: torch.device, tag: int = _PP_TAG_NOISE) -> torch.Tensor:
    return _recv_tensor(src, group, tag, device)


class P2pAsyncSend:
    """Non-blocking P2P send; keeps tensor refs alive until wait()."""

    def __init__(self) -> None:
        self._works: list[dist.Work] = []
        self._keepalive: list[torch.Tensor] = []

    def _isend(self, tensor: torch.Tensor, dst: int, group, tag: int) -> None:
        t = tensor.contiguous()
        self._keepalive.append(t)
        self._works.append(dist.isend(t, dst=_global_rank(group, dst), group=group, tag=tag))

    def send_activation(self, x: torch.Tensor, dst: int, group) -> None:
        shape = torch.zeros(8, dtype=torch.int64, device=x.device)
        shape[0] = x.ndim
        for i, s in enumerate(x.shape):
            shape[i + 1] = s
        meta = _encode_dtype(x.dtype, x.device)
        self._keepalive.extend([shape, meta])
        self._isend(shape, dst, group, _PP_TAG_ACTIVATION)
        self._isend(meta, dst, group, _PP_TAG_ACTIVATION)
        self._isend(x, dst, group, _PP_TAG_ACTIVATION)

    def wait(self) -> None:
        for work in self._works:
            work.wait()
        self._works.clear()
        self._keepalive.clear()
