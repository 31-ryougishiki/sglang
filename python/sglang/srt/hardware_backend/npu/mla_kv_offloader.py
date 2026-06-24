"""
Cross-rank KV buffer sharing for MLA (Multi-head Latent Attention) on Ascend NPU.

In MLA with tensor parallelism, all ranks compute identical KV cache data
(MLA has 1 KV head, not split by TP). This module distributes buffer ownership
across TP ranks to save HBM via async send/recv pipelining.

Usage::

    offloader = MLAKVOffloader(
        layer_num=60,
        tp_rank=dist.get_rank(),
        tp_size=dist.get_world_size(),
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        store_dtype=torch.bfloat16,
        device="npu:0",
        size=131072,
        page_size=1,
    )
    # In set_kv_buffer:
    offloader.pre_load(local_layer_id)
    # write K/V to offloader.k_buffer[local_layer_id] / v_buffer[local_layer_id]
    ...
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist

from sglang.srt.utils import get_bool_env_var

logger = logging.getLogger(__name__)


def _get_tensor_size_bytes(t: torch.Tensor) -> int:
    """Return the number of bytes consumed by *t* in memory."""
    return t.numel() * t.element_size()


class MLAKVOffloader:
    """Cross-rank KV buffer sharing manager for MLA on Ascend NPU.

    **Layer ownership** is round-robin: rank *R* owns layers where
    ``layer_id % tp_size == R``.  Owned layers are stored in dedicated
    tensors; non-owned layers share a small pool of reusable buffers whose
    content is refreshed via async ``isend`` / ``irecv``.

    **Pipeline** (one layer ahead):

    * *Owned layer*: wait pending sends → post 1 ``irecv`` for the next
      non-owned layer → compute.
    * *Non-owned layer*: wait ``irecv`` for this layer → schedule 1
      ``isend`` of the previous owned layer to the next peer → compute.

    This keeps at most one receive in-flight at any time (single shared
    buffer per buffer type), which is safe because layers are processed
    sequentially and data is consumed before the next overwrite.

    Parameters
    ----------
    layer_num:
        Number of MLA layers managed by this offloader (local to this PP
        rank, i.e. ``end_layer - start_layer``).
    tp_rank:
        Rank within the tensor-parallel group.
    tp_size:
        Total number of ranks in the tensor-parallel group (TP ≥ 2).
    kv_lora_rank:
        Dimension of the compressed KV latent (k_buffer).
    qk_rope_head_dim:
        Dimension of the RoPE-decoupled key component (v_buffer).
    store_dtype:
        Storage data type for the buffers (typically ``dtype``, or
        ``torch.uint8`` when the compute dtype is FP8).
    device:
        Torch device string (e.g. ``"npu:0"``).
    size:
        Maximum number of token slots in the pool.
    page_size:
        Page granularity (1 for MLA, 64 for MHA).
    index_head_dim:
        Optional dimension for the indexer K buffer (DSA / DeepSeek).
    enable_k_buffer_sharing:
        When *False* every rank allocates a full *k_buffer* for every
        layer (same as the non-shared baseline).
    enable_v_buffer_sharing:
        When *False* every rank allocates a full *v_buffer* for every
        layer.
    enable_index_k_sharing:
        When *False* every rank allocates a full *index_k_buffer* for
        every layer.
    offload_group:
        Optional pre-created ``ProcessGroup``.  If *None* a new group
        containing ranks ``[0, tp_size)`` is created.

    Environment variables
    ---------------------
    ``ASCEND_MLA_KV_SHARING`` (bool, default ``"True"``):
        Master switch – when ``"False"`` all sharing is disabled and every
        rank allocates full, dedicated buffers for all layers.
    ``ASCEND_MLA_KV_K_SHARING`` (bool, default ``"True"``):
        Enable *k_buffer* sharing.
    ``ASCEND_MLA_KV_V_SHARING`` (bool, default ``"True"``):
        Enable *v_buffer* sharing.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        layer_num: int,
        tp_rank: int,
        tp_size: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        store_dtype: torch.dtype,
        device: str,
        size: int,
        page_size: int,
        index_head_dim: Optional[int] = None,
        enable_k_buffer_sharing: bool = True,
        enable_v_buffer_sharing: bool = True,
        enable_index_k_sharing: bool = True,
        offload_group: Optional[dist.ProcessGroup] = None,
    ):
        # -- basic parameters --
        self.layer_num = layer_num
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.store_dtype = store_dtype
        self.device = device
        self.size = size
        self.page_size = page_size
        self.index_head_dim = index_head_dim

        # -- master switch --
        self._enabled = get_bool_env_var("ASCEND_MLA_KV_SHARING", "True")
        if not self._enabled:
            enable_k_buffer_sharing = False
            enable_v_buffer_sharing = False
            enable_index_k_sharing = False

        # -- per-buffer-type switches --
        self._share_k = enable_k_buffer_sharing and tp_size > 1
        self._share_v = enable_v_buffer_sharing and tp_size > 1
        self._share_ik = enable_index_k_sharing and index_head_dim is not None and tp_size > 1

        # -- layer ownership (round-robin) --
        self._owned_layers: List[int] = sorted(
            i for i in range(layer_num) if i % tp_size == tp_rank
        )
        self._recv_layers: List[int] = sorted(
            i for i in range(layer_num) if i % tp_size != tp_rank
        )

        # -- communication group --
        if tp_size > 1:
            self._offload_group: dist.ProcessGroup = (
                offload_group
                if offload_group is not None
                else dist.new_group(list(range(tp_size)))
            )
            self._peer_ranks: List[int] = [r for r in range(tp_size) if r != tp_rank]
        else:
            self._offload_group = None
            self._peer_ranks = []

        # -- pipeline state --
        self._pending_sends: List[dist.Work] = []
        self._pending_recv: Optional[List[dist.Work]] = None
        self._pending_recv_layer: Optional[int] = None

        # Indices for round-robin scheduling
        self._next_owned_idx: int = 0  # index into _owned_layers (for sending)
        self._next_recv_idx: int = 0   # index into _recv_layers (for receiving)
        self._next_peer_idx: int = 0   # index into _peer_ranks (round-robin peers)

        # -- allocate buffers --
        self.k_buffer: List[torch.Tensor]
        self.v_buffer: List[torch.Tensor]
        self.index_k_buffer: Optional[List[torch.Tensor]]
        self._allocate_buffers()

    # ------------------------------------------------------------------
    # Buffer allocation
    # ------------------------------------------------------------------

    def _allocate_buffers(self) -> None:
        """Create per-layer buffer lists with shared tensors for non-owned layers."""
        num_pages = self.size // self.page_size + 1
        k_shape = (num_pages, self.page_size, 1, self.kv_lora_rank)
        v_shape = (num_pages, self.page_size, 1, self.qk_rope_head_dim)

        # Shared tensors – one per buffer type, reused by all non-owned layers
        shared_k = torch.zeros(k_shape, dtype=self.store_dtype, device=self.device)
        shared_v = torch.zeros(v_shape, dtype=self.store_dtype, device=self.device)
        shared_ik: Optional[torch.Tensor] = None
        if self.index_head_dim is not None:
            shared_ik = torch.zeros(
                (num_pages, self.page_size, 1, self.index_head_dim),
                dtype=self.store_dtype,
                device=self.device,
            )

        self.k_buffer = []
        self.v_buffer = []
        self.index_k_buffer = [] if self.index_head_dim is not None else None

        for local_id in range(self.layer_num):
            if local_id in self._owned_layers:
                # Owned layer → dedicated tensor
                k_buf = torch.zeros(k_shape, dtype=self.store_dtype, device=self.device)
                v_buf = torch.zeros(v_shape, dtype=self.store_dtype, device=self.device)
            else:
                # Non-owned layer → shared tensor (or dedicated if sharing disabled)
                k_buf = (
                    shared_k
                    if self._share_k
                    else torch.zeros(k_shape, dtype=self.store_dtype, device=self.device)
                )
                v_buf = (
                    shared_v
                    if self._share_v
                    else torch.zeros(v_shape, dtype=self.store_dtype, device=self.device)
                )

            self.k_buffer.append(k_buf)
            self.v_buffer.append(v_buf)

            if self.index_head_dim is not None:
                if local_id in self._owned_layers:
                    ik = torch.zeros(
                        (num_pages, self.page_size, 1, self.index_head_dim),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                else:
                    ik = (
                        shared_ik
                        if self._share_ik
                        else torch.zeros(
                            (num_pages, self.page_size, 1, self.index_head_dim),
                            dtype=self.store_dtype,
                            device=self.device,
                        )
                    )
                self.index_k_buffer.append(ik)

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def pre_load(self, local_layer_id: int) -> None:
        """Call **before** ``set_kv_buffer`` writes to *local_layer_id*.

        Manages the async send / recv pipeline so that non-owned layer
        data is available when attention needs it.
        """
        if self.tp_size <= 1:
            return

        if local_layer_id in self._owned_layers:
            self._pre_load_owned()
        else:
            self._pre_load_recv(local_layer_id)

    def _pre_load_owned(self) -> None:
        """Pipeline step for a layer whose buffer is owned locally."""
        # 1. Drain all pending sends from the previous block of owned layers
        self._wait_pending_sends()

        # 2. Post one irecv for the next non-owned layer (pre-fetch 1 ahead)
        self._post_recv_ahead()

    def _pre_load_recv(self, local_layer_id: int) -> None:
        """Pipeline step for a layer whose buffer must be received from a peer."""
        # 1. Wait for this layer's data to arrive (if it was posted)
        self._wait_pending_recv(local_layer_id)

        # 2. Schedule one isend: send the current owned layer to the next peer
        self._schedule_send()

        # 3. Post irecv for the next non-owned layer (pre-fetch 1 ahead)
        self._post_recv_ahead()

    # ------------------------------------------------------------------
    # Internal: pipeline helpers
    # ------------------------------------------------------------------

    def _wait_pending_sends(self) -> None:
        """Block until all in-flight *isend* operations complete."""
        for handle in self._pending_sends:
            handle.wait()
        self._pending_sends.clear()

    def _wait_pending_recv(self, local_layer_id: int) -> None:
        """Wait for the previously posted irecv if it targets *local_layer_id*."""
        if self._pending_recv_layer == local_layer_id and self._pending_recv is not None:
            for handle in self._pending_recv:
                handle.wait()
            self._pending_recv = None
            self._pending_recv_layer = None
        # else: the recv hasn't been posted yet (happens for the first few
        # layers of prefill).  This is safe because prefill writes local K/V
        # directly and only reads the positions it just wrote.

    def _schedule_send(self) -> None:
        """Send the current owned layer's buffers to the next peer rank.

        The *current* owned layer is the one pointed to by
        ``_next_owned_idx``.  Its data may be from the previous decode
        step (if it has not been computed yet in this step), which is the
        correct version for the receiver at this step.
        """
        if not self._peer_ranks or not self._owned_layers:
            return

        peer = self._peer_ranks[self._next_peer_idx]
        owned_layer = self._owned_layers[self._next_owned_idx]

        # Send k_buffer
        if self._share_k:
            self._pending_sends.append(
                dist.isend(
                    self.k_buffer[owned_layer], peer, group=self._offload_group
                )
            )
        # Send v_buffer
        if self._share_v:
            self._pending_sends.append(
                dist.isend(
                    self.v_buffer[owned_layer], peer, group=self._offload_group
                )
            )
        # Send index_k_buffer
        if self._share_ik and self.index_k_buffer is not None:
            self._pending_sends.append(
                dist.isend(
                    self.index_k_buffer[owned_layer],
                    peer,
                    group=self._offload_group,
                )
            )

        # Advance peer index; when we've sent to all peers, advance the
        # owned layer index.
        self._next_peer_idx += 1
        if self._next_peer_idx >= len(self._peer_ranks):
            self._next_peer_idx = 0
            self._next_owned_idx = (self._next_owned_idx + 1) % len(self._owned_layers)

    def _post_recv_ahead(self) -> None:
        """Post one ``irecv`` for the next non-owned layer, if any.

        Only one recv is in-flight at a time because all non-owned layers
        share the same buffer tensors.  Sequential layer processing
        guarantees the buffer is consumed before the next overwrite.
        """
        if self._pending_recv is not None:
            return  # already have one in-flight

        if not self._recv_layers:
            return

        recv_layer = self._recv_layers[self._next_recv_idx]
        owner = recv_layer % self.tp_size

        handles: List[dist.Work] = []
        if self._share_k:
            handles.append(
                dist.irecv(
                    self.k_buffer[recv_layer], owner, group=self._offload_group
                )
            )
        if self._share_v:
            handles.append(
                dist.irecv(
                    self.v_buffer[recv_layer], owner, group=self._offload_group
                )
            )
        if self._share_ik and self.index_k_buffer is not None:
            handles.append(
                dist.irecv(
                    self.index_k_buffer[recv_layer],
                    owner,
                    group=self._offload_group,
                )
            )

        if handles:
            self._pending_recv = handles
            self._pending_recv_layer = recv_layer

        self._next_recv_idx = (self._next_recv_idx + 1) % len(self._recv_layers)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _unique_tensors(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """Return unique tensors by identity (deduplicate shared ones)."""
        seen: set[int] = set()
        unique: List[torch.Tensor] = []
        for t in tensors:
            tid = id(t)
            if tid not in seen:
                seen.add(tid)
                unique.append(t)
        return unique

    def get_kv_size_bytes(self) -> int:
        """Total bytes consumed by *k_buffer*, *v_buffer* and (optionally)
        *index_k_buffer*, with shared tensors counted only once."""
        total = 0
        for t in self._unique_tensors(self.k_buffer):
            total += _get_tensor_size_bytes(t)
        for t in self._unique_tensors(self.v_buffer):
            total += _get_tensor_size_bytes(t)
        if self.index_k_buffer is not None:
            for t in self._unique_tensors(self.index_k_buffer):
                total += _get_tensor_size_bytes(t)
        return total

    def get_contiguous_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        """Return ``(data_ptrs, data_lens, item_lens)`` for disaggregation RDMA.

        Each list has ``layer_num`` entries for *k_buffer* followed by
        ``layer_num`` entries for *v_buffer*, and optionally another
        ``layer_num`` for *index_k_buffer*.
        """
        kv_data_ptrs: List[int] = [b.data_ptr() for b in self.k_buffer]
        kv_data_lens: List[int] = [b.nbytes for b in self.k_buffer]
        kv_item_lens: List[int] = [b[0].nbytes for b in self.k_buffer]

        kv_data_ptrs += [b.data_ptr() for b in self.v_buffer]
        kv_data_lens += [b.nbytes for b in self.v_buffer]
        kv_item_lens += [b[0].nbytes for b in self.v_buffer]

        if self.index_k_buffer is not None:
            kv_data_ptrs += [b.data_ptr() for b in self.index_k_buffer]
            kv_data_lens += [b.nbytes for b in self.index_k_buffer]
            kv_item_lens += [b[0].nbytes for b in self.index_k_buffer]

        return kv_data_ptrs, kv_data_lens, kv_item_lens
