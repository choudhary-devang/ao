# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
Compressed-Sparse-Row (CSR) layout support for *affine quantized tensors* in
TorchAO.

This file mirrors the structure of ``SemiSparseLayout`` but targets a *general
unstructured* sparsity pattern encoded with the CSR format.  It enables INT8
weights packed as CSR to participate in the dynamic-activation / INT8-weight
workflow on CPU back-ends (FBGEMM / oneDNN) and provides a fall-back path that
relies on PyTorch's native ``torch.sparse.mm`` when no vendor kernel is
available.

Key pieces:

* ``CSRLayout`` – a stateless ``Layout`` subclass signalling that a tensor is
  stored in CSR form.
* ``CSR_AQTTensorImpl`` – a ``TensorImpl`` that actually holds the compressed
  data and quantization parameters (scale / zero-point).
* Helper ``_linear_int8_act_int8_weight_csr_sparse_check``
  and ``_linear_int8_act_int8_weight_csr_sparse_impl`` which will be invoked by
  TorchDispatch guards when ``aten.linear`` sees csr-packed weights.  These are
  minimal and will be expanded by later PRs to call vendor kernels.

NOTE: This file keeps CUDA-only pieces out – the CPU path is the first target.
Vendor kernels are *optional*; the reference implementation lowers to
``torch.sparse.mm`` so functional correctness is guaranteed even on exotic
architectures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.dtypes.affine_quantized_tensor import (
    AffineQuantizedTensor,
    register_layout,
)
from torchao.dtypes.uintx.plain_layout import (
    PlainAQTTensorImpl,
    _aqt_is_int8_reduced_range,
)
from torchao.dtypes.utils import Layout, PlainLayout

aten = torch.ops.aten

# -----------------------------------------------------------------------------
# Op-level helpers – dispatched during aten.linear if both inputs satisfy the
# predicate.  In the first version we delegate to ``torch.sparse.mm``; follow-up
# patches may swap in vendor SpMM kernels (FBGEMM / oneDNN).
# -----------------------------------------------------------------------------

def _linear_int8_act_int8_weight_csr_sparse_check(
    input_tensor: torch.Tensor, weight_tensor: torch.Tensor, bias: Optional[torch.Tensor]
) -> bool:
    """FX-based guard – true if we can execute csr-optimised path."""
    return (
        isinstance(input_tensor, AffineQuantizedTensor)
        and _aqt_is_int8_reduced_range(input_tensor)
        and isinstance(weight_tensor, AffineQuantizedTensor)
        and input_tensor.dtype == weight_tensor.dtype
        and isinstance(input_tensor._layout, PlainLayout)
        and isinstance(weight_tensor._layout, CSRLayout)
    )


def _linear_int8_act_int8_weight_csr_sparse_impl(
    input_tensor: AffineQuantizedTensor,
    weight_tensor: AffineQuantizedTensor,
    bias: Optional[torch.Tensor],
):
    """Reference implementation using ``torch.sparse.mm``.

    * ``input_tensor`` is **plain** INT8 AQT with per-tensor scale.
    * ``weight_tensor`` is **CSR** INT8 AQT – compressed row representation.
    * Produces *fp32* output multiplied by both quant scales, then casts back
      to original activation dtype.
    """
    x_vals_int8 = input_tensor.tensor_impl.int_data
    x_scale = input_tensor.tensor_impl.scale  # shape: (1,)

    # The weight CSR tensor is stored as ``int_data`` in CSR layout.
    w_csr = weight_tensor.tensor_impl.int_data  # 2-D sparse CSR INT8
    w_scale = weight_tensor.tensor_impl.scale   # shape: (1,) or (row,)

    # Reshape activations to 2-D (batch*seqlen, in_features)
    x2d = x_vals_int8.reshape(-1, x_vals_int8.shape[-1]).to(torch.int32)

    A = torch.rand(768, 768, dtype=torch.float32)
    mask = torch.rand(768, 768) < (1 - 0.70)
    A = A * mask
    S = A.to_sparse_csr()
    B = torch.rand(768, 128, dtype=torch.float32)
    # Sparse matmul (int8*int8 -> int32).  PyTorch up-casts INT8 CSR to INT32
    # during mm; if vendor kernel is registered this will be replaced at runtime.
    _=torch.mm(S,B)
    y_int32 = torch.mm(w_csr.to(torch.float32), x2d.t().to(torch.float32).contiguous()).t()

    # Dequantise: y = (x_scale * w_scale) * y_int32
    y_fp32 = (y_int32.to(torch.float32) * x_scale * w_scale).reshape(
        *x_vals_int8.shape[:-1], y_int32.shape[-1]
    )

    if bias is not None:
        y_fp32 += bias

    # Cast back to activation dtype (usually fp32 or bf16)
    return y_fp32.to(input_tensor.dtype).contiguous()




@dataclass(frozen=True)
class CSRLayout(Layout):
    """Layout marker for *Compressed Sparse Row* INT8 weights.

    The layout itself is **stateless**; all structural metadata (crow_indices,
    col_indices) lives inside the associated ``TensorImpl``.
    """
    def pre_process(self, input: torch.Tensor) -> torch.Tensor:
            """Magnitude‑based pruning prior to CSR packing.

            Parameters
            ----------
            input : torch.Tensor
                A **dense** weight matrix (any dtype) that we intend to quantise
                and pack as CSR.  The function returns a *copy* with elements below
                a global magnitude threshold set to zero so that subsequent
                `torch._to_csr` compression produces the desired sparsity.

            Heuristic
            ---------
            *Prune* the smallest‑magnitude values until we reach the *target
            sparsity* fraction.  The target defaults to **90 %** but can be
            overridden by setting the environment variable
            ``TORCHAO_CSR_TARGET_SPARSITY`` (float in 0 – 1).  If the target is
            outside that range the function becomes a no‑op and simply returns the
            input unchanged.
            """
            print("we are in pre_process function")
            import os

            target = float(os.getenv("TORCHAO_CSR_TARGET_SPARSITY", "0.9"))
            # Validate target range; fall back to no pruning if mis‑configured
            if not (0.0 < target < 1.0):
                return input

            # Clone to avoid mutating the caller's tensor in‑place
            temp = input.detach().clone()

            # Compute global threshold that keeps the largest (1‑target) fraction
            flat = temp.abs().view(-1)
            k = int(flat.numel() * (1 - target))
            if k <= 0:  # requested sparsity so high that everything would be zero
                return temp.zero_()

            # `kthvalue` is 1‑indexed: kthvalue(1) gives minimum, kthvalue(n) max
            thresh = flat.kthvalue(k).values
            mask = temp.abs() >= thresh
            temp.mul_(mask)
            return temp


@register_layout(CSRLayout)
class CSR_AQTTensorImpl(PlainAQTTensorImpl):
    """TensorImpl for CSR-compressed INT8 weights inside an AffineQuantizedTensor.

    The internal representation follows PyTorch's native CSR layout: we store
    *one* 2-D ``torch.sparse_csr_tensor`` holding the INT8 values, plus the
    usual quantisation scale & zero-point.
    """

    # NEW — provide a Python attribute so AffineQuantizedTensor
    # can fetch the layout without touching the C++ side.
    @property
    def layout(self):
        """
        Return a *torch.layout* enum so that
        `AffineQuantizedTensor.__new__` can forward it to
        `torch.Tensor._make_wrapper_subclass` without type errors.

        Note: the **custom** sparsity marker we care about
        (`CSRLayout()` instance) is still stored in `self._layout`.
        """
        return torch.sparse_csr
    
    # ------------------------------------------------------------------
    # Override torch dispatch for basic ops we need (detach, to_plain).
    # ------------------------------------------------------------------

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        kwargs = kwargs or {}
        print("in __torch_dispatch function")
        if func is aten.detach.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
            )

        raise NotImplementedError(
            f"CSR_AQTTensorImpl dispatch: attempted to run {func}, not supported."
        )

    # ------------------------------ helpers ---------------------------

    def get_plain(self):
        """Return dense INT8 matrix alongside scale/zero_point.

        This is a slow path used by debugging utilities (e.g. ``to_dense()``).
        We materialise the CSR tensor to dense form.
        """
        print("we are in get_plain function\n")
        int_data_expanded = self.int_data.to_dense()
        return int_data_expanded, self.scale, self.zero_point

    # --------------------------- construction -------------------------

    @classmethod
    def from_plain(
        cls,
        int_data: torch.Tensor,  # expected *dense* INT8 matrix (out, in)
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
        _layout: Layout,
    ):
        print("we are in from_plain")
        """Pack a *dense* INT8 matrix ``int_data`` into CSR layout."""
        assert isinstance(_layout, CSRLayout), "layout must be CSRLayout"

        # Use PyTorch util to convert to CSR; keep on same device / dtype.
        # Suggestion: ensure row-major order (out_features × in_features).

        # 1) optional magnitude pruning (user-controlled env var)
        pruned = _layout.pre_process(int_data)     # returns dense tensor
        csr_tensor = pruned.to_sparse_csr() 
        print(csr_tensor)
        return cls(csr_tensor, scale, zero_point, _layout)
