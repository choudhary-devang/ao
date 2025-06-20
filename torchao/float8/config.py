# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import enum
import logging
from dataclasses import dataclass
from typing import Optional, Union

import torch

from torchao.utils import is_MI300

logger: logging.Logger = logging.getLogger()


class ScalingType(enum.Enum):
    DYNAMIC = "dynamic"
    # ScalingType.DISABLED means "skip scaling for this tensor, leave it in
    # its original precision.
    DISABLED = "disabled"

    def short_str(self):
        if self is ScalingType.DYNAMIC:
            return "dyn"
        else:
            assert self is ScalingType.DISABLED
            return "dis"


class ScalingGranularity(enum.Enum):
    """
    Defines the granularity of scaling strategies for casting to float8
    """

    # A single scaling factor for the entire tensor
    TENSORWISE = "tensorwise"
    # Scaling factors computed along one axis of the tensor, reducing it to
    # size 1.
    AXISWISE = "axiswise"

    def short_str(self):
        if self is ScalingGranularity.TENSORWISE:
            return "ten"
        else:
            assert self is ScalingGranularity.AXISWISE
            return "axs"


@dataclass
class Float8TypeConfig:
    """
    Configuration for selecting the preferred float8 type pair, either e4m3fn/e5m2 or e4m3fnuz/e5m2fnuz.

    Currently, ROCm supports 1. fnuz variants in MI300. 2. OCP F8 variants in MI350/Navi4.
    """

    # The preferred e4m3 type.
    e4m3_dtype = torch.float8_e4m3fn

    # The preferred e5m2 type.
    e5m2_dtype = torch.float8_e5m2

    def __post_init__(self):
        if torch.version.hip and torch.cuda.is_available() and is_MI300():
            self.e4m3_dtype = torch.float8_e4m3fnuz
            self.e5m2_dtype = torch.float8_e5m2fnuz


# User defined type for using the individual F8 type based on config
type_config = Float8TypeConfig()
e4m3_dtype = type_config.e4m3_dtype
e5m2_dtype = type_config.e5m2_dtype


@dataclass(frozen=True)
class CastConfig:
    """
    Configuration for maybe casting a single tensor to float8
    """

    scaling_type: ScalingType = ScalingType.DYNAMIC
    scaling_granularity: ScalingGranularity = ScalingGranularity.TENSORWISE
    target_dtype: Optional[torch.dtype] = None

    def short_str(self):
        dtype = {e4m3_dtype: "e4m3", e5m2_dtype: "e5m2"}[self.target_dtype]
        return f"{self.scaling_type.short_str()}_{self.scaling_granularity.short_str()}_{dtype}"

    def __post_init__(self):
        if self.scaling_granularity is ScalingGranularity.AXISWISE:
            assert self.scaling_type is ScalingType.DYNAMIC, (
                "only dynamic scaling type is supported for axiswise scaling granularity"
            )
        assert self.target_dtype is None or (
            self.target_dtype.is_floating_point and self.target_dtype.itemsize == 1
        ), "must specify a 8-bit floating-point dtype"


@dataclass(frozen=True)
class Float8GemmConfig:
    """
    Configuration for a float8 gemm.
    """

    # If True, fast accumulation in lower precision is used.
    # Note: this flag is currently a no-op if emulation is turned on.
    use_fast_accum: bool = False


# Pre-made recipes for common configurations
class Float8LinearRecipeName(enum.Enum):
    # Default, dynamic per-tensor scaling with the cuBLAS tensorwise kernel
    TENSORWISE = "tensorwise"

    # dynamic rowwise scaling with the CUTLASS rowwise kernel
    # * e4m3 for activations, weights, gradients
    # * scales rounded (floor) to the nearest power of two for increased accuracy
    ROWWISE = "rowwise"

    # lw's recipe for a modification on rowwise scaling:
    #
    #   output_hp = input_fp8_rowwise_dim0 @ weight_t_rowwise_dim1
    #   grad_input_hp = grad_output_fp8_rowwise_dim0 @ weight_fp8_tensorwise
    #   grad_weight_hp = input_t_hp @ grad_output_hp
    #
    # key characteristics:
    #   * increased accuracy for grad_weight
    #   * `input`, `weight` and `grad_output` now only need to be scaled
    #     rowwise across a single dim compared to vanilla rowwise,
    #     which is more amenable to fast kernels
    #   * the e4m3 dtype is used across the board, including for gradients
    ROWWISE_WITH_GW_HP = "rowwise_with_gw_hp"


@dataclass(frozen=True)
class Float8LinearConfig:
    """
    Configuration for converting a `torch.nn.Linear` module to float8
    for training.
    """

    #
    # Per-tensor configuration for casting of `input`, `weight`, `grad_output`
    # for the operands of gemms calculating `output`, `grad_weight`, and `grad_input`.
    #
    # Note:
    # 1. if `cast_config_input_for_grad_weight` is None, then
    #    `cast_config_input` is used for scaling `input` for both gemms that
    #    use `input.
    # 2. if `cast_config_input_for_grad_weight` is specified, then
    #    a. `cast_config_input` is used for scaling `input` for the gemm that calculates
    #       `output`
    #    b. `cast_config_input_for_grad_weight` is used for scaling `input` for
    #       the gemm that calculates `grad_weight`
    # 3. the same behavior holds for `cast_config_weight` and `cast_config_grad_output`.
    #
    # `input`
    cast_config_input: CastConfig = CastConfig()
    cast_config_input_for_grad_weight: Optional[CastConfig] = None
    # `weight`
    cast_config_weight: CastConfig = CastConfig()
    cast_config_weight_for_grad_input: Optional[CastConfig] = None
    # `grad_output`
    cast_config_grad_output: CastConfig = CastConfig()
    cast_config_grad_output_for_grad_weight: Optional[CastConfig] = None

    #
    # Per-gemm configuration for gemms calculating `output`, `grad_input` and
    # `grad_weight`
    #
    gemm_config_output: Float8GemmConfig = Float8GemmConfig(use_fast_accum=True)
    gemm_config_grad_input: Float8GemmConfig = Float8GemmConfig()
    gemm_config_grad_weight: Float8GemmConfig = Float8GemmConfig()

    #
    # Per-linear configuration
    #

    # If True, then uses a tensor subclass for the float8 linear module's weight that
    # implements pre/post-all-gather methods to do float8 all-gather with FSDP2.
    enable_fsdp_float8_all_gather: bool = False

    # If True, then prior to performing the fp8 scaled mamtmul we will pad the
    # inner dimension of a (dim 1) and b (dim 2) with 0s. This is needed for matmuls
    # _scaled_mm since it has the strong constraint that for M,N,K  N, K must be a multiple of 16.
    # This can cause a memory spike however so we keep this off by default.
    pad_inner_dim: bool = False

    # If True, emulation is used instead of hardware accelerated gemm
    emulate: bool = False

    # This flag is deprecated and currently has no effect. It will be removed
    # in a future release. Please see https://github.com/pytorch/ao/issues/2251
    # for more context.
    force_recompute_fp8_weight_in_bwd: bool = False

    # If this option is enabled, the scaling factor used for float8 quantization
    # will be rounded down to the nearest power of 2. This has been shown to help
    # reduce quantization error by avoiding rounding errors when multiplying/dividing
    # by the scaling factor, as well as ensuring large values are quantized to the
    # same value in the forward pass as the backward passes.
    round_scales_to_power_of_2: bool = False

    def __post_init__(self):
        # Populate the additional cast overrides, if the user did not specify them
        # Note: this hacks around the frozen-ness of this dataclass
        # by using `object.__setattr__`.  This is fine, as what we really need
        # is for this object to be frozen after `__post_init__` for torch.compile
        # to work.
        # Source of hack: https://stackoverflow.com/a/65959419/
        if self.cast_config_input_for_grad_weight is None:
            object.__setattr__(
                self, "cast_config_input_for_grad_weight", self.cast_config_input
            )
        if self.cast_config_weight_for_grad_input is None:
            object.__setattr__(
                self, "cast_config_weight_for_grad_input", self.cast_config_weight
            )
        if self.cast_config_grad_output_for_grad_weight is None:
            object.__setattr__(
                self,
                "cast_config_grad_output_for_grad_weight",
                self.cast_config_grad_output,
            )

        # float8 all-gather only supports tensorwise, in the future may support blockwise
        if self.cast_config_weight.scaling_granularity != ScalingGranularity.TENSORWISE:
            assert not self.enable_fsdp_float8_all_gather, (
                f"enable_fsdp_float8_all_gather only supports tensorwise scaling granularity, got {self.cast_config_weight.scaling_granularity}"
            )

        # save some characters in the compatibility checks below
        cc_i = self.cast_config_input
        cc_w = self.cast_config_weight
        cc_go = self.cast_config_grad_output
        cc_i_gw = self.cast_config_input_for_grad_weight
        cc_w_gi = self.cast_config_weight_for_grad_input
        cc_go_gw = self.cast_config_grad_output_for_grad_weight
        # for now, we only have gemm kernels where both operands are either both
        # in high precision, or both in float8. In the future, this may be relaxed.
        # TODO(future): make the float8 check more precise with the specific dtypes.
        for cc1, cc2, gemm_name in (
            (cc_i, cc_w, "output"),
            (cc_go, cc_w_gi, "grad_input"),
            (cc_i_gw, cc_go_gw, "grad_weight"),
        ):
            is_disabled_1 = cc1.scaling_type is ScalingType.DISABLED
            is_disabled_2 = cc1.scaling_type is ScalingType.DISABLED
            assert is_disabled_1 == is_disabled_2, (
                f"incompatible operand precision for {gemm_name}"
            )

        for cc1, cc2, operand_name, default_dtype in [
            (cc_i, cc_i_gw, "input", e4m3_dtype),
            (cc_w, cc_w_gi, "weight", e4m3_dtype),
            (cc_go, cc_go_gw, "grad_output", e5m2_dtype),
        ]:
            # Override the dataclass being frozen
            if cc1.target_dtype is None:
                object.__setattr__(cc1, "target_dtype", default_dtype)
            if cc2.target_dtype is None:
                object.__setattr__(cc2, "target_dtype", default_dtype)
            assert cc1.target_dtype == cc2.target_dtype, (
                f"{operand_name} must be cast to the same dtype in both matmuls it's used in"
            )

        if self.force_recompute_fp8_weight_in_bwd:
            logger.warning(
                "`config.force_recompute_fp8_weight_in_bwd` is deprecated and will be removed in a future release. Please see https://github.com/pytorch/ao/issues/2251 for more details."
            )

    @staticmethod
    def from_recipe_name(
        recipe_name: Union[Float8LinearRecipeName, str],
    ) -> "Float8LinearConfig":
        """
        Input: `Float8LinearRecipeName` value, or a string representing a `Float8LinearRecipeName` value
        Output: a `Float8LinearConfig` configured to implement the specified recipe
        """
        if type(recipe_name) == str:
            valid_names = [n.value for n in Float8LinearRecipeName]
            assert recipe_name in valid_names, (
                f"recipe_name {recipe_name} not in valid names {valid_names}"
            )
            recipe_name = Float8LinearRecipeName(recipe_name)

        if recipe_name is Float8LinearRecipeName.TENSORWISE:
            return Float8LinearConfig()

        elif recipe_name is Float8LinearRecipeName.ROWWISE:
            cc_i = CastConfig(
                scaling_granularity=ScalingGranularity.AXISWISE, target_dtype=e4m3_dtype
            )
            cc_w = CastConfig(
                scaling_granularity=ScalingGranularity.AXISWISE, target_dtype=e4m3_dtype
            )
            cc_go = CastConfig(
                scaling_granularity=ScalingGranularity.AXISWISE, target_dtype=e4m3_dtype
            )

            return Float8LinearConfig(
                cast_config_input=cc_i,
                cast_config_weight=cc_w,
                cast_config_grad_output=cc_go,
                # enable power of 2 scaling factors by default for row-wise scaling
                round_scales_to_power_of_2=True,
            )

        elif recipe_name is Float8LinearRecipeName.ROWWISE_WITH_GW_HP:
            # output_hp = input_fp8_axiswise_dim0 @ weight_t_axiswise_dim1
            cc_i = CastConfig(scaling_granularity=ScalingGranularity.AXISWISE)
            cc_w = CastConfig(scaling_granularity=ScalingGranularity.AXISWISE)

            # grad_input_hp = grad_output_fp8_axiswise_dim0 @ weight_fp8_tensorwise
            cc_go = CastConfig(
                scaling_granularity=ScalingGranularity.AXISWISE, target_dtype=e4m3_dtype
            )
            cc_w_gi = CastConfig(scaling_granularity=ScalingGranularity.TENSORWISE)

            # grad_weight_hp = input_t_hp @ grad_output_hp
            cc_i_gw = CastConfig(scaling_type=ScalingType.DISABLED)
            cc_go_gw = CastConfig(
                scaling_type=ScalingType.DISABLED, target_dtype=e4m3_dtype
            )

            return Float8LinearConfig(
                cast_config_input=cc_i,
                cast_config_weight=cc_w,
                cast_config_grad_output=cc_go,
                cast_config_input_for_grad_weight=cc_i_gw,
                cast_config_weight_for_grad_input=cc_w_gi,
                cast_config_grad_output_for_grad_weight=cc_go_gw,
            )

        else:
            raise AssertionError(f"unknown recipe_name {recipe_name}")
