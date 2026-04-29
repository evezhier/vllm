# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for InternVL's multimodal preprocessing and CUDA graph support."""

from collections.abc import Mapping
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from transformers import PretrainedConfig

from vllm.model_executor.models.internvl import InternVLChatModel
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.image import rescale_image_size
from vllm.multimodal.processing import BaseMultiModalProcessor

from ....conftest import ImageTestAssets
from ...utils import build_model_context


def _get_expected_num_patches(
    config: PretrainedConfig,
    image: Image.Image,
    num_imgs: int,
    min_num: int,
    max_num: int,
):
    from vllm.transformers_utils.processors.internvl import (
        calculate_internvl_targets,
        get_internvl_target_ratios,
    )

    width, height = image.size

    blocks, _, _ = calculate_internvl_targets(
        orig_width=width,
        orig_height=height,
        target_ratios=get_internvl_target_ratios(
            min_num,
            max_num,
        ),
        image_size=config.vision_config.image_size,
        use_thumbnail=False,
    )
    expected_num_patches = blocks

    if config.use_thumbnail and expected_num_patches > 1:
        expected_num_patches += 1

    return expected_num_patches


def _run_check(
    processor: BaseMultiModalProcessor,
    images: list[Image.Image],
    min_num: int,
    max_num: int,
    mm_processor_kwargs: Mapping[str, object],
):
    tokenizer = processor.info.get_tokenizer()
    config = processor.info.get_hf_config()

    prompt = "<image>" * len(images)
    mm_data = {"image": images}

    total_expected_num_patches = sum(
        _get_expected_num_patches(config, image, len(images), min_num, max_num)
        for image in images
    )

    processed_inputs = processor(
        prompt,
        mm_items=processor.info.parse_mm_data(mm_data),
        hf_processor_mm_kwargs=mm_processor_kwargs,
    )

    # Ensure we have the right number of placeholders per num_crops size
    image_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    img_tok_count = processed_inputs["prompt_token_ids"].count(image_token_id)
    pixel_shape = processed_inputs["mm_kwargs"].get_data()["pixel_values_flat"].shape

    assert img_tok_count == 256 * total_expected_num_patches
    assert pixel_shape[0] == total_expected_num_patches


@pytest.mark.parametrize("model_id", ["OpenGVLab/InternVL2-2B"])
@pytest.mark.parametrize(
    "size_factors",
    [
        # Single-scale
        [1.0],
        # Single-scale, batched
        [1.0, 1.0, 1.0],
        # Multi-scale
        [0.25, 0.5, 1.0],
        [4.0, 2.0, 1.0],
    ],
)
@pytest.mark.parametrize(
    ("min_dynamic_patch", "max_dynamic_patch"),
    [(1, 1), (1, 2), (1, 4), (1, 8), (2, 4), (4, 8)],
)
@pytest.mark.parametrize("dynamic_image_size", [True, False])
@pytest.mark.parametrize("kwargs_on_init", [True, False])
def test_processor_override(
    model_id: str,
    image_assets: ImageTestAssets,
    size_factors: list[int],
    min_dynamic_patch: int,
    max_dynamic_patch: int,
    dynamic_image_size: bool | None,
    kwargs_on_init: bool,
):
    mm_processor_kwargs = {
        "min_dynamic_patch": min_dynamic_patch,
        "max_dynamic_patch": max_dynamic_patch,
        "dynamic_image_size": dynamic_image_size,
    }

    ctx = build_model_context(
        model_id,
        mm_processor_kwargs=mm_processor_kwargs if kwargs_on_init else None,
        limit_mm_per_prompt={"image": len(size_factors)},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    hf_processor_mm_kwargs = {} if kwargs_on_init else mm_processor_kwargs

    min_num = min_dynamic_patch if dynamic_image_size else 1
    max_num = max_dynamic_patch if dynamic_image_size else 1

    _run_check(
        processor,
        [rescale_image_size(image_assets[0].pil_image, f) for f in size_factors],
        min_num,
        max_num,
        hf_processor_mm_kwargs,
    )

_IMAGE_SIZE = 2
_VIT_HIDDEN = 1
_LLM_HIDDEN = 1
_NUM_IMAGE_TOKEN = 1  # int((2//2)^2 * 1.0^2)


class _VisionModelStub(torch.nn.Module):
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        N = pixel_values.shape[0]
        val = pixel_values.mean(dim=(1, 2, 3)).view(N, 1, 1)
        return val.expand(N, (_IMAGE_SIZE // 2) ** 2 + 1, _VIT_HIDDEN).contiguous()


class _ProjectorStub(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1


def _make_internvl_model() -> InternVLChatModel:
    model = object.__new__(InternVLChatModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        force_image_size=_IMAGE_SIZE,
        vision_config=SimpleNamespace(image_size=_IMAGE_SIZE),
        text_config=SimpleNamespace(hidden_size=_LLM_HIDDEN),
    )
    model.num_image_token = _NUM_IMAGE_TOKEN
    model.patch_tokens = (_IMAGE_SIZE // 2) ** 2
    model.downsample_ratio = 1.0
    model.ps_version = "v2"
    model.multimodal_config = SimpleNamespace(
        get_limit_per_prompt=lambda modality: 0,
    )
    model.vision_model = _VisionModelStub()
    model.mlp1 = _ProjectorStub()
    return model


def test_encoder_cudagraph_forward_matches_eager():
    model = _make_internvl_model()
    mm_kwargs = {
        "pixel_values_flat": torch.stack(
            [
                torch.full((3, _IMAGE_SIZE, _IMAGE_SIZE), 1.0),
                torch.full((3, _IMAGE_SIZE, _IMAGE_SIZE), 2.0),
            ]
        ),
    }

    eager = model.encoder_eager_forward(mm_kwargs)
    graph = model.encoder_cudagraph_forward(mm_kwargs, buffers={})

    expected = torch.tensor([[2.0], [3.0]])
    assert torch.equal(eager, expected)
    assert torch.equal(eager, graph)
