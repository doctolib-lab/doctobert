# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import random
import time

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

# Add tests folder root to path to allow us to use relative imports regardless of what directory the script is run from
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Add folder root to path to allow us to use relative imports regardless of what directory the script is run from
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import main
from test_utils import SynthTextDirectory
from src.bert_layers.initialization import InitFnType

IMPL_USE_FLASH2 = False
IMPL_USE_FLASH3 = False
try:
    import flash_attn

    IMPL_USE_FLASH2 = True
except ImportError:
    pass

try:
    import flash_attn_interface

    IMPL_USE_FLASH3 = True
except ImportError:
    pass

layer_combinations = [
    ("prenorm", "absolute_pos", "base", "mlp"),
    ("prenorm", "sans_pos", "rope", "mlp"),
    ("postnorm", "absolute_pos", "base", "glu"),
    ("postnorm", "sans_pos", "rope", "glu"),
    ("parallel_prenorm", "absolute_pos", "parallel", "parallel_glu"),
    ("parallel_prenorm", "sans_pos", "rope_parallel", "parallel_glu"),
]


@pytest.mark.skipif(not IMPL_USE_FLASH2 and not IMPL_USE_FLASH3, reason="Flash Attention is not installed")
@pytest.mark.parametrize("different_first_layer", [False, True], ids=lambda x: "different_first_layer" if x else "_")
@pytest.mark.parametrize("sliding_window", [False, True], ids=lambda x: "sliding_window" if x else "_")
@pytest.mark.parametrize("unpad_embeddings", [False, True], ids=lambda x: "unpad_embeddings" if x else "_")
@pytest.mark.parametrize("pad_logits", [False, True], ids=lambda x: "pad_logits" if x else "_")
@pytest.mark.parametrize("partial_compile", [False, True], ids=lambda x: "compile" if x else "_")
@pytest.mark.parametrize("layer,embedding,attention,mlp", layer_combinations)
def test_trainer(
    synth_text_dir: str,
    layer: str,
    embedding: str,
    attention: str,
    mlp: str,
    different_first_layer: bool,
    sliding_window: bool,
    unpad_embeddings: bool,
    pad_logits: bool,
    partial_compile: bool,
):
    if not unpad_embeddings and pad_logits:
        pytest.skip("Pad logits requires unpadded embeddings.")

    with open("yamls/defaults.yaml") as f:
        default_cfg = OmegaConf.load(f)
    with open("yamls/models/flex_bert.yaml") as f:
        model_cfg = OmegaConf.load(f)
    with open("tests/smoketest_config_sdpa_fa.yaml") as f:
        test_config = OmegaConf.load(f)
    config = OmegaConf.merge(default_cfg, model_cfg, test_config)
    assert isinstance(config, DictConfig)

    config.model.name = "flex_bert"
    config.seed = 42
    config.model.model_config.bert_layer = layer
    config.model.model_config.embedding_layer = embedding
    config.model.model_config.attention_layer = attention
    config.model.model_config.mlp_layer = mlp
    if partial_compile and layer == "prenorm":
        config.model.model_config.partial_compile = True
    elif partial_compile:
        pytest.skip("Only prenorm can be partially compiled")
    if partial_compile and embedding == "absolute_pos":
        pytest.skip("Only absolute pos embeddings can be partially compiled")

    if layer == "postnorm":
        config.model.model_config.final_norm = False

    if sliding_window:
        config.model.model_config.sliding_window = 64
        config.model.model_config.num_hidden_layers = 3
        config.model.model_config.global_attn_every_n_layers = random.choice([-1, 2])

    if different_first_layer:
        if layer != "parallel_prenorm":
            pytest.skip("Only parallel_prenorm needs a different first layer")
        config.model.model_config.initial_attention_layer = "rope" if attention == "rope_parallel" else "base"
        config.model.model_config.initial_bert_layer = "prenorm"
        config.model.model_config.initial_mlp_layer = "glu"
        config.model.model_config.num_initial_layers = random.randint(1, 2)

    if config.model.model_config.bert_layer in ["parallel_prenorm", "prenorm"] and random.random() < 0.5:
        config.model.model_config.skip_first_prenorm = True
        config.model.model_config.embed_norm = True

    # pick a random init type for testing
    config.model.model_config.init_fn = random.choice([member.value for member in InitFnType])
    if config.model.model_config.init_fn != InitFnType.full_megatron:
        config.model.model_config.init_small_embedding = random.choice([True, False])

    # pick a random loss function for testing
    config.model.model_config.loss_function = random.choice(["cross_entropy", "fa_cross_entropy"])
    if config.model.model_config.loss_function == "fa_cross_entropy":
        config.model.model_config.loss_kwargs["lse_square_scale"] = random.choice([1e-4, 0])

    with SynthTextDirectory() as tmp_datadir:
        config.model.model_config.use_fa = True
        config.model.model_config.unpad_embeddings = True
        config.model.model_config.pad_logits = pad_logits
        config.train_loader.dataset.remote = tmp_datadir
        config.train_loader.dataset.local = os.path.join(tmp_datadir, "tr-local1")
        config.eval_loader.dataset.remote = tmp_datadir
        config.eval_loader.dataset.local = os.path.join(tmp_datadir, "ev-local1")

        # Train with FA2/FA3
        trainer1 = main(config, return_trainer=True)
        assert trainer1 is not None
        model1 = trainer1.state.model.model

        # if sliding window is set, there might only be one attention layer with a sliding window
        if sliding_window:
            # fmt: off
            if config.model.model_config.initial_attention_layer in ["rope", "rope_parallel"]:
                config.model.model_config.local_attn_rotary_emb_base = random.choice([1000, -1])
            if config.model.model_config.global_attn_every_n_layers == 2:
                assert model1.bert.encoder.layers[1].attn.sliding_window == [32, 32], f"Sliding window not set for second layer: {model1.bert.encoder.layers[1].attn}"
            else:
                assert model1.bert.encoder.layers[0].attn.sliding_window == [32, 32], f"Sliding window not set for first layer: {model1.bert.encoder.layers[0].attn}"
                assert model1.bert.encoder.layers[1].attn.sliding_window == [32, 32], f"Sliding window not set for second layer: {model1.bert.encoder.layers[1].attn}"
                assert model1.bert.encoder.layers[2].attn.sliding_window == [32, 32], f"Sliding window not set for third layer: {model1.bert.encoder.layers[2].attn}"
            # fmt: on
        # SDPA doesn't have sliding window implemented, so skip the test
        else:
            config.model.model_config.sliding_window = -1
            config.model.model_config.use_fa = False
            config.model.model_config.use_sdpa_attn_mask = True
            config.train_loader.dataset.local = os.path.join(tmp_datadir, "tr-local2")
            config.eval_loader.dataset.local = os.path.join(tmp_datadir, "ev-local2")

            # Train with SDPA
            trainer2 = main(config, return_trainer=True)
            assert trainer2 is not None
            model2 = trainer2.state.model.model

    # SDPA doesn't have sliding window impleemnted, so skip the comparison
    if not sliding_window:
        for param1, param2 in zip(model1.parameters(), model2.parameters()):
            torch.testing.assert_close(param1, param2, rtol=1e-2, atol=1e-3)

        if different_first_layer:
            nl = config.model.model_config.num_initial_layers
            for m in range(2):
                model = model1 if m == 0 else model2
                for i in range(nl - 1):
                    assert isinstance(model.bert.encoder.layers[i], type(model.bert.encoder.layers[i + 1]))
                if nl < len(model.bert.encoder.layers):
                    assert not isinstance(model.bert.encoder.layers[nl - 1], type(model.bert.encoder.layers[nl]))
                    for i in range(nl, len(model1.bert.encoder.layers) - 1):
                        assert isinstance(model.bert.encoder.layers[i], type(model.bert.encoder.layers[i + 1]))
