#!/usr/bin/env python3
"""
Convert HuggingFace ModernBERT checkpoint to FlexBERT format for unpadded training.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/ModernBERT")

import argparse
import torch
from transformers import AutoConfig, AutoTokenizer
import os

from src.bert_layers.configuration_bert import FlexBertConfig
from src.bert_layers.model import FlexBertForMaskedLM


def create_flexbert_config_from_hf(hf_config):
    """Convert HuggingFace ModernBertConfig to FlexBertConfig."""
    
    # Create the FlexBERT config with parameters that exactly match the YAML file
    flex_config = FlexBertConfig(
        # Core architecture parameters (from YAML)
        vocab_size=hf_config.vocab_size,
        hidden_size=hf_config.hidden_size,
        num_hidden_layers=hf_config.num_hidden_layers,
        num_attention_heads=hf_config.num_attention_heads,
        intermediate_size=hf_config.intermediate_size,
        
        # Initialization (from YAML)
        init_method="full_megatron",
        
        # Attention configuration (from YAML)
        attention_layer="rope",
        attention_probs_dropout_prob=hf_config.attention_dropout,
        attn_out_bias=hf_config.attention_bias,
        attn_out_dropout_prob=hf_config.attention_dropout,
        attn_qkv_bias=hf_config.attention_bias,
        
        # Layer configuration (from YAML)
        bert_layer="prenorm",
        skip_first_prenorm=True,
        final_norm=True,
        
        # Embedding configuration (from YAML)
        embedding_layer="sans_pos",
        embed_norm=True,
        embed_dropout_prob=hf_config.embedding_dropout,
        
        # Loss configuration (from YAML)
        loss_function="fa_cross_entropy",
        loss_kwargs={"reduction": "mean"},
        
        # MLP configuration (from YAML)
        mlp_dropout_prob=hf_config.mlp_dropout,
        mlp_in_bias=hf_config.mlp_bias,
        mlp_layer="glu",
        mlp_out_bias=hf_config.mlp_bias,
        
        # Normalization (from YAML)
        normalization="layernorm",
        norm_kwargs={"eps": hf_config.norm_eps, "bias": hf_config.norm_bias},
        
        # Activation functions (from YAML)
        hidden_act=hf_config.hidden_activation,
        head_pred_act=hf_config.classifier_activation,
        activation_function="gelu",  # This is in YAML but was missing
        
        # Padding and positioning (from YAML)
        padding="unpadded",  # This was missing
        rotary_emb_dim=None,  # This was missing
        rotary_emb_base=10000,
        rotary_emb_scale_base=None,  # This was missing
        rotary_emb_interleaved=False,  # This was missing
        allow_embedding_resizing=True,  # This was missing
        
        # ModernBERT specific: sliding window and global attention (from YAML)
        sliding_window=hf_config.local_attention,
        global_attn_every_n_layers=hf_config.global_attn_every_n_layers,
        
        # Training optimizations (from YAML)
        unpad_embeddings=True,
        compile_model=True,  # This was missing
        masked_prediction=True,
        pad_logits=False,
        partial_compile=False,  # This was False in original, should be True per YAML
    )
    
    return flex_config


def convert_state_dict_hf_to_flexbert(hf_state_dict):
    """Convert HuggingFace state dict to FlexBERT format."""
    
    flex_state_dict = {}
    
    for key, value in hf_state_dict.items():
        new_key = key
        
        # Map HuggingFace keys to FlexBERT keys
        if key.startswith("model."):
            new_key = key.replace("model.", "bert.")
        
        # Layer mappings for encoder
        if "layers." in key:
            new_key = new_key.replace("layers.", "encoder.layers.")
        
        flex_state_dict[new_key] = value
    
    return flex_state_dict


def main():
    parser = argparse.ArgumentParser(description="Convert HuggingFace ModernBERT to FlexBERT")
    parser.add_argument("--hf_model_path", type=str, required=True, 
                       help="Path to HuggingFace ModernBERT model")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Output path for FlexBERT checkpoint")
    parser.add_argument("--config_only", action="store_true",
                       help="Only create config, don't convert weights")
    
    args = parser.parse_args()
    
    print(f"Loading HuggingFace model from: {args.hf_model_path}")
    
    # Load HuggingFace model and config
    hf_config = AutoConfig.from_pretrained(args.hf_model_path)
    
    print(f"HF Config: {hf_config}")
    
    # Create FlexBERT config
    flex_config = create_flexbert_config_from_hf(hf_config)
    
    # Save FlexBERT config
    config_path = os.path.join(args.output_path, "config.json")
    os.makedirs(args.output_path, exist_ok=True)
    flex_config.save_pretrained(args.output_path)
    print(f"Saved FlexBERT config to: {config_path}")
    
    if not args.config_only:
        # Load HuggingFace model weights
        from transformers import AutoModelForMaskedLM
        hf_model = AutoModelForMaskedLM.from_pretrained(args.hf_model_path)
        
        # Create FlexBERT model
        flex_model = FlexBertForMaskedLM(flex_config)
        
        # Convert state dict
        hf_state_dict = hf_model.state_dict()
        flex_state_dict = convert_state_dict_hf_to_flexbert(hf_state_dict)
        
        # Load converted weights
        missing_keys, unexpected_keys = flex_model.load_state_dict(flex_state_dict, strict=False)
        
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
        
        # Save FlexBERT model
        model_path = os.path.join(args.output_path, "pytorch_model.bin")
        torch.save(flex_model.state_dict(), model_path)
        print(f"Saved FlexBERT model to: {model_path}")
        
        # Copy tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.hf_model_path)
        tokenizer.save_pretrained(args.output_path)
        print(f"Saved tokenizer to: {args.output_path}")
    
    print("Conversion completed!")

if __name__ == "__main__":
    main() 