"""
Trim tokenizer vocabulary and embedding layers to keep only specified tokens.
Not working with SPM tokenizers.
"""

import torch
from transformers import AutoTokenizer, AutoModel


def main(
    model_name_or_path: str,
    vocab_to_keep: list = None,
    vocab_to_trim: list = None,
    output_tokenizer_path: str = None,
    output_model_path: str = None,
    round_to: int = 128
):
    """
    Trim tokenizer vocabulary and embedding layers.
    
    Args:
        model_name_or_path: Path to the original model
        vocab_to_keep: List of tokens to keep (preferred)
        vocab_to_trim: List of tokens to remove (alternative to vocab_to_keep)
        output_tokenizer_path: Path to save the trimmed tokenizer (optional)
        output_model_path: Path to save the trimmed model (optional)
        round_to: Round vocabulary size to this number (default: 128)
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path)
    
    # Determine token IDs to keep
    token_ids_to_keep = []
    if vocab_to_keep is not None:
        for token in vocab_to_keep:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id != tokenizer.unk_token_id:
                token_ids_to_keep.append(token_id)
    elif vocab_to_trim is not None:
        token_ids_to_trim = []
        for token in vocab_to_trim:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id != tokenizer.unk_token_id:
                token_ids_to_trim.append(token_id)
        all_token_ids = set(range(len(tokenizer)))
        token_ids_to_keep = list(all_token_ids - set(token_ids_to_trim))
    else:
        raise ValueError("Provide either vocab_to_keep or vocab_to_trim")
    
    # Sort token IDs
    token_ids_to_keep = sorted(set(token_ids_to_keep))
    
    # Calculate new vocab size rounded to specified value
    base_vocab_size = len(token_ids_to_keep)
    new_vocab_size = (base_vocab_size // round_to + (base_vocab_size % round_to > 0)) * round_to
    num_new_tokens = new_vocab_size - base_vocab_size
    
    print(f"Original vocab size: {len(tokenizer)}")
    print(f"Tokens to keep: {len(token_ids_to_keep)}")
    print(f"New vocab size (rounded): {new_vocab_size}")
    print(f"Padding tokens needed: {num_new_tokens}")
    
    # Create new tokenizer with trimmed vocabulary
    new_vocab = {}
    
    # Add kept tokens with new IDs
    for new_id, old_id in enumerate(token_ids_to_keep):
        token = tokenizer.convert_ids_to_tokens(old_id)
        new_vocab[token] = new_id
    
    # Add padding tokens if needed
    if num_new_tokens > 0:
        for i in range(num_new_tokens):
            new_vocab[f"PRESERVED_TOKEN_{i:03d}"] = base_vocab_size + i
    
    # Create new tokenizer
    modified_tokenizer = tokenizer.__class__(
        vocab_file=None,
        merges_file=None if not hasattr(tokenizer, 'merges_file') else tokenizer.merges_file,
        **tokenizer.init_kwargs
    )
    modified_tokenizer.vocab = new_vocab
    
    # Trim model embeddings
    with torch.no_grad():
        old_embeddings = model.get_input_embeddings().weight.data
        new_embeddings = torch.zeros(new_vocab_size, old_embeddings.size(1))
        
        # Copy embeddings for kept tokens
        for new_id, old_id in enumerate(token_ids_to_keep):
            new_embeddings[new_id] = old_embeddings[old_id]
        
        # Initialize padding token embeddings with mean of kept tokens
        if num_new_tokens > 0:
            mean_embedding = new_embeddings[:base_vocab_size].mean(dim=0, keepdim=True)
            new_embeddings[base_vocab_size:] = mean_embedding
    
    # Update model embeddings
    model.resize_token_embeddings(new_vocab_size)
    model.get_input_embeddings().weight.data = new_embeddings
    
    # Update model config
    model.config.vocab_size = new_vocab_size
    
    print(f"Final embedding shape: {model.get_input_embeddings().weight.shape}")
    
    # Save if paths provided
    if output_tokenizer_path:
        modified_tokenizer.save_pretrained(output_tokenizer_path)
        print(f"Tokenizer saved to: {output_tokenizer_path}")
    
    if output_model_path:
        model.save_pretrained(output_model_path)
        print(f"Model saved to: {output_model_path}")
    
    return modified_tokenizer, model


if __name__ == "__main__":
    import fire

    fire.Fire(main)
