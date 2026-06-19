"""Add new tokens to tokenizer and embedding layer."""

import torch
from transformers import AutoModel, AutoTokenizer


def main(
    model_path: str,
    new_vocab: list,
    output_tokenizer_path: str = None,
    output_model_path: str = None,
    round_to: int = 128,
    embedding_init: str = "mean",
):
    """
    Add new tokens to tokenizer and embedding layer.

    Args:
        model_path: Path to the original model
        new_vocab: List of new vocabulary tokens to add
        output_tokenizer_path: Path to save the updated tokenizer (optional)
        output_model_path: Path to save the updated model (optional)
        round_to: Round vocabulary size to this number (default: 128)
        embedding_init: How to initialize new embeddings ("mean" or "scratch")
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)

    print(f"Original vocab size: {len(tokenizer)}")
    print(f"Original model vocab size: {model.config.vocab_size}")

    # Add new tokens to tokenizer
    num_added_tokens = tokenizer.add_tokens(new_vocab)
    print(f"Added {num_added_tokens} new tokens")
    print(f"New tokenizer vocab size: {len(tokenizer)}")

    # Calculate rounded vocab size
    current_vocab_size = len(tokenizer)
    rounded_vocab_size = (current_vocab_size // round_to + (current_vocab_size % round_to > 0)) * round_to
    num_padding_tokens = rounded_vocab_size - current_vocab_size

    print(f"Rounded vocab size: {rounded_vocab_size}")
    print(f"Padding tokens needed: {num_padding_tokens}")

    # Add padding tokens if needed
    if num_padding_tokens > 0:
        padding_tokens = [f"PRESERVED_TOKEN_{i:03d}" for i in range(num_padding_tokens)]
        num_padding_added = tokenizer.add_tokens(padding_tokens)
        print(f"Added {num_padding_added} padding tokens")
        assert num_padding_added == num_padding_tokens

    final_vocab_size = len(tokenizer)
    print(f"Final tokenizer vocab size: {final_vocab_size}")

    # Resize model embeddings
    model.resize_token_embeddings(final_vocab_size)

    # Initialize new embeddings
    with torch.no_grad():
        input_embeddings = model.get_input_embeddings().weight.data
        original_vocab_size = model.config.vocab_size

        if embedding_init == "mean":
            # Initialize with mean of existing embeddings
            mean_embedding = input_embeddings[:original_vocab_size].mean(dim=0, keepdim=True)
            input_embeddings[original_vocab_size:] = mean_embedding
            print("Initialized new embeddings with mean of existing embeddings")
        elif embedding_init == "scratch":
            # Initialize from scratch (random)
            embedding_dim = input_embeddings.size(1)
            new_embeddings = torch.randn(final_vocab_size - original_vocab_size, embedding_dim)
            new_embeddings *= input_embeddings[:original_vocab_size].std()
            input_embeddings[original_vocab_size:] = new_embeddings
            print("Initialized new embeddings from scratch")
        else:
            raise ValueError(f"Invalid embedding_init: {embedding_init}. Choose 'mean' or 'scratch'")

    # Update model config
    model.config.vocab_size = final_vocab_size

    print(f"Final embedding shape: {model.get_input_embeddings().weight.shape}")

    # Save if paths provided
    if output_tokenizer_path:
        tokenizer.save_pretrained(output_tokenizer_path)
        print(f"Tokenizer saved to: {output_tokenizer_path}")

    if output_model_path:
        model.save_pretrained(output_model_path)
        print(f"Model saved to: {output_model_path}")

    return tokenizer, model


if __name__ == "__main__":
    import fire

    fire.Fire(main)
