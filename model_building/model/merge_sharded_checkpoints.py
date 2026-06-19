"""
Merge sharded checkpoints into a single checkpoint.

python -m torch.distributed.checkpoint.format_utils dcp_to_torch <checkpoint_dir> <output_checkpoint_path>
"""

import torch
import torch.distributed.checkpoint as dcp
# from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


def main(checkpoint_dir: str, output_checkpoint_path: str):
    state_dict = {}
    dcp.load_state_dict(
        state_dict=state_dict,
        storage_reader=dcp.FileSystemReader(checkpoint_dir)
    )

    # Save as regular checkpoint
    torch.save(state_dict, output_checkpoint_path)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
