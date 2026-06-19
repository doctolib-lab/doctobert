# Adapted from https://github.com/huggingface/datatrove/blob/main/examples/minhash_deduplication.py

from typing import Callable, Literal

from datatrove.executor import LocalPipelineExecutor
from datatrove.io import DataFileLike, DataFolderLike
from datatrove.pipeline.dedup import MinhashDedupSignature
from datatrove.pipeline.dedup.minhash import (
    MinhashConfig,
    MinhashDedupBuckets,
    MinhashDedupCluster,
    MinhashDedupFilter,
)
from datatrove.pipeline.readers.base import BaseDiskReader
from datatrove.pipeline.tokens import TokensCounter
from datatrove.pipeline.writers import JsonlWriter, ParquetWriter
from datatrove.utils.hashing import HashConfig
from datatrove.utils.typeshelper import Languages


class TxtReader(BaseDiskReader):
    """Read data from newline-delimited TXT files.
       Each (optionally stripped) line is yielded as a separate document.

    Args:
        data_folder: a str, tuple or DataFolder object representing a path/filesystem
        paths_file: optionally provide a file with one path per line (without the `data_folder` prefix) to read.
        compression: the compression to use (default: "infer")
        limit: limit the number of documents to read. Useful for debugging
        skip: skip the first n rows/lines (applied per file)
        file_progress: show progress bar for files
        doc_progress: show progress bar for documents
        adapter: function to adapt the data dict from the source to a Document.
            Takes as input: (self, data: dict, path: str, id_in_file: int | str)
                self allows access to self.text_key and self.id_key
            Returns: a dict with at least "text" and "id" keys
        text_key: the key containing the text data (default: "text")
        id_key: the key containing the id for each sample (default: "id")
        default_metadata: a dictionary with any data that should be added to all samples' metadata
        recursive: whether to search files recursively. Ignored if paths_file is provided
        glob_pattern: pattern that all files must match exactly to be included (relative to data_folder). Ignored if paths_file is provided
        shuffle_files: shuffle the files within the returned shard. Mostly used for data viz. purposes, do not use with dedup blocks
        strip_whitespace: strip leading/trailing whitespace from each line (default: True)
        skip_empty: skip empty/whitespace-only lines (default: True)
    """

    name = "📄 Txt"

    def __init__(
        self,
        data_folder: DataFolderLike,
        paths_file: DataFileLike | None = None,
        compression: Literal["infer", "gzip", "zstd"] | None = "infer",
        limit: int = -1,
        skip: int = 0,
        file_progress: bool = False,
        doc_progress: bool = False,
        adapter: Callable | None = None,
        text_key: str = "text",
        id_key: str = "id",
        default_metadata: dict | None = None,
        recursive: bool = True,
        glob_pattern: str | None = None,
        shuffle_files: bool = False,
        strip_whitespace: bool = True,
        skip_empty: bool = True,
    ):
        super().__init__(
            data_folder,
            paths_file,
            limit,
            skip,
            file_progress,
            doc_progress,
            adapter,
            text_key,
            id_key,
            default_metadata,
            recursive,
            glob_pattern,
            shuffle_files,
        )
        self.compression = compression
        self.strip_whitespace = strip_whitespace
        self.skip_empty = skip_empty
        self.empty_warning = False  # kept for parity with CsvReader

    def read_file(self, filepath: str):
        with self.data_folder.open(filepath, "r", compression=self.compression) as f:
            for li, raw_line in enumerate(f):
                if li < self.skip:
                    continue
                line = raw_line.rstrip("\n\r")
                if self.strip_whitespace:
                    line = line.strip()
                if self.skip_empty and (not line):
                    continue

                # Build a row-like dict, mirroring CsvReader behavior
                row = {self.text_key: line, self.id_key: li}

                with self.track_time():
                    document = self.get_document_from_dict(row, filepath, li)
                    if not document:
                        continue
                yield document


def main(
    input_path: str,
    output_path: str | None = None,
    intermediate_data_path: str | None = None,
    logs_folder: str | None = None,
    language: str = "french",
    total_tasks: int = 16,
):
    # I/O
    if output_path is None:
        output_path = f"{input_path}_deduplicated"
    if intermediate_data_path is None:
        intermediate_data_path = f"{input_path}_minhash"
    if logs_folder is None:
        logs_folder = f"{intermediate_data_path}/logs"

    # you can also change ngrams or the number of buckets and their size here
    minhash_config = MinhashConfig(
        hash_config=HashConfig(precision=64),
        num_buckets=14,
        hashes_per_bucket=8,
    )  # better precision -> fewer false positives (collisions)

    # stage 1 computes minhash signatures for each task (each task gets a set of files)
    stage1 = LocalPipelineExecutor(
        pipeline=[
            TxtReader(
                input_path,
                shuffle_files=True,  # shuffle to balance long lines
                # glob_pattern="shard_0.txt",  # debug
            ),
            MinhashDedupSignature(
                output_folder=f"{intermediate_data_path}/signatures",
                config=minhash_config,
                language=getattr(Languages, language),
            ),
        ],
        tasks=total_tasks,
        workers=-1,
        logging_dir=f"{logs_folder}/signatures",
    )

    # stage 2 finds matches between signatures in each bucket
    stage2 = LocalPipelineExecutor(
        pipeline=[
            MinhashDedupBuckets(
                input_folder=f"{intermediate_data_path}/signatures",
                output_folder=f"{intermediate_data_path}/buckets",
                config=minhash_config,
            ),
        ],
        tasks=minhash_config.num_buckets,
        workers=-1,
        logging_dir=f"{logs_folder}/buckets",
    )

    # stage 3 creates clusters of duplicates using the results from all buckets
    stage3 = LocalPipelineExecutor(
        pipeline=[
            MinhashDedupCluster(
                input_folder=f"{intermediate_data_path}/buckets",
                output_folder=f"{intermediate_data_path}/remove_ids",
                config=minhash_config,
            ),
        ],
        tasks=1,
        workers=-1,
        logging_dir=f"{logs_folder}/clusters",
    )

    # stage 4 reads the original input data and removes all but 1 sample per duplicate cluster
    # the data must match exactly stage 1, so number of tasks and the input source must be the same
    stage4 = LocalPipelineExecutor(
        pipeline=[
            input_reader,
            TokensCounter(),  # nice way to see how many tokens we had before and after deduplication
            MinhashDedupFilter(
                input_folder=f"{intermediate_data_path}/remove_ids",
                exclusion_writer=JsonlWriter(f"{intermediate_data_path}/removed"),
            ),
            # JsonlWriter(output_folder=f"{intermediate_data_path}/deduplicated_output"),
            ParquetWriter(output_folder=output_path),
        ],
        tasks=total_tasks,
        workers=-1,
        logging_dir=f"{logs_folder}/filter",
    )

    # Run the pipeline stages sequentially.
    stage1.run()
    stage2.run()
    stage3.run()
    stage4.run()


if __name__ == "__main__":
    import fire

    fire.Fire(main)
