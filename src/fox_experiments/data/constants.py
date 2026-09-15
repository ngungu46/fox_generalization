"""Pinned source and checksum of the verified native-token pilot subset."""

DOCUMENT_LENGTH = 65536
REVISION = "5e2f38e601ca7c16fa304f30536d9e2945bb2a32"
SOURCE_BASE = f"https://huggingface.co/datasets/clankur/longcrawl64/resolve/{REVISION}/heldout.zarr/"
SOURCE_URL = SOURCE_BASE
DATA_FILENAME = "native_longcrawl64_pilot.u16"
DEFAULT_SUBSET_SHA256 = (
    "4f12e470db96593c281e194f25658d755c1603c3c39df9debabf869e2fc0b2a2"
)
SOURCE_SHAPE = [52131, 65536]
CHUNK_SHAPE = [2048, 2048]
