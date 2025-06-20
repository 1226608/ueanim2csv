# -*- coding: utf-8 -*-
"""
Standalone Python script to batch convert .ueanim files to .csv.

This script extracts Shape Key (Curve) animation data from .ueanim files
and saves it as a CSV file, with each row representing a frame and each
column representing a shape key's value.

It is designed to run independently of Blender.

Requirements:
    pip install zstandard numpy

Usage:
    python ueanim_to_csv.py "path/to/your/ueanim/folder"
"""
from __future__ import annotations
import argparse
import csv
import gzip
import io
import os
import struct
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum, auto
from pathlib import Path
from types import TracebackType
from typing import (TYPE_CHECKING, Any, BinaryIO, ClassVar, Generic, Literal,
                    TypeVar, cast, overload)

import numpy as np
import numpy.typing as npt
import zstandard as zstd

# =============================================================================
# COPIED AND MODIFIED FROM io_scene_ueformat
#
# The following code is adapted from the io_scene_ueformat Blender addon.
# All dependencies on Blender's 'bpy' and 'mathutils' have been removed
# to allow for standalone execution.
#
# Original addon structure has been flattened into this single script
# for simplicity.
# =============================================================================

# region logging.py
class Log:
    INFO = "\u001b[36m"
    WARN = "\u001b[33m"
    ERROR = "\u001b[31m"
    RESET = "\u001b[0m"

    NoLog: bool = False
    timers: ClassVar[dict[str, float]] = {}

    @classmethod
    def info(cls, message: str) -> None:
        if not cls.NoLog:
            print(f"{cls.INFO}[UEFORMAT] {cls.RESET}{message}")

    @classmethod
    def warn(cls, message: str) -> None:
        if not cls.NoLog:
            print(f"{cls.WARN}[UEFORMAT] {cls.RESET}{message}")

    @classmethod
    def error(cls, message: str) -> None:
        if not cls.NoLog:
            print(f"{cls.ERROR}[UEFORMAT] {cls.RESET}{message}")

    @classmethod
    def time_start(cls, name: str) -> None:
        if not cls.NoLog:
            cls.timers[name] = time.time()

    @classmethod
    def time_end(cls, name: str) -> None:
        if cls.NoLog:
            return
        start_time = cls.timers.pop(name, None)
        if start_time is None:
            cls.error(f"Timer {name} does not exist")
        else:
            cls.info(f"{name} took {time.time() - start_time:.4f} seconds")
# endregion

# region importer/utils.py (minimal)
def bytes_to_str(in_bytes: bytes) -> str:
    return in_bytes.rstrip(b"\x00").decode(errors='ignore')
# endregion

# region importer/reader.py
R = TypeVar("R")

class FArchiveReader:
    def __init__(self, data: bytes) -> None:
        self.data: BinaryIO = io.BytesIO(data)
        self.size = len(data)
        self.data.seek(0)
        self.file_version = EUEFormatVersion.BeforeCustomVersionWasAdded
        self.metadata = {}

    def __enter__(self) -> FArchiveReader:
        self.data.seek(0)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.data.close()

    def eof(self) -> bool:
        return self.data.tell() >= self.size

    def read(self, size: int) -> bytes:
        return self.data.read(size)

    def read_to_end(self) -> bytes:
        return self.data.read(self.size - self.data.tell())

    def read_bool(self) -> bool:
        return struct.unpack("?", self.data.read(1))[0]

    def read_string(self, size: int) -> str:
        string = self.data.read(size)
        return bytes_to_str(string)

    def read_fstring(self) -> str:
        try:
            (size,) = struct.unpack("i", self.data.read(4))
            if size > self.size - self.data.tell() or size < -self.size: # Basic sanity check
                raise struct.error("Invalid fstring size")
            string = self.data.read(size)
            return bytes_to_str(string)
        except struct.error as e:
            Log.error(f"Failed to read fstring: {e}")
            return ""

    def read_int(self) -> int:
        return struct.unpack("i", self.data.read(4))[0]

    def read_float(self) -> float:
        return struct.unpack("f", self.data.read(4))[0]

    def read_byte(self) -> bytes:
        return struct.unpack("c", self.data.read(1))[0]

    def skip(self, size: int) -> None:
        self.data.seek(size, 1)

    def read_serialized_array(self, predicate: Callable[[FArchiveReader], R]) -> list[R]:
        count = self.read_int()
        return self.read_array(count, predicate)

    def read_array(
        self,
        count: int,
        predicate: Callable[[FArchiveReader], R],
    ) -> list[R]:
        return [predicate(self) for _ in range(count)]

    def chunk(self, size: int) -> FArchiveReader:
        new_reader = FArchiveReader(self.read(size))
        new_reader.file_version = self.file_version
        new_reader.metadata = self.metadata
        return new_reader

# endregion

# region importer/classes.py (minimal)
MAGIC = "UEFORMAT"
ANIM_IDENTIFIER = "UEANIM"

class EUEFormatVersion(IntEnum):
    BeforeCustomVersionWasAdded = 0
    SerializeBinormalSign = 1
    AddMultipleVertexColors = 2
    AddConvexCollisionGeom = 3
    LevelOfDetailFormatRestructure = 4
    SerializeVirtualBones = 5
    SerializeMaterialPath = 6
    SerializeAssetMetadata = 7
    PreserveOriginalTransforms = 8
    AddPoseExport = 9
    VersionPlusOne = auto()
    LatestVersion = VersionPlusOne - 1

@dataclass(slots=True)
class AnimKey:
    frame: int

    @classmethod
    def from_archive(cls, ar: FArchiveReader) -> AnimKey:
        return cls(frame=ar.read_int())

@dataclass(slots=True)
class FloatKey(AnimKey):
    value: float

    @classmethod
    def from_archive(cls, ar: FArchiveReader) -> FloatKey:
        return cls(
            frame=ar.read_int(),
            value=ar.read_float(),
        )

@dataclass(slots=True)
class Curve:
    name: str
    keys: list[FloatKey]

    @classmethod
    def from_archive(cls, ar: FArchiveReader) -> Curve:
        return cls(
            name=ar.read_fstring(),
            keys=ar.read_serialized_array(lambda ar: FloatKey.from_archive(ar)),
        )

class EAdditiveAnimationType(IntEnum):
    AAT_None = 0
    AAT_LocalSpaceBase = 1
    AAT_RotationOffsetMeshSpace = 2
    AAT_MAX = 3

class EAdditiveBasePoseType(IntEnum):
    ABPT_None = 0
    ABPT_RefPose = 1
    ABPT_AnimScaled = 2
    ABPT_AnimFrame = 3
    ABPT_LocalAnimFrame = 4
    ABPT_MAX = 5

@dataclass(slots=True)
class UEAnimMetadata:
    num_frames: int
    frames_per_second: float
    ref_pose_path: str = ""
    additive_anim_type: EAdditiveAnimationType = EAdditiveAnimationType.AAT_None
    ref_pose_type: EAdditiveBasePoseType = EAdditiveBasePoseType.ABPT_None
    ref_frame_index: int = 0

    @classmethod
    def from_archive(cls, ar: FArchiveReader) -> UEAnimMetadata:
        return cls(
            num_frames=ar.read_int(),
            frames_per_second=ar.read_float(),
            ref_pose_path=ar.read_fstring(),
            additive_anim_type=EAdditiveAnimationType(int.from_bytes(ar.read_byte(), byteorder="big")),
            ref_pose_type=EAdditiveBasePoseType(int.from_bytes(ar.read_byte(), byteorder="big")),
            ref_frame_index=ar.read_int()
        )

@dataclass(slots=True)
class UEAnim:
    metadata: UEAnimMetadata = None
    curves: list[Curve] = field(default_factory=list)

    @classmethod
    def from_archive(cls, ar: FArchiveReader) -> UEAnim:
        data = cls()
        if ar.file_version < EUEFormatVersion.SerializeAssetMetadata:
            data.metadata = UEAnimMetadata(num_frames=ar.read_int(), frames_per_second=ar.read_float())

        while not ar.eof():
            section_name = ar.read_fstring()
            array_size = ar.read_int()
            byte_size = ar.read_int()
            pos = ar.data.tell()

            match section_name:
                case "METADATA":
                    data.metadata = UEAnimMetadata.from_archive(ar)
                case "CURVES":
                    data.curves = ar.read_array(array_size, lambda ar: Curve.from_archive(ar))
                case "TRACKS":
                    ar.skip(byte_size) # Skip bone track data
                case _:
                    Log.warn(f"Unknown Animation Data: {section_name}, skipping.")
                    ar.skip(byte_size)
            
            # Ensure we've moved to the next chunk correctly
            current_pos = ar.data.tell()
            if current_pos != pos + byte_size:
                ar.data.seek(pos + byte_size)

        return data
# endregion

# =============================================================================
# STANDALONE SCRIPT LOGIC
# =============================================================================

def parse_ueanim_file(file_path: Path) -> UEAnim | None:
    """Reads a .ueanim file and parses it into a UEAnim object."""
    Log.info(f"Processing: {file_path.name}")
    try:
        with file_path.open("rb") as file:
            data = file.read()
    except IOError as e:
        Log.error(f"Could not read file {file_path}: {e}")
        return None

    with FArchiveReader(data) as ar:
        magic = ar.read_string(len(MAGIC))
        if magic != MAGIC:
            Log.error(f"Invalid magic in {file_path.name}. Expected '{MAGIC}', got '{magic}'.")
            return None

        identifier = ar.read_fstring()
        if identifier != ANIM_IDENTIFIER:
            Log.error(f"Not a .ueanim file (identifier: {identifier}). Skipping.")
            return None

        file_version = EUEFormatVersion(int.from_bytes(ar.read_byte(), byteorder="big"))
        if file_version > EUEFormatVersion.LatestVersion:
            Log.error(f"File version {file_version} not supported. Skipping.")
            return None

        object_name = ar.read_fstring()
        is_compressed = ar.read_bool()
        read_archive = ar

        if is_compressed:
            compression_type = ar.read_fstring()
            uncompressed_size = ar.read_int()
            _compressed_size = ar.read_int()
            compressed_data = ar.read_to_end()
            
            try:
                if compression_type == "GZIP":
                    decompressed_data = gzip.decompress(compressed_data)
                elif compression_type == "ZSTD":
                    zstd_decompressor = zstd.ZstdDecompressor()
                    decompressed_data = zstd_decompressor.decompress(compressed_data, uncompressed_size)
                else:
                    Log.error(f"Unknown compression type: {compression_type}. Skipping.")
                    return None
                
                if len(decompressed_data) != uncompressed_size:
                    Log.warn(f"Decompressed size mismatch for {file_path.name}")

                read_archive = FArchiveReader(decompressed_data)

            except Exception as e:
                Log.error(f"Decompression failed for {file_path.name}: {e}")
                return None

        read_archive.file_version = file_version
        read_archive.metadata["scale"] = 0.01 # Default value, not used for curves

        return UEAnim.from_archive(read_archive)

def process_and_write_csv(ueanim_data: UEAnim, output_path: Path):
    """Processes UEAnim data and writes it to a CSV file."""
    if not ueanim_data.curves:
        Log.warn(f"No curve data found in {output_path.stem}. Skipping CSV creation.")
        return

    num_frames = ueanim_data.metadata.num_frames
    header = [curve.name for curve in ueanim_data.curves]
    
    # Create a data table (list of lists) initialized to 0.0
    # Rows: frames, Columns: curves
    data_table = [[0.0] * len(header) for _ in range(num_frames)]

    # Populate the table with keyframe data
    for i, curve in enumerate(ueanim_data.curves):
        if not curve.keys:
            continue

        key_iter = iter(sorted(curve.keys, key=lambda k: k.frame))
        current_key = next(key_iter)
        next_key = next(key_iter, None)
        
        for frame_idx in range(num_frames):
            if next_key and frame_idx >= next_key.frame:
                current_key = next_key
                next_key = next(key_iter, None)
            
            data_table[frame_idx][i] = current_key.value

    # Write to CSV
    try:
        with output_path.open('w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerows(data_table)
        Log.info(f"Successfully created: {output_path.name}")
    except IOError as e:
        Log.error(f"Could not write to file {output_path}: {e}")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Batch convert .ueanim files to CSV for shape key animation.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to the folder containing .ueanim files."
    )
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    if not input_path.is_dir():
        Log.error(f"Error: Provided path '{input_path}' is not a valid directory.")
        sys.exit(1)

    Log.time_start("Total processing time")
    ueanim_files = list(input_path.rglob("*.ueanim"))

    if not ueanim_files:
        Log.warn(f"No .ueanim files found in '{input_path}'.")
        sys.exit(0)

    Log.info(f"Found {len(ueanim_files)} .ueanim file(s). Starting conversion...")
    
    processed_count = 0
    for file_path in ueanim_files:
        ueanim_data = parse_ueanim_file(file_path)
        if ueanim_data and ueanim_data.metadata:
            output_csv_path = file_path.with_suffix('.csv')
            process_and_write_csv(ueanim_data, output_csv_path)
            processed_count += 1
        else:
             Log.error(f"Failed to parse or missing metadata in {file_path.name}. Skipping.")

    Log.info(f"Finished. Processed {processed_count}/{len(ueanim_files)} files.")
    Log.time_end("Total processing time")


if __name__ == "__main__":
    main()