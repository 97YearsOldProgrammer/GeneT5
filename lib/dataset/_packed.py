import struct
import json
import math
import pathlib


#######################
#####  Constants  #####
#######################


PACKED_MAGIC   = b'GT5P'
PACKED_VERSION = 1


#########################
#####  PackedSample #####
#########################


class PackedSample:
    """
    Pre-packed training sample ready for direct streaming

    Contains concatenated and padded sequences from multiple chunks,
    with isolation padding and segment info pre-computed.
    """

    def __init__(
        self,
        input_ids,
        labels,
        segment_ids,
        segment_starts,
        segment_ends,
        num_segments,
        source_chunks   = None,
        total_input_len = None,
        total_label_len = None,
    ):

        self.input_ids       = input_ids
        self.labels          = labels
        self.segment_ids     = segment_ids
        self.segment_starts  = segment_starts
        self.segment_ends    = segment_ends
        self.num_segments    = num_segments
        self.source_chunks   = source_chunks or []
        self.total_input_len = total_input_len or len(input_ids)
        self.total_label_len = total_label_len or len(labels)

    def to_bytes(self):
        """Serialize packed sample to bytes"""

        meta = {
            "segment_starts":  self.segment_starts,
            "segment_ends":    self.segment_ends,
            "num_segments":    self.num_segments,
            "source_chunks":   self.source_chunks,
            "total_input_len": self.total_input_len,
            "total_label_len": self.total_label_len,
        }

        meta_json = json.dumps(meta).encode('utf-8')

        # Pack token IDs as int32
        input_bytes   = struct.pack(f'<{len(self.input_ids)}i', *self.input_ids)
        labels_bytes  = struct.pack(f'<{len(self.labels)}i', *self.labels)
        seg_ids_bytes = struct.pack(f'<{len(self.segment_ids)}i', *self.segment_ids)

        meta_len      = len(meta_json)
        input_len     = len(self.input_ids)
        labels_len    = len(self.labels)
        seg_ids_len   = len(self.segment_ids)

        header = struct.pack('<4I', meta_len, input_len, labels_len, seg_ids_len)
        data   = header + meta_json + input_bytes + labels_bytes + seg_ids_bytes

        return data

    @classmethod
    def from_bytes(cls, data):
        """Deserialize packed sample from bytes"""

        header_size                                = struct.calcsize('<4I')
        meta_len, input_len, labels_len, seg_ids_len = struct.unpack('<4I', data[:header_size])

        offset    = header_size
        meta_json = data[offset:offset + meta_len].decode('utf-8')
        offset   += meta_len

        input_bytes = data[offset:offset + input_len * 4]
        input_ids   = list(struct.unpack(f'<{input_len}i', input_bytes))
        offset     += input_len * 4

        labels_bytes = data[offset:offset + labels_len * 4]
        labels       = list(struct.unpack(f'<{labels_len}i', labels_bytes))
        offset      += labels_len * 4

        seg_ids_bytes = data[offset:offset + seg_ids_len * 4]
        segment_ids   = list(struct.unpack(f'<{seg_ids_len}i', seg_ids_bytes))

        meta = json.loads(meta_json)

        return cls(
            input_ids       = input_ids,
            labels          = labels,
            segment_ids     = segment_ids,
            segment_starts  = meta["segment_starts"],
            segment_ends    = meta["segment_ends"],
            num_segments    = meta["num_segments"],
            source_chunks   = meta.get("source_chunks", []),
            total_input_len = meta.get("total_input_len"),
            total_label_len = meta.get("total_label_len"),
        )


#######################
#####  Packing    #####
#######################


def align_to_block(length, block_size=64):
    """Align length to next block boundary"""

    return math.ceil(length / block_size) * block_size


def pack_chunks_to_sample(
    chunks,
    tokenizer,
    block_size   = 64,
    window_size  = 256,
    label_pad_id = -100,
):
    """
    Pack multiple chunks into a single PackedSample

    Applies isolation padding between segments to prevent attention leakage.
    Uses pre-tokenized IDs when available to skip redundant tokenization.
    """

    pad_token_id = tokenizer.pad_token_id
    sep_token_id = tokenizer.convert_tokens_to_ids("[SEP]")

    # Check if chunks have pre-tokenized IDs
    has_pretokenized = all(c.input_ids is not None and c.target_ids is not None for c in chunks)

    if has_pretokenized:
        # Use pre-tokenized IDs directly (fast path)
        input_ids_list  = [c.input_ids for c in chunks]
        target_ids_list = [c.target_ids for c in chunks]
    else:
        # Fall back to tokenization (legacy path)
        input_texts  = [c.get_input_text() for c in chunks]
        target_texts = [c.get_target_text() for c in chunks]

        input_enc       = tokenizer(input_texts, add_special_tokens=False)
        target_enc      = tokenizer(target_texts, add_special_tokens=False)
        input_ids_list  = input_enc["input_ids"]
        target_ids_list = target_enc["input_ids"]

    # Pack inputs with isolation
    packed_input   = []
    packed_labels  = []
    segment_ids    = []
    segment_starts = []
    segment_ends   = []

    for i in range(len(chunks)):
        inp_ids = input_ids_list[i]
        lbl_ids = target_ids_list[i]

        # Record segment start
        segment_starts.append(len(packed_input))

        # Add input tokens
        packed_input.extend(inp_ids)
        segment_ids.extend([i] * len(inp_ids))

        # Add label tokens
        packed_labels.extend(lbl_ids)

        # Record segment end
        segment_ends.append(len(packed_input))

        # Add isolation padding (except after last segment)
        if i < len(chunks) - 1:
            # Separator token
            packed_input.append(sep_token_id)
            segment_ids.append(-1)
            packed_labels.append(label_pad_id)

            # Compute isolation padding
            current_pos = len(packed_input)
            target_pos  = align_to_block(current_pos + window_size + 1, block_size)
            pad_len     = target_pos - current_pos

            packed_input.extend([pad_token_id] * pad_len)
            segment_ids.extend([-1] * pad_len)
            packed_labels.extend([label_pad_id] * pad_len)

    # Source chunk info for debugging
    source_chunks = [
        {"seqid": c.seqid, "start": c.start, "end": c.end}
        for c in chunks
    ]

    return PackedSample(
        input_ids       = packed_input,
        labels          = packed_labels,
        segment_ids     = segment_ids,
        segment_starts  = segment_starts,
        segment_ends    = segment_ends,
        num_segments    = len(chunks),
        source_chunks   = source_chunks,
        total_input_len = len(packed_input),
        total_label_len = len(packed_labels),
    )


#################
#####  I/O  #####
#################


def write_packed(samples, output_path, show_progress=True):
    """Write packed samples to binary file"""

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_samples = len(samples)

    with open(output_path, 'wb') as f:
        f.write(PACKED_MAGIC)
        f.write(struct.pack('<B', PACKED_VERSION))
        f.write(struct.pack('<I', num_samples))

        # Reserve space for offset table (64-bit offsets, 32-bit lengths)
        offset_table_pos = f.tell()
        f.write(b'\x00' * (num_samples * 12))

        offsets = []

        for i, sample in enumerate(samples):
            current_offset = f.tell()
            sample_bytes   = sample.to_bytes()

            f.write(sample_bytes)
            offsets.append((current_offset, len(sample_bytes)))

            if show_progress and (i + 1) % 10000 == 0:
                pct = 100 * (i + 1) / num_samples
                print(f"    Writing: {i + 1:,}/{num_samples:,} ({pct:.1f}%)", end='\r')

        if show_progress and num_samples > 10000:
            print(f"    Writing: {num_samples:,}/{num_samples:,} (100.0%)")

        # Write offset table
        f.seek(offset_table_pos)
        for offset, length in offsets:
            f.write(struct.pack('<QI', offset, length))

    return output_path


def read_packed(input_path):
    """Read all packed samples from binary file"""

    with open(input_path, 'rb') as f:
        magic = f.read(4)
        if magic != PACKED_MAGIC:
            raise ValueError(f"Invalid magic header: {magic}")

        version     = struct.unpack('<B', f.read(1))[0]
        num_samples = struct.unpack('<I', f.read(4))[0]

        offsets = []
        for _ in range(num_samples):
            offset, length = struct.unpack('<QI', f.read(12))
            offsets.append((offset, length))

        samples = []
        for offset, length in offsets:
            f.seek(offset)
            data   = f.read(length)
            sample = PackedSample.from_bytes(data)
            samples.append(sample)

    return samples


def read_packed_at_index(input_path, index):
    """Read single packed sample by index"""

    with open(input_path, 'rb') as f:
        magic = f.read(4)
        if magic != PACKED_MAGIC:
            raise ValueError(f"Invalid magic header: {magic}")

        version     = struct.unpack('<B', f.read(1))[0]
        num_samples = struct.unpack('<I', f.read(4))[0]

        if index >= num_samples:
            raise IndexError(f"Index {index} out of range (total: {num_samples})")

        # Seek to offset table entry
        f.seek(9 + index * 12)
        offset, length = struct.unpack('<QI', f.read(12))

        f.seek(offset)
        data = f.read(length)

        return PackedSample.from_bytes(data)


def get_packed_info(input_path):
    """Get metadata about packed dataset"""

    with open(input_path, 'rb') as f:
        magic = f.read(4)
        if magic != PACKED_MAGIC:
            raise ValueError(f"Invalid magic header: {magic}")

        version     = struct.unpack('<B', f.read(1))[0]
        num_samples = struct.unpack('<I', f.read(4))[0]

        offsets = []
        for _ in range(num_samples):
            offset, length = struct.unpack('<QI', f.read(12))
            offsets.append((offset, length))

        total_size = sum(length for _, length in offsets)

    return {
        "version":     version,
        "num_samples": num_samples,
        "total_size":  total_size,
        "file_path":   str(input_path),
    }


####################################
#####  Streaming Write (NEW)   #####
####################################


class PackedWriter:
    """
    Streaming writer for packed samples - writes directly to disk

    Memory: O(offset_table) + O(1 sample) instead of O(all samples)
    """

    def __init__(self, output_path, num_samples):

        self.output_path = pathlib.Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.num_samples      = num_samples
        self.offsets          = []
        self.samples_written  = 0
        self.offset_table_pos = None

        self.file = open(self.output_path, 'wb')

        self.file.write(PACKED_MAGIC)
        self.file.write(struct.pack('<B', PACKED_VERSION))
        self.file.write(struct.pack('<I', num_samples))

        self.offset_table_pos = self.file.tell()
        self.file.write(b'\x00' * (num_samples * 12))

    def write_sample(self, sample):
        """Write a single packed sample"""

        current_offset = self.file.tell()
        sample_bytes   = sample.to_bytes()

        self.file.write(sample_bytes)
        self.offsets.append((current_offset, len(sample_bytes)))
        self.samples_written += 1

    def finalize(self):
        """Write offset table and close file"""

        if self.samples_written != self.num_samples:
            raise ValueError(
                f"Expected {self.num_samples} samples, wrote {self.samples_written}"
            )

        self.file.seek(self.offset_table_pos)
        for offset, length in self.offsets:
            self.file.write(struct.pack('<QI', offset, length))

        self.file.close()
        return self.output_path

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.file.closed:
            self.finalize()
        return False
