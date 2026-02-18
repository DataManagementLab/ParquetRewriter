# ParquetRewriter

This tool is a general-purpose Parquet Rewriter: it takes an input Parquet file and a set of specified configuration options, and produces an output Parquet file with the same logical content but different physical layout. The rewriter is built entirely on top of Apache Arrow Rust :heart:, with the goal of offering far more flexible configurability than [existing tools](https://github.com/apache/arrow-rs/blob/main/parquet/src/bin/parquet-rewrite.rs).

This tool represents our first step toward studying how Parquet configuration choices influence read performance during query execution. We also believe that default configurations from other rewriters do not fit all use cases and are often suboptimal. We also expect that different hardware -- CPUs, GPUs, FPGAs -- may benefit from different configurations. Therefore, we welcome pull requests for additional options tailored to your hardware.

## CLI Reference

```Bash
# Basic Example
cargo run --release -- --input <FILE> --output <FILE> [OPTIONS]
```

**Essential Options:**

- `--input`, `--output`: Paths to source and destination files.
- `--page-count <N>`: Target pages per row group (Default: `0` [auto]).
- `--row-group-precise-row-count <N>`: Target exact rows per group.
- `--compression <STRATEGY>`: `lightweight` (default), `optimal`, or specific codecs.
- `--encodings <STRATEGY>`: `parquet_v2` (default), `parquet_v1`, or `plain`.
- `--optimizer <STRATEGY>`: `brute-force` (default) checks every combination; `threshold` uses heuristics for speed.
- `--full-report-path <PATH>`: Exports a CSV detailing the decision process for every column.

For full options: `cargo run --release -- --help`

### Detailed Configuration Strategies

#### How to optimize Row Groups?

Larger chunks reduce metadata overhead. Options are mutually exclusive:

* `--row-group-precise-row-count <N>` **(Recommended)**: Forces exactly N rows per group.
* `--row-group-size <SIZE>`: Sets approximate target size (e.g., `1G`, `256M`).
* `--row-group-count <N>`: Splits file into N equal parts.

#### How to select Encodings?

- `--encodings parquet_v2`: **(Default)** Uses modern, efficient encodings (`DELTA_BINARY_PACKED`, `BYTE_STREAM_SPLIT`).
- `--encodings parquet_v1`: Restricts to legacy encodings (`PLAIN`, `RLE_DICTIONARY`).
- `--encodings plain`: Forces raw `PLAIN` encoding.

#### How to choose the best Compression?

Benchmarks multiple codecs per column to find the local optimum.

- `--compression lightweight`: **(Default)** Tests `uncompressed` and `snappy`.
- `--compression optimal`: Brute-force tests *all* supported algorithms (`zstd`, `lz4`, `gzip`, `brotli`).
- **Custom List:** e.g., `--compression "uncompressed,zstd(3)"` allows you to target specific algorithms.

