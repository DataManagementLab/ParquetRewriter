use crate::optimizer::ColumnOptimizer;
use clap::{ColorChoice, Parser};
use log::{info, LevelFilter};
use parquet::basic::{BrotliLevel, GzipLevel, ZstdLevel, Encoding};
use parquet::{basic::Compression, errors::Result};
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::{path::PathBuf, str::FromStr};

// !! Command-line arguments and enums for compression and optimization strategies

#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq)]
pub enum OptimizerStrategy {
    /// Exhaustively check every combination to find the absolute smallest size.
    BruteForce,
    /// Use separate thresholds to decide if encoding and compression are worthwhile.
    Threshold,
    /// Only test a specific subset of encodings.
    Lightweight,
}

/// A builder for creating a list of encodings from a string.
pub struct EncodingListBuilder(String);

impl EncodingListBuilder {
    /// Creates a new builder with the input string.
    pub fn new(s: String) -> Self {
        Self(s)
    }

    /// Builds the list of applicable encodings based on the predefined strategy or a custom list.
    pub fn build(&self, physical_type: parquet::basic::Type) -> Vec<Encoding> {
        match self.0.to_lowercase().as_str() {
            "parquet_v2" => get_v2_encodings(physical_type),
            "parquet_v1" => get_v1_encodings(physical_type),
            "plain" => vec![Encoding::PLAIN],
            custom => {
                // Custom list
                let requested_encodings: Vec<_> = custom
                    .split(',')
                    .map(|s| s.trim().to_uppercase())
                    .filter_map(|s| Encoding::from_str(&s).ok())
                    .collect();

                // Filter them by what's applicable for the physical type
                let applicable_encodings = get_v2_encodings(physical_type);
                applicable_encodings
                    .into_iter()
                    .filter(|enc| requested_encodings.contains(enc))
                    .collect()
            }
        }
    }
}

/// Returns encodings applicable for Parquet V2.
fn get_v2_encodings(physical_type: parquet::basic::Type) -> Vec<Encoding> {
    use parquet::basic::Type::*;
    match physical_type {
        BOOLEAN => vec![Encoding::PLAIN, Encoding::RLE],
        INT32 | INT64 => vec![
            Encoding::PLAIN,
            Encoding::RLE_DICTIONARY,
            Encoding::DELTA_BINARY_PACKED,
        ],
        INT96 => vec![Encoding::PLAIN, Encoding::RLE_DICTIONARY],
        FLOAT | DOUBLE => vec![
            Encoding::PLAIN,
            Encoding::RLE_DICTIONARY,
            Encoding::DELTA_BINARY_PACKED,
            Encoding::BYTE_STREAM_SPLIT,
        ],
        BYTE_ARRAY | FIXED_LEN_BYTE_ARRAY => vec![
            Encoding::PLAIN,
            Encoding::RLE_DICTIONARY,
            Encoding::DELTA_LENGTH_BYTE_ARRAY,
            Encoding::DELTA_BYTE_ARRAY,
        ],
    }
}

/// Returns encodings applicable for Parquet V1.
fn get_v1_encodings(physical_type: parquet::basic::Type) -> Vec<Encoding> {
    use parquet::basic::Type::*;
    let v1_encodings = vec![
        Encoding::PLAIN,
        Encoding::RLE,
        Encoding::RLE_DICTIONARY,
    ];
    let all_encodings = get_v2_encodings(physical_type);
    all_encodings
        .into_iter()
        .filter(|e| v1_encodings.contains(e))
        .collect()
}

#[derive(Debug, Parser)]
#[clap(author, version, about("Optimizes a Parquet file by choosing the best encoding for each column chunk."), long_about = None)]
#[clap(group(
    clap::ArgGroup::new("row_group_strategy")
        .args(&["row_group_count", "row_group_size", "row_group_precise_row_count"]),
))]
pub struct Args {
    /// Path to input parquet file.
    #[clap(short, long)]
    pub input: String,

    /// Path to output parquet file.
    #[clap(short, long)]
    pub output: String,

    /// Target number of pages per row group.
    /// If 0, uses the default data page row count limit (20,000).
    #[clap(long, default_value = "0")]
    pub page_count: usize,

    /// Target number of row groups in the output file. Mutually exclusive with --row-group-size and --row-group-precise-row-count.
    #[clap(long)]
    pub row_group_count: Option<usize>,

    /// Target best-effort **lower-bound** size for each row group (e.g., "256M", "1G"). Mutually exclusive with --row-group-count and --row-group-precise-row-count.
    #[clap(long)]
    pub row_group_size: Option<String>,

    /// Target exact number of rows for each row group (best-effort). Mutually exclusive with --row-group-count and --row-group-size.
    #[clap(long)]
    pub row_group_precise_row_count: Option<usize>,

    /// Set the verbosity of the log output.
    #[arg(long, default_value_t = LevelFilter::Info)]
    pub log_level: LevelFilter,

    /// Control when to use color for log messages.
    #[arg(long, value_name = "WHEN", default_value_t = ColorChoice::Auto)]
    pub color: ColorChoice,

    /// The compression strategy/strategies to apply.
    /// Can be a single value or a comma-separated list.
    /// Examples: --compression "snappy"; --compression "zstd(5)"; --compression "gzip(9),brotli(4)".
    /// Special values: 'lightweight' (uncompressed,snappy) and
    /// 'optimal' (all supported compressions with default levels).
    #[clap(long, value_delimiter = ',', default_value = "uncompressed")]
    pub compression: Vec<String>,

    /// Optional path for a full CSV report enumerating all combinations.
    /// The unit (KB, MB, etc.) will be appended to the filename automatically.
    #[clap(
        long,
        help = "Path for the full enumeration report CSV file.",
        value_name = "REPORT_PATH"
    )]
    pub full_report_path: Option<String>,

    /// The optimization strategy to use.
    #[clap(long, value_enum, default_value = "brute-force")]
    pub optimizer: OptimizerStrategy,

    /// Options for the Threshold optimizer.
    #[clap(flatten)]
    pub threshold_opts: ThresholdOptions,

    /// Specifies the encoding strategy or a custom list of encodings to test.
    ///
    /// Predefined strategies:
    /// - `parquet_v2`: (Default) Use modern encodings like DELTA_BINARY_PACKED.
    /// - `parquet_v1`: Use a restricted set (PLAIN, RLE, RLE_DICTIONARY).
    /// - `plain`: Use only PLAIN encoding.
    ///
    /// Custom list (comma-separated, case-insensitive):
    /// e.g., `plain,delta_binary_packed`
    #[clap(long, default_value = "parquet_v2", value_name = "STRATEGY_OR_LIST")]
    pub encodings: String,
}

/// A wrapper for `parquet::basic::Compression` to make it hashable for deduplication.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HashableCompression(Compression);

impl Hash for HashableCompression {
    fn hash<H: Hasher>(&self, state: &mut H) {
        core::mem::discriminant(&self.0).hash(state);
        match self.0 {
            Compression::GZIP(level) => level.hash(state),
            Compression::BROTLI(level) => level.hash(state),
            Compression::ZSTD(level) => level.hash(state),
            _ => {} // Other variants have no extra data
        }
    }
}

/// Parses a size string (e.g., "256M", "1G") into a number of bytes.
pub fn parse_size_string(size_str: &str) -> Result<usize> {
    let s = size_str.trim().to_uppercase();
    let (num_str, unit) = s.split_at(s.char_indices().find(|&(_, c)| c.is_alphabetic()).map_or(s.len(), |(i, _)| i));
    let num: usize = num_str.parse().map_err(|_| {
        parquet::errors::ParquetError::General(format!("Invalid size number: '{}'", num_str))
    })?;

    let multiplier = match unit {
        "" | "B" => 1,
        "K" | "KB" => 1024,
        "M" | "MB" => 1024 * 1024,
        "G" | "GB" => 1024 * 1024 * 1024,
        _ => {
            return Err(parquet::errors::ParquetError::General(format!(
                "Invalid size unit: '{}'",
                unit
            )))
        }
    };

    Ok(num * multiplier)
}

/// Holds the options for the Threshold optimizer strategy.
/// These arguments are only used if `--optimizer threshold` is specified.
#[derive(Debug, Parser, Clone)]
pub struct ThresholdOptions {
    /// A percentage (0-100) used by the Threshold optimizer. An encoding is only
    /// considered if it is at least this much smaller than the PLAIN encoding.
    #[clap(long, default_value = "1.0")]
    pub encoding_threshold: f64,

    /// A percentage (0-100) used by the Threshold optimizer. A compression
    /// is only chosen if it is at least this much smaller than the uncompressed
    /// size for the same encoding.
    #[clap(long, default_value = "1.0")]
    pub compression_threshold: f64,
}

/// Configures and initializes the global logger.
pub fn setup_logger(level: LevelFilter, color: ColorChoice) {
    let mut log_builder = env_logger::Builder::new();
    log_builder.filter_level(level);
    log_builder.write_style(match color {
        ColorChoice::Always => env_logger::WriteStyle::Always,
        ColorChoice::Auto => env_logger::WriteStyle::Auto,
        ColorChoice::Never => env_logger::WriteStyle::Never,
    });
    log_builder.init();
}

fn parse_compression_string(s: &str) -> Result<Compression> {
    log::info!("Parsing compression string: {}", s);
    let s_lower = s.to_lowercase();
    let (codec, level_str) = if let Some(pos) = s_lower.find('(') {
        let (codec, rest) = s_lower.split_at(pos);
        (codec, Some(rest.trim_matches(|p| p == '(' || p == ')')))
    } else {
        (s_lower.as_str(), None)
    };

    match codec {
        "uncompressed" => Ok(Compression::UNCOMPRESSED),
        "snappy" => Ok(Compression::SNAPPY),
        "lz4_raw" => Ok(Compression::LZ4_RAW),
        "gzip" => {
            let level = level_str
                .map(|l| l.parse::<u32>())
                .transpose()
                .map_err(|e| {
                    parquet::errors::ParquetError::General(format!(
                        "Invalid gzip compression level: {}",
                        e
                    ))
                })?
                .unwrap_or_else(|| GzipLevel::default().compression_level());
            Ok(Compression::GZIP(GzipLevel::try_new(level)?))
        }
        "brotli" => {
            let level = level_str
                .map(|l| l.parse::<u32>())
                .transpose()
                .map_err(|e| {
                    parquet::errors::ParquetError::General(format!(
                        "Invalid brotli compression level: {}",
                        e
                    ))
                })?
                .unwrap_or_else(|| BrotliLevel::default().compression_level());
            Ok(Compression::BROTLI(BrotliLevel::try_new(level)?))
        }
        "zstd" => {
            let level = level_str
                .map(|l| l.parse::<i32>())
                .transpose()
                .map_err(|e| {
                    parquet::errors::ParquetError::General(format!(
                        "Invalid zstd compression level: {}",
                        e
                    ))
                })?
                .unwrap_or_else(|| ZstdLevel::default().compression_level());
            Ok(Compression::ZSTD(ZstdLevel::try_new(level)?))
        }
        "lz4" => {
            info!("Note: Lz4 is not supported in the Rust Parquet library as of version 14.0.0 and will be skipped.");
            Err(parquet::errors::ParquetError::General(
                "Lz4 is not supported".to_string(),
            ))
        }
        _ => Err(parquet::errors::ParquetError::General(format!(
            "Unknown compression codec: {}",
            s
        ))),
    }
}

/// Parses the user's compression choices, expands meta-strategies, and returns a deduplicated list.
fn parse_compression_list(choices: &[String]) -> Vec<Compression> {
    let mut final_compressions = HashSet::new();

    for choice_str in choices {
        match choice_str.to_lowercase().as_str() {
            "lightweight" => {
                final_compressions.insert(HashableCompression(Compression::UNCOMPRESSED));
                final_compressions.insert(HashableCompression(Compression::SNAPPY));
            }
            "optimal" => {
                final_compressions.insert(HashableCompression(Compression::UNCOMPRESSED));
                final_compressions.insert(HashableCompression(Compression::SNAPPY));
                final_compressions
                    .insert(HashableCompression(Compression::GZIP(Default::default())));
                final_compressions.insert(HashableCompression(Compression::BROTLI(
                    Default::default(),
                )));
                final_compressions
                    .insert(HashableCompression(Compression::ZSTD(Default::default())));
                final_compressions.insert(HashableCompression(Compression::LZ4_RAW));
            }
            s => match parse_compression_string(s) {
                Ok(compression) => {
                    final_compressions.insert(HashableCompression(compression));
                }
                Err(e) => {
                    log::warn!("Skipping invalid compression option '{}': {}", s, e);
                }
            },
        }
    }

    final_compressions.into_iter().map(|hc| hc.0).collect()
}
/// Builds the list of compression codecs to test based on the command-line strategy.
pub fn build_compressions_list(
    optimizer_strategy: OptimizerStrategy,
    compression_choices: &[String],
) -> Vec<Compression> {
    let compressions = parse_compression_list(compression_choices);
    // if the optimizer_strategy is threshold, we should ensure that UNCOMPRESSED is included for a fair baseline comparison, even if the user didn't specify it.
    if optimizer_strategy == OptimizerStrategy::Threshold
        && !compressions.iter().any(|c| std::mem::discriminant(c) == std::mem::discriminant(&Compression::UNCOMPRESSED))
    {
        log::info!("Threshold optimizer requires uncompressed baseline. Adding UNCOMPRESSED to the compression list.");
        let mut extended_compressions = compressions.clone();
        extended_compressions.push(Compression::UNCOMPRESSED);
        return extended_compressions;
    }
    info!(
        "Compressions to be tested: {:?}",
        compressions
            .iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
    );

    compressions
}

/// Instantiates the correct optimizer based on the command-line arguments.
pub fn build_optimizer(args: &Args) -> Result<Box<dyn ColumnOptimizer + Send + Sync>> {
    let optimizer: Box<dyn ColumnOptimizer + Send + Sync> = match args.optimizer {
        OptimizerStrategy::BruteForce => Box::new(super::optimizer::BruteForceOptimizer),
        OptimizerStrategy::Lightweight => Box::new(super::optimizer::LightweightOptimizer),
        OptimizerStrategy::Threshold => {
            let opts = &args.threshold_opts;
            if !(0.0..=100.0).contains(&opts.encoding_threshold)
                || !(0.0..=100.0).contains(&opts.compression_threshold)
            {
                panic!("Threshold values must be between 0 and 100.");
            }
            Box::new(super::optimizer::ThresholdOptimizer {
                encoding_threshold: opts.encoding_threshold,
                compression_threshold: opts.compression_threshold,
            })
        }
    };
    Ok(optimizer)
}

/// Constructs the final output file path, embedding the optimizer name.
pub fn build_output_path(base_path: &str, _optimizer_name: &str) -> std::path::PathBuf {
    let p = std::path::Path::new(base_path);
    let parent = p.parent().unwrap_or_else(|| std::path::Path::new(""));
    let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
    let extension = p.extension().and_then(|s| s.to_str()).unwrap_or("parquet");
    parent.join(format!("{}.{}", stem, extension))
}