use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{Field, Schema};
use parquet::{
    arrow::arrow_writer::ArrowWriter,
    basic::{Compression, Encoding},
    errors::Result,
    file::properties::WriterProperties,
    schema::types::ColumnPath,
};
use rayon::prelude::*;
use std::{collections::HashMap, mem::Discriminant, sync::Arc};

/// The results of analyzing a single column for the best encoding/compression.
pub struct ColumnOptimizationResult {
    /// All combinations of encoding and compression and their estimated sizes.
    pub all_results: Vec<(Encoding, HashMap<Discriminant<Compression>, usize>)>,
    pub best_encoding: Encoding,
    pub best_compression: Compression,
    pub min_size: usize,
}

/// A trait for defining a strategy to find the best encoding/compression for a column.
pub trait ColumnOptimizer {
    /// Returns the name of the optimizer.
    fn name(&self) -> &'static str;

    /// Finds the best configuration for a given column.
    fn find_best_config(
        &self,
        rg_idx: usize,
        field: &Field,
        column_array: &ArrayRef,
        applicable_encodings: &[Encoding],
        compressions_to_test: &[Compression],
    ) -> Result<ColumnOptimizationResult>;
}

/// An optimizer that exhaustively checks every combination of encoding and compression.
pub struct BruteForceOptimizer;
impl ColumnOptimizer for BruteForceOptimizer {
    fn name(&self) -> &'static str {
        "BruteForceOptimizer"
    }

    fn find_best_config(
        &self,
        _rg_idx: usize,
        field: &Field,
        column_array: &ArrayRef,
        applicable_encodings: &[Encoding],
        compressions_to_test: &[Compression],
    ) -> Result<ColumnOptimizationResult> {
        // --- Full Enumeration ---
        let all_results = enumerate_all_combinations(
            field,
            column_array,
            applicable_encodings,
            compressions_to_test,
        )?;

        // --- Find Best Combination ---
        let ((best_encoding, best_compression), min_size) = all_results
            .iter()
            .flat_map(|(encoding, sizes)| {
                // We need to find the original `Compression` value for the best config.
                sizes.iter().map(move |(disc, size)| {
                    let compression = *compressions_to_test
                        .iter()
                        .find(|c| std::mem::discriminant(*c) == *disc)
                        .unwrap(); // This is safe as the map is built from this vec
                    ((*encoding, compression), *size)
                })
            })
            .min_by_key(|&(_, size)| size)
            .unwrap_or(((Encoding::PLAIN, Compression::UNCOMPRESSED), usize::MAX));

        Ok(ColumnOptimizationResult {
            all_results,
            best_encoding,
            best_compression,
            min_size,
        })
    }
}

/// An optimizer that uses a threshold to decide if compression is worthwhile.
pub struct ThresholdOptimizer {
    pub encoding_threshold: f64,
    pub compression_threshold: f64,
}

impl ColumnOptimizer for ThresholdOptimizer {
    fn name(&self) -> &'static str {
        "ThresholdOptimizer"
    }

    fn find_best_config(
        &self,
        rg_idx: usize,
        field: &Field,
        column_array: &ArrayRef,
        applicable_encodings: &[Encoding],
        compressions_to_test: &[Compression],
    ) -> Result<ColumnOptimizationResult> {
        let all_results = enumerate_all_combinations(
            field,
            column_array,
            applicable_encodings,
            compressions_to_test,
        )?;

        let mut final_candidates = Vec::new();

        // Find the baseline size using PLAIN encoding, uncompressed.
        let plain_uncompressed_size = all_results
            .iter()
            .find(|(e, _)| *e == Encoding::PLAIN)
            .and_then(|(_, sizes)| sizes.get(&std::mem::discriminant(&Compression::UNCOMPRESSED)))
            .cloned()
            .unwrap_or(usize::MAX);

        for (encoding, compression_sizes) in &all_results {
            let encoded_uncompressed_size = compression_sizes
                .get(&std::mem::discriminant(&Compression::UNCOMPRESSED))
                .cloned()
                .unwrap_or(usize::MAX);

            // Step 1: Check if the encoding itself is worthwhile compared to PLAIN.
            if encoded_uncompressed_size
                >= (plain_uncompressed_size as f64 * (1.0 - self.encoding_threshold / 100.0))
                    as usize
            {
                if *encoding != Encoding::PLAIN {
                    // Don't log for the baseline itself
                    log::trace!(
                            "[{}] RG {}, Col '{}': Encoding {:?} did not meet the {:.1}% size reduction threshold over PLAIN. Consider the compression is useful, but the encoding is not.",
                            self.name(), rg_idx, field.name(), encoding, self.encoding_threshold
                        );
                }
                // NOTE: Consider the compression is useful, but the encoding is not.
                // continue; // Skip to the next encoding
            }

            // Step 2: For this worthy encoding, check which compressions are worthwhile.
            let worthy_compressions: Vec<_> = compression_sizes
                .iter()
                .filter(|&(_, &size)| {
                    size <= (encoded_uncompressed_size as f64
                        * (1.0 - self.compression_threshold / 100.0))
                        as usize
                })
                .collect();

            if worthy_compressions.is_empty() {
                log::trace!(
                        "[{}] RG {}, Col '{}': For encoding {:?}, no compression met the {:.1}% threshold. Using UNCOMPRESSED.",
                        self.name(), rg_idx, field.name(), encoding, self.compression_threshold
                    );
                // If no compression is "worth it", the candidate is the uncompressed version of this encoding.
                final_candidates.push((
                    (*encoding, Compression::UNCOMPRESSED),
                    encoded_uncompressed_size,
                ));
            } else {
                // Add all "worthy" compressions as candidates
                for (disc, size) in worthy_compressions {
                    let compression = *compressions_to_test
                        .iter()
                        .find(|c| std::mem::discriminant(*c) == *disc)
                        .unwrap();
                    final_candidates.push(((*encoding, compression), *size));
                }
            }
        }

        if final_candidates.is_empty() {
            log::trace!(
                    "[{}] RG {}, Col '{}': No encoding/compression combination offered a benefit. Defaulting to PLAIN/UNCOMPRESSED.",
                    self.name(),
                    rg_idx,
                    field.name()
                );
            // Ensure there's always at least the default option
            final_candidates.push((
                (Encoding::PLAIN, Compression::UNCOMPRESSED),
                plain_uncompressed_size,
            ));
        }

        // From all candidates, pick the absolute best one.
        let ((best_encoding, best_compression), min_size) = final_candidates
            .into_iter()
            .min_by_key(|&(_, size)| size)
            .unwrap(); // Safe due to the push above

        Ok(ColumnOptimizationResult {
            all_results,
            best_encoding,
            best_compression,
            min_size,
        })
    }
}

/// An optimizer that only tests a specific subset of encodings.
pub struct LightweightOptimizer;

impl ColumnOptimizer for LightweightOptimizer {
    fn name(&self) -> &'static str {
        "LightweightOptimizer"
    }

    fn find_best_config(
        &self,
        _rg_idx: usize,
        field: &Field,
        column_array: &ArrayRef,
        applicable_encodings: &[Encoding],
        compressions_to_test: &[Compression],
    ) -> Result<ColumnOptimizationResult> {
        // The filtering of compressions happens before this function is called.
        // Therefore, its implementation is identical to the BruteForceOptimizer.
        let all_results = enumerate_all_combinations(
            field,
            column_array,
            applicable_encodings,
            compressions_to_test,
        )?;

        let ((best_encoding, best_compression), min_size) = all_results
            .iter()
            .flat_map(|(encoding, sizes)| {
                sizes.iter().map(move |(disc, size)| {
                    let compression = *compressions_to_test
                        .iter()
                        .find(|c| std::mem::discriminant(*c) == *disc)
                        .unwrap();
                    ((*encoding, compression), *size)
                })
            })
            .min_by_key(|&(_, size)| size)
            .unwrap_or(((Encoding::PLAIN, Compression::UNCOMPRESSED), usize::MAX));

        Ok(ColumnOptimizationResult {
            all_results,
            best_encoding,
            best_compression,
            min_size,
        })
    }
}

/// Helper function to exhaustively test all encoding/compression combinations for a column.
///
/// This can be reused by different `ColumnOptimizer` implementations.
fn enumerate_all_combinations(
    field: &Field,
    column_array: &ArrayRef,
    applicable_encodings: &[Encoding],
    compressions_to_test: &[Compression],
) -> Result<Vec<(Encoding, HashMap<Discriminant<Compression>, usize>)>> {
    let results: Vec<_> = applicable_encodings
            .into_par_iter()
            .map(|&encoding| {
                let mut compression_sizes = HashMap::new();
                for &compression in compressions_to_test {
                    match estimate_column_chunk_size(field, column_array, encoding, compression) {
                        Ok(size) => {
                            compression_sizes.insert(std::mem::discriminant(&compression), size);
                        }
                        Err(e) => {
                            log::warn!(
                                "Could not estimate size for column '{}' with encoding={:?} and compression={:?}. Skipping. Reason: {}",
                                field.name(),
                                encoding,
                                compression,
                                e
                            );
                        }
                    }
                }
                (encoding, compression_sizes)
            })
            .collect();
    Ok(results)
}

fn estimate_column_chunk_size(
    field: &Field,
    column_array: &ArrayRef,
    encoding: Encoding,
    compression: Compression,
) -> Result<usize> {
    // Clone data to be moved into the panic-catching closure.
    // This is necessary because `catch_unwind` requires the closure to own its data
    // to be considered safe to run after a potential panic.
    let field_clone = field.clone();
    let column_array_clone = column_array.clone();
    let schema = Arc::new(Schema::new(vec![field_clone]));
    let column_path = ColumnPath::from(schema.field(0).name().as_str());

    let mut props_builder = WriterProperties::builder();
    if encoding == Encoding::RLE_DICTIONARY {
        props_builder = props_builder
            .set_column_dictionary_enabled(column_path.clone(), true)
            .set_dictionary_page_size_limit(usize::MAX);
    } else {
        props_builder = props_builder
            .set_column_dictionary_enabled(column_path.clone(), false)
            .set_column_encoding(column_path.clone(), encoding);
    }

    props_builder = props_builder.set_column_compression(column_path, compression);
    let props = props_builder.build();

    let mut buffer = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut buffer, schema.clone(), Some(props))?;
    let batch_to_write =
        RecordBatch::try_from_iter(vec![(schema.field(0).name().as_str(), column_array_clone)])?;

    // The write and close calls can panic with unsupported codecs.
    writer.write(&batch_to_write)?;
    writer.close()?;

    Ok(buffer.len())
}