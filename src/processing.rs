use crate::{
    cli::{parse_size_string, Args, EncodingListBuilder},
    optimizer::ColumnOptimizer,
    reporting::RowGroupReport,
};
use arrow_array::RecordBatch;
use arrow_schema::Schema;
use log::{info, trace};
use parquet::{
    arrow::arrow_reader::{ParquetRecordBatchReaderBuilder, RowSelection, RowSelector},
    basic::{Compression, Encoding, Type},
    errors::Result,
    file::{
        metadata::{ColumnChunkMetaData, ParquetMetaData},
        properties::{EnabledStatistics, WriterProperties},
    },
    schema::types::ColumnPath,
};
use rayon::prelude::*;
use std::{fs::File, path::Path, sync::Arc};


/// Holds the optimal configuration for a single column.
pub struct ColumnConfig {
    pub encoding: Encoding,
    pub compression: Compression,
}

/// The strategy for partitioning rows into groups, holding the calculated rows per group.
#[derive(Debug, Clone, Copy)]
pub enum RowPartitioningStrategy {
    /// Distribute rows as evenly as possible. The value is the calculated number of rows per group.
    Average(usize),
    /// Set a fixed number of rows for each group. The value is the user-specified number of rows per group.
    Exact(usize),
}

// ! For reference: Get applicable encodings for a physical type
// pub fn get_applicable_encodings(physical_type: Type) -> Vec<Encoding> {
//     match physical_type {
//         Type::BOOLEAN => vec![Encoding::PLAIN, Encoding::RLE],
//         Type::INT32 | Type::INT64 => {
//             vec![
//                 Encoding::PLAIN,
//                 Encoding::RLE_DICTIONARY,
//                 Encoding::DELTA_BINARY_PACKED,
//             ]
//         }
//         Type::INT96 => vec![Encoding::PLAIN, Encoding::RLE_DICTIONARY], // E.g., for legacy timestamps
//         Type::FLOAT | Type::DOUBLE => {
//             vec![
//                 Encoding::PLAIN,
//                 Encoding::RLE_DICTIONARY,
//                 Encoding::DELTA_BINARY_PACKED,
//                 Encoding::BYTE_STREAM_SPLIT,
//             ]
//         }
//         Type::BYTE_ARRAY | Type::FIXED_LEN_BYTE_ARRAY => {
//             vec![
//                 Encoding::PLAIN,
//                 Encoding::RLE_DICTIONARY,
//                 Encoding::DELTA_LENGTH_BYTE_ARRAY,
//                 Encoding::DELTA_BYTE_ARRAY,
//             ]
//         }
//     }
// }

// ! 1. Determine Row Grouping Strategy

/// A plan for reading the data for a new, remapped row group.
#[derive(Debug)]
pub struct RowGroupPlan {
    /// The index of the new row group.
    pub new_rg_idx: usize,
    /// The indices of the original row groups to read from.
    pub original_rg_indices: Vec<usize>,
    /// The selection of rows to read from the combined original row groups.
    pub selection: RowSelection,
    /// The metadata for the columns in the first original row group, used for analysis.
    pub column_metadatas: Vec<ColumnChunkMetaData>,
}


/// Analyzes CLI arguments and file metadata to create a row group processing plan.
///
/// This function centralizes the logic for deciding whether to remap row groups.
///
/// - If no row group arguments (`--row-group-count`, `--row-group-size`, `--row-group-precise-row-count`)
///   are provided, it returns `Ok(None)`, indicating the original row group structure should be preserved.
/// - If any of those arguments are provided, it calculates the new row group layout
///   and returns `Ok(Some(Vec<RowGroupPlan>))`.
///
/// # Returns
/// - `Ok(None)`: Use the original row group structure.
/// - `Ok(Some(plans))`: Use the new row group structure defined by the plans.
/// - `Err(e)`: An error occurred during planning (e.g., during file sampling).
pub fn create_row_group_remapping_plan(
    args: &Args,
    file_metadata: &ParquetMetaData,
    arrow_schema: &Schema,
    optimizer: &(dyn ColumnOptimizer + Send + Sync),
    compressions_to_test: &[Compression],
) -> Result<Option<Vec<RowGroupPlan>>> {
    let use_default_layout = args.row_group_count.is_none()
        && args.row_group_size.is_none()
        && args.row_group_precise_row_count.is_none();

    if use_default_layout {
        info!("No row group arguments provided. Preserving original row group structure.");
        return Ok(None);
    }

    // determine_row_group_count is now internal to this function's logic path
    let final_rg_count = determine_row_group_count(
        args,
        file_metadata,
        arrow_schema,
        optimizer,
        compressions_to_test,
    )?;

    let total_rows = file_metadata.file_metadata().num_rows() as usize;
    let sizing_strategy = if let Some(rows) = args.row_group_precise_row_count {
        RowPartitioningStrategy::Exact(rows)
    } else {
        let rows_per_rg = if final_rg_count > 0 {
            (total_rows as f64 / final_rg_count as f64).ceil() as usize
        } else {
            total_rows // Avoid division by zero, effectively one group
        };
        RowPartitioningStrategy::Average(rows_per_rg)
    };

    let plans = plan_row_groups(final_rg_count, sizing_strategy, file_metadata);
    Ok(Some(plans))
}


/// Determines the target number of row groups based on command-line arguments.
///
/// If `--row-group-size` is specified, it samples the file to estimate the
/// optimal number of rows per group to meet the target size, and from that,
/// infers the total number of row groups.
///
/// Otherwise, it returns the value of `--row-group-count`.
fn determine_row_group_count(
    args: &Args,
    file_metadata: &ParquetMetaData,
    arrow_schema: &Schema,
    optimizer: &(dyn ColumnOptimizer + Send + Sync),
    compressions_to_test: &[Compression],
) -> Result<usize> {
    if let Some(size_str) = &args.row_group_size {
        // --- Sampling Phase for Size-based Row Grouping ---
        info!("--row-group-size specified. Starting sampling phase to determine row count...");
        let target_rg_size = parse_size_string(size_str)?;
        info!(
            "Target row group size: {} bytes ({:.2} MiB)",
            target_rg_size,
            target_rg_size as f64 / (1024.0 * 1024.0)
        );

        const NUM_SAMPLES: usize = 5;
        let num_row_groups = file_metadata.num_row_groups();
        let sample_indices: Vec<usize> = if num_row_groups <= NUM_SAMPLES {
            (0..num_row_groups).collect()
        } else {
            // Pick evenly spaced samples
            (0..NUM_SAMPLES)
                .map(|i| (i * (num_row_groups - 1)) / (NUM_SAMPLES - 1))
                .collect()
        };
        info!(
            "Taking {} samples from row groups at indices: {:?}",
            sample_indices.len(),
            sample_indices
        );

        let sample_results: Vec<(usize, usize)> = sample_indices
            .par_iter()
            .map(|&rg_idx| {
                let input_file = File::open(&args.input)?;
                let builder = ParquetRecordBatchReaderBuilder::try_new(input_file)?;
                let num_rows = builder.metadata().row_group(rg_idx).num_rows() as usize;
                let mut reader = builder
                    .with_row_groups(vec![rg_idx])
                    .with_batch_size(num_rows)
                    .build()?;
                let batch = reader.next().unwrap()?;

                let mut estimated_rg_size = 0;
                let encoding_builder = EncodingListBuilder::new(args.encodings.clone());
                for (col_idx, col_array) in batch.columns().iter().enumerate() {
                    let field = arrow_schema.field(col_idx);
                    let physical_type =
                        file_metadata.row_group(rg_idx).column(col_idx).column_type();
                    let encodings = encoding_builder.build(physical_type);
                    let optim_result = optimizer.find_best_config(
                        rg_idx,
                        field,
                        col_array,
                        &encodings,
                        compressions_to_test,
                    )?;
                    estimated_rg_size += optim_result.min_size;
                }
                Ok((batch.num_rows(), estimated_rg_size))
            })
            .collect::<Result<Vec<_>>>()?;

        let total_sample_rows: usize = sample_results.iter().map(|(r, _)| r).sum();
        let total_sample_size: usize = sample_results.iter().map(|(_, s)| s).sum();

        if total_sample_rows == 0 {
            panic!("Cannot determine row group size: no rows found in samples.");
        }

        let avg_bytes_per_row = total_sample_size as f64 / total_sample_rows as f64;
        info!("Estimated average bytes per row: {:.2}", avg_bytes_per_row);

        let target_rows_per_rg = (target_rg_size as f64 / avg_bytes_per_row).ceil() as usize;
        info!("Calculated target rows per group: {}", target_rows_per_rg);

        let total_rows = file_metadata.file_metadata().num_rows() as usize;
        // TODO: 如果是 upper bound, 需要修改这里
        let inferred_rg_count = (total_rows) / target_rows_per_rg;
        info!("Inferred target row group count: {}", inferred_rg_count);
        Ok(inferred_rg_count)
    } else if let Some(rg_count) = args.row_group_count {
        Ok(rg_count)
    } else if let Some(rows_per_rg) = args.row_group_precise_row_count {
        let total_rows = file_metadata.file_metadata().num_rows() as usize;
        if total_rows == 0 {
            return Ok(0);
        }
        let inferred_rg_count = (total_rows + rows_per_rg - 1) / rows_per_rg; // Ceiling division
        Ok(inferred_rg_count)
    } else {
        // This case should not be hit if called from plan_new_row_groups
        panic!("No row group strategy specified. Please provide --row-group-count, --row-group-size, or --row-group-precise-row-count.");
    }
}

/// Creates a plan to remap the original row groups into a new layout.
fn plan_row_groups(
    target_rg_count: usize,
    sizing_strategy: RowPartitioningStrategy,
    file_metadata: &ParquetMetaData,
) -> Vec<RowGroupPlan> {
    let original_rgs = file_metadata.row_groups();
    let total_rows = file_metadata.file_metadata().num_rows() as usize;

    // if target_rg_count == 0 || target_rg_count > total_rows {
    //     // If no remapping is requested, create a 1-to-1 plan.
    //     return original_rgs
    //         .iter()
    //         .enumerate()
    //         .map(|(i, rg_meta)| RowGroupPlan {
    //             new_rg_idx: i,
    //             original_rg_indices: vec![i],
    //             selection: RowSelection::from(vec![]), // Empty selection means read all.
    //             column_metadatas: rg_meta.columns().to_vec(),
    //         })
    //         .collect();
    // }

    let mut plans = Vec::with_capacity(target_rg_count);
    let rows_per_rg = match sizing_strategy {
        RowPartitioningStrategy::Average(rows) => rows,
        RowPartitioningStrategy::Exact(rows) => rows,
    };

    let mut file_row_offset = 0;
    let mut original_rg_idx = 0;
    let mut rows_in_original_rg_offset = 0;

    for new_rg_idx in 0..target_rg_count {
        let start_file_row = file_row_offset;
        let mut end_file_row = (start_file_row + rows_per_rg).min(total_rows);
        if new_rg_idx == target_rg_count - 1 {
            end_file_row = total_rows; // Ensure the last group includes all remaining rows.
        }

        if start_file_row >= end_file_row {
            break;
        }

        let mut plan = RowGroupPlan {
            new_rg_idx,
            original_rg_indices: vec![],
            selection: RowSelection::from(vec![]), // Will be built below.
            column_metadatas: vec![],
        };

        let mut rows_to_select_in_plan = end_file_row - start_file_row;

        // Find the original row groups that contain the rows for this new row group.
        let mut temp_original_rg_idx = original_rg_idx;
        let mut temp_rows_in_original_rg_offset = rows_in_original_rg_offset;

        loop {
            if temp_original_rg_idx >= original_rgs.len() {
                break;
            }
            plan.original_rg_indices
                .push(temp_original_rg_idx);
            if plan.column_metadatas.is_empty() {
                plan.column_metadatas = original_rgs[temp_original_rg_idx].columns().to_vec();
            }

            let original_rg_rows = original_rgs[temp_original_rg_idx].num_rows() as usize;
            let rows_left_in_original_rg = original_rg_rows - temp_rows_in_original_rg_offset;

            if rows_left_in_original_rg >= rows_to_select_in_plan {
                break; // This original RG has enough rows to complete the plan.
            } else {
                rows_to_select_in_plan -= rows_left_in_original_rg;
                temp_rows_in_original_rg_offset = 0;
                temp_original_rg_idx += 1;
            }
        }

        // Build the RowSelection for the identified original row groups.
        let rows_skipped_in_plan = rows_in_original_rg_offset;
        let selection = RowSelection::from_consecutive_ranges(
            [rows_skipped_in_plan..(rows_skipped_in_plan + (end_file_row - start_file_row))]
                .into_iter(),
            total_rows, // This is an upper bound, which is fine.
        );
        plan.selection = selection;

        plans.push(plan);

        // Update offsets for the next new row group.
        file_row_offset = end_file_row;
        let mut rows_to_advance = end_file_row - start_file_row;
        while rows_to_advance > 0 {
            let rows_left_in_rg = original_rgs[original_rg_idx].num_rows() as usize - rows_in_original_rg_offset;
            if rows_to_advance >= rows_left_in_rg {
                rows_to_advance -= rows_left_in_rg;
                original_rg_idx += 1;
                rows_in_original_rg_offset = 0;
            } else {
                rows_in_original_rg_offset += rows_to_advance;
                rows_to_advance = 0;
            }
        }
    }

    plans
}


// ! 2. Parallel Processing of Row Groups

/// A struct to hold the results of processing a single row group in a worker thread.
pub struct RowGroupRecipe {
    pub rg_idx: usize,
    pub batch: RecordBatch,
    pub rg_props: Arc<WriterProperties>,
}

/// Takes row group plans and processes them in parallel to produce optimized
/// row group recipes and reports.
pub fn generate_recipes_from_plan(
    args: &Args,
    rg_plans_option: Option<Vec<RowGroupPlan>>,
    file_metadata: &ParquetMetaData,
    arrow_schema: &Schema,
    compressions_to_test: &[Compression],
    optimizer: &(dyn ColumnOptimizer + Send + Sync),
) -> Result<(Vec<RowGroupRecipe>, Vec<RowGroupReport>)> {
    // Create a plan for how to construct the new row groups.
    // If no remapping is specified (rg_plans_option is None), create a 1-to-1 mapping.
    let (rg_plans, is_remapping) = if let Some(plans) = rg_plans_option {
        info!(
            "Remapping {} original row group(s) into {} new row group(s).",
            file_metadata.num_row_groups(),
            plans.len()
        );
        (plans, true)
    } else {
        info!("No row group remapping specified; using 1-to-1 mapping.");
        let default_plans = file_metadata
            .row_groups()
            .iter()
            .enumerate()
            .map(|(i, rg_meta)| RowGroupPlan {
                new_rg_idx: i,
                original_rg_indices: vec![i],
                selection: RowSelection::from(vec![]),
                column_metadatas: rg_meta.columns().to_vec(),
            })
            .collect();
        (default_plans, false)
    };

    info!("Beginning creating row group recipes in parallel...");
    let encoding_builder = EncodingListBuilder::new(args.encodings.clone());
    let results: Vec<_> = rg_plans
        .into_par_iter()
        .map(|plan| {
            let input_file_path = std::path::PathBuf::from(&args.input);

            trace!(
                "Reading batches for new row group {}...",
                plan.new_rg_idx
            );

            let batch = if is_remapping && plan.selection.selects_any() {
                read_batch_in_parallel_chunks(
                    &input_file_path,
                    &plan.original_rg_indices,
                    &plan.selection,
                )?
            } else {
                let input_file = File::open(&input_file_path)?;
                let mut builder = ParquetRecordBatchReaderBuilder::try_new(input_file)?
                    .with_row_groups(plan.original_rg_indices);

                if is_remapping {
                    builder = builder.with_row_selection(plan.selection.clone());
                }

                let reader = builder.build()?;
                reader
                    .map(|res| res.expect("Failed to read record batch during remapping"))
                    .reduce(|a, b| {
                        arrow::compute::concat_batches(&a.schema(), [&a, &b])
                            .expect("Failed to concatenate batches")
                    })
                    .ok_or_else(|| {
                        parquet::errors::ParquetError::General(format!(
                            "No record batches found for new row group {}",
                            plan.new_rg_idx
                        ))
                    })?
            };

            trace!(
                "Finished reading data for new row group {}.",
                plan.new_rg_idx
            );

            create_rg_recipe(
                plan.new_rg_idx,
                batch,
                args.page_count,
                &plan.column_metadatas,
                arrow_schema,
                compressions_to_test,
                optimizer,
                &encoding_builder,
            )
        })
        .collect::<Result<Vec<_>>>()?;

    let (mut rd_recipes, mut reports): (Vec<_>, Vec<_>) = results.into_iter().unzip();
    // Sort results to ensure original row group order
    rd_recipes.sort_by_key(|r| r.rg_idx);
    reports.sort_by_key(|r| r.rg_idx);

    Ok((rd_recipes, reports))
}

/// Reads a RecordBatch from a Parquet file by breaking a large selection into
/// smaller chunks and reading them in parallel.
///
/// This is useful when a `RowGroupPlan` requires reading a large, non-contiguous
/// number of rows that can be processed independently.
fn read_batch_in_parallel_chunks(
    input_path: &Path,
    original_rg_indices: &[usize],
    selection: &RowSelection,
) -> Result<RecordBatch> {
    const CHUNK_SIZE: usize = 100_000;
    let total_rows_to_select = selection.row_count();

    if total_rows_to_select == 0 {
        return Err(parquet::errors::ParquetError::General(
            "Cannot read batch with an empty selection.".to_string(),
        ));
    }

    // 1. Create a vector of chunk-specific RowSelections.
    let mut chunk_selections = Vec::new();
    let mut offset = 0;
    while offset < total_rows_to_select {
        let current_chunk_size = (total_rows_to_select - offset).min(CHUNK_SIZE);

        // Create a "mask" to select only this chunk from the master selection.
        let mask_selectors = vec![
            RowSelector::skip(offset),
            RowSelector::select(current_chunk_size),
            RowSelector::skip(total_rows_to_select - offset - current_chunk_size),
        ];
        let mask_selection: RowSelection = mask_selectors.into();

        // Apply the mask to the master selection to get the final selection for this chunk.
        let chunk_selection = selection.and_then(&mask_selection);
        chunk_selections.push(chunk_selection);

        offset += current_chunk_size;
    }

    // 2. Read each chunk's RecordBatch in parallel using Rayon.
    let batches: Vec<_> = chunk_selections
        .into_par_iter()
        .map(|chunk_sel| {
            let file = File::open(input_path)?;
            let builder = ParquetRecordBatchReaderBuilder::try_new(file)?
                .with_row_groups(original_rg_indices.to_vec())
                .with_row_selection(chunk_sel);

            let reader = builder.build()?;
            // Each chunk might produce multiple batches, so we concatenate them.
            let batch = reader
                .map(|res| res.expect("Failed to read record batch in chunk"))
                .reduce(|a, b| {
                    arrow::compute::concat_batches(&a.schema(), [&a, &b])
                        .expect("Failed to concatenate batches in chunk")
                })
                .ok_or_else(|| {
                    parquet::errors::ParquetError::General(
                        "No record batches found in chunk".to_string(),
                    )
                })?;
            Ok(Some(batch))
        })
        .collect::<Result<Vec<_>>>()?;

    // Filter out any None results and get a Vec of RecordBatches.
    let non_empty_batches: Vec<_> = batches.into_iter().flatten().collect();

    if non_empty_batches.is_empty() {
        return Err(parquet::errors::ParquetError::General(
            "Parallel chunk read resulted in no data.".to_string(),
        ));
    }

    // 3. Merge the collected RecordBatches into a single one.
    let schema = non_empty_batches[0].schema();
    let batch_refs: Vec<_> = non_empty_batches.iter().collect();
    arrow::compute::concat_batches(&schema, batch_refs)
        .map_err(|e| parquet::errors::ParquetError::ArrowError(e.to_string()))
}

/// Finds the optimal encoding and compression for each column in a record batch.
///
/// This function parallelizes the optimization process across all columns in the batch.
///
/// # Returns
/// A tuple containing:
/// - A vector of `ColumnConfig` with the best settings for each column.
/// - A `RowGroupReport` detailing the analysis for this row group.
fn find_best_column_configs(
    rg_idx: usize,
    batch: &RecordBatch,
    column_metadatas: &[ColumnChunkMetaData],
    arrow_schema: &Schema,
    compressions_to_test: &[Compression],
    optimizer: &(dyn ColumnOptimizer + Send + Sync),
    encoding_builder: &EncodingListBuilder,
) -> Result<(Vec<ColumnConfig>, RowGroupReport)> {
    let mut report = RowGroupReport::new(rg_idx);

    log::trace!("Starting optimization for row group {}", rg_idx);
    // Combine data sources for parallel processing.
    let columns_to_process: Vec<_> = column_metadatas
        .iter()
        .zip(arrow_schema.fields().iter())
        .zip(batch.columns().iter())
        .enumerate()
        .collect();
    log::trace!(
        "Row group {} has {} columns to process.",
        rg_idx,
        columns_to_process.len()
    );

    // Parallelize the optimization process for each column.
    let optimization_results: Vec<_> = columns_to_process
        .par_iter()
        .map(
            |&(
                _col_idx, // Not needed here, but part of the enumerated tuple
                ((col_chunk_meta, field), column_array),
            )| {
                let physical_type = col_chunk_meta.column_descr().physical_type();
                let applicable_encodings = encoding_builder.build(physical_type);

                // The optimizer finds the best configuration for this column.
                let optim_result = optimizer.find_best_config(
                    rg_idx,
                    field,
                    column_array,
                    &applicable_encodings,
                    compressions_to_test,
                )?;

                Ok(optim_result)
            },
        )
        .collect::<Result<Vec<_>>>()?;

    // --- Sequential Processing of Results ---
    let mut best_configs: Vec<ColumnConfig> = Vec::with_capacity(optimization_results.len());
    for (col_idx, optim_result) in optimization_results.into_iter().enumerate() {
        let column_name = arrow_schema.field(col_idx).name();

        // Add data to the report.
        report.add_full_report_data(column_name, &optim_result);

        // Store the best configuration for building writer properties.
        best_configs.push(ColumnConfig {
            encoding: optim_result.best_encoding,
            compression: optim_result.best_compression,
        });
    }

    log::debug!("Completed optimization for row group {}", rg_idx);

    Ok((best_configs, report))
}

/// Returns `None` if the default limit should be used.
/// Otherwise, returns `Some(limit)` with the smallest row count per page
/// that ensures the number of pages does not exceed the target.
fn calculate_page_row_limit(
    total_rows: usize,
    target_page_count: usize,
) -> Option<usize> {
    if target_page_count == 0 || total_rows == 0 {
        // Use default if target is 0 or there are no rows.
        return None;
    }
    // Ceiling division to find the smallest limit that meets the page count.
    Some((total_rows + target_page_count - 1) / target_page_count)
}

/// Processes a single row group: reads it, finds the best encoding for each
/// column, and returns the processed data and optimal properties.
pub fn create_rg_recipe(
    rg_idx: usize,
    batch: RecordBatch,
    page_count: usize,
    column_metadatas: &[ColumnChunkMetaData],
    arrow_schema: &Schema,
    compressions_to_test: &[Compression],
    optimizer: &(dyn ColumnOptimizer + Send + Sync),
    encoding_builder: &EncodingListBuilder,
) -> Result<(RowGroupRecipe, RowGroupReport)> {
    
    // --- Find Best Configurations ---
    let (best_configs, report) = find_best_column_configs(
        rg_idx,
        &batch,
        column_metadatas,
        arrow_schema,
        compressions_to_test,
        optimizer,
        encoding_builder,
    )?;

    // --- Property Building ---
    let mut props_builder = WriterProperties::builder();

    // Calculate and set the data page row count limit if specified.
    if let Some(limit) = calculate_page_row_limit(batch.num_rows(), page_count) {
        props_builder = props_builder.set_data_page_row_count_limit(limit).set_data_page_size_limit(1 * 1024 * 1024 * 1024); // large page size to 1GB magic fix number
    }

    for (col_idx, config) in best_configs.iter().enumerate() {
        let field = arrow_schema.field(col_idx);
        let path = ColumnPath::from(field.name().as_str());

        props_builder = props_builder.set_column_statistics_enabled(path.clone(), EnabledStatistics::Chunk);
        // props_builder = props_builder.set_offset_index_disabled(true);

        if config.encoding == Encoding::RLE_DICTIONARY {
            props_builder = props_builder
                .set_column_dictionary_enabled(path.clone(), true)
                .set_dictionary_page_size_limit(usize::MAX);
        } else {
            props_builder = props_builder
                .set_column_dictionary_enabled(path.clone(), false)
                .set_column_encoding(path.clone(), config.encoding);
        }
        // TODO: data_page_size_limit cli?
        props_builder = props_builder
            .set_column_compression(path, config.compression)
            .set_data_page_size_limit(usize::MAX);
    }
    let rg_props = Arc::new(props_builder.build());

    let recipe = RowGroupRecipe {
        rg_idx,
        batch,
        rg_props,
    };

    Ok((recipe, report))
}
