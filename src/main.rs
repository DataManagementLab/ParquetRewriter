// filepath: /parquet-column-grained-rewriter/parquet-column-grained-rewriter/src/main.rs
// This file is the entry point of the application, utilizing components defined in lib.rs.

mod cli;
mod optimizer;
mod processing;
mod reporting;

use crate::{
    cli::{build_compressions_list, build_optimizer, build_output_path, setup_logger, Args},
    processing::{create_row_group_remapping_plan, generate_recipes_from_plan},
    reporting::ReportGenerator,
};
use clap::Parser;
use log::{info, trace};
use parquet::{
    arrow::{
        arrow_reader::ParquetRecordBatchReaderBuilder, arrow_writer::{compute_leaves, get_column_writers}, ArrowSchemaConverter
    },
    errors::Result,
    file::{
        properties::{WriterProperties, WriterVersion},
        writer::SerializedFileWriter,
    },
};
use rayon::prelude::*;
use std::{
    fs::File,
    sync::Arc,
    time::{Duration, Instant},
};

fn main() -> Result<()> {
    // env::set_var("RUST_BACKTRACE", "1");
    let args = Args::parse();

    // --- Setup ---
    setup_logger(args.log_level, args.color);
    let program_start = Instant::now();
    info!("Starting Parquet optimization...");
    info!("Input: {}", args.input);

    // === 0. Read metadata and set up configurations ===
    let input_file_for_metadata = File::open(&args.input)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(input_file_for_metadata)?;
    let num_row_groups = builder.metadata().num_row_groups();
    let arrow_schema = builder.schema().clone();
    let file_metadata = builder.metadata().clone();
    info!("Input file has {} row group(s).", num_row_groups);

    let optimizer = build_optimizer(&args)?;
    let compressions_to_test = build_compressions_list(args.optimizer, &args.compression);
    let output_path = build_output_path(&args.output, optimizer.name());

    info!("Optimizer: {}", optimizer.name());
    info!("Output: {}", output_path.display());

    // === 1. Determine Row Grouping Strategy ===
    let rg_plans_option = create_row_group_remapping_plan(
        &args,
        &file_metadata,
        &arrow_schema,
        optimizer.as_ref(),
        &compressions_to_test,
    )?;

    // === 2. Parallel Processing of Row Groups ===
    let parallel_start = Instant::now();
    let (rd_recipes, reports) = generate_recipes_from_plan(
        &args,
        rg_plans_option,
        &file_metadata,
        &arrow_schema,
        &compressions_to_test,
        optimizer.as_ref(),
    )?;
    let parallel_time = parallel_start.elapsed();
    info!(
        "Finished row group recipes creation in {:.2?}.",
        parallel_time
    );

    // === 3. Report Generation ===
    let report_generator = ReportGenerator::new(&reports, &compressions_to_test);
    let optimizer_name = optimizer.name();

    if let Some(path) = &args.full_report_path {
        if let Err(e) = report_generator.write_full_report(path, optimizer_name) {
            log::error!("Failed to write full enumeration report: {}", e);
        }
    }

    // === 4. Sequential Writing ===
    let mut total_write_time = Duration::ZERO;
    let parquet_schema = ArrowSchemaConverter::new().convert(&arrow_schema)?;
    let output_file = File::create(&output_path)?;
    let base_props = WriterProperties::builder()
        .set_writer_version(WriterVersion::PARQUET_2_0)
        .build();
    let root_schema = parquet_schema.root_schema_ptr();
    let mut writer = SerializedFileWriter::new(output_file, root_schema, Arc::new(base_props))?;

    info!("Writing optimized Parquet file...");
    for rd_recipe in rd_recipes.iter() {
        trace!("Writing row group {}...", rd_recipe.rg_idx);
        let write_start = Instant::now();
        let mut row_group_writer = writer.next_row_group()?;
        let col_writers = get_column_writers(&parquet_schema, &rd_recipe.rg_props, &arrow_schema)?;

        let processing_data: Vec<_> = rd_recipe
            .batch
            .columns()
            .iter()
            .zip(arrow_schema.fields().iter())
            .zip(col_writers.into_iter())
            .collect();

        let chunks: Result<Vec<_>> = processing_data
            .into_par_iter()
            .map(|((column_array, field), mut col_writer)| {
                let leaves = compute_leaves(field, column_array)?;
                for leaf in leaves {
                    col_writer.write(&leaf)?;
                }
                col_writer.close()
            })
            .collect();

        for chunk in chunks? {
            chunk.append_to_row_group(&mut row_group_writer)?;
        }

        row_group_writer.close()?;
        total_write_time += write_start.elapsed();
    }
    info!("Finished writing optimized Parquet file.");

    writer.close()?;
    let total_program_time = program_start.elapsed();

    // --- Print Performance Summary ---
    info!("--- Performance Summary ---");
    info!("Parallel processing took: {:.2?}", parallel_time);
    info!("Total write time: {:.2?}", total_write_time);
    info!("---------------------------------");
    info!("Total program time: {:.2?}", total_program_time);
    info!("Total row groups processed: {}", rd_recipes.len());
    info!("Best effort Page count per row group: {}", args.page_count);
    info!("---------------------------------");
    // Get file sizes directly from the filesystem metadata
    let original_size = File::open(&args.input)?.metadata()?.len();
    let optimized_size = File::open(&output_path)?.metadata()?.len();

    info!(
        "Original file size : {:.2} MB",
        original_size as f64 / (1024.0 * 1024.0)
    );
    info!(
        "Optimized file size: {:.2} MB",
        optimized_size as f64 / (1024.0 * 1024.0)
    );

    println!(
        "\nSuccessfully wrote optimized parquet file to {}",
        output_path.display()
    );

    Ok(())
}
