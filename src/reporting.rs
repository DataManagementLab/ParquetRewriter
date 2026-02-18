use super::optimizer;
use chrono::Local;
use log::info;
use parquet::{
    basic::{Compression, Encoding},
    errors::Result,
};
use std::fmt::Write as FmtWrite;
use std::fs::File;
use std::io::Write;

/// A struct to hold the reporting data for a single row group.
pub struct RowGroupReport {
    pub rg_idx: usize,
    pub full_report_rows: Vec<FullReportRow>,
}

impl RowGroupReport {
    /// Creates a new, empty report for a row group.
    pub fn new(rg_idx: usize) -> Self {
        Self {
            rg_idx,
            full_report_rows: Vec::new(),
        }
    }

    /// Adds the data for the full enumeration report for a single column.
    pub fn add_full_report_data(
        &mut self,
        column_name: &str,
        optim_result: &optimizer::ColumnOptimizationResult,
    ) {
        for (encoding, compression_sizes) in &optim_result.all_results {
            self.full_report_rows.push(FullReportRow {
                row_group_index: self.rg_idx,
                column_name: column_name.to_string(),
                encoding: *encoding,
                compression_sizes: compression_sizes.clone(),
            });
        }
    }
}

/// Holds the data for one row of the full enumeration report.
pub struct FullReportRow {
    pub row_group_index: usize,
    pub column_name: String,
    pub encoding: Encoding,
    /// Maps a compression type to its estimated size in bytes.
    pub compression_sizes: std::collections::HashMap<std::mem::Discriminant<Compression>, usize>,
}

/// A struct to handle the generation of all reports.
pub struct ReportGenerator<'a> {
    reports: &'a [RowGroupReport],
    compressions_tested: &'a [Compression],
}

impl<'a> ReportGenerator<'a> {
    /// Creates a new ReportGenerator.
    pub fn new(reports: &'a [RowGroupReport], compressions_tested: &'a [Compression]) -> Self {
        Self {
            reports,
            compressions_tested,
        }
    }

    /// Determines the best unit (B, KB, MB, GB) and divisor for a given max size.
    fn determine_unit_and_divisor(max_size: i64) -> (&'static str, f64) {
        const KB: i64 = 1000;
        const MB: i64 = KB * 1000;
        const GB: i64 = MB * 1000;

        if max_size >= 100 * GB {
            ("GB", GB as f64)
        } else if max_size >= 100 * MB {
            ("MB", MB as f64)
        } else if max_size >= 100 * KB {
            ("KB", KB as f64)
        } else {
            ("B", 1.0)
        }
    }

    /// Constructs the final report path with the unit appended.
    fn construct_report_path(
        base_path: &str,
        unit_str: &str,
        _optimizer_name: &str,
    ) -> std::path::PathBuf {
        let timestamp = Local::now().format("%Y%m%d_%H%M").to_string();
        let p = std::path::Path::new(base_path);
        let parent = p.parent().unwrap_or_else(|| std::path::Path::new(""));
        let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("report");

        let new_filename = format!(
            "parKRWer_{}_{}_{}.csv",
            stem, timestamp, unit_str
        );
        parent.join(new_filename)
    }
    
    /// Writes the full enumeration report.
    pub fn write_full_report(&self, path: &str, optimizer_name: &str) -> Result<()> {
        if self.reports.is_empty() {
            return Ok(());
        }

        let max_size = self
            .reports
            .iter()
            .flat_map(|r| &r.full_report_rows)
            .flat_map(|fr| fr.compression_sizes.values())
            .max()
            .cloned()
            .unwrap_or(0) as i64;

        let (unit_str, divisor) = Self::determine_unit_and_divisor(max_size);
        let final_path = Self::construct_report_path(path, unit_str, optimizer_name);

        info!(
            "Writing full enumeration report to: {}",
            final_path.display()
        );
        let mut writer = File::create(final_path)?;

        let mut header = "RowGroupIndex,Column,Encoding".to_string();
        for comp in self.compressions_tested {
            write!(header, ",{:?}", comp).expect("Failed to write compression type to header");
        }
        writeln!(writer, "{}", header).expect("Failed to write header");

        for report in self.reports {
            for row_data in &report.full_report_rows {
                let mut line = format!(
                    "{},{},{:?}",
                    row_data.row_group_index, row_data.column_name, row_data.encoding
                );
                for comp in self.compressions_tested {
                    let size_val = row_data
                        .compression_sizes
                        .get(&std::mem::discriminant(comp));
                    match size_val {
                        Some(size) => write!(line, ",{:.2}", *size as f64 / divisor)
                            .expect("Failed to write size"),
                        None => write!(line, ",N/A").expect("Failed to write N/A"),
                    }
                }
                writeln!(writer, "{}", line).expect("Failed to write full report row");
            }
        }
        Ok(())
    }
}
