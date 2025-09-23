use anyhow::Result;
use std::path::Path;
use tracing::instrument;


// Elusion prelude expõe DataFrame, CsvReadOptions, etc.
use elusion::prelude::*;


/// Lê CSV em DataFrame (Elusion) com inferência de schema.
#[instrument(skip(path))]
pub fn read_csv<P: AsRef<Path>>(path: P, has_header: bool, delimiter: u8) -> Result<DataFrame> {
let opts = CsvReadOptions::default()
.has_header(has_header)
.delimiter(delimiter);
let df = DataFrame::read_csv(path.as_ref(), opts)?;
Ok(df)
}


/// Salva DataFrame em Parquet.
#[instrument]
pub fn write_parquet<P: AsRef<Path>>(df: &DataFrame, path: P, overwrite: bool) -> Result<()> {
let opts = DataFrameWriteOptions::default().overwrite(overwrite);
df.write_parquet(path.as_ref(), opts)?;
Ok(())
}


/// Operações analíticas comuns (exemplos): describe, null counts, select/projeções.
#[instrument]
pub fn null_profile(df: &DataFrame) -> Result<DataFrame> {
// Elusion traz helpers p/ estatísticas/quality.
let profile = df.null_analysis()?; // retorna DF com contagem de nulos por coluna
Ok(profile)
}


#[instrument]
pub fn basic_stats(df: &DataFrame, cols: &[&str]) -> Result<DataFrame> {
// exemplo: média, min, max por coluna
let mut agg = df.clone();
for c in cols {
agg = agg
.with_column(format!("{}_mean", c), Expr::col(c).mean())
.with_column(format!("{}_min", c), Expr::col(c).min())
.with_column(format!("{}_max", c), Expr::col(c).max());
}
Ok(agg)
}


#[instrument]
pub fn filter_and_project(df: &DataFrame, predicate: Expr, select_cols: &[&str]) -> Result<DataFrame> {
let out = df
.clone()
.filter(predicate)?
.select(select_cols)?;
Ok(out)
}


use anyhow::Result;
use tokio_cron_scheduler::{Job, JobScheduler};
use tracing::{info, instrument};


use crate::data_science::pipeline::Pipeline;


#[instrument]
pub async fn schedule_pipeline(cron: &str, pipeline: Pipeline) -> Result<JobScheduler> {
let scheduler = JobScheduler::new().await?;


let job = Job::new_async(cron, move |_uuid, _l| {
let p = pipeline.clone();
Box::pin(async move {
if let Err(e) = p.execute().await {
eprintln!("pipeline failed: {e:?}");
}
})
})?;


scheduler.add(job).await?;
scheduler.start().await?;


info!(cron, pipeline = %pipeline.name, "scheduler started");
Ok(scheduler)
}

use anyhow::Result;
use elusion::prelude::{Job, JobScheduler};
use tracing::instrument;
use crate::data_science::pipeline::Pipeline;


#[instrument]
pub async fn schedule_with_elusion(cron: &str, pipeline: Pipeline) -> Result<JobScheduler> {
let mut sched = JobScheduler::new();
let job = Job::new_async(cron, move |_uuid, _l| {
let p = pipeline.clone();
Box::pin(async move { let _ = p.execute().await; })
})?;
sched.add(job)?;
sched.start().await?;
Ok(sched)
}