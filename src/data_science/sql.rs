use anyhow::Result;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::prelude::*;
use tracing::instrument;


/// Cria um SessionContext do DataFusion com config padrão.
pub fn new_session() -> SessionContext {
let mut cfg = SessionConfig::new();
cfg = cfg.with_information_schema(true);
SessionContext::new_with_config(cfg)
}


/// Registra um CSV como tabela.
#[instrument(skip(ctx, path))]
pub async fn register_csv(ctx: &SessionContext, table: &str, path: &str, has_header: bool) -> Result<()> {
let options = CsvReadOptions::new().has_header(has_header);
ctx.register_csv(table, path, options).await?;
Ok(())
}


/// Registra um Parquet como tabela.
#[instrument(skip(ctx))]
pub async fn register_parquet(ctx: &SessionContext, table: &str, path: &str) -> Result<()> {
ctx.register_parquet(table, path, ParquetReadOptions::default()).await?;
Ok(())
}


/// Executa SQL e retorna os RecordBatches.
#[instrument(skip(ctx, sql))]
pub async fn query(ctx: &SessionContext, sql: &str) -> Result<Vec<RecordBatch>> {
let df = ctx.sql(sql).await?;
let results = df.collect().await?;
Ok(results)
}


/// Helper: executa SQL e imprime como tabela.
#[instrument(skip(ctx, sql))]
pub async fn query_print(ctx: &SessionContext, sql: &str) -> Result<()> {
let batches = query(ctx, sql).await?;
datafusion::arrow::util::pretty::print_batches(&batches)?;
Ok(())
}