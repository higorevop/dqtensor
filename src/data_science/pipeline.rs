use anyhow::Result;
use std::sync::Arc;
use tracing::{info, instrument};


/// Semântica simples de pipeline: uma sequência nomeada de passos async.
pub type StepFn = Arc<dyn Fn() -> futures::future::BoxFuture<'static, Result<()>> + Send + Sync>;


#[derive(Clone)]
pub struct Step {
pub name: String,
pub run: StepFn,
}


#[derive(Default, Clone)]
pub struct Pipeline {
pub name: String,
pub steps: Vec<Step>,
}


impl Pipeline {
pub fn new(name: impl Into<String>) -> Self { Self { name: name.into(), steps: vec![] } }
pub fn step(mut self, name: impl Into<String>, f: StepFn) -> Self { self.steps.push(Step { name: name.into(), run: f }); self }


#[instrument]
pub async fn execute(&self) -> Result<()> {
info!(pipeline = %self.name, total_steps = self.steps.len(), "starting pipeline");
for s in &self.steps {
info!(step = %s.name, "running step");
(s.run)().await?;
}
info!(pipeline = %self.name, "pipeline finished");
Ok(())
}
}


/// Helper para envolver async fns como StepFn.
pub fn step_fn<Fut, F>(f: F) -> StepFn
where
Fut: std::future::Future<Output = Result<()>> + Send + 'static,
F: Fn() -> Fut + Send + Sync + 'static,
{
Arc::new(move || Box::pin(f()))
}