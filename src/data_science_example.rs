use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use rand::prelude::*;

#[derive(Debug, Clone)]
pub struct WineSample {
    pub fixed_acidity: f64,
    pub volatile_acidity: f64,
    pub citric_acid: f64,
    pub residual_sugar: f64,
    pub chlorides: f64,
    pub free_sulfur_dioxide: f64,
    pub total_sulfur_dioxide: f64,
    pub density: f64,
    pub ph: f64,
    pub sulphates: f64,
    pub alcohol: f64,
    pub quality: i32,
    pub wine_type: String,
}

#[derive(Debug)]
pub struct WineDataset {
    pub samples: Vec<WineSample>,
    pub name: String,
}

impl WineDataset {
    pub fn new(name: &str) -> Self {
        Self {
            samples: Vec::new(),
            name: name.to_string(),
        }
    }

    /// Carrega dados de um arquivo CSV
    pub fn load_from_csv(&mut self, filepath: &str, wine_type: &str) -> Result<(), Box<dyn std::error::Error>> {
        println!(" ... ... Carregando dataset de vinhos {} de {}...", wine_type, filepath);
        
        let file = File::open(filepath)?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();
        
        // Pular cabeçalho
        lines.next();
        
        let mut count = 0;
        for line in lines {
            let line = line?;
            let fields: Vec<&str> = line.split(';').collect();
            
            if fields.len() >= 12 {
                let sample = WineSample {
                    fixed_acidity: fields[0].trim_matches('"').parse().unwrap_or(0.0),
                    volatile_acidity: fields[1].trim_matches('"').parse().unwrap_or(0.0),
                    citric_acid: fields[2].trim_matches('"').parse().unwrap_or(0.0),
                    residual_sugar: fields[3].trim_matches('"').parse().unwrap_or(0.0),
                    chlorides: fields[4].trim_matches('"').parse().unwrap_or(0.0),
                    free_sulfur_dioxide: fields[5].trim_matches('"').parse().unwrap_or(0.0),
                    total_sulfur_dioxide: fields[6].trim_matches('"').parse().unwrap_or(0.0),
                    density: fields[7].trim_matches('"').parse().unwrap_or(0.0),
                    ph: fields[8].trim_matches('"').parse().unwrap_or(0.0),
                    sulphates: fields[9].trim_matches('"').parse().unwrap_or(0.0),
                    alcohol: fields[10].trim_matches('"').parse().unwrap_or(0.0),
                    quality: fields[11].trim_matches('"').parse().unwrap_or(0),
                    wine_type: wine_type.to_string(),
                };
                self.samples.push(sample);
                count += 1;
            }
        }
        
        println!(" ### Carregadas {} amostras de vinho {}", count, wine_type);
        Ok(())
    }

    /// Análise exploratória completa
    pub fn comprehensive_analysis(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("\n\n ### EXEMPLO DE ANÁLISE EXPLORATÓRIA ###\n");
        
        // Informações básicas
        let total = self.samples.len();
        let red_count = self.samples.iter().filter(|s| s.wine_type == "red").count();
        let white_count = total - red_count;
        
        println!(" ### Informações Gerais:");
        println!("  # Total de amostras: {}", total);
        println!("  # Vinhos tintos: {} ({:.1}%)", red_count, (red_count as f64 / total as f64) * 100.0);
        println!("  # Vinhos brancos: {} ({:.1}%) \n", white_count, (white_count as f64 / total as f64) * 100.0);
        
        // Distribuição de qualidade
        println!("\n ### Distribuição de Qualidade:");
        let mut quality_counts = HashMap::new();
        for sample in &self.samples {
            *quality_counts.entry(sample.quality).or_insert(0) += 1;
        }
        
        for quality in 3..=9 {
            if let Some(&count) = quality_counts.get(&quality) {
                let percentage = (count as f64 / total as f64) * 100.0;
                let bar = "█".repeat((percentage * 2.0) as usize);
                println!("  # Qualidade {}: {} amostras ({:.1}%) {}, \n", quality, count, percentage, bar);
            }
        }
        
        // Estatísticas por variável
        println!("\n ### Estatísticas por Variável:");
        self.print_variable_stats(" # Álcool", &self.samples.iter().map(|s| s.alcohol).collect::<Vec<_>>());
        self.print_variable_stats(" # Acidez Fixa", &self.samples.iter().map(|s| s.fixed_acidity).collect::<Vec<_>>());
        self.print_variable_stats(" # Acidez Volátil", &self.samples.iter().map(|s| s.volatile_acidity).collect::<Vec<_>>());
        self.print_variable_stats( " # pH", &self.samples.iter().map(|s| s.ph).collect::<Vec<_>>());
        self.print_variable_stats(" # Sulfatos", &self.samples.iter().map(|s| s.sulphates).collect::<Vec<_>>());
        
        // Correlações com qualidade
        println!("\n \n ### Correlações com Qualidade:");
        let quality_values: Vec<f64> = self.samples.iter().map(|s| s.quality as f64).collect();
        
        let alcohol_values: Vec<f64> = self.samples.iter().map(|s| s.alcohol).collect();
        let alcohol_corr = correlation(&quality_values, &alcohol_values);
        
        let acidity_values: Vec<f64> = self.samples.iter().map(|s| s.volatile_acidity).collect();
        let acidity_corr = correlation(&quality_values, &acidity_values);
        
        let citric_values: Vec<f64> = self.samples.iter().map(|s| s.citric_acid).collect();
        let citric_corr = correlation(&quality_values, &citric_values);
        
        println!("  # Álcool < - > Qualidade: {:.3} {}", alcohol_corr, if alcohol_corr > 0.0 { " subiu" } else { "desceu" });
        println!("  # Acidez Volátil < - > Qualidade: {:.3} {}", acidity_corr, if acidity_corr > 0.0 { "subiu" } else { "desceu" });
        println!("  # Ácido Cítrico < - > Qualidade: {:.3} {}", citric_corr, if citric_corr > 0.0 { "subiu" } else { "desceu" });
        
        Ok(())
    }
    
    fn print_variable_stats(&self, name: &str, values: &[f64]) {
        let (mean, std, min, max) = calculate_stats(values);
        println!("\n  # {}: μ={:.3}, σ={:.3}, min={:.3}, max={:.3} \n\n", name, mean, std, min, max);
    }

    /// Análise comparativa por tipo de vinho
    pub fn wine_type_comparison(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("\n ### COMPARAÇÃO POR TIPO DE VINHO \n");
        
        let red_wines: Vec<_> = self.samples.iter().filter(|s| s.wine_type == "red").collect();
        let white_wines: Vec<_> = self.samples.iter().filter(|s| s.wine_type == "white").collect();
        
        if !red_wines.is_empty() && !white_wines.is_empty() {
            let red_quality = red_wines.iter().map(|s| s.quality as f64).sum::<f64>() / red_wines.len() as f64;
            let white_quality = white_wines.iter().map(|s| s.quality as f64).sum::<f64>() / white_wines.len() as f64;
            
            let red_alcohol = red_wines.iter().map(|s| s.alcohol).sum::<f64>() / red_wines.len() as f64;
            let white_alcohol = white_wines.iter().map(|s| s.alcohol).sum::<f64>() / white_wines.len() as f64;
            
            println!(" ### Comparação de médias:");
            println!("  # Qualidade: Tinto {:.2} vs Branco {:.2}", red_quality, white_quality);
            println!("  # Álcool: Tinto {:.2}% vs Branco {:.2}%", red_alcohol, white_alcohol);
            
            // Histograma visual simples
            println!("\n ### Distribuição de Qualidade por Tipo:");
            self.print_quality_distribution(&red_wines, "Tinto");
            self.print_quality_distribution(&white_wines, "Branco");
        }
        
        Ok(())
    }
    
    fn print_quality_distribution(&self, wines: &[&WineSample], wine_type: &str) {
        let mut quality_counts = HashMap::new();
        for wine in wines {
            *quality_counts.entry(wine.quality).or_insert(0) += 1;
        }
        
        println!("  {} ({})", wine_type, wines.len());
        for quality in 3..=9 {
            if let Some(&count) = quality_counts.get(&quality) {
                let percentage = (count as f64 / wines.len() as f64) * 100.0;
                let bar = "█".repeat((percentage / 5.0) as usize);
                println!("    Q{}: {:3} ({:4.1}%) {}", quality, count, percentage, bar);
            }
        }
    }
}

/// Calcula estatísticas básicas
fn calculate_stats(data: &[f64]) -> (f64, f64, f64, f64) {
    if data.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    
    let sum: f64 = data.iter().sum();
    let mean = sum / data.len() as f64;
    
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
    let std_dev = variance.sqrt();
    
    let min = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    
    (mean, std_dev, min, max)
}

/// Calcula correlação de Pearson
fn correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }
    
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;
    
    let mut numerator = 0.0;
    let mut sum_sq_x = 0.0;
    let mut sum_sq_y = 0.0;
    
    for (xi, yi) in x.iter().zip(y.iter()) {
        let diff_x = xi - mean_x;
        let diff_y = yi - mean_y;
        numerator += diff_x * diff_y;
        sum_sq_x += diff_x * diff_x;
        sum_sq_y += diff_y * diff_y;
    }
    
    let denominator = (sum_sq_x * sum_sq_y).sqrt();
    if denominator == 0.0 { 0.0 } else { numerator / denominator }
}

// ############################################
// EXEMPLOS DE ALGORITMOS DE MACHINE LEARNING
// ############################################

/// 1. Classificador k-NN para qualidade de vinhos
pub struct WineKNNClassifier {
    k: usize,
    train_samples: Vec<WineSample>,
}

impl WineKNNClassifier {
    pub fn new(k: usize) -> Self {
        Self { k, train_samples: Vec::new() }
    }
    
    pub fn fit(&mut self, samples: Vec<WineSample>) {
        self.train_samples = samples;
        println!(" ### k-NN treinado com {} amostras (k={})", self.train_samples.len(), self.k);
    }
    
    pub fn predict(&self, samples: &[WineSample]) -> Vec<i32> {
        samples.iter().map(|sample| self.predict_single(sample)).collect()
    }
    
    fn predict_single(&self, sample: &WineSample) -> i32 {
        let mut distances: Vec<(f64, i32)> = self.train_samples
            .iter()
            .map(|train_sample| {
                let dist = self.wine_distance(sample, train_sample);
                (dist, train_sample.quality)
            })
            .collect();
        
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        
        let k_nearest = &distances[..self.k.min(distances.len())];
        let mut quality_counts = HashMap::new();
        
        for (_, quality) in k_nearest {
            *quality_counts.entry(*quality).or_insert(0) += 1;
        }
        
        quality_counts.iter()
            .max_by_key(|(_, &count)| count)
            .map(|(&quality, _)| quality)
            .unwrap_or(5)
    }
    
    fn wine_distance(&self, a: &WineSample, b: &WineSample) -> f64 {
        // Distância euclidiana normalizada
        let features_a = [a.alcohol / 15.0, a.fixed_acidity / 15.0, a.ph / 4.0];
        let features_b = [b.alcohol / 15.0, b.fixed_acidity / 15.0, b.ph / 4.0];
        
        features_a.iter()
            .zip(features_b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

/// 2. Classificador Naive Bayes para tipo de vinho
pub struct WineNaiveBayes {
    red_alcohol_mean: f64,
    red_alcohol_std: f64,
    white_alcohol_mean: f64,
    white_alcohol_std: f64,
    red_prior: f64,
    white_prior: f64,
}

impl WineNaiveBayes {
    pub fn new() -> Self {
        Self {
            red_alcohol_mean: 0.0,
            red_alcohol_std: 1.0,
            white_alcohol_mean: 0.0,
            white_alcohol_std: 1.0,
            red_prior: 0.5,
            white_prior: 0.5,
        }
    }
    
    pub fn fit(&mut self, samples: &[WineSample]) {
        let red_samples: Vec<_> = samples.iter().filter(|s| s.wine_type == "red").collect();
        let white_samples: Vec<_> = samples.iter().filter(|s| s.wine_type == "white").collect();
        
        self.red_prior = red_samples.len() as f64 / samples.len() as f64;
        self.white_prior = white_samples.len() as f64 / samples.len() as f64;
        
        // Calcular estatísticas para álcool
        let red_alcohol: Vec<f64> = red_samples.iter().map(|s| s.alcohol).collect();
        let white_alcohol: Vec<f64> = white_samples.iter().map(|s| s.alcohol).collect();
        
        let (red_mean, red_std, _, _) = calculate_stats(&red_alcohol);
        let (white_mean, white_std, _, _) = calculate_stats(&white_alcohol);
        
        self.red_alcohol_mean = red_mean;
        self.red_alcohol_std = red_std;
        self.white_alcohol_mean = white_mean;
        self.white_alcohol_std = white_std;
        
        println!(" ### Naive Bayes treinado: {} tintos, {} brancos", red_samples.len(), white_samples.len());
    }
    
    pub fn predict(&self, samples: &[WineSample]) -> Vec<String> {
        samples.iter().map(|sample| self.predict_single(sample)).collect()
    }
    
    fn predict_single(&self, sample: &WineSample) -> String {
        let red_prob = self.red_prior.ln() + 
            gaussian_log_likelihood(sample.alcohol, self.red_alcohol_mean, self.red_alcohol_std);
        let white_prob = self.white_prior.ln() + 
            gaussian_log_likelihood(sample.alcohol, self.white_alcohol_mean, self.white_alcohol_std);
        
        if red_prob > white_prob { "red".to_string() } else { "white".to_string() }
    }
}

fn gaussian_log_likelihood(x: f64, mean: f64, std: f64) -> f64 {
    let variance = std * std;
    -0.5 * ((x - mean).powi(2) / variance + variance.ln() + 2.0 * std::f64::consts::PI.ln())
}

/// 3. Árvore de decisão simples
pub struct SimpleDecisionTree {
    alcohol_threshold: f64,
    acidity_threshold: f64,
}

impl SimpleDecisionTree {
    pub fn new() -> Self {
        Self {
            alcohol_threshold: 10.5,
            acidity_threshold: 0.6,
        }
    }
    
    pub fn fit(&mut self, samples: &[WineSample]) {
        // Calcular thresholds baseados na mediana
        let mut alcohols: Vec<f64> = samples.iter().map(|s| s.alcohol).collect();
        let mut acidities: Vec<f64> = samples.iter().map(|s| s.volatile_acidity).collect();
        
        alcohols.sort_by(|a, b| a.partial_cmp(b).unwrap());
        acidities.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        self.alcohol_threshold = alcohols[alcohols.len() / 2];
        self.acidity_threshold = acidities[acidities.len() / 2];
        
        println!(" ### Árvore de decisão: álcool>{:.1}, acidez<{:.2}", 
               self.alcohol_threshold, self.acidity_threshold);
    }
    
    pub fn predict(&self, samples: &[WineSample]) -> Vec<i32> {
        samples.iter().map(|sample| self.predict_single(sample)).collect()
    }
    
    fn predict_single(&self, sample: &WineSample) -> i32 {
        let mut score = 5;
        
        if sample.alcohol > self.alcohol_threshold {
            score += 1;
        }
        
        if sample.volatile_acidity < self.acidity_threshold {
            score += 1;
        }
        
        score.max(3).min(8)
    }
}

// ############################################
// EXEMPLO DE AGENDAMENTO DE PIPELINE DE DADOS
// ############################################

/// Pipeline de processamento inspirado no Elusion
pub struct WineDataPipeline {
    pub name: String,
    pub steps: Vec<String>,
    pub schedule: String,
}

impl WineDataPipeline {
    pub fn new(name: &str, schedule: &str) -> Self {
        Self {
            name: name.to_string(),
            steps: Vec::new(),
            schedule: schedule.to_string(),
        }
    }
    
    pub fn add_step(&mut self, step: &str) -> &mut Self {
        self.steps.push(step.to_string());
        self
    }
    
    pub fn execute(&self, dataset: &mut WineDataset) -> Result<(), Box<dyn std::error::Error>> {
        println!("\n ... ### EXECUTANDO PIPELINE: {}", self.name);
        println!(" ... ... Agendamento: {}", self.schedule);
        
        for (i, step) in self.steps.iter().enumerate() {
            println!(" ... Passo {}: {}", i + 1, step);
            
            match step.as_str() {
                "load_data" => {
                    dataset.load_from_csv("wine_dataset/winequality-red.csv", "red")?;
                    dataset.load_from_csv("wine_dataset/winequality-white.csv", "white")?;
                }
                "comprehensive_analysis" => {
                    dataset.comprehensive_analysis()?;
                }
                "wine_type_comparison" => {
                    dataset.wine_type_comparison()?;
                }
                "export_summary" => {
                    self.export_summary(dataset)?;
                }
                _ => println!("  !!! ###  Passo desconhecido: {} ### !!!", step),
            }
            
            println!(" ###  Passo concluído ###");
        }
        
        println!(" ### Pipeline '{}' executado com sucesso!", self.name);
        Ok(())
    }
    
    fn export_summary(&self, dataset: &WineDataset) -> Result<(), Box<dyn std::error::Error>> {
        let mut file = File::create("wine_pipeline_summary.txt")?;
        
        writeln!(file, "\n\n ### RESUMO DO PIPELINE DE VINHOS \n")?;
        writeln!(file, " #          Pipeline: {}", self.name)?;
        writeln!(file, " #       Agendamento: {}", self.schedule)?;
        writeln!(file, " # Total de amostras: {}", dataset.samples.len())?;
        
        let red_count = dataset.samples.iter().filter(|s| s.wine_type == "red").count();
        let white_count = dataset.samples.len() - red_count;
        
        writeln!(file, " # Vinhos tintos: {}", red_count)?;
        writeln!(file, " # Vinhos brancos: {}", white_count)?;
        
        let avg_quality: f64 = dataset.samples.iter().map(|s| s.quality as f64).sum::<f64>() / dataset.samples.len() as f64;
        writeln!(file, " # Qualidade média: {:.2}", avg_quality)?;
        
        println!(" \n ### Resumo exportado para wine_pipeline_summary.txt");
        Ok(())
    }
}

/// Agendador de pipelines como no Elusion
pub struct PipelineScheduler {
    pipelines: Vec<WineDataPipeline>,
}

impl PipelineScheduler {
    pub fn new() -> Self {
        Self { pipelines: Vec::new() }
    }
    
    pub fn add_pipeline(&mut self, pipeline: WineDataPipeline) -> &mut Self {
        self.pipelines.push(pipeline);
        self
    }
    
    pub fn run_all(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("\n ... \n ... ### EXECUTANDO PIPELINES AGENDADOS");
        
        for pipeline in &self.pipelines {
            println!(" ... # Executando pipeline: {} ({})", pipeline.name, pipeline.schedule);
            
            let mut dataset = WineDataset::new(&format!("Dataset para {}", pipeline.name));
            pipeline.execute(&mut dataset)?;
            
            println!(" ### Pipeline concluído, aguardando próximo...\n");
        }
        
        println!(" Todos os pipelines foram executados! \n");
        Ok(())
    }
}

// ########################################
// QUERIES ANALÍTICAS NO APACHE DATAFUSION
// ########################################

/// Engine de queries analíticas para dados de vinhos
pub struct WineAnalyticsEngine {
    dataset: WineDataset,
}

impl WineAnalyticsEngine {
    pub fn new(dataset: WineDataset) -> Self {
        Self { dataset }
    }
    
    /// Query SQL-like: SELECT * FROM wines WHERE quality >= 7 GROUP BY wine_type
    pub fn high_quality_wines_by_type(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("\n #### QUERY: VINHOS DE ALTA QUALIDADE POR TIPO \n");
        println!("SQL: SELECT wine_type, COUNT(*), AVG(quality) FROM wines WHERE quality >= 7 GROUP BY wine_type \n");
        
        let high_quality: Vec<_> = self.dataset.samples.iter()
            .filter(|s| s.quality >= 7)
            .collect();
        
        let mut results = HashMap::new();
        for sample in &high_quality {
            let entry = results.entry(&sample.wine_type).or_insert((0, 0.0));
            entry.0 += 1;
            entry.1 += sample.quality as f64;
        }
        
        println!(" ### Resultados:");
        for (wine_type, (count, total_quality)) in results {
            let avg_quality = total_quality / count as f64;
            println!("  • {}: {} vinhos, qualidade média: {:.2} \n\n", wine_type, count, avg_quality);
        }
        
        Ok(())
    }
    
    /// Query SQL-like: SELECT * FROM wines ORDER BY alcohol DESC LIMIT 10
    pub fn top_alcoholic_wines(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("\n ### QUERY: TOP 10 VINHOS POR ÁLCOOL ");
        println!("SQL: SELECT wine_type, alcohol, quality FROM wines ORDER BY alcohol DESC LIMIT 10 \n");
        
        let mut wines_by_alcohol: Vec<_> = self.dataset.samples.iter().collect();
        wines_by_alcohol.sort_by(|a, b| b.alcohol.partial_cmp(&a.alcohol).unwrap());
        
        println!(" ### ... Resultados:");
        for (i, wine) in wines_by_alcohol.iter().take(10).enumerate() {
            println!("  {}. {} - {:.1}% álcool, qualidade: {} \n", 
                   i + 1, wine.wine_type, wine.alcohol, wine.quality);
        }
        
        Ok(())
    }
    
    /// Query SQL-like com CASE WHEN para categorização
    pub fn quality_categories(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("\n ### QUERY: CATEGORIZAÇÃO DE QUALIDADE");
        println!("SQL: SELECT CASE WHEN quality <= 4 THEN 'Baixa' WHEN quality <= 6 THEN 'Média' ELSE 'Alta' END as categoria, COUNT(*) \n");
        
        let mut categories = HashMap::new();
        
        for sample in &self.dataset.samples {
            let category = match sample.quality {
                0..=4 => "Baixa",
                5..=6 => "Média",
                _ => "Alta",
            };
            *categories.entry(category).or_insert(0) += 1;
        }
        
        println!(" ### Resultados:");
        for (category, count) in &categories {
            let percentage = (*count as f64 / self.dataset.samples.len() as f64) * 100.0;
            println!("  • Qualidade {}: {} vinhos ({:.1}%)", category, count, percentage);
        }
        
        Ok(())
    }
    
    /// Query com JOIN simulado - correlação entre variáveis
    pub fn correlation_analysis(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("\n ### QUERY: ANÁLISE DE CORRELAÇÕES ");
        println!("SQL: SELECT corr(alcohol, quality), corr(volatile_acidity, quality) FROM wines \n");
        
        let quality_values: Vec<f64> = self.dataset.samples.iter().map(|s| s.quality as f64).collect();
        let alcohol_values: Vec<f64> = self.dataset.samples.iter().map(|s| s.alcohol).collect();
        let acidity_values: Vec<f64> = self.dataset.samples.iter().map(|s| s.volatile_acidity).collect();
        
        let alcohol_corr = correlation(&quality_values, &alcohol_values);
        let acidity_corr = correlation(&quality_values, &acidity_values);
        
        println!(" ### Resultados:");
        println!("  # corr(álcool, qualidade): {:.3}", alcohol_corr);
        println!("  # corr(acidez_volátil, qualidade): {:.3}", acidity_corr);
        
        Ok(())
    }
}

// #########################
// FUNÇÕES PRINCIPAIS
// #########################

/// Executa análise completa do dataset de vinhos
pub fn run_wine_data_analysis() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "=".repeat(80));
    println!("\n ### ANÁLISE DO DATASET WINE QUALITY ... ");
    println!("\n");
    
    let mut dataset = WineDataset::new("Wine Quality Dataset");
    dataset.load_from_csv("wine_dataset/winequality-red.csv", "red")?;
    dataset.load_from_csv("wine_dataset/winequality-white.csv", "white")?;
    
    dataset.comprehensive_analysis()?;
    dataset.wine_type_comparison()?;
    
    // Queries analíticas
    let analytics = WineAnalyticsEngine::new(dataset);
    analytics.high_quality_wines_by_type()?;
    analytics.top_alcoholic_wines()?;
    analytics.quality_categories()?;
    analytics.correlation_analysis()?;
    
    println!("\n Análise concluída!");
    Ok(())
}

/// Executa exemplos de algoritmos de machine learning
pub fn run_wine_ml_algorithms() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n");
    println!(" ### EXEMPLOS DE ALGORITMOS DE MACHINE LEARNING ...\n \n");
    
    let mut dataset = WineDataset::new("Wine ML Dataset");
    dataset.load_from_csv("wine_dataset/winequality-red.csv", "red")?;
    dataset.load_from_csv("wine_dataset/winequality-white.csv", "white")?;
    
    // Dividir em treino e teste
    let mut rng = thread_rng();
    dataset.samples.shuffle(&mut rng);
    
    let split_idx = (dataset.samples.len() as f64 * 0.8) as usize;
    let train_samples = dataset.samples[..split_idx].to_vec();
    let test_samples = &dataset.samples[split_idx..];
    
    println!(" ... Dataset: {} treino, {} teste", train_samples.len(), test_samples.len());
    
    // 1. k-NN para qualidade
    println!("\n ### ALGORITMO 1: k-NN PARA QUALIDADE \n ");
    let mut knn = WineKNNClassifier::new(5);
    knn.fit(train_samples.clone());
    
    let knn_predictions = knn.predict(test_samples);
    let knn_accuracy = calculate_accuracy(&test_samples.iter().map(|s| s.quality).collect::<Vec<_>>(), &knn_predictions);
    
    println!(" ---> k-NN Resultados: Acurácia {:.1}%", knn_accuracy * 100.0);
    
    // 2. Naive Bayes para tipo
    println!("\n ### ALGORITMO 2: NAIVE BAYES PARA TIPO \n");
    let mut nb = WineNaiveBayes::new();
    nb.fit(&train_samples);
    
    let nb_predictions = nb.predict(test_samples);
    let nb_accuracy = calculate_string_accuracy(&test_samples.iter().map(|s| s.wine_type.clone()).collect::<Vec<_>>(), &nb_predictions);
    
    println!(" --- > Naive Bayes Resultados: Acurácia {:.1}%", nb_accuracy * 100.0);
    
    // 3. Árvore de decisão
    println!("\n ### ALGORITMO 3: ÁRVORE DE DECISÃO \n");
    let mut dt = SimpleDecisionTree::new();
    dt.fit(&train_samples);
    
    let dt_predictions = dt.predict(test_samples);
    let dt_accuracy = calculate_accuracy(&test_samples.iter().map(|s| s.quality).collect::<Vec<_>>(), &dt_predictions);
    
    println!("\n --- > Árvore Resultados: Acurácia {:.1}%", dt_accuracy * 100.0);
    
    // Comparação final
    println!("\n ### COMPARAÇÃO DE ALGORITMOS ");
    println!("  #   k-NN (qualidade): {:.1}%", knn_accuracy * 100.0);
    println!("  # Naive Bayes (tipo): {:.1}%", nb_accuracy * 100.0);
    println!("  # Árvore (qualidade): {:.1}%", dt_accuracy * 100.0);
    
    println!("\n FIM DA Análise.\n\n");
    Ok(())
}

/// Demonstra pipeline com agendamento
pub fn run_pipeline_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!(" ### ### ### EXEMPLO DE PIPELINE COM AGENDAMENTO");
    
    let mut scheduler = PipelineScheduler::new();
    
    // Pipeline diário
    let mut daily_pipeline = WineDataPipeline::new("Análise Diária", "0 8 * * *");
    daily_pipeline
        .add_step("load_data")
        .add_step("comprehensive_analysis")
        .add_step("export_summary");
    
    // Pipeline semanal
    let mut weekly_pipeline = WineDataPipeline::new("Relatório Semanal", "0 9 * * 1");
    weekly_pipeline
        .add_step("load_data")
        .add_step("comprehensive_analysis")
        .add_step("wine_type_comparison")
        .add_step("export_summary");
    
    scheduler
        .add_pipeline(daily_pipeline)
        .add_pipeline(weekly_pipeline);
    
    scheduler.run_all()?;
    
    println!(" Fim da pipeline. \n\n");
    Ok(())
}

/// Função principal que executa tudo
pub fn run_complete_wine_analysis() -> Result<(), Box<dyn std::error::Error>> {
    run_wine_data_analysis()?;
    run_wine_ml_algorithms()?;
    run_pipeline_demo()?;
    
    println!("\n ### FIM DA ANÁLISE.");
    println!(" --- > Arquivos gerados: wine_pipeline_summary.txt");
    
    Ok(())
}

// Funções auxiliares
fn calculate_accuracy(true_values: &[i32], predictions: &[i32]) -> f64 {
    if true_values.len() != predictions.len() || true_values.is_empty() {
        return 0.0;
    }
    
    let correct = true_values.iter()
        .zip(predictions.iter())
        .filter(|(true_val, pred)| true_val == pred)
        .count();
    
    correct as f64 / true_values.len() as f64
}

fn calculate_string_accuracy(true_values: &[String], predictions: &[String]) -> f64 {
    if true_values.len() != predictions.len() || true_values.is_empty() {
        return 0.0;
    }
    
    let correct = true_values.iter()
        .zip(predictions.iter())
        .filter(|(true_val, pred)| true_val == pred)
        .count();
    
    correct as f64 / true_values.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wine_sample_creation() {
        let sample = WineSample {
            fixed_acidity: 7.4, volatile_acidity: 0.7, citric_acid: 0.0,
            residual_sugar: 1.9, chlorides: 0.076, free_sulfur_dioxide: 11.0,
            total_sulfur_dioxide: 34.0, density: 0.9978, ph: 3.51,
            sulphates: 0.56, alcohol: 9.4, quality: 5, wine_type: "red".to_string(),
        };
        
        assert_eq!(sample.quality, 5);
        assert_eq!(sample.wine_type, "red");
    }

    #[test]
    fn test_stats_calculation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (mean, _std, min, max) = calculate_stats(&data);
        
        assert_eq!(mean, 3.0);
        assert_eq!(min, 1.0);
        assert_eq!(max, 5.0);
    }

    #[test]
    fn test_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let corr = correlation(&x, &y);
        
        assert!((corr - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_knn_classifier() {
        let mut knn = WineKNNClassifier::new(3);
        let train_samples = vec![
            WineSample {
                fixed_acidity: 7.0, volatile_acidity: 0.5, citric_acid: 0.1,
                residual_sugar: 2.0, chlorides: 0.08, free_sulfur_dioxide: 15.0,
                total_sulfur_dioxide: 40.0, density: 0.995, ph: 3.2,
                sulphates: 0.6, alcohol: 10.0, quality: 6, wine_type: "red".to_string(),
            },
        ];
        
        knn.fit(train_samples);
        assert_eq!(knn.train_samples.len(), 1);
    }
}
