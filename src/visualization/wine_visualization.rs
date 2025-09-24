

use crate::data_science_example::{WineSample, WineDataset};
use std::collections::HashMap;

pub struct WineVisualizer {
    pub dataset: Vec<WineSample>,
}

impl WineVisualizer {
    pub fn new(dataset: Vec<WineSample>) -> Self {
        Self { dataset }
    }

    pub fn quality_histogram(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("\n ### HISTOGRAMA DE QUALIDADE");
        
        let mut quality_counts = HashMap::new();
        for sample in &self.dataset {
            *quality_counts.entry(sample.quality).or_insert(0) += 1;
        }
        
        let max_count = *quality_counts.values().max().unwrap_or(&0);
        let scale = 50.0 / max_count as f64; // escalar para 50 caracteres max
        
        println!("Qualidade | Frequência | Distribuição");
        println!("{}", "-".repeat(50));
        
        for quality in 3..=9 {
            let count = quality_counts.get(&quality).unwrap_or(&0);
            let bar_length = (*count as f64 * scale) as usize;
            let bar = "█".repeat(bar_length);
            let percentage = (*count as f64 / self.dataset.len() as f64) * 100.0;
            
            println!("    {}     |    {:4}     | {} ({:.1}%)", 
                   quality, count, bar, percentage);
        }
        
        Ok(())
    }

    /// Cria um gráfico de dispersão ASCII entre álcool e qualidade
    pub fn alcohol_quality_scatter(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("\n ### GRÁFICO ÁLCOOL vs QUALIDADE ");
        
        // Criar grid 20x10 para o gráfico
        let mut grid = vec![vec![' '; 60]; 12];
        
        // Encontrar min/max para normalização
        let min_alcohol = self.dataset.iter().map(|s| s.alcohol).fold(f64::INFINITY, f64::min);
        let max_alcohol = self.dataset.iter().map(|s| s.alcohol).fold(f64::NEG_INFINITY, f64::max);
        
        // Plotar pontos
        for sample in &self.dataset {
            let x = ((sample.alcohol - min_alcohol) / (max_alcohol - min_alcohol) * 58.0) as usize;
            let y = (11 - sample.quality as usize).min(11);
            
            if x < 60 && y < 12 {
                grid[y][x] = match sample.wine_type.as_str() {
                    "red" => '●',
                    "white" => '○',
                    _ => '·',
                };
            }
        }
        
        // Imprimir grid
        println!("Qualidade");
        for (i, row) in grid.iter().enumerate() {
            let quality_label = if i == 0 { "10" } else if i == 11 { " 3" } else { "  " };
            print!("{}  |", quality_label);
            for &cell in row {
                print!("{}", cell);
            }
            println!("|");
        }
        
        println!("   +{}+", "-".repeat(60));
        println!("   {:.1}{}Álcool (%){}  {:.1}", 
               min_alcohol, " ".repeat(25), " ".repeat(20), max_alcohol);
        
        println!("\nLegenda: ● Vinho Tinto  ○ Vinho Branco");
        
        Ok(())
    }

    /// Cria um gráfico de barras comparativo por tipo de vinho
    pub fn wine_type_comparison(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("\n ### COMPARAÇÃO POR TIPO DE VINHO");
        
        let red_wines: Vec<_> = self.dataset.iter().filter(|s| s.wine_type == "red").collect();
        let white_wines: Vec<_> = self.dataset.iter().filter(|s| s.wine_type == "white").collect();
        
        if red_wines.is_empty() || white_wines.is_empty() {
            println!(" !!! ###  Dados insuficientes para comparação. ### !!! ");
            return Ok(());
        }
        
        // Calcular médias
        let red_alcohol = red_wines.iter().map(|s| s.alcohol).sum::<f64>() / red_wines.len() as f64;
        let white_alcohol = white_wines.iter().map(|s| s.alcohol).sum::<f64>() / white_wines.len() as f64;
        
        let red_quality = red_wines.iter().map(|s| s.quality as f64).sum::<f64>() / red_wines.len() as f64;
        let white_quality = white_wines.iter().map(|s| s.quality as f64).sum::<f64>() / white_wines.len() as f64;
        
        let red_acidity = red_wines.iter().map(|s| s.fixed_acidity).sum::<f64>() / red_wines.len() as f64;
        let white_acidity = white_wines.iter().map(|s| s.fixed_acidity).sum::<f64>() / white_wines.len() as f64;
        
        // Criar gráfico de barras
        println!("Métrica        | Tinto    | Branco   | Comparação");
        println!("{}", "-".repeat(55));
        
        self.print_comparison_bar("Álcool (%)", red_alcohol, white_alcohol, 15.0);
        self.print_comparison_bar("Qualidade", red_quality, white_quality, 10.0);
        self.print_comparison_bar("Acidez", red_acidity, white_acidity, 15.0);
        
        Ok(())
    }
    
    fn print_comparison_bar(&self, metric: &str, red_val: f64, white_val: f64, max_val: f64) {
        let red_bar_len = (red_val / max_val * 20.0) as usize;
        let white_bar_len = (white_val / max_val * 20.0) as usize;
        
        let red_bar = "█".repeat(red_bar_len);
        let white_bar = "░".repeat(white_bar_len);
        
        println!("{:12}   | {:6.2}   | {:6.2}   | {}{}",
               metric, red_val, white_val, red_bar, white_bar);
    }

    /// Cria uma matriz de correlação visual
    pub fn correlation_matrix(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("\n ### MATRIZ DE CORRELAÇÃO VISUAL ###");
        
        let features = vec![
            ("Álcool", self.dataset.iter().map(|s| s.alcohol).collect::<Vec<_>>()),
            ("Acidez", self.dataset.iter().map(|s| s.fixed_acidity).collect::<Vec<_>>()),
            ("pH", self.dataset.iter().map(|s| s.ph).collect::<Vec<_>>()),
            ("Qualidade", self.dataset.iter().map(|s| s.quality as f64).collect::<Vec<_>>()),
        ];
        
        println!("           | Álcool | Acidez |   pH   |Qualidade");
        println!("{}", "-".repeat(45));
        
        for (i, (name1, values1)) in features.iter().enumerate() {
            print!("{:10} |", name1);
            
            for (j, (_, values2)) in features.iter().enumerate() {
                if i == j {
                    print!("  1.00  |");
                } else {
                    let corr = self.calculate_correlation(values1, values2);
                    let _symbol = match corr.abs() {
                        x if x > 0.7 => "██",
                        x if x > 0.5 => "▓▓",
                        x if x > 0.3 => "▒▒",
                        x if x > 0.1 => "░░",
                        _ => "  ",
                    };
                    print!(" {:5.2} |", corr);
                }
            }
            println!();
        }
        
        println!("\nLegenda: ██ Forte (>0.7)  ▓▓ Moderada (>0.5)  ▒▒ Fraca (>0.3)  ░░ Muito fraca (>0.1)");
        
        Ok(())
    }
    
    fn calculate_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
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
        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Cria um dashboard resumo com múltiplas visualizações
    pub fn create_dashboard(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("\n ########### \n DASHBOARD DE VISUALIZAÇÃO - DATASET WINE QUALITY \n ############");
        
        // Estatísticas gerais
        let total_samples = self.dataset.len();
        let red_count = self.dataset.iter().filter(|s| s.wine_type == "red").count();
        let white_count = total_samples - red_count;
        let avg_quality = self.dataset.iter().map(|s| s.quality as f64).sum::<f64>() / total_samples as f64;
        let avg_alcohol = self.dataset.iter().map(|s| s.alcohol).sum::<f64>() / total_samples as f64;
        
        println!(" ### RESUMO:");
        println!("  # Total de amostras: {}", total_samples);
        println!("  # Vinhos tintos: {} ({:.1}%)", red_count, (red_count as f64 / total_samples as f64) * 100.0);
        println!("  # Vinhos brancos: {} ({:.1}%)", white_count, (white_count as f64 / total_samples as f64) * 100.0);
        println!("  # Qualidade média: {:.2}", avg_quality);
        println!("  # Teor alcoólico médio: {:.2}%", avg_alcohol);
        
        // Visualizações
        self.quality_histogram()?;
        self.wine_type_comparison()?;
        self.alcohol_quality_scatter()?;
        self.correlation_matrix()?;
        
        println!("\n ### Fim da visualização. \n");
        Ok(())
    }
}

/// Função para criar visualizações do dataset de vinhos
pub fn create_wine_visualizations(samples: Vec<WineSample>) -> Result<(), Box<dyn std::error::Error>> {
    let visualizer = WineVisualizer::new(samples);
    visualizer.create_dashboard()?;
    Ok(())
}
