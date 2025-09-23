//! Módulo básico de machine learning
//! 
//! Fornece implementações simples de algoritmos de ML e utilitários
//! para análise de dados.

use std::collections::HashMap;

/// Estrutura para armazenar dados de treino
#[derive(Debug, Clone)]
pub struct Dataset {
    pub features: Vec<Vec<f64>>,
    pub labels: Vec<i32>,
}

impl Dataset {
    /// Cria um novo dataset
    pub fn new(features: Vec<Vec<f64>>, labels: Vec<i32>) -> Self {
        Self { features, labels }
    }
    
    /// Divide o dataset em treino e teste
    pub fn train_test_split(&self, test_ratio: f64) -> (Dataset, Dataset) {
        let test_size = (self.features.len() as f64 * test_ratio) as usize;
        let train_size = self.features.len() - test_size;
        
        let train_features = self.features[..train_size].to_vec();
        let train_labels = self.labels[..train_size].to_vec();
        
        let test_features = self.features[train_size..].to_vec();
        let test_labels = self.labels[train_size..].to_vec();
        
        (
            Dataset::new(train_features, train_labels),
            Dataset::new(test_features, test_labels)
        )
    }
}

/// Classificador k-NN simples
pub struct KNNClassifier {
    k: usize,
    train_data: Option<Dataset>,
}

impl KNNClassifier {
    /// Cria um novo classificador k-NN
    pub fn new(k: usize) -> Self {
        Self { k, train_data: None }
    }
    
    /// Treina o classificador
    pub fn fit(&mut self, dataset: Dataset) {
        self.train_data = Some(dataset);
    }
    
    /// Faz predições
    pub fn predict(&self, features: &[Vec<f64>]) -> Vec<i32> {
        if let Some(ref train_data) = self.train_data {
            features.iter()
                .map(|feature| self.predict_single(feature, train_data))
                .collect()
        } else {
            vec![]
        }
    }
    
    fn predict_single(&self, feature: &[f64], train_data: &Dataset) -> i32 {
        let mut distances: Vec<(f64, i32)> = train_data.features
            .iter()
            .zip(train_data.labels.iter())
            .map(|(train_feature, &label)| {
                let distance = euclidean_distance(feature, train_feature);
                (distance, label)
            })
            .collect();
        
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        
        let k_nearest = &distances[..self.k.min(distances.len())];
        let mut label_counts = HashMap::new();
        
        for (_, label) in k_nearest {
            *label_counts.entry(*label).or_insert(0) += 1;
        }
        
        label_counts.iter()
            .max_by_key(|(_, &count)| count)
            .map(|(&label, _)| label)
            .unwrap_or(0)
    }
}

/// Calcula a distância euclidiana entre dois vetores
fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Calcula a acurácia de predições
pub fn accuracy(y_true: &[i32], y_pred: &[i32]) -> f64 {
    if y_true.len() != y_pred.len() || y_true.is_empty() {
        return 0.0;
    }
    
    let correct = y_true.iter()
        .zip(y_pred.iter())
        .filter(|(true_label, pred_label)| true_label == pred_label)
        .count();
    
    correct as f64 / y_true.len() as f64
}

/// Calcula estatísticas básicas de um vetor
pub fn basic_stats(data: &[f64]) -> (f64, f64, f64, f64) {
    if data.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    
    let sum: f64 = data.iter().sum();
    let mean = sum / data.len() as f64;
    
    let min = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    
    let variance = data.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / data.len() as f64;
    let std_dev = variance.sqrt();
    
    (mean, std_dev, min, max)
}
