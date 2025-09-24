#![allow(warnings)]

use crate::algebra_linear::laguerre::{LaguerreWindowTransformer};
use crate::algebra_linear::lagrange::{LagrangeWindowTransformer};

mod lstm_test;
mod mlp_test;
mod mlp_test2;
mod mamba_test;
mod algebra_linear;
mod data_science_example;

mod visualization;
use visualization::wine_visualization;


fn main() {

    //if let Err(e) = mamba_test::main() {
    //    eprintln!("mamba test: {}", e);
    //}

    //  if let Err(e) = mlp_test::run_experiment() {
    //      eprintln!("Experiment MLP Iris: {}", e);
    //  }

    //  if let Err(e) = mlp_test2::run_experiment_wine() {
    //      eprintln!("Experiment MLP Wine: {}", e);
    //  }

    // if let Err(e) = lstm_test::main() {
    //     eprintln!("Experiment LSTM Jena Climate: {}", e);
    // }

    
    // println!("\n####### TESTE, ARQUIVO: data_science_example.rs");
    // println!("####### TESTE,  MODULO: src/data_science/mod.rs \n");


    // if let Err(e) = data_science_example::run_complete_wine_analysis() {
    //     eprintln!(" !!! ### Erro na análise completa: {} ### !!! ", e);
    // }


    if let Err(e) = wine_visualization::run_wine_visualization_demo(){
        eprintln!("Erro na visualização (wine): {}", e);
    }

   


 }
