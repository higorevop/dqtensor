use crate::algebra_linear::laguerre::{LaguerreWindowTransformer};
use crate::algebra_linear::lagrange::{LagrangeWindowTransformer};

mod lstm_test;
mod mlp_test;
mod mlp_test2;
mod mamba_test;
mod algebra_linear;

fn main() {

     if let Err(e) = mamba_test::main() {
         eprintln!("mamba test: {}", e);
     }

    //  if let Err(e) = mlp_test::run_experiment() {
    //      eprintln!("Experiment MLP Iris: {}", e);
    //  }

    //  if let Err(e) = mlp_test2::run_experiment_wine() {
    //      eprintln!("Experiment MLP Wine: {}", e);
    //  }

    //   if let Err(e) = lstm_test::main() {
    //      eprintln!("Experiment LSTM Jena Climate: {}", e);
    //  }

    // teste laguerre

    let mut lt = LaguerreWindowTransformer::new(3, 8, 6.0, 1);
    let data = [0f64, 1.,2.,3.,4.,3.,2.,1.,0.,0.,1.,2.,3.,4.];
    for x in data{
        let coefs = lt.update_scalar(x);
        println!("{coefs:?}");
    }

    let sep: &str = "----------------------";
    println!("{}",sep);

    let mut lt2 = LaguerreWindowTransformer::new(2, 5, 5.0, 2);
    let data2 = [[1.,10.],[2.,9.],[3.,8.],[4.,7.],[5.,6.],[6.,5.]];
    for x in data2 {
        let coefs = lt2.update_vector(&x);
        println!("{coefs:?}");
    }

    let sep: &str = "############################---------";
    println!("{}",sep);

    let mut lag = LagrangeWindowTransformer::new(3, 4, 1);
    for x in [0., 1., 2., 3., 4., 3., 2., 1.] {
        let coefs = lag.update_scalar(x);
        println!("a = {:?}", coefs); // [a0, a1, a2, a3]
    }

    let sep: &str = "----------------------";
    println!("{}",sep);


    let mut lag2 = LagrangeWindowTransformer::new(2, 6, 2);
    let data2 = [[1., 10.],[2., 9.],[3., 8.],[4., 7.],[5., 6.]];
    for x in data2 {
        let coefs = lag2.update_vector(&x);
        println!("a = {:?}", coefs); // concat dos 2 conjuntos de coeficientes
    }


 }
