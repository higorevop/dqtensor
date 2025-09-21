use nalgebra::{DMatrix, Dvector};
use std::collections::VecDeque;

#[inline]
fn check_nonneg_int(name: &str, val: usize){
    if name == "window" && val == 0{
        panic!("{} deve ser maior ou igual a 1")
    }
}

// Laguerre via recorrencia
pub fn laguerre_eval(n: usize, x: f64, alpha: usize) -> f64 {
    if n == 0 {
        return 1.0;
    }

    if n == 1{
        return (alpha as f64) + 1.0 - x;
    }

    let mut l_nm2 = 1.0;
    let mut l_nm1 = (alpha as f64) + 1.0 - x;

    for k in 2..=n {
        let kf = k as f64;
        let num1 = (2.0 * kf - 1.0 + (alpha as f64)) - x;
        let a = num1 / kf;
        let b = ((k - 1 + alpha) as f64) / kf;
        let l_n = a * l_nm1 - b * l_nm2;

        l_nm2 = l_nm1;
        l_nm1 = l_n;

    }
    l_nm1
}