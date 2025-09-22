use nalgebra::{DMatrix, DVector};
use std::collections::VecDeque;

#[inline]
fn ensure_min(val: usize, min: usize, name: &str) {
    if val < min {
        panic!("{} deve ser maior do que {}, recebido {}", name, min, val);
    }
}

// construir os tempos discretos da janela: t_i = i (0..w-1)
// i=0 é o mais recente, vai comprimir os dados mais velhos nos índices maiores
fn build_t_grid(window: usize) -> Vec<f64> {
    (0..window).map(|i| i as f64).collect()
}

// construir matriz de Vandermonde
fn build_vandermonde(window: usize, degree: usize) -> DMatrix<f64>{
    let t = build_t_grid(window);
    let cols = degree + 1;
    let mut data = Vec::with_capacity(window * cols);
    for &ti in &t{
        let mut p = 1.0;
        for _k in 0..cols {
            data.push(p);
            p *= ti;

        }
    }
    DMatrix::from_row_slice(window, cols, &data)
}

pub struct LagrangeWindowTransformer {
    degree: usize,
    window: usize,
    dims: usize,
    ridge: f64,
    buffers: Vec<VecDeque<f64>>,
    v_mat: DMatrix<f64>,
    projector: DMatrix<f64>
}

impl LagrangeWindowTransformer {
    pub fn new(degree: usize, window: usize, dims: usize) -> Self{
        ensure_min(dims, 1, "dims");
        ensure_min(window, degree + 1, "window precisa ser maior que degree+1");

        let ridge = 1e-10;
        let v_mat = build_vandermonde(window, degree);

        let cols = degree + 1;
        let mut vt_v = &v_mat.transpose() * &v_mat;

        for i in 0..cols {
            vt_v[(i, i)] += ridge;
        }

        let vt_v_inv = vt_v.try_inverse().expect("matriz singular, ajuste janela-grau-lambda");
        let projector = vt_v_inv * v_mat.transpose();


        let mut buffers = Vec::with_capacity(dims);
        for _ in 0..dims{
            buffers.push(VecDeque::with_capacity(window));
        }
        
        Self{
            degree,
            window,
            dims,
            ridge,
            buffers,
            v_mat,
            projector,
        }
    }

    pub fn n_coefs_per_dim(&self) -> usize{
        self.degree + 1
    }

    pub fn update_scalar(&mut self, x_t: f64) -> Vec<f64> {
        assert!(self.dims == 1, "update_vector para dims > 1");
        self.push_val(0, x_t);
        self.solve_all_dims()
    }

    pub fn update_vector(&mut self, x_t: &[f64]) -> Vec<f64> {
        assert!(x_t.len() == self.dims, "dimensao da entrada = {} diferente da dimensao {}", x_t.len(), self.dims);

        for (d, &val) in x_t.iter().enumerate(){
            self.push_val(d, val);
        }
        self.solve_all_dims()

    }

    pub fn push_val(&mut self, dim: usize, val: f64) {
        let buf = &mut self.buffers[dim];
        buf.push_front(val);
        if buf.len() > self.window{
            buf.pop_back();
        }
    }

    fn y_for_dim(&self, dim: usize) -> DVector<f64> {
        let mut y = vec![0.0f64; self.window];
        let buf = &self.buffers[dim];
        for (i, &val) in buf.iter().enumerate() {
            if i < self.window {
                y[i] = val;
            } else {
                break;
            }
        }
        DVector::from_vec(y)
    }

    fn solve_all_dims(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.dims * (self.degree +1));
        for d in 0..self.dims {
            let y = self.y_for_dim(d);
            let a = &self.projector * y;
            out.extend(a.iter().copied());
        }
        out
    }

    pub fn eval_poly_dim_at(&self, dim: usize, t: f64) -> f64{
        let coefs = {
            let y = self.y_for_dim(dim);
            (&self.projector * y).data.as_vec().clone()
        };

        let mut acc = 0.0;
        for &c in coefs.iter().rev(){
            acc = acc * t + c;

        }
        acc
    }
}