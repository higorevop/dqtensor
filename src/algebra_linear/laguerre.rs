use nalgebra::{DMatrix, DVector};
use std::collections::VecDeque;

#[inline]
fn check_nonneg_int(name: &str, val: usize){
    if name == "window" && val == 0{
        panic!("{} deve ser maior ou igual a 1", name)
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

// criar s_i pontos no intervalo [0, S], com W amostras

fn make_s_grid(window: usize, s_total: f64) -> Vec<f64> {
    if window == 1 {
        return vec![0.0];
    }
    let ds = s_total / ((window - 1) as f64);
    (0..window).map(|i| (i as f64) * ds).collect()
}

// construir uma matriz de base B (W x (k+1))
fn build_basis(window: usize, order: usize, s_total: f64, alpha: usize) -> DMatrix<f64> {
    let s = make_s_grid(window, s_total);
    let mut data = Vec::with_capacity(window * (order + 1));
    for &si in &s {
        for k in 0..=order {
            data.push(laguerre_eval(k, si, alpha));
        }
    }
    DMatrix::from_row_slice(window, order+1, &data)
}


// construir o vetor de pesos w_i = exp(-s_i)
fn build_weights(window: usize, s_total: f64) -> DVector<f64> {
    let s = make_s_grid(window, s_total);
    let w: Vec<f64> = s.into_iter().map(|si| (-si).exp()).collect();
    DVector::from_vec(w)
}

fn add_ridge(mat: &mut DMatrix<f64>, ridge: f64) {
    let n = mat.nrows().min(mat.ncols());
    for i in 0..n{
        mat[(i, i)] += ridge;
    }
}

// transformador de janela com base de lagerre.
// retorna os coeficientes contatenados por dimensao a cada update
pub struct LaguerreWindowTransformer {
    order: usize,      //K
    window: usize,     //W
    s_total: f64,      //S
    dims: usize,       //D
    alpha: usize,      // parametr do laguerre (normalmente 0)
    ridge: f64,        // regularização para BtWB

    buffers: Vec<VecDeque<f64>>,

    b_mat: DMatrix<f64>,
    w_vec: DVector<f64>,

}

impl LaguerreWindowTransformer {
    pub fn new(order: usize, window: usize, s_total: f64, dims: usize) -> Self {
        check_nonneg_int("order", order);
        check_nonneg_int("window", window.max(1)); // >=1
        check_nonneg_int("dims", dims.max(1));
        let alpha = 0usize;
        let ridge = 1e-8; // muito baixo, apenas para fins de estabilidade
        let b_mat = build_basis(window, order, s_total, alpha);
        let w_vec = build_weights(window, s_total);

        let mut buffers = Vec::with_capacity(dims);
        for _ in 0..dims{
            buffers.push(VecDeque::with_capacity(window));
        }

        Self{
            order,
            window,
            s_total,
            dims,
            alpha,
            ridge,
            buffers,
            b_mat,
            w_vec,
        }
    }

    pub fn n_coefs_per_dim(&self) -> usize {
        self.order + 1
    }

    pub fn update_scalar(&mut self, x: f64) -> Vec<f64> {
        assert!(self.dims == 1, "para D>1 tem que usar update_vector");
        self.push_to_buffer(0, x);
        self.solve_all_dims()
    }

    pub fn update_vector(&mut self, x: &[f64]) -> Vec<f64> {
        assert!(x.len() == self.dims, "dimensão de entrada ({}) diferente das dimensoes ({})", x.len(), self.dims );

        for (d, &val) in x.iter().enumerate() {
            self.push_to_buffer(d, val);
        }
        self.solve_all_dims()
    }

    fn push_to_buffer(&mut self, dim: usize, val: f64) {
        let buf = &mut self.buffers[dim];
        buf.push_front(val);
        if buf.len() > self.window {
            buf.pop_back();
        }
    }

    fn solve_all_dims(&self) -> Vec<f64> {
        let w_diag = self.w_vec.clone();
        let b = &self.b_mat;

        let mut bt_w_b = {
            let mut bw = DMatrix::zeros(self.window, self.order + 1);
            for i in 0..self.window {
                let wi_sqrt = w_diag[i].sqrt();
                for j in 0..(self.order + 1){
                    bw[(i, j)] = wi_sqrt * b[(i, j)];
                }
            }
            &bw.transpose() * &bw
        };
        add_ridge(&mut bt_w_b, self.ridge);


        // fatorar uma vez, para cada dimensão

        let bt_w_b_inv = bt_w_b.clone().try_inverse().expect("matriz (bˆT W B) singular; ajuste janela/ordem/ridge");

        let mut out = Vec::with_capacity(self.dims * (self.order + 1));
        for d in 0..self.dims{
            let y = self.build_y_for_dim(d); // W
            let mut bw_ty = DVector::zeros(self.order + 1);
            for i in 0..self.window {
                let wi = w_diag[i];
                let yi = y[i];
                if wi == 0.0 || yi == 0.0 {
                    continue;
                }
                for k in 0..(self.order + 1) {
                    bw_ty[k] += b[(i, k)] * wi * yi
                }
            }
            let c = &bt_w_b_inv * bw_ty;
            out.extend(c.iter().copied());
        }
        out
    }

    fn build_y_for_dim(&self, dim:usize) -> DVector<f64> {
        let mut y = vec![0.0f64; self.window];
        let buf = &self.buffers[dim];
        for (i, &val) in buf.iter().enumerate(){
            if i < self.window{
                y[i] = val;
            } else {
                break;
            }
        }
        DVector::from_vec(y)
    }
}

