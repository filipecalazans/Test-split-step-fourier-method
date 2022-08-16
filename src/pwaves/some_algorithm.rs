use rustfft::{Fft, num_complex::Complex};
use super::operations;
use std::sync::Arc;

// To create an initial condition
pub fn step_initial_condition(x: &Vec<f64>, u_left: f64, u_right: f64, x0: f64) -> Vec<Complex<f64>> {
    let mut output = vec![Complex{re:0.0f64, im:0.0f64}; x.len()];
    for i in 0..x.len(){
        output[i].im = 0.0;
        if x[i] <= x0 {
            output[i].re = u_left;
        }
         else{
            output[i].re = u_right;
        }
    }

    output
}

pub fn freq_bin(size: usize, dx: f64) -> Vec<Complex<f64>> {
    let mut v_left : Vec<Complex<f64>> = Vec::new();
    let mut v_right : Vec<Complex<f64>> = Vec::new();
    
    if size % 2 == 0 {
        for i in 0..size/2{
            v_left.push(Complex{re: (i as f64), im: 0.0f64});
            v_right.push(Complex{re: -((size/2) as f64 - (i as f64)), im: 0.0f64});
        }

        v_left.append(&mut v_right);
        operations::multiply_num_vec(&operations::num_real_to_complex(1.0/((size as f64)*dx)), &v_left)
    }
    else{
        v_left.push(Complex{re:0.0f64, im:0.0f64});
        for i in 0..(size - 1)/2{
            v_left.push(Complex{re: (i as f64), im: 0.0f64});
            v_right.push(Complex{re: -(((size - 1)/2) as f64  - (i as f64)), im: 0.0f64});
        }
        v_left.append(&mut v_right);
        operations::multiply_num_vec(&operations::num_real_to_complex(1.0/((size as f64)*dx)), &v_left)
    }
}

pub fn linspace(left: f64, right: f64, size: usize) -> Vec<Complex<f64>> {
    let mut vec:Vec<Complex<f64>> = Vec::new();
    let dx: f64 = (right - left)/(size as f64 - 1.0);
    for i in 0..size{
        vec.push(operations::num_real_to_complex(left + (i as f64)*dx));
    }
    vec
}

pub fn fft(u: &Vec<Complex<f64>>, fft:&Arc<dyn Fft<f64>>) -> Vec<Complex<f64>>{
    let mut buffer = u.clone();
    fft.process(&mut buffer);
    buffer
}

pub fn ifft(u: &Vec<Complex<f64>>, ifft: &Arc<dyn Fft<f64>>) -> Vec<Complex<f64>>{
    let mut buffer = u.clone();
    ifft.process(&mut buffer);
    buffer
}

pub fn next_step(u: &Vec<Complex<f64>>, k: &Vec<Complex<f64>>, alpha: f64,
    fft_a: &Arc<dyn Fft<f64>>, ifft_a: &Arc<dyn Fft<f64>>) -> Vec<Complex<f64>>{
    
    let u1 = operations::multiply_num_vec(&Complex{re:0.0f64, im:-3.0f64},
    &operations::multiply_vec_vec(k, &fft(&operations::vec_fade_complex_part(
    &operations::pow(&ifft(&u, ifft_a), 2.0)), fft_a)));
    
    let u2 = operations::multiply_num_vec(&Complex{re:0.0f64, im:2.0*alpha},
    &operations::multiply_vec_vec(k,&fft(&operations::vec_fade_complex_part(
    &operations::pow(&ifft(&u, ifft_a), 3.0)), fft_a)));
    
    operations::multiply_vec_vec(&u1, &u2)
}

// u1 = u*np.exp((1j*k**3 - rhi*k**2)*dt)
pub fn runge_kutta_split_step(atual_wave_vec: &Vec<Complex<f64>>, k: &Vec<Complex<f64>>, 
    dt: f64, rhi : f64, alpha: f64, fft: &Arc<dyn Fft<f64>>, ifft: &Arc<dyn Fft<f64>>) -> Vec<Complex<f64>> {
    
    let u1 = operations::multiply_vec_vec(&atual_wave_vec, 
        &operations::exp(&operations::add_vec_vec(&operations::multiply_num_vec(&Complex{re:0.0f64, im:1.0f64}, 
        &operations::pow(k, 3.0)), 
        &operations::multiply_num_vec(&Complex{re:-rhi , im:0.0},
        &operations::pow(k, 2.0)))));

    let k1 = next_step(&u1, k, alpha, fft, ifft);
    
    let k2 = next_step(&operations::add_vec_vec(&u1,
    &operations::multiply_num_vec(&operations::num_real_to_complex(0.5*dt), &k1),
    ), k, alpha, fft, ifft);
    
    let k3 = next_step(&operations::add_vec_vec(&u1,
        &operations::multiply_num_vec(&operations::num_real_to_complex(0.5*dt), &k2),
        ), k, alpha, fft, ifft);
    
        let k4 = next_step(&operations::add_vec_vec(&u1,
        &operations::multiply_num_vec(&operations::num_real_to_complex(dt), &k3),
        ), k, alpha, fft, ifft);
    
    operations::add_vec_vec(&u1,
    &operations::multiply_num_vec(&operations::num_real_to_complex(dt/6.0),
    &operations::add_vec_vec(&operations::add_vec_vec(&k1, &operations::multiply_num_vec(&operations::num_real_to_complex(2.0), &k2)),
    &operations::add_vec_vec(&k4, &operations::multiply_num_vec(&operations::num_real_to_complex(2.0), &k3))
    )))
}