use rustfft::{Fft, num_complex::Complex};
use std::sync::Arc;
use std::thread;
use std::f32::consts::PI;
use std::sync::mpsc;
use ndarray::Array1;


pub fn step_initial_condition(x: &Array1<f32>, u_left: f32, u_right: f32, x0: f32) -> Array1<Complex<f32>> {
    let mut output = Array1::from_vec(vec![Complex{re:0.0f32, im:0.0f32}; x.len()]);
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

pub fn freq_bin(size: usize, dx: f32) -> Array1<Complex<f32>> {
    let mut v_left : Vec<Complex<f32>> = Vec::new();
    let mut v_right : Vec<Complex<f32>> = Vec::new();
    
    if size % 2 == 0 {
        for i in 0..size/2{
            v_left.push(Complex{re: (i as f32), im: 0.0f32});
            v_right.push(Complex{re: -((size/2) as f32 - (i as f32)), im: 0.0f32});
        }

        v_left.append(&mut v_right);
        let output = Array1::from(v_left);
        Complex{re:2.0*PI/((size as f32)*dx), im: 0.0f32}*output
    }
    else{
        v_left.push(Complex{re:0.0f32, im:0.0f32});
        for i in 0..(size - 1)/2{
            v_left.push(Complex{re: (i as f32), im: 0.0f32});
            v_right.push(Complex{re: -(((size - 1)/2) as f32  - (i as f32)), im: 0.0f32});
        }
        v_left.append(&mut v_right);
        let output = Array1::from(v_left);
        Complex{re:2.0*PI/((size as f32)*dx), im: 0.0f32}*output
    }
}

pub fn fft(u: &Array1<Complex<f32>>, fft:&Arc<dyn Fft<f32>>) -> Array1<Complex<f32>>{
    let mut buffer = u.clone().into_raw_vec();
    fft.process(&mut buffer);
    Array1::from(buffer)
}

pub fn ifft(u: &Array1<Complex<f32>>, ifft: &Arc<dyn Fft<f32>>) -> Array1<Complex<f32>>{
    let mut buffer = u.clone().into_raw_vec();
    buffer = buffer.iter().map(|x| x*Complex{re:1.0/(buffer.len() as f32), im:0.0f32}).collect::<Vec<Complex<f32>>>();
    ifft.process(&mut buffer);
    Array1::from(buffer)
}

pub fn vec_fade_complex_part(vec:&Array1<Complex<f32>>) -> Array1<Complex<f32>>{
    vec.iter().map(|x| Complex{re: x.re, im: 0.0f32}).collect::<Array1<Complex<f32>>>()
}

pub fn next_step(u: &Array1<Complex<f32>>, k: &Array1<Complex<f32>>, alpha: f32,
    fft_a: &Arc<dyn Fft<f32>>, ifft_a: &Arc<dyn Fft<f32>>) -> Array1<Complex<f32>>{
    
    let u_copy = u.clone();
    let k_copy = k.clone();
    let fft_copy = fft_a.clone();
    let ifft_copy = ifft_a.clone();

    let t1 = thread::spawn(move ||{
        let u1 = ifft(&u_copy, &ifft_copy);
        Complex{re: 0.0f32, im:-3.0f32}*&k_copy*fft(&vec_fade_complex_part(&(&u1*&u1)), &fft_copy)
    });

    let u_copy = u.clone();
    let k_copy = k.clone();
    let fft_copy = fft_a.clone();
    let ifft_copy = ifft_a.clone();

    let t2 = thread::spawn(move ||{
        let u2 = ifft(&u_copy, &ifft_copy);
        Complex{re: 0.0f32, im:2.0f32}*alpha*&k_copy*fft(&vec_fade_complex_part(&(&u2*&u2*&u2)), &fft_copy)
    });
    
    let u1 = t1.join().unwrap();
    let u2 = t2.join().unwrap();
    
    &u1 + &u2
}

pub fn runge_kutta_split_step(atual_wave_vec: &Array1<Complex<f32>>, k: &Array1<Complex<f32>>, 
    dt: f32, rhi : f32, alpha: f32, fft_a: &Arc<dyn Fft<f32>>, ifft_a: &Arc<dyn Fft<f32>>) -> Array1<Complex<f32>> {
    
    let u1 = atual_wave_vec*(((Complex{re:0.0f32, im:1.0f32}*k*k*k - Complex{re:rhi, im:0.0f32}*k*k)*Complex{re:dt, im:0.0f32}).mapv_into(Complex::exp));
    let k1 = next_step(&u1, k, alpha, fft_a, ifft_a);
    
    let k2 = next_step(&(&u1 + Complex{re:dt/2_f32, im:0.0f32}*&k1), k, alpha, fft_a, ifft_a);
    
    let k3 = next_step(&(&u1 + Complex{re:dt/2_f32, im:0.0f32}*&k2), k, alpha, fft_a, ifft_a);
    
    let k4 = next_step(&(&u1 + Complex{re:dt, im:0.0f32}*&k3), k, alpha, fft_a, ifft_a);
    
        
    let u_final = u1 + Complex{re:dt/6_f32, im:0.0f32}*(k1 + Complex{re:2.0f32, im:0.0f32}*k2 + Complex{re:2.0f32, im:0.0f32}*k3 + k4);
    ifft(&u_final, ifft_a)

}