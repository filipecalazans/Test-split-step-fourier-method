use rustfft::num_complex::Complex;
use rustfft::num_traits::{Pow};


pub fn vec_complex_to_real(vec: &Vec<Complex<f64>>) -> Vec<f64>{
    vec.iter().map(|x| x.re).collect::<Vec<f64>>()
}

pub fn vec_real_to_complex(vec:&Vec<f64>) -> Vec<Complex<f64>>{
    vec.iter().map(|x| num_real_to_complex(*x)).collect::<Vec<Complex<f64>>>()
}

pub fn num_real_to_complex(num: f64) -> Complex<f64>{
    Complex{re: num, im:0.0f64}
}

pub fn vec_fade_complex_part(vec:&Vec<Complex<f64>>) -> Vec<Complex<f64>>{
    vec.iter().map(|x| Complex{re: x.re, im: 0.0f64}).collect::<Vec<Complex<f64>>>()
}

pub fn multiply_num_vec(num: &Complex<f64>, vec: &Vec<Complex<f64>>) -> Vec<Complex<f64>>{
    vec.iter().map(|x| num*x).collect::<Vec<Complex<f64>>>()
}

pub fn multiply_vec_vec(vec1: &Vec<Complex<f64>>, vec2: &Vec<Complex<f64>>) -> Vec<Complex<f64>>{
    let mut pos_vec2 = 0;
    vec1.iter().map(|x| {pos_vec2 = 1 + pos_vec2; x*vec2[pos_vec2 - 1]}).collect::<Vec<Complex<f64>>>()
}

pub fn add_vec_vec(vec1: &Vec<Complex<f64>>, vec2: &Vec<Complex<f64>>) -> Vec<Complex<f64>>{
    let mut pos_vec2 = 0;
    vec1.iter().map(|x| {pos_vec2 = 1 + pos_vec2; x + vec2[pos_vec2 - 1]}).collect::<Vec<Complex<f64>>>()
}

pub fn exp(vec: &Vec<Complex<f64>>) -> Vec<Complex<f64>>{
    vec.iter().map(|x| x.exp()).collect::<Vec<Complex<f64>>>()
}

pub fn pow(vec: &Vec<Complex<f64>>, exponent: f64) -> Vec<Complex<f64>>{
    vec.iter().map(|x| x.pow(exponent)).collect::<Vec<Complex<f64>>>()
}