mod pwaves;
//use rustfft::{Fft, FftPlanner, num_complex::Complex, algorithm::Radix4, FftDirection};
use crate::pwaves::wave_simulation::*;
use crate::pwaves::some_algorithm::*;
use ndarray::Array1;

fn main() { 
    let size = 32768;
    let x = Array1::<f32>::linspace(-6000., 6000., size); let dx = x[1] - x[0];
    let u0 = step_initial_condition(&x, 0.3, 0.1, 0.0);
    let k = freq_bin(size, dx);
    let mut simulation = WaveSimulation::configure_simulation(false , u0, k, x, 0.00001, 0.0, 1.0, 0.05);
    simulation.run();
    println!("Finished!");
}
