mod pwaves;

use crate::pwaves::wave_simulation::*;
use crate::pwaves::operations::*;
use crate::pwaves::some_algorithm::*;
//use rustfft::{FftPlanner, num_complex::Complex};

fn main() { 
    let size = 32768;
    let x = linspace(-6000., 6000., size); let dx = x[1].re - x[0].re;
    let u0 = step_initial_condition(&vec_complex_to_real(&x), 0.3, 0.1, 0.0);
    let k = freq_bin(size, dx);
    let mut simulation = WaveSimulation::configure_simulation(false , u0, k, 0.00001, 0.0, 1.0, 0.05);
    simulation.run();
    println!("Finished!");
}
