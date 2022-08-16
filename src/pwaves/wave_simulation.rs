use rustfft::{Fft, FftPlanner, num_complex::Complex};
use std::sync::Arc;
use super::some_algorithm as sm;

pub struct WaveState{
    wave_vector : Vec<Complex<f64>>,
    t: f64,
}

impl WaveState{
    fn initialize_wave(wave_vector: &Vec<Complex<f64>>, t:f64) -> WaveState {
        WaveState{
            wave_vector: wave_vector.clone(),
            t : t,
        }
    }
    fn set_wave(&mut self, wave_vector: &Vec<Complex<f64>>, t:f64){
        self.wave_vector = wave_vector.clone();
        self.t = t;
    }
    fn wave_vector(self) -> Vec<Complex<f64>>{ // Porque tenho que colocar self em vez de &self(pq perde o ownership?)
        self.wave_vector
    }
    fn t(&self) -> f64{
        self.t
    }
}

pub struct WaveSimulation{
    wave: WaveState, // implement simulation_animation and simulation_final
    animation: bool,
    k: Vec<Complex<f64>>,
    dt: f64, 
    rhi: f64, 
    alpha: f64,
    t_final: f64,
    fft: Arc<dyn Fft<f64>>,
    ifft: Arc<dyn Fft<f64>>,
}

impl WaveSimulation{ 
    pub fn configure_simulation(animation: bool, initial_wave: Vec<Complex<f64>>, 
        k: Vec<Complex<f64>>, dt: f64, rhi:f64, alpha: f64, t_final: f64) -> WaveSimulation {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(initial_wave.len());
        let ifft = planner.plan_fft_inverse(initial_wave.len());
        WaveSimulation{
            wave : WaveState::initialize_wave(&initial_wave, 0.0f64),
            animation : animation,
            k : k,
            rhi : rhi,
            alpha : alpha,
            dt : dt,
            t_final : t_final,
            fft : fft,
            ifft : ifft,
        }
    }

    fn time_step(&mut self){
        self.wave.set_wave(&sm::runge_kutta_split_step(&self.wave.wave_vector, &self.k, self.dt, self.rhi, self.alpha, &self.fft, &self.ifft),
        self.wave.t + self.dt);
        //println!("{}", self.wave.t  + self.dt);
    }

    pub fn run(&mut self){
        if self.animation == false {
            while self.wave.t <= self.t_final{
                self.time_step();
                //println!("{}",self.wave.t);
            }
        }
    }
}

