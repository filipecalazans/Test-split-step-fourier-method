use rustfft::{Fft, FftDirection, FftPlanner, num_complex::Complex};
use std::sync::Arc;
use ndarray::Array1;
use super::some_algorithm as sm;
use std::fs::File;
use std::io::Write;

pub struct WaveState{
    wave_vector : Array1<Complex<f32>>,
    t: f32,
}

impl WaveState{
    fn new(wave_vector: &Array1<Complex<f32>>, t:f32) -> Self {
        Self{
            wave_vector: wave_vector.clone(),
            t : t,
        }
    }
    fn set_wave(&mut self, wave_vector: &Array1<Complex<f32>>, t:f32){
        self.wave_vector = wave_vector.clone();
        self.t = t;
    }
    fn wave_vector(&self) -> Array1<Complex<f32>>{ // Porque tenho que colocar self em vez de &self(pq perde o ownership?)
        self.wave_vector.clone()
    }
    fn t(&self) -> f32{
        self.t
    }
}

pub struct WaveSimulation{
    wave: WaveState, // implement simulation_animation and simulation_final
    animation: bool,
    k: Array1<Complex<f32>>,
    x: Array1<f32>,
    dt: f32, 
    rhi: f32, 
    alpha: f32,
    t_final: f32,
    fft: Arc<dyn Fft<f32>>,
    ifft: Arc<dyn Fft<f32>>,
}

impl WaveSimulation{ 
    pub fn configure_simulation(animation: bool, initial_wave: Array1<Complex<f32>>, 
        k: Array1<Complex<f32>>, x: Array1<f32>, dt: f32, rhi:f32, alpha: f32, t_final: f32) -> Self {
        let mut planner = FftPlanner::new();
        println!("From waveSimulation: dim={}", initial_wave.dim());
        let ifft = planner.plan_fft(initial_wave.dim(), FftDirection::Inverse);
        let fft = planner.plan_fft(initial_wave.dim(), FftDirection::Forward);
        WaveSimulation{
            wave : WaveState::new(&initial_wave, 0.0f32),
            animation : animation,
            k : k,
            x : x,
            rhi : rhi,
            alpha : alpha,
            dt : dt,
            t_final : t_final,
            fft : fft,
            ifft : ifft,
        }
    }

    fn time_step(&mut self){
        self.wave.set_wave(&sm::runge_kutta_split_step(&sm::fft(&self.wave.wave_vector, &self.fft), &self.k, self.dt, self.rhi, self.alpha, &self.fft, &self.ifft),
        self.wave.t + self.dt);
        //println!("{}", self.wave.t  + self.dt);
    }
    fn save_data(&self){
        let mut fx = File::create("x_data.dat").expect("Unable to create a file!");
        let mut fy = File::create("y_data.dat").expect("Unable to create a file!");
        println!("The file was criated!");
        let wave_copy = self.wave.wave_vector();
        for i in 0..self.x.len(){
            write!(fx, "{} ", &self.x[i]).expect("Unable to write a file!");
            write!(fy, "{} ", wave_copy[i].re).expect("Unable to write a file!");
        }
    }
    pub fn run(&mut self){
        if self.animation == false {
            while self.wave.t <= self.t_final{
                self.time_step();
                println!("{}",self.wave.t);
            }
            self.save_data();
        }
    }
}

