extern crate wavetoy;
use wavetoy::wavetoy as w;

fn main() {
    println!("WaveToy");
    let n = 11;
    let cfl = 4;
    let niters = cfl * (n - 1);
    let dt = 1.0 / (cfl * (n - 1)) as f64;

    let mut iter = 0;
    let mut s = w::init(0.0, n);
    w::output(iter, &s);
    for _ in 0..niters {
        iter = iter + 1;
        s = w::step(&s, dt);
    }
    w::output(iter, &s);
    println!("Done.")
}
