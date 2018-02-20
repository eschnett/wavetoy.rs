pub mod wavetoy {

    use std::f64;

    pub struct State {
        time: f64,
        u: Vec<f64>,
        udot: Vec<f64>,
    }

    pub fn output(iter: i64, s: &State) {
        println!("Iteration {}, time {}", iter, s.time);
        let n = s.u.len();
        for i in 0..n {
            println!("    {}: {} {}", i, s.u[i], s.udot[i]);
        }
    }

    pub fn init(t: f64, n: usize) -> State {
        let mut s = State { time: t, u: vec![0.0; n], udot: vec![0.0; n], };
        for i in 0..n {
            let pi = f64::consts::PI;
            let x = i as f64 / (n - 1) as f64;
            s.u[i] = (2.0 * pi * x).sin();
            s.udot[i] = 2.0 * pi * (2.0 * pi * x).cos();
        }
        return s;
    }

    pub fn rhs(s: &State) -> State {
        let n = s.u.len();
        let dx = 1.0 / (n - 1) as f64;
        let dx2 = dx * dx;
        let mut r = State { time: 1.0, u: vec![0.0; n], udot: vec![0.0; n], };
        for i in 0..n {
            if i == 0 {
                r.u[i] = 0.0;
                r.udot[i] = 0.0;
            } else if i == n - 1 {
                r.u[i] = 0.0;
                r.udot[i] = 0.0;
            } else {
                r.u[i] = s.udot[i];
                r.udot[i] = (s.u[i+1] - 2.0 * s.u[i] + s.u[i-1]) / dx2;
            }
        }
        return r;
    }

    fn vadd(u: &Vec<f64>, a: f64, v: &Vec<f64>) -> Vec<f64> {
        let n = u.len();
        assert!(v.len() == n);
        let mut r: Vec<f64> = vec![0.0; n];
        for i in 0..n {
            r[i] = u[i] + a * v[i];
        }
        return r;
    }

    fn add(s: &State, a: f64, r: &State) -> State {
        return State {
            time: s.time + a * r.time,
            u: vadd(&s.u, a, &r.u),
            udot: vadd(&s.udot, a, &r.udot),
        };
    }

    fn rk4(rhs: fn(&State) -> State, dt: f64, s0: &State) -> State
    {
        let r0 = rhs(s0);
        let s1 = add(s0, 0.5 * dt, &r0);
        let r1 = rhs(&s1);
        let s2 = add(s0, dt, &r1);
        return s2;
    }

    pub fn step(s: &State, dt: f64) -> State {
        return rk4(rhs, dt, &s);
    }

}
