extern crate hdf5_sys;
extern crate libc;

pub mod wavetoy {

    use std::f64;
    use std::ffi;
    use libc;
    use hdf5_sys as hdf5;

    pub struct State {
        time: f64,
        u: Vec<f64>,
        udot: Vec<f64>,
    }

    pub fn output(iter: i64, s: &State)
    {
        println!("Iteration {}, time {}", iter, s.time);
        let n = s.u.len();
        for i in 0..n {
            println!("    {}: {} {}", i, s.u[i], s.udot[i]);
        }
    }

    // Low-level functions

    unsafe fn create_file(name: &str) -> hdf5::hid_t {
        let cname = ffi::CString::new(name).unwrap();
        // TODO: use strong close degree
        let flags = hdf5::H5F_ACC_TRUNC;
        let fcpl = hdf5::H5P_DEFAULT;
        let fapl = hdf5::H5P_DEFAULT;
        let file = hdf5::H5Fcreate(cname.as_ptr(), flags, fcpl, fapl);
        assert!(file >= 0);
        return file;
    }

    unsafe fn open_file(name: &str) -> hdf5::hid_t {
        let cname = ffi::CString::new(name).unwrap();
        // TODO: use strong close degree
        let flags = hdf5::H5F_ACC_RDWR;
        let fapl = hdf5::H5P_DEFAULT;
        let file = hdf5::H5Fopen(cname.as_ptr(), flags, fapl);
        assert!(file >= 0);
        return file;
    }

    unsafe fn close_file(file: hdf5::hid_t) {
        let herr = hdf5::H5Fclose(file);
        assert!(herr >= 0);
    }

    unsafe fn create_group(location: hdf5::hid_t, name: &str) -> hdf5::hid_t {
        let cname = ffi::CString::new(name).unwrap();
        let lcpl = hdf5::H5P_DEFAULT;
        let gcpl = hdf5::H5P_DEFAULT;
        let gapl = hdf5::H5P_DEFAULT;
        let group = hdf5::H5Gcreate2
            (location, cname.as_ptr(), lcpl, gcpl, gapl);
        assert!(group >= 0);
        return group;
    }

    unsafe fn close_group(group: hdf5::hid_t) {
        let herr = hdf5::H5Gclose(group);
        assert!(herr >= 0);
    }

    unsafe fn create_simple_dataspace(npoints: usize) -> hdf5::hid_t {
        let dataspace = hdf5::H5Screate(hdf5::H5S_SIMPLE);
        assert!(dataspace >= 0);
        const RANK: usize = 1;
        let dims: [hdf5::hsize_t; RANK] = [npoints as hdf5::hsize_t];
        let herr = hdf5::H5Sset_extent_simple
            (dataspace, RANK as i32, dims.as_ptr(), dims.as_ptr());
        assert!(herr >= 0);
        return dataspace;
    }

    unsafe fn close_dataspace(dataspace: hdf5::hid_t) {
        let herr = hdf5::H5Sclose(dataspace);
        assert!(herr >= 0);
    }

    unsafe fn create_dataset(location: hdf5::hid_t, name: &str,
                             datatype: hdf5::hid_t,
                             dataspace: hdf5::hid_t,
                             ) -> hdf5::hid_t {
        let cname = ffi::CString::new(name).unwrap();
        let lcpl = hdf5::H5P_DEFAULT;
        let dcpl = hdf5::H5P_DEFAULT;
        let dapl = hdf5::H5P_DEFAULT;
        let dataset = hdf5::H5Dcreate2
            (location, cname.as_ptr(), datatype, dataspace, lcpl, dcpl, dapl);
        assert!(dataset >= 0);
        return dataset;
    }

    unsafe fn write_dataset_f64(dataset: hdf5::hid_t, data: &Vec<f64>) {
        let mem_type = hdf5::H5T_NATIVE_DOUBLE;
        let mem_space = hdf5::H5S_ALL;
        let file_space = hdf5::H5S_ALL;
        let xpl = hdf5::H5P_DEFAULT;
        let herr = hdf5::H5Dwrite
            (dataset, mem_type, mem_space, file_space, xpl,
             data.as_ptr() as *const libc::c_void);
        assert!(herr >= 0);
    }

    unsafe fn close_dataset(dataset: hdf5::hid_t) {
        let herr = hdf5::H5Dclose(dataset);
        assert!(herr >= 0);
    }

    // High-level functions

    unsafe fn with_file<F>(name: &str, do_create: bool, run: F)
        where F: Fn(hdf5::hid_t)
    {
        let file = if do_create {
            create_file(name)
        } else {
            open_file(name)
        };
        run(file);
        close_file(file);
    }

    unsafe fn with_group<F>(location: hdf5::hid_t, name: &str, run: F)
        where F: Fn(hdf5::hid_t)
    {
        let group = create_group(location, name);
        run(group);
        close_group(group);
    }

    unsafe fn attach_dataset_f64(location: hdf5::hid_t, name: &str,
                                 data: &Vec<f64>) {
        let npoints = data.len();
        let dataspace = create_simple_dataspace(npoints);
        let dataset = create_dataset
            (location, name, hdf5::H5T_NATIVE_DOUBLE, dataspace);
        write_dataset_f64(dataset, data);
        close_dataset(dataset);
        close_dataspace(dataspace);
    }

    unsafe fn attach_attribute_i64(location: hdf5::hid_t, name: &str,
                                   value: i64) {
        let cname = ffi::CString::new(name).unwrap();
        let attr_type = hdf5::H5T_NATIVE_INT64;
        let attr_space = hdf5::H5Screate(hdf5::H5S_SCALAR);
        let acpl = hdf5::H5P_DEFAULT;
        let aapl = hdf5::H5P_DEFAULT;
        let attribute = hdf5::H5Acreate2
            (location, cname.as_ptr(), attr_type, attr_space, acpl, aapl);
        assert!(attribute >= 0);
        let mem_type = hdf5::H5T_NATIVE_INT64;
        let buf: [i64; 1] = [value];
        let herr = hdf5::H5Awrite
            (attribute, mem_type, buf.as_ptr() as *const libc::c_void);
        assert!(herr >= 0);
        let herr = hdf5::H5Sclose(attr_space);
        assert!(herr >= 0);
        let herr = hdf5::H5Aclose(attribute);
        assert!(herr >= 0);
    }

    unsafe fn attach_attribute_f64(location: hdf5::hid_t, name: &str,
                                   value: f64) {
        let cname = ffi::CString::new(name).unwrap();
        let attr_type = hdf5::H5T_NATIVE_DOUBLE;
        let attr_space = hdf5::H5Screate(hdf5::H5S_SCALAR);
        let acpl = hdf5::H5P_DEFAULT;
        let aapl = hdf5::H5P_DEFAULT;
        let attribute = hdf5::H5Acreate2
            (location, cname.as_ptr(), attr_type, attr_space, acpl, aapl);
        assert!(attribute >= 0);
        let mem_type = hdf5::H5T_NATIVE_INT64;
        let buf: [f64; 1] = [value];
        let herr = hdf5::H5Awrite
            (attribute, mem_type, buf.as_ptr() as *const libc::c_void);
        assert!(herr >= 0);
        let herr = hdf5::H5Sclose(attr_space);
        assert!(herr >= 0);
        let herr = hdf5::H5Aclose(attribute);
        assert!(herr >= 0);
    }

    pub fn output_hdf5(iter: i64, s: &State) {
        println!("HDF5 output at iteration {}, time {}", iter, s.time);
        unsafe {
            with_file("wavetoy.h5", iter == 0, |file: hdf5::hid_t| {
                let groupname = format!("wavetoy.iteration-{:010}", iter);
                with_group(file, &groupname, |group: hdf5::hid_t| {
                    attach_attribute_i64(group, "iteration", iter);
                    attach_attribute_f64(group, "time", s.time);
                    attach_dataset_f64(group, "u", &s.u);
                    attach_dataset_f64(group, "udot", &s.udot);
                });
            });
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

    fn rk4<F>(rhs: F, dt: f64, s0: &State) -> State
        where F: Fn(&State) -> State
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
