use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

type SyncFunc = Option<Box<dyn Fn() + Send + Sync>>;

#[derive(Debug)]
struct Timer {
    name: String,
    started: bool,
    start_time: Option<Instant>,
    start_times: Vec<Instant>,
    stop_times: Vec<Instant>,
    costs: Vec<Duration>,
}

impl Timer {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            started: false,
            start_time: None,
            start_times: Vec::new(),
            stop_times: Vec::new(),
            costs: Vec::new(),
        }
    }

    fn start(&mut self, sync_func: SyncFunc) {
        assert!(!self.started, "timer {} has already been started.", self.name);
        if let Some(func) = sync_func {
            func();
        }

        let now = Instant::now();
        self.start_time = Some(now);
        self.start_times.push(now);
        self.started = true;
    }

    fn stop(&mut self, sync_func: SyncFunc) {
        assert!(self.started, "timer {} is not started.", self.name);
        if let Some(func) = sync_func {
            func();
        }

        let stop_time = Instant::now();
        if let Some(start_time) = self.start_time {
            self.costs.push(stop_time.duration_since(start_time));
        }
        self.stop_times.push(stop_time);
        self.started = false;
    }

    fn reset(&mut self) {
        self.started = false;
        self.start_time = None;
        self.start_times.clear();
        self.stop_times.clear();
        self.costs.clear();
    }

    fn elapsed(&self, mode: &str) -> f64 {
        if self.costs.is_empty() {
            return 0.0;
        }

        match mode {
            "average" => {
                let total: Duration = self.costs.iter().sum();
                total.as_secs_f64() / (self.costs.len() as f64)
            }
            "sum" => {
                let total: Duration = self.costs.iter().sum();
                total.as_secs_f64()
            }
            _ => panic!("Supported mode is: average | sum"),
        }
    }
}

#[derive(Debug, Default)]
struct Timers {
    timers: HashMap<String, Timer>,
}

impl Timers {
    fn get_mut(&mut self, name: &str) -> &mut Timer {
        self.timers.entry(name.to_string()).or_insert_with(|| Timer::new(name))
    }

    fn contains(&self, name: &str) -> bool {
        self.timers.contains_key(name)
    }
}

// Global singleton
lazy_static::lazy_static! {
    pub static ref TIMERS: Arc<Mutex<Timers>> = Arc::new(Mutex::new(Timers::default()));
}

#[derive(Debug, Clone)]
struct Event {
    tstamp: Instant,
    name: String,
    info: String,
}

#[derive(Debug, Default)]
struct Tracer {
    events: Vec<Event>,
}

impl Tracer {
    fn log(&mut self, name: &str, info: &str, sync_func: SyncFunc) {
        if let Some(func) = sync_func {
            func();
        }

        self.events.push(Event {
            tstamp: Instant::now(),
            name: name.to_string(),
            info: info.to_string(),
        });
    }
}

// Global tracer instance
lazy_static::lazy_static! {
    pub static ref TRACER: Arc<Mutex<Tracer>> = Arc::new(Mutex::new(Tracer::default()));
}
