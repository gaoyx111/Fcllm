use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct ArcCache {
    max_size: usize,
    p: usize, // 自适应阈值
    t1: VecDeque<usize>, // 最近使用列表
    t2: VecDeque<usize>, // 频繁使用列表
    b1: VecDeque<usize>, // T1 ghost
    b2: VecDeque<usize>, // T2 ghost
}

impl ArcCache {
    pub fn new(max_size: usize) -> Self {
        let mut arc = Self {
            max_size,
            p: 0,
            t1: VecDeque::new(),
            t2: VecDeque::new(),
            b1: VecDeque::new(),
            b2: VecDeque::new(),
        };

        // 初始化填满缓存
        for expert_id in 0..max_size {
            arc.update(expert_id);
        }

        arc
    }

    pub fn is_evicted(&self, expert_id: usize) -> bool {
        !(self.t1.contains(&expert_id) || self.t2.contains(&expert_id))
    }

    pub fn update_list(&mut self, expert_ids: &[usize]) -> Vec<usize> {
        let mut evicted = vec![];
        for &expert_id in expert_ids {
            if let Some(eid) = self.update(expert_id) {
                if !expert_ids.contains(&eid) && !evicted.contains(&eid) {
                    evicted.push(eid);
                }
            }
        }
        evicted
    }

    pub fn update(&mut self, expert_id: usize) -> Option<usize> {
        if self.t1.contains(&expert_id) {
            self.t1.retain(|&x| x != expert_id);
            self.t2.push_back(expert_id);
            return None;
        } else if self.t2.contains(&expert_id) {
            self.t2.retain(|&x| x != expert_id);
            self.t2.push_back(expert_id);
            return None;
        } else if self.b1.contains(&expert_id) {
            self.adjust_p((self.t1.len() as isize).min(self.max_size as isize));
            let evicted = self.replace(expert_id);
            self.b1.retain(|&x| x != expert_id);
            self.t2.push_back(expert_id);
            return evicted;
        } else if self.b2.contains(&expert_id) {
            self.adjust_p(-(self.t2.len() as isize).min(self.max_size as isize));
            let evicted = self.replace(expert_id);
            self.b2.retain(|&x| x != expert_id);
            self.t2.push_back(expert_id);
            return evicted;
        } else {
            let mut evicted: Option<usize> = None;

            if self.t1.len() + self.b1.len() >= self.max_size {
                if self.t1.len() < self.max_size {
                    self.b1.pop_front();
                    evicted = self.replace(expert_id);
                } else {
                    evicted = self.t1.pop_front();
                }
            } else if self.t1.len() + self.t2.len() + self.b1.len() + self.b2.len() >= self.max_size {
                if self.t1.len() + self.t2.len() + self.b1.len() + self.b2.len() >= 2 * self.max_size {
                    if !self.b1.is_empty() {
                        self.b1.pop_front();
                    } else {
                        self.b2.pop_front();
                    }
                }
                evicted = self.replace(expert_id);
            }

            self.t1.push_back(expert_id);
            evicted
        }
    }

    fn adjust_p(&mut self, delta: isize) {
        let p = self.p as isize + delta;
        self.p = p.clamp(0, self.max_size as isize) as usize;
    }

    fn replace(&mut self, expert_id: usize) -> Option<usize> {
        if !self.t1.is_empty() &&
            ((self.b2.contains(&expert_id) && self.t1.len() > self.p)
            || self.t1.len() > self.p)
        {
            let victim = self.t1.pop_front()?;
            self.b1.push_back(victim);
            Some(victim)
        } else if !self.t2.is_empty() {
            let victim = self.t2.pop_front()?;
            self.b2.push_back(victim);
            Some(victim)
        } else if !self.t1.is_empty() {
            let victim = self.t1.pop_front()?;
            self.b1.push_back(victim);
            Some(victim)
        } else {
            None
        }
    }
}
