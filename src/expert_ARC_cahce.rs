use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct ARC_Cache {
    max_size: usize,
    p: usize, // 自适应参数，用于调整T1和T2的大小

    t1: VecDeque<usize>, // 最近使用列表，保存专家ID
    t2: VecDeque<usize>, // 频繁使用列表，保存专家ID
    b1: VecDeque<usize>, // T1的Ghost列表，保存专家ID
    b2: VecDeque<usize>, // T2的Ghost列表，保存专家ID
}

impl ARC_Cache {
    pub fn new(max_size: usize) -> Self {
        let mut cache = Self {
            max_size,
            p: 0,
            t1: VecDeque::new(),
            t2: VecDeque::new(),
            b1: VecDeque::new(),
            b2: VecDeque::new(),
        };
        for expert_id in 0..max_size {
            cache.update(expert_id);
        }
        cache
    }

    pub fn is_evicted(&self, expert_id: usize) -> bool {
        !(self.t1.contains(&expert_id) || self.t2.contains(&expert_id))
    }

    pub fn update_list(&mut self, expert_list: &[usize]) -> Vec<usize> {
        let mut evicted_list = Vec::new();
        for &expert_id in expert_list {
            if let Some(evicted_id) = self.update(expert_id) {
                if !expert_list.contains(&evicted_id) && !evicted_list.contains(&evicted_id) {
                    evicted_list.push(evicted_id);
                }
            }
        }
        evicted_list
    }

    pub fn update(&mut self, expert_id: usize) -> Option<usize> {
        if let Some(pos) = self.t1.iter().position(|&x| x == expert_id) {
            self.t1.remove(pos);
            self.t2.push_back(expert_id);
            return None;
        } else if let Some(pos) = self.t2.iter().position(|&x| x == expert_id) {
            self.t2.remove(pos);
            self.t2.push_back(expert_id);
            return None;
        } else if let Some(pos) = self.b1.iter().position(|&x| x == expert_id) {
            self.adjust_p(self.t1.len().min(self.max_size) as isize);
            let evicted_id = self.replace(expert_id);
            self.b1.remove(pos);
            self.t2.push_back(expert_id);
            return evicted_id;
        } else if let Some(pos) = self.b2.iter().position(|&x| x == expert_id) {
            self.adjust_p(-(self.t2.len().min(self.max_size) as isize));
            let evicted_id = self.replace(expert_id);
            self.b2.remove(pos);
            self.t2.push_back(expert_id);
            return evicted_id;
        } else {
            let mut evicted_id = None;

            if self.t1.len() + self.b1.len() == self.max_size {
                if self.t1.len() < self.max_size {
                    self.b1.pop_front();
                    evicted_id = self.replace(expert_id);
                } else {
                    evicted_id = self.t1.pop_front();
                }
            } else if self.total_len() >= self.max_size {
                if self.total_len() >= 2 * self.max_size {
                    if !self.b1.is_empty() {
                        self.b1.pop_front();
                    } else {
                        self.b2.pop_front();
                    }
                }
                evicted_id = self.replace(expert_id);
            }

            self.t1.push_back(expert_id);
            return evicted_id;
        }
    }

    fn total_len(&self) -> usize {
        self.t1.len() + self.t2.len() + self.b1.len() + self.b2.len()
    }

    fn adjust_p(&mut self, delta: isize) {
        let new_p = self.p as isize + delta;
        self.p = new_p.clamp(0, self.max_size as isize) as usize;
    }

    fn replace(&mut self, expert_id: usize) -> Option<usize> {
        if !self.t1.is_empty()
            && ((self.b2.contains(&expert_id) && self.t1.len() > self.p) || self.t1.len() > self.p)
        {
            let id = self.t1.pop_front().unwrap();
            self.b1.push_back(id);
            Some(id)
        } else if !self.t2.is_empty() {
            let id = self.t2.pop_front().unwrap();
            self.b2.push_back(id);
            Some(id)
        } else if !self.t1.is_empty() {
            let id = self.t1.pop_front().unwrap();
            self.b1.push_back(id);
            Some(id)
        } else {
            None
        }
    }
}
