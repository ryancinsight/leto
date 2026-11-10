//! Sorted adjacency rows in one relocatable arena.
//!
//! Row-local spare capacity avoids moving unrelated rows on every fill edge.
//! Geometric row growth leaves retired segments; compaction reclaims them once
//! their total exceeds half of live row capacity. Thus occupied arena length is
//! bounded by 1.5 times live capacity after every edit. Row capacity grows by
//! 1.5 and the shared allocation by 1.25, balancing copy cost against slack.
//! Compaction moves rows in source order within the existing allocation; its
//! reusable ordering scratch contains at most one pair of indices per row.

#[derive(Clone, Copy)]
struct Row {
    start: usize,
    len: usize,
    capacity: usize,
}

pub(super) struct Adjacency {
    data: Vec<usize>,
    rows: Vec<Row>,
    live_capacity: usize,
    compaction_order: Vec<(usize, usize)>,
}

impl Adjacency {
    pub(super) fn new(data: Vec<usize>, offsets: &[usize]) -> Self {
        let live_capacity = data.len();
        let rows = offsets
            .windows(2)
            .map(|p| Row {
                start: p[0],
                len: p[1] - p[0],
                capacity: p[1] - p[0],
            })
            .collect();
        Self {
            data,
            rows,
            live_capacity,
            compaction_order: Vec::new(),
        }
    }

    pub(super) fn row(&self, vertex: usize) -> &[usize] {
        let row = self.rows[vertex];
        &self.data[row.start..row.start + row.len]
    }

    pub(super) fn contains(&self, vertex: usize, value: usize) -> bool {
        self.row(vertex).binary_search(&value).is_ok()
    }

    pub(super) fn clear(&mut self, vertex: usize) {
        self.live_capacity -= self.rows[vertex].capacity;
        self.rows[vertex] = Row {
            start: 0,
            len: 0,
            capacity: 0,
        };
        self.compact_if_needed();
    }

    pub(super) fn insert(&mut self, vertex: usize, value: usize) {
        let Err(index) = self.row(vertex).binary_search(&value) else {
            return;
        };
        let mut row = self.rows[vertex];
        if row.len == row.capacity {
            let capacity = row.len.saturating_add((row.len / 2).max(1));
            let start = self.data.len();
            let required = start + capacity;
            if required > self.data.capacity() {
                let reserve = required.max(
                    self.data
                        .capacity()
                        .saturating_add(self.data.capacity() / 4),
                );
                self.data.reserve_exact(reserve - self.data.len());
            }
            self.data.extend_from_within(row.start..row.start + row.len);
            self.data.resize(start + capacity, 0);
            self.live_capacity += capacity - row.capacity;
            row.start = start;
            row.capacity = capacity;
        }
        self.data.copy_within(
            row.start + index..row.start + row.len,
            row.start + index + 1,
        );
        self.data[row.start + index] = value;
        row.len += 1;
        self.rows[vertex] = row;
        self.compact_if_needed();
    }

    fn compact_if_needed(&mut self) {
        if self.data.len() <= self.live_capacity.saturating_add(self.live_capacity / 2) {
            return;
        }
        for row in &mut self.rows {
            if row.capacity == 0 {
                row.start = 0;
            }
        }
        self.compaction_order.clear();
        self.compaction_order.extend(
            self.rows
                .iter()
                .enumerate()
                .filter(|(_, row)| row.capacity != 0)
                .map(|(vertex, row)| (row.start, vertex)),
        );
        self.compaction_order.sort_unstable();
        let mut write = 0;
        for &(_, vertex) in &self.compaction_order {
            let row = &mut self.rows[vertex];
            // In increasing source order, retained capacity cannot exceed the
            // original prefix. Each copy therefore stays behind unread rows.
            debug_assert!(write <= row.start);
            self.data.copy_within(row.start..row.start + row.len, write);
            row.start = write;
            write += row.capacity;
        }
        self.data.truncate(write);
        if self.data.capacity() > self.live_capacity.saturating_mul(2) {
            self.data.shrink_to(self.live_capacity);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn verify(adjacency: &Adjacency, reference: &[BTreeSet<usize>]) {
        for (vertex, row) in reference.iter().enumerate() {
            assert_eq!(
                adjacency.row(vertex),
                row.iter().copied().collect::<Vec<_>>()
            );
            for value in 0..reference.len() {
                assert_eq!(adjacency.contains(vertex, value), row.contains(&value));
            }
        }
        assert_eq!(
            adjacency.live_capacity,
            adjacency.rows.iter().map(|r| r.capacity).sum::<usize>()
        );
        assert!(adjacency.data.len() <= adjacency.live_capacity + adjacency.live_capacity / 2);
    }

    #[test]
    fn relocations_and_compaction_preserve_unrelated_rows() {
        let mut adjacency = Adjacency::new(vec![1, 3, 0, 2], &[0, 2, 3, 3, 4]);
        let mut reference: Vec<BTreeSet<usize>> = [vec![1, 3], vec![0], vec![], vec![2]]
            .into_iter()
            .map(|r| r.into_iter().collect())
            .collect();
        verify(&adjacency, &reference);
        for value in (0..128).rev() {
            for vertex in [3, 1, 0, 2] {
                adjacency.insert(vertex, value);
                reference[vertex].insert(value);
                verify(&adjacency, &reference);
            }
        }
        // Every retirement changes live capacity; the final retirement reclaims
        // every slot. Regrowth from that empty state must remain well formed.
        for vertex in [1, 3, 0, 2] {
            adjacency.clear(vertex);
            reference[vertex].clear();
            verify(&adjacency, &reference);
        }
        assert!(adjacency.data.is_empty());
        adjacency.insert(2, 7);
        reference[2].insert(7);
        verify(&adjacency, &reference);
    }

    #[test]
    fn compaction_preserves_initially_empty_rows_and_out_of_order_segments() {
        let mut adjacency = Adjacency::new(vec![1, 2, 3, 4], &[0, 0, 4, 4]);
        adjacency.clear(1);
        assert_eq!(adjacency.row(0), &[]);
        assert_eq!(adjacency.row(2), &[]);
        let mut reference = vec![BTreeSet::new(); 3];
        for round in 0..30 {
            for vertex in [2, 0, 1] {
                if round % 7 == 0 {
                    adjacency.clear(vertex);
                    reference[vertex].clear();
                }
                for value in (0..round).rev() {
                    adjacency.insert(vertex, value);
                    reference[vertex].insert(value);
                }
                verify(&adjacency, &reference);
            }
        }
    }

    #[test]
    fn repeated_duplicates_do_not_allocate_or_change_rows() {
        let mut adjacency = Adjacency::new(vec![2, 5, 7], &[0, 3]);
        let ptr = adjacency.data.as_ptr();
        let capacity = adjacency.data.capacity();
        for _ in 0..128 {
            for value in [7, 2, 5] {
                adjacency.insert(0, value);
            }
        }
        assert_eq!(adjacency.row(0), &[2, 5, 7]);
        assert_eq!(adjacency.data.as_ptr(), ptr);
        assert_eq!(adjacency.data.capacity(), capacity);
    }
}
