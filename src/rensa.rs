use rand::prelude::*;
use rustc_hash::FxHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

// ADAPTED FROM https://github.com/beowolx/rensa/blob/main/src/lib.rs

/// RMinHash implements the MinHash algorithm for efficient similarity estimation.
pub struct RMinHash {
    num_perm: usize,
    hash_values: Vec<u32>,
    permutations: Vec<(u64, u64)>,
}

impl RMinHash {
    /// Creates a new RMinHash instance.
    ///
    /// # Arguments
    ///
    /// * `num_perm` - The number of permutations to use in the MinHash algorithm.
    /// * `seed` - A seed value for the random number generator.
    pub fn new(num_perm: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let permutations: Vec<(u64, u64)> = (0..num_perm).map(|_| (rng.gen(), rng.gen())).collect();

        RMinHash {
            num_perm,
            hash_values: vec![u32::MAX; num_perm],
            permutations,
        }
    }

    /// Updates the MinHash with a new set of items.
    ///
    /// # Arguments
    ///
    /// * `items` - A vector of strings to be hashed and incorporated into the MinHash.
    pub fn update(&mut self, items: Vec<&str>) {
        for item in items {
            let item_hash = calculate_hash(&item);
            for (i, &(a, b)) in self.permutations.iter().enumerate() {
                let hash = permute_hash(item_hash, a, b);
                self.hash_values[i] = self.hash_values[i].min(hash);
            }
        }
    }

    /// Returns the current MinHash digest.
    ///
    /// # Returns
    ///
    /// A vector of u32 values representing the MinHash signature.
    pub fn digest(&self) -> Vec<u32> {
        self.hash_values.clone()
    }

    /// Calculates the Jaccard similarity between this MinHash and another.
    ///
    /// # Arguments
    ///
    /// * `other` - Another RMinHash instance to compare with.
    ///
    /// # Returns
    ///
    /// A float value representing the estimated Jaccard similarity.
    pub fn jaccard(&self, other: &RMinHash) -> f64 {
        let equal_count = self
            .hash_values
            .iter()
            .zip(&other.hash_values)
            .filter(|&(&a, &b)| a == b)
            .count();
        equal_count as f64 / self.num_perm as f64
    }
}

/// RMinHashLSH implements Locality-Sensitive Hashing using MinHash for efficient similarity search.
pub struct RMinHashLSH {
    threshold: f64,
    num_perm: usize,
    num_bands: usize,
    band_size: usize,
    hash_tables: Vec<HashMap<u64, Vec<usize>>>,
}

impl RMinHashLSH {
    /// Creates a new RMinHashLSH instance.
    ///
    /// # Arguments
    ///
    /// * `threshold` - The similarity threshold for considering items as similar.
    /// * `num_perm` - The number of permutations used in the MinHash algorithm.
    /// * `num_bands` - The number of bands for the LSH algorithm.
    pub fn new(threshold: f64, num_perm: usize, num_bands: usize) -> Self {
        RMinHashLSH {
            threshold,
            num_perm,
            num_bands,
            band_size: num_perm / num_bands,
            hash_tables: vec![HashMap::new(); num_bands],
        }
    }

    /// Inserts a MinHash into the LSH index.
    ///
    /// # Arguments
    ///
    /// * `key` - A unique identifier for the MinHash.
    /// * `minhash` - The RMinHash instance to be inserted.
    pub fn insert(&mut self, key: usize, minhash: &RMinHash) {
        let digest = minhash.digest();
        for (i, table) in self.hash_tables.iter_mut().enumerate() {
            let start = i * self.band_size;
            let end = start + self.band_size;
            let band_hash = calculate_band_hash(&digest[start..end]);
            table.entry(band_hash).or_insert_with(Vec::new).push(key);
        }
    }

    /// Queries the LSH index for similar items.
    ///
    /// # Arguments
    ///
    /// * `minhash` - The RMinHash instance to query for.
    ///
    /// # Returns
    ///
    /// A vector of keys (usize) of potentially similar items.
    pub fn query(&self, minhash: &RMinHash) -> Vec<usize> {
        let digest = minhash.digest();
        let mut candidates = Vec::new();
        for (i, table) in self.hash_tables.iter().enumerate() {
            let start = i * self.band_size;
            let end = start + self.band_size;
            let band_hash = calculate_band_hash(&digest[start..end]);
            if let Some(keys) = table.get(&band_hash) {
                candidates.extend(keys);
            }
        }
        candidates.sort_unstable();
        candidates.dedup();
        candidates
    }

    pub fn any_matches(&self, minhash: &RMinHash) -> Option<&Vec<usize>> {
        let digest = minhash.digest();
        for (i, table) in self.hash_tables.iter().enumerate() {
            let start = i * self.band_size;
            let end = start + self.band_size;
            let band_hash = calculate_band_hash(&digest[start..end]);
            if let Some(keys) = table.get(&band_hash) {
                return Some(keys);
            }
        }
        None
    }

    /// Checks if two MinHashes are similar based on the LSH threshold.
    ///
    /// # Arguments
    ///
    /// * `minhash1` - The first RMinHash instance.
    /// * `minhash2` - The second RMinHash instance.
    ///
    /// # Returns
    ///
    /// A boolean indicating whether the MinHashes are considered similar.
    fn is_similar(&self, minhash1: &RMinHash, minhash2: &RMinHash) -> bool {
        minhash1.jaccard(minhash2) >= self.threshold
    }

    /// Returns the number of permutations used in the LSH index.
    fn get_num_perm(&self) -> usize {
        self.num_perm
    }

    /// Returns the number of bands used in the LSH index.
    fn get_num_bands(&self) -> usize {
        self.num_bands
    }
}

/// Calculates a hash value for a given item.
fn calculate_hash<T: Hash>(t: &T) -> u64 {
    let mut s = FxHasher::default();
    t.hash(&mut s);
    s.finish()
}

/// Applies a permutation to a hash value.
fn permute_hash(hash: u64, a: u64, b: u64) -> u32 {
    ((a.wrapping_mul(hash).wrapping_add(b)) >> 32) as u32
}

/// Calculates a hash value for a band of MinHash values.
fn calculate_band_hash(band: &[u32]) -> u64 {
    let mut hasher = FxHasher::default();
    for &value in band {
        hasher.write_u32(value);
    }
    hasher.finish()
}
