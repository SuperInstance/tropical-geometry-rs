//! Harmonic analysis via tropical geometry.
//!
//! Uses the max-plus (and min-plus) tropical semiring to reason about voice
//! leadings and chord spaces. Tropical distance replaces Euclidean distance
//! when searching for shortest voice leadings between chords, yielding
//! piecewise-linear optimal paths that are natural in the tropical setting.

// TropicalSemiring is re-exported at the crate level for external consumers.

/// Result of an optimal voice-leading search between two chords.
#[derive(Debug, Clone, PartialEq)]
pub struct VoiceLeading {
    /// Starting chord pitches (MIDI-like integers).
    pub from_chord: Vec<i32>,
    /// Target chord pitches.
    pub to_chord: Vec<i32>,
    /// Per-voice displacements (to - from) for the optimal mapping.
    pub displacements: Vec<i32>,
    /// Tropical (max-plus) distance of the optimal voice leading.
    pub tropical_distance: f64,
    /// Permutation mapping: voice i in `from_chord` maps to `permutation[i]` in `to_chord`.
    pub permutation: Vec<usize>,
}

impl VoiceLeading {
    /// The maximal absolute displacement (voice-leading "width" in tropical metric).
    pub fn maximal_displacement(&self) -> i32 {
        self.displacements.iter().map(|d| d.abs()).max().unwrap_or(0)
    }

    /// Sum of absolute displacements (L1 norm).
    pub fn total_displacement(&self) -> i32 {
        self.displacements.iter().map(|d| d.abs()).sum()
    }
}

/// A chord embedded in tropical space.
#[derive(Debug, Clone, PartialEq)]
pub struct TropicalChordSpace {
    /// Chord pitches sorted ascending.
    pub pitches: Vec<i32>,
    /// Dimension of the embedding (equals pitches.len()).
    pub dim: usize,
}

impl TropicalChordSpace {
    /// Create a new tropical chord from raw pitches (will be sorted).
    pub fn new(pitches: Vec<i32>) -> Self {
        let mut p = pitches;
        p.sort();
        Self { dim: p.len(), pitches: p }
    }

    /// Tropical (max-plus) distance to another chord of the same dimension.
    ///
    /// `d_trop(a,b) = max_i |a_i - b_i|` after aligning via the optimal permutation.
    pub fn tropical_distance_to(&self, other: &TropicalChordSpace) -> f64 {
        tropical_chord_distance(self.pitches.clone(), other.pitches.clone())
    }

    /// Project onto a lower-dimensional tropical subspace by dropping voices.
    pub fn project(&self, keep_indices: &[usize]) -> TropicalChordSpace {
        let mut p: Vec<i32> = keep_indices.iter().map(|&i| self.pitches[i]).collect();
        let dim = p.len();
        p.sort();
        TropicalChordSpace { pitches: p, dim }
    }
}

/// Harmonic analysis of a chord progression in tropical geometry.
#[derive(Debug, Clone)]
pub struct TropicalHarmonicAnalysis {
    /// The chord progression.
    pub progression: Vec<TropicalChordSpace>,
    /// Pairwise tropical distances.
    pub pairwise_distances: Vec<Vec<f64>>,
}

impl TropicalHarmonicAnalysis {
    /// Analyse a chord progression.
    pub fn from_progression(chords: Vec<Vec<i32>>) -> Self {
        let progression: Vec<TropicalChordSpace> =
            chords.into_iter().map(TropicalChordSpace::new).collect();
        let n = progression.len();
        let mut pairwise = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let d = progression[i].tropical_distance_to(&progression[j]);
                pairwise[i][j] = d;
                pairwise[j][i] = d;
            }
        }
        Self {
            progression,
            pairwise_distances: pairwise,
        }
    }

    /// Total tropical length of the progression (sum of consecutive distances).
    pub fn total_tropical_length(&self) -> f64 {
        let mut total = 0.0;
        for i in 1..self.progression.len() {
            total += self.pairwise_distances[i - 1][i];
        }
        total
    }

    /// The most "distant" voice leading (max consecutive tropical distance).
    pub fn max_voice_leading_distance(&self) -> f64 {
        let mut max_d = 0.0;
        for i in 1..self.progression.len() {
            let d = self.pairwise_distances[i - 1][i];
            if d > max_d {
                max_d = d;
            }
        }
        max_d
    }

    /// Decompose the progression into tropical polytope faces.
    ///
    /// Returns segments [start..end] where each segment spans a locally
    /// "smooth" voice leading (consecutive tropical distance ≤ threshold).
    pub fn decompose_polytope_faces(&self, threshold: f64) -> Vec<(usize, usize)> {
        if self.progression.is_empty() {
            return vec![];
        }
        let mut faces = Vec::new();
        let mut start = 0;
        for i in 1..self.progression.len() {
            if self.pairwise_distances[i - 1][i] > threshold {
                faces.push((start, i));
                start = i;
            }
        }
        faces.push((start, self.progression.len()));
        faces
    }
}

// ---------------------------------------------------------------------------
// Core functions
// ---------------------------------------------------------------------------

/// Compute the tropical (max-plus) distance between two chords.
///
/// Tries all permutations (small chords only) and returns the minimum
/// tropical distance across permutations.
///
/// `d_trop = min_{σ} max_i |c1[i] - c2[σ(i)]|`
pub fn tropical_chord_distance(c1: Vec<i32>, c2: Vec<i32>) -> f64 {
    assert_eq!(c1.len(), c2.len(), "chords must have same number of voices");
    let n = c1.len();
    if n == 0 {
        return 0.0;
    }
    let mut best = f64::MAX;
    let mut perm: Vec<usize> = (0..n).collect();
    loop {
        let d = (0..n)
            .map(|i| (c1[i] - c2[perm[i]]).abs() as f64)
            .fold(0.0_f64, f64::max);
        if d < best {
            best = d;
        }
        // next permutation
        if !next_permutation(&mut perm) {
            break;
        }
    }
    best
}

/// Find the optimal voice leading between two chords using tropical distance.
///
/// Returns the permutation and displacements that minimise the tropical
/// (max-plus) distance.
pub fn optimal_voice_leading(from_chord: Vec<i32>, to_chord: Vec<i32>) -> VoiceLeading {
    assert_eq!(
        from_chord.len(),
        to_chord.len(),
        "chords must have same number of voices"
    );
    let n = from_chord.len();
    if n == 0 {
        return VoiceLeading {
            from_chord: vec![],
            to_chord: vec![],
            displacements: vec![],
            tropical_distance: 0.0,
            permutation: vec![],
        };
    }

    let mut best_dist = f64::MAX;
    let mut best_perm: Vec<usize> = (0..n).collect();
    let mut perm: Vec<usize> = (0..n).collect();

    loop {
        let d = (0..n)
            .map(|i| (from_chord[i] - to_chord[perm[i]]).abs() as f64)
            .fold(0.0_f64, f64::max);
        if d < best_dist {
            best_dist = d;
            best_perm = perm.clone();
        }
        if !next_permutation(&mut perm) {
            break;
        }
    }

    let displacements: Vec<i32> = (0..n)
        .map(|i| to_chord[best_perm[i]] - from_chord[i])
        .collect();

    VoiceLeading {
        from_chord,
        to_chord,
        displacements,
        tropical_distance: best_dist,
        permutation: best_perm,
    }
}

/// Tropical addition of two vectors (elementwise max).
pub fn tropical_add(a: &[i32], b: &[i32]) -> Vec<i32> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(&x, &y)| x.max(y)).collect()
}

/// Tropical multiplication of two vectors (elementwise addition).
pub fn tropical_mul(a: &[i32], b: &[i32]) -> Vec<i32> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
}

/// Tropical scalar multiplication (add scalar to each element).
pub fn tropical_scalar_mul(scalar: i32, v: &[i32]) -> Vec<i32> {
    v.iter().map(|&x| x + scalar).collect()
}

/// Compute the tropical convex hull indicator: for each pair of chords,
/// whether one tropically dominates the other.
///
/// Returns a matrix `dom[i][j] = true` if chord i tropically dominates chord j.
pub fn tropical_dominance(chords: &[Vec<i32>]) -> Vec<Vec<bool>> {
    let n = chords.len();
    let mut dom = vec![vec![false; n]; n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            // i dominates j if for all k: chords[i][k] >= chords[j][k]
            let dominates = chords[i]
                .iter()
                .zip(chords[j].iter())
                .all(|(&a, &b)| a >= b);
            dom[i][j] = dominates;
        }
    }
    dom
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn next_permutation(perm: &mut [usize]) -> bool {
    let n = perm.len();
    if n < 2 {
        return false;
    }
    // find largest i such that perm[i] < perm[i+1]
    let mut i = n - 2;
    while perm[i] >= perm[i + 1] {
        if i == 0 {
            return false;
        }
        i -= 1;
    }
    // find largest j > i such that perm[i] < perm[j]
    let mut j = n - 1;
    while perm[i] >= perm[j] {
        j -= 1;
    }
    perm.swap(i, j);
    perm[i + 1..].reverse();
    true
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tropical_chord_distance_identity() {
        let c = vec![60, 64, 67]; // C major
        assert_eq!(tropical_chord_distance(c.clone(), c), 0.0);
    }

    #[test]
    fn test_tropical_chord_distance_c_major_to_c_minor() {
        let c_maj = vec![60, 64, 67];
        let c_min = vec![60, 63, 67];
        let d = tropical_chord_distance(c_maj, c_min);
        // |64-63| = 1 is the max displacement
        assert_eq!(d, 1.0);
    }

    #[test]
    fn test_optimal_voice_leading_same_chord() {
        let c = vec![60, 64, 67];
        let vl = optimal_voice_leading(c.clone(), c);
        assert_eq!(vl.tropical_distance, 0.0);
        assert!(vl.displacements.iter().all(|&d| d == 0));
    }

    #[test]
    fn test_optimal_voice_leading_c_to_f() {
        // C major [60,64,67] → F major [65,69,72] — all voices move up
        let vl = optimal_voice_leading(vec![60, 64, 67], vec![65, 69, 72]);
        assert_eq!(vl.maximal_displacement(), 5);
        assert_eq!(vl.tropical_distance, 5.0);
    }

    #[test]
    fn test_voice_leading_displacements() {
        let vl = optimal_voice_leading(vec![0, 4, 7], vec![0, 3, 7]);
        assert_eq!(vl.tropical_distance, 1.0);
        assert!(vl.displacements.contains(&-1));
    }

    #[test]
    fn test_tropical_chord_space_distance() {
        let c1 = TropicalChordSpace::new(vec![60, 64, 67]);
        let c2 = TropicalChordSpace::new(vec![60, 63, 67]);
        assert_eq!(c1.tropical_distance_to(&c2), 1.0);
    }

    #[test]
    fn test_tropical_chord_space_project() {
        let cs = TropicalChordSpace::new(vec![60, 64, 67]);
        let proj = cs.project(&[0, 2]);
        assert_eq!(proj.pitches, vec![60, 67]);
    }

    #[test]
    fn test_harmonic_analysis_total_length() {
        let analysis = TropicalHarmonicAnalysis::from_progression(vec![
            vec![60, 64, 67], // C
            vec![60, 63, 67], // Cm
            vec![60, 64, 67], // C
        ]);
        // C→Cm = 1, Cm→C = 1
        assert_eq!(analysis.total_tropical_length(), 2.0);
        assert_eq!(analysis.max_voice_leading_distance(), 1.0);
    }

    #[test]
    fn test_polytope_face_decomposition() {
        let analysis = TropicalHarmonicAnalysis::from_progression(vec![
            vec![60, 64, 67], // C
            vec![60, 63, 67], // Cm (dist 1)
            vec![55, 59, 62], // G (large jump)
            vec![55, 59, 62], // G (dist 0)
        ]);
        // threshold 2.0: C→Cm (1) ok, Cm→G (>2) split, G→G (0) ok
        let faces = analysis.decompose_polytope_faces(2.0);
        assert!(faces.len() >= 2);
    }

    #[test]
    fn test_tropical_add() {
        let a = vec![1, 4, 2];
        let b = vec![3, 2, 5];
        assert_eq!(tropical_add(&a, &b), vec![3, 4, 5]);
    }

    #[test]
    fn test_tropical_mul() {
        let a = vec![1, 4, 2];
        let b = vec![3, 2, 5];
        assert_eq!(tropical_mul(&a, &b), vec![4, 6, 7]);
    }

    #[test]
    fn test_tropical_scalar_mul() {
        assert_eq!(tropical_scalar_mul(3, &[1, 4, 2]), vec![4, 7, 5]);
    }

    #[test]
    fn test_tropical_dominance() {
        let chords = vec![vec![1, 2], vec![3, 4], vec![0, 1]];
        let dom = tropical_dominance(&chords);
        assert!(dom[1][0]); // [3,4] dominates [1,2]
        assert!(dom[1][2]); // [3,4] dominates [0,1]
        assert!(!dom[0][1]); // [1,2] does not dominate [3,4]
    }

    #[test]
    fn test_empty_chords() {
        let vl = optimal_voice_leading(vec![], vec![]);
        assert_eq!(vl.tropical_distance, 0.0);
        assert!(vl.displacements.is_empty());
    }
}
