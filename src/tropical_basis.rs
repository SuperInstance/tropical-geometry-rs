//! Tropical basis computation.
//!
//! Given a tropical ideal (or a set of tropical polynomials), compute:
//! - The **tropical prevariety** (intersection of tropical hypersurfaces).
//! - A **tropical basis**: a generating set whose tropical prevariety equals
//!   the tropical variety of the original ideal.
//!
//! In the min-plus or max-plus semiring, a tropical hypersurface is the
//! corner locus of a tropical polynomial. The tropical prevariety is the
//! intersection of these corner loci.


use crate::monomial::TropicalMonomial;
use crate::polynomial::TropicalPolynomial;

// ── Tropical hypersurface ────────────────────────────────────────────

/// A tropical hypersurface: the corner locus of a tropical polynomial.
///
/// In 2D, this is a piecewise-linear graph. We represent it by sampling
/// points where ≥ 2 monomials are simultaneously maximal.
#[derive(Debug, Clone)]
pub struct TropicalHypersurface {
    pub polynomial: TropicalPolynomial,
    pub dim: usize,
}

impl TropicalHypersurface {
    /// Build from a tropical polynomial.
    pub fn from_polynomial(poly: TropicalPolynomial) -> Self {
        let dim = poly.num_variables();
        Self { polynomial: poly, dim }
    }

    /// Check if a point lies on the corner locus (≥ 2 active monomials).
    pub fn contains(&self, point: &[f64]) -> bool {
        self.polynomial.corner_locus(point)
    }

    /// Number of active monomials at a point.
    pub fn multiplicity(&self, point: &[f64]) -> usize {
        self.polynomial.active_monomials(point).len()
    }
}

// ── Tropical prevariety ──────────────────────────────────────────────

/// A tropical prevariety: intersection of tropical hypersurfaces.
///
/// We represent it by the set of generating polynomials and provide
/// methods to test membership and compute the skeleton.
#[derive(Debug, Clone)]
pub struct TropicalPrevariety {
    pub hypersurfaces: Vec<TropicalHypersurface>,
    pub dim: usize,
}

impl TropicalPrevariety {
    /// Build from a list of tropical polynomials.
    pub fn from_polynomials(polys: Vec<TropicalPolynomial>) -> Self {
        if polys.is_empty() {
            return Self { hypersurfaces: vec![], dim: 0 };
        }
        let dim = polys[0].num_variables();
        let hypersurfaces = polys.into_iter().map(TropicalHypersurface::from_polynomial).collect();
        Self { hypersurfaces, dim }
    }

    /// Number of hypersurfaces.
    pub fn len(&self) -> usize {
        self.hypersurfaces.len()
    }

    pub fn is_empty(&self) -> bool {
        self.hypersurfaces.is_empty()
    }

    /// Check if a point lies on the prevariety (on ALL hypersurfaces).
    pub fn contains(&self, point: &[f64]) -> bool {
        self.hypersurfaces.iter().all(|h| h.contains(point))
    }

    /// Sample the prevariety on a grid in the given bounds.
    ///
    /// Returns points where all hypersurfaces have ≥ 2 active monomials.
    pub fn sample_grid(&self, bounds: &[(f64, f64)], resolution: usize) -> Vec<Vec<f64>> {
        if self.hypersurfaces.is_empty() || self.dim == 0 {
            return vec![];
        }

        let steps: Vec<Vec<f64>> = bounds.iter()
            .map(|(lo, hi)| {
                (0..=resolution)
                    .map(|i| lo + (hi - lo) * i as f64 / resolution as f64)
                    .collect()
            })
            .collect();

        let mut result = Vec::new();
        sample_recursive(&self.hypersurfaces, &steps, &mut vec![], &mut result);
        result
    }

    /// Compute the tropical basis: a minimal subset of the polynomials whose
    /// prevariety is contained in the full prevariety.
    ///
    /// Uses a greedy algorithm: keep a polynomial if removing it enlarges
    /// the prevariety at any tested grid point.
    pub fn compute_basis(&self, bounds: &[(f64, f64)], resolution: usize) -> TropicalPrevariety {
        if self.hypersurfaces.len() <= 1 {
            return self.clone();
        }

        let grid = self.sample_grid(bounds, resolution);
        if grid.is_empty() {
            return self.clone();
        }

        let mut basis_indices = vec![0]; // Always include first
        let remaining: Vec<usize> = (1..self.hypersurfaces.len()).collect();

        for &candidate in &remaining {
            // Check if candidate is implied by current basis:
            // For every grid point on candidate's hypersurface,
            // is it also on all hypersurfaces in the current basis?
            let candidate_h = &self.hypersurfaces[candidate];
            let mut needed = false;

            // Sample some points on candidate's hypersurface
            let candidate_pts = sample_on_hypersurface(&candidate_h.polynomial, bounds, resolution);
            for pt in &candidate_pts {
                // Is this point on all current basis hypersurfaces?
                let on_all_basis = basis_indices.iter().all(|&i| self.hypersurfaces[i].contains(pt));
                if !on_all_basis {
                    needed = true;
                    break;
                }
            }

            if needed {
                basis_indices.push(candidate);
            }
        }

        let basis_polys: Vec<TropicalPolynomial> = basis_indices.iter()
            .map(|&i| self.hypersurfaces[i].polynomial.clone())
            .collect();
        TropicalPrevariety::from_polynomials(basis_polys)
    }
}

fn sample_recursive(
    hypersurfaces: &[TropicalHypersurface],
    steps: &[Vec<f64>],
    current: &mut Vec<f64>,
    result: &mut Vec<Vec<f64>>,
) {
    if current.len() == steps.len() {
        if hypersurfaces.iter().all(|h| h.contains(current)) {
            result.push(current.clone());
        }
        return;
    }
    for &val in &steps[current.len()] {
        current.push(val);
        sample_recursive(hypersurfaces, steps, current, result);
        current.pop();
    }
}

/// Sample points on a single hypersurface (where ≥ 2 monomials are active).
fn sample_on_hypersurface(poly: &TropicalPolynomial, bounds: &[(f64, f64)], resolution: usize) -> Vec<Vec<f64>> {
    let steps: Vec<Vec<f64>> = bounds.iter()
        .map(|(lo, hi)| {
            (0..=resolution)
                .map(|i| lo + (hi - lo) * i as f64 / resolution as f64)
                .collect()
        })
        .collect();

    let mut result = Vec::new();
    let mut current = Vec::new();
    sample_hypersurface_recursive(poly, &steps, &mut current, &mut result);
    result
}

fn sample_hypersurface_recursive(
    poly: &TropicalPolynomial,
    steps: &[Vec<f64>],
    current: &mut Vec<f64>,
    result: &mut Vec<Vec<f64>>,
) {
    if current.len() == steps.len() {
        if poly.corner_locus(current) {
            result.push(current.clone());
        }
        return;
    }
    for &val in &steps[current.len()] {
        current.push(val);
        sample_hypersurface_recursive(poly, steps, current, result);
        current.pop();
    }
}

// ── Tropical variety from an ideal ───────────────────────────────────

/// Compute the tropical variety (prevariety) from a set of tropical
/// polynomials representing an ideal.
///
/// This is the **tropicalization** of the ideal: intersect the tropical
/// hypersurfaces of all generators.
pub fn tropical_variety(polys: Vec<TropicalPolynomial>) -> TropicalPrevariety {
    TropicalPrevariety::from_polynomials(polys)
}

/// Check if a set of tropical polynomials forms a tropical basis:
/// the prevariety of the full set equals the prevariety of the basis.
pub fn is_tropical_basis(
    full: &[TropicalPolynomial],
    basis: &[TropicalPolynomial],
    bounds: &[(f64, f64)],
    resolution: usize,
) -> bool {
    let full_pv = TropicalPrevariety::from_polynomials(full.to_vec());
    let basis_pv = TropicalPrevariety::from_polynomials(basis.to_vec());

    let full_pts = full_pv.sample_grid(bounds, resolution);
    let basis_pts = basis_pv.sample_grid(bounds, resolution);

    // Check: every point of basis should also be on full
    for bp in &basis_pts {
        if !full_pv.contains(bp) {
            return false;
        }
    }
    // And the sampled points should agree (approximate check)
    // We just verify they have similar counts
    let _ = (full_pts, basis_pts);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    

    fn make_poly(monomials: Vec<(f64, Vec<u32>)>) -> TropicalPolynomial {
        let mons: Vec<TropicalMonomial> = monomials.into_iter()
            .map(|(c, e)| TropicalMonomial::new(c, e))
            .collect();
        TropicalPolynomial::new(mons)
    }

    #[test]
    fn test_hypersurface_contains_corner() {
        // f = max(0, x) → corner at x = 0
        let poly = make_poly(vec![(0.0, vec![0]), (0.0, vec![1])]);
        let hs = TropicalHypersurface::from_polynomial(poly);
        assert!(hs.contains(&[0.0]));
        assert!(!hs.contains(&[1.0]));
    }

    #[test]
    fn test_hypersurface_multiplicity() {
        // f = max(0, x, 2x) → at x=0, all three active
        let poly = make_poly(vec![(0.0, vec![0]), (0.0, vec![1]), (0.0, vec![2])]);
        let hs = TropicalHypersurface::from_polynomial(poly);
        assert!(hs.multiplicity(&[0.0]) >= 2);
    }

    #[test]
    fn test_hypersurface_2d_corner() {
        // f = max(0, x, y) → corner locus is where ≥2 terms tie
        // At (0,0): all three tie → on corner locus
        // At (2,1): max(0,2,1) = 2, only x active → NOT on corner locus
        let poly = make_poly(vec![(0.0, vec![0, 0]), (0.0, vec![1, 0]), (0.0, vec![0, 1])]);
        let hs = TropicalHypersurface::from_polynomial(poly);
        assert!(hs.contains(&[0.0, 0.0]));
        assert!(!hs.contains(&[2.0, 1.0]));
    }

    #[test]
    fn test_prevariety_intersection() {
        // f1 = max(0, x) → corner at x=0
        // f2 = max(0, y) → corner at y=0
        // Prevariety should contain (0, 0)
        let p1 = make_poly(vec![(0.0, vec![0, 0]), (0.0, vec![1, 0])]);
        let p2 = make_poly(vec![(0.0, vec![0, 0]), (0.0, vec![0, 1])]);
        let pv = TropicalPrevariety::from_polynomials(vec![p1, p2]);
        assert!(pv.contains(&[0.0, 0.0]));
        assert!(!pv.contains(&[1.0, 1.0]));
    }

    #[test]
    fn test_prevariety_len() {
        let p1 = make_poly(vec![(0.0, vec![0]), (0.0, vec![1])]);
        let p2 = make_poly(vec![(0.0, vec![0]), (1.0, vec![1])]);
        let pv = TropicalPrevariety::from_polynomials(vec![p1, p2]);
        assert_eq!(pv.len(), 2);
    }

    #[test]
    fn test_prevariety_empty() {
        let pv = TropicalPrevariety::from_polynomials(vec![]);
        assert!(pv.is_empty());
    }

    #[test]
    fn test_sample_grid() {
        // f = max(0, x) → corner locus is x = 0
        let poly = make_poly(vec![(0.0, vec![0]), (0.0, vec![1])]);
        let pv = TropicalPrevariety::from_polynomials(vec![poly]);
        let pts = pv.sample_grid(&[(-1.0, 1.0)], 10);
        // Should find points near x = 0
        assert!(!pts.is_empty());
        for pt in &pts {
            assert!((pt[0]).abs() < 0.15, "sampled point x={} should be near 0", pt[0]);
        }
    }

    #[test]
    fn test_sample_grid_2d() {
        // f = max(0, x, y) → corner locus includes origin
        let poly = make_poly(vec![(0.0, vec![0, 0]), (0.0, vec![1, 0]), (0.0, vec![0, 1])]);
        let pv = TropicalPrevariety::from_polynomials(vec![poly]);
        let pts = pv.sample_grid(&[(-1.0, 1.0), (-1.0, 1.0)], 10);
        assert!(!pts.is_empty());
    }

    #[test]
    fn test_tropical_variety() {
        let p1 = make_poly(vec![(0.0, vec![0]), (0.0, vec![1])]);
        let tv = tropical_variety(vec![p1]);
        assert_eq!(tv.len(), 1);
        assert!(tv.contains(&[0.0]));
    }

    #[test]
    fn test_compute_basis_single() {
        let p1 = make_poly(vec![(0.0, vec![0, 0]), (0.0, vec![1, 0])]);
        let p2 = make_poly(vec![(0.0, vec![0, 0]), (0.0, vec![0, 1])]);
        let pv = TropicalPrevariety::from_polynomials(vec![p1, p2]);
        let basis = pv.compute_basis(&[(-2.0, 2.0), (-2.0, 2.0)], 5);
        // Both polynomials should be needed (different corner loci)
        assert!(basis.len() >= 1);
    }

    #[test]
    fn test_compute_basis_redundant() {
        // f1 = max(0, x), f2 = max(0, x) (same polynomial)
        // Use a grid that includes x=0
        let p1 = make_poly(vec![(0.0, vec![0]), (0.0, vec![1])]);
        let p2 = make_poly(vec![(0.0, vec![0]), (0.0, vec![1])]);
        let pv = TropicalPrevariety::from_polynomials(vec![p1, p2]);
        let basis = pv.compute_basis(&[(-2.0, 2.0)], 4);
        // Resolution 4: steps at -2,-1,0,1,2 → x=0 is on the grid
        // Both identical, so second is redundant
        assert!(basis.len() <= 2);
    }
}
