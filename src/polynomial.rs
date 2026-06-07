use serde::{Deserialize, Serialize};
use crate::monomial::TropicalMonomial;
use crate::polytope::NewtonPolytope;

/// A tropical polynomial: the max of several tropical monomials.
///
/// `f(x) = max_i (coeff_i + Σ_j exponents_{ij} · x_j)`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TropicalPolynomial {
    pub monomials: Vec<TropicalMonomial>,
}

impl TropicalPolynomial {
    /// Create a polynomial from monomials, normalizing exponents to the correct dimension.
    pub fn new(monomials: Vec<TropicalMonomial>) -> Self {
        let nvar = monomials.iter().map(|m| m.num_variables()).max().unwrap_or(0);
        let monomials = monomials
            .into_iter()
            .map(|mut m| {
                m.exponents.resize(nvar, 0);
                m
            })
            .collect();
        Self { monomials }
    }

    /// Number of monomials.
    pub fn len(&self) -> usize {
        self.monomials.len()
    }

    /// Is the polynomial empty?
    pub fn is_empty(&self) -> bool {
        self.monomials.is_empty()
    }

    /// Number of variables (from the first monomial; all should agree).
    pub fn num_variables(&self) -> usize {
        self.monomials.first().map(|m| m.num_variables()).unwrap_or(0)
    }

    /// Total degree: max degree among monomials.
    pub fn degree(&self) -> u32 {
        self.monomials.iter().map(|m| m.degree()).max().unwrap_or(0)
    }

    /// Evaluate: `max_i monomial_i(point)`.
    pub fn evaluate(&self, point: &[f64]) -> f64 {
        self.monomials
            .iter()
            .map(|m| m.evaluate(point))
            .fold(f64::NEG_INFINITY, |a, b| a.max(b))
    }

    /// Indices of monomials achieving the maximum value at `point`.
    pub fn active_monomials(&self, point: &[f64]) -> Vec<usize> {
        let vals: Vec<f64> = self.monomials.iter().map(|m| m.evaluate(point)).collect();
        let max_val = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        vals.iter()
            .enumerate()
            .filter(|(_, v)| (**v - max_val).abs() < 1e-10)
            .map(|(i, _)| i)
            .collect()
    }

    /// A point is *smooth* if exactly 2 monomials are active.
    pub fn is_smooth_point(&self, point: &[f64]) -> bool {
        self.active_monomials(point).len() == 2
    }

    /// A point is on the *corner locus* if ≥ 2 monomials are active.
    pub fn corner_locus(&self, point: &[f64]) -> bool {
        self.active_monomials(point).len() >= 2
    }

    /// Compute the Newton polytope (convex hull of exponent vectors).
    pub fn newton_polytope(&self) -> NewtonPolytope {
        let points: Vec<Vec<u32>> = self.monomials.iter().map(|m| m.exponents.clone()).collect();
        NewtonPolytope::from_points(&points)
    }

    /// Remove duplicate monomials (same exponent vector, keep the one with larger coeff).
    pub fn simplify(&mut self) {
        use std::collections::HashMap;
        let mut best: HashMap<Vec<u32>, f64> = HashMap::new();
        for m in &self.monomials {
            best.entry(m.exponents.clone())
                .and_modify(|c| {
                    if m.coeff > *c {
                        *c = m.coeff;
                    }
                })
                .or_insert(m.coeff);
        }
        self.monomials = best
            .into_iter()
            .map(|(exp, coeff)| TropicalMonomial::new(coeff, exp))
            .collect();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_linear_2var() -> TropicalPolynomial {
        // max(0 + x, 0 + y, 0) — i.e. max(x, y, 0)
        TropicalPolynomial::new(vec![
            TropicalMonomial::new(0.0, vec![1, 0]),
            TropicalMonomial::new(0.0, vec![0, 1]),
            TropicalMonomial::constant(0.0),
        ])
    }

    #[test]
    fn test_evaluate_simple() {
        let p = make_linear_2var();
        // max(3, 5, 0) = 5
        assert_eq!(p.evaluate(&[3.0, 5.0]), 5.0);
        // max(-1, -2, 0) = 0
        assert_eq!(p.evaluate(&[-1.0, -2.0]), 0.0);
    }

    #[test]
    fn test_active_monomials_corner() {
        let p = make_linear_2var();
        // At (0, 0): all three monomials give 0
        let active = p.active_monomials(&[0.0, 0.0]);
        assert_eq!(active.len(), 3);
    }

    #[test]
    fn test_active_monomials_generic() {
        let p = make_linear_2var();
        // At (5, 3): only x=5 is max
        let active = p.active_monomials(&[5.0, 3.0]);
        assert_eq!(active, vec![0]);
    }

    #[test]
    fn test_corner_locus() {
        let p = make_linear_2var();
        assert!(p.corner_locus(&[0.0, 0.0]));
        assert!(!p.corner_locus(&[5.0, 3.0]));
    }

    #[test]
    fn test_smooth_point() {
        let p = make_linear_2var();
        // At (0, -1): max(0, -1, 0) = 0, achieved by monomials 0 and 2
        assert!(p.is_smooth_point(&[0.0, -1.0]));
        // At (0,0): 3 monomials active, not smooth
        assert!(!p.is_smooth_point(&[0.0, 0.0]));
    }

    #[test]
    fn test_degree() {
        let p = TropicalPolynomial::new(vec![
            TropicalMonomial::new(0.0, vec![2, 1]),
            TropicalMonomial::new(1.0, vec![1, 0]),
            TropicalMonomial::constant(0.0),
        ]);
        assert_eq!(p.degree(), 3);
    }

    #[test]
    fn test_quadratic_evaluate() {
        // max(x², y², 0) — the tropical conic
        let p = TropicalPolynomial::new(vec![
            TropicalMonomial::new(0.0, vec![2, 0]),
            TropicalMonomial::new(0.0, vec![0, 2]),
            TropicalMonomial::constant(0.0),
        ]);
        // At (3, 1): max(6, 2, 0) = 6
        assert_eq!(p.evaluate(&[3.0, 1.0]), 6.0);
    }

    #[test]
    fn test_newton_polytope() {
        let p = TropicalPolynomial::new(vec![
            TropicalMonomial::new(0.0, vec![2, 0]),
            TropicalMonomial::new(0.0, vec![0, 2]),
            TropicalMonomial::new(0.0, vec![1, 1]),
            TropicalMonomial::constant(0.0),
        ]);
        let np = p.newton_polytope();
        assert!(np.vertices.len() >= 3);
    }

    #[test]
    fn test_simplify() {
        let mut p = TropicalPolynomial::new(vec![
            TropicalMonomial::new(1.0, vec![1, 0]),
            TropicalMonomial::new(3.0, vec![1, 0]), // duplicate exponent, higher coeff
        ]);
        p.simplify();
        assert_eq!(p.len(), 1);
        assert_eq!(p.monomials[0].coeff, 3.0);
    }

    #[test]
    fn test_one_variable_polynomial() {
        // max(0, x, 2x) — tropical polynomial in 1 variable
        let p = TropicalPolynomial::new(vec![
            TropicalMonomial::constant(0.0),
            TropicalMonomial::new(0.0, vec![1]),
            TropicalMonomial::new(0.0, vec![2]),
        ]);
        // At x=3: max(0, 3, 6) = 6
        assert_eq!(p.evaluate(&[3.0]), 6.0);
        // At x=-1: max(0, -1, -2) = 0
        assert_eq!(p.evaluate(&[-1.0]), 0.0);
        // Corner at x=0: max(0, 0, 0) = 0, all active
        assert!(p.corner_locus(&[0.0]));
    }

    #[test]
    fn test_num_variables() {
        let p = TropicalPolynomial::new(vec![
            TropicalMonomial::new(0.0, vec![1, 0, 2]),
            TropicalMonomial::constant(1.0),
        ]);
        assert_eq!(p.num_variables(), 3);
    }
}
