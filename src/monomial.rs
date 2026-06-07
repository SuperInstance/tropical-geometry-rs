use serde::{Deserialize, Serialize};
use crate::semiring::TropicalSemiring;

/// A single tropical monomial: `coeff + Σ exponents[i] · point[i]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TropicalMonomial {
    pub coeff: f64,
    pub exponents: Vec<u32>,
}

impl TropicalMonomial {
    /// Create a new tropical monomial with the given coefficient and exponents.
    pub fn new(coeff: f64, exponents: Vec<u32>) -> Self {
        Self { coeff, exponents }
    }

    /// Constant monomial (no variables).
    pub fn constant(coeff: f64) -> Self {
        Self {
            coeff,
            exponents: vec![],
        }
    }

    /// Evaluate at a point: `coeff + Σ exponents[i] * point[i]`.
    pub fn evaluate(&self, point: &[f64]) -> f64 {
        TropicalSemiring::mul(
            self.coeff,
            self.exponents
                .iter()
                .zip(point.iter())
                .map(|(e, p)| TropicalSemiring::pow(*p, *e))
                .fold(TropicalSemiring::one(), TropicalSemiring::mul),
        )
    }

    /// Total degree: sum of all exponents.
    pub fn degree(&self) -> u32 {
        self.exponents.iter().sum()
    }

    /// Number of variables.
    pub fn num_variables(&self) -> usize {
        self.exponents.len()
    }

    /// The exponent vector as a reference.
    pub fn exponent_vector(&self) -> &[u32] {
        &self.exponents
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluate_constant() {
        let m = TropicalMonomial::constant(5.0);
        assert_eq!(m.evaluate(&[]), 5.0);
    }

    #[test]
    fn test_evaluate_single_var() {
        // 3 + 2*x, at x=4 => 3 + 2*4 = 11
        let m = TropicalMonomial::new(3.0, vec![2]);
        assert_eq!(m.evaluate(&[4.0]), 11.0);
    }

    #[test]
    fn test_evaluate_multi_var() {
        // 1 + x^2 + y^3, at (2,3) => 1 + 2*2 + 3*3 = 1+4+9 = 14
        let m = TropicalMonomial::new(1.0, vec![2, 3]);
        assert_eq!(m.evaluate(&[2.0, 3.0]), 14.0);
    }

    #[test]
    fn test_degree() {
        let m = TropicalMonomial::new(0.0, vec![2, 3, 1]);
        assert_eq!(m.degree(), 6);
    }

    #[test]
    fn test_num_variables() {
        let m = TropicalMonomial::new(0.0, vec![1, 0, 2, 3]);
        assert_eq!(m.num_variables(), 4);
    }

    #[test]
    fn test_evaluate_zero_exponents() {
        // coeff + 0*everything = coeff
        let m = TropicalMonomial::new(7.0, vec![0, 0]);
        assert_eq!(m.evaluate(&[100.0, -100.0]), 7.0);
    }

    #[test]
    fn test_evaluate_negative_point() {
        // 0 + 3*x at x=-2 => 0 + 3*(-2) = -6
        let m = TropicalMonomial::new(0.0, vec![3]);
        assert_eq!(m.evaluate(&[-2.0]), -6.0);
    }
}
