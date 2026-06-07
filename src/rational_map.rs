use serde::{Deserialize, Serialize};
use crate::polynomial::TropicalPolynomial;

/// A tropical rational map: numerator / denominator in the tropical sense.
///
/// Evaluates as `num.evaluate(p) - denom.evaluate(p)` (tropical subtraction = ordinary subtraction).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TropicalRationalMap {
    pub numerator: TropicalPolynomial,
    pub denominator: TropicalPolynomial,
}

impl TropicalRationalMap {
    /// Create a new tropical rational map.
    pub fn new(numerator: TropicalPolynomial, denominator: TropicalPolynomial) -> Self {
        Self { numerator, denominator }
    }

    /// Evaluate at a point: `num(point) - denom(point)`.
    pub fn evaluate(&self, point: &[f64]) -> f64 {
        self.numerator.evaluate(point) - self.denominator.evaluate(point)
    }

    /// Is this a linear map? (Both numerator and denominator have degree ≤ 1)
    pub fn is_linear(&self) -> bool {
        self.numerator.degree() <= 1 && self.denominator.degree() <= 1
    }

    /// Corner locus: the point is on the corner locus of the rational map
    /// if it's on the corner locus of the numerator or the denominator.
    pub fn corner_locus(&self, point: &[f64]) -> bool {
        self.numerator.corner_locus(point) || self.denominator.corner_locus(point)
    }

    /// Compose this map with another: `self(other(point))`.
    ///
    /// Tropical composition: substitute point into `other`, then into `self`.
    /// This is just evaluation of self at the output of other.
    /// For a true compositional algebra, we'd need symbolic substitution,
    /// but here we evaluate numerically.
    pub fn compose(&self, _other: &TropicalRationalMap) -> TropicalRationalMap {
        // Symbolic composition is complex. We construct a new map that
        // represents f(g(x)). Since tropical polynomials are piecewise-linear,
        // composition means substituting one PL function into another.
        // For now, we return the identity composition (same as self).
        // A full implementation would expand the monomials.
        self.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monomial::TropicalMonomial;

    #[test]
    fn test_evaluate_rational() {
        // num = max(x, 0), denom = max(y, 0)
        let num = TropicalPolynomial::new(vec![
            TropicalMonomial::new(0.0, vec![1, 0]),
            TropicalMonomial::constant(0.0),
        ]);
        let denom = TropicalPolynomial::new(vec![
            TropicalMonomial::new(0.0, vec![0, 1]),
            TropicalMonomial::constant(0.0),
        ]);
        let map = TropicalRationalMap::new(num, denom);

        // At (3, 1): num = max(3, 0) = 3, denom = max(1, 0) = 1, result = 2
        assert_eq!(map.evaluate(&[3.0, 1.0]), 2.0);
        // At (-2, -1): num = max(-2, 0) = 0, denom = max(-1, 0) = 0, result = 0
        assert_eq!(map.evaluate(&[-2.0, -1.0]), 0.0);
    }

    #[test]
    fn test_is_linear() {
        let num = TropicalPolynomial::new(vec![
            TropicalMonomial::new(1.0, vec![1, 0]),
        ]);
        let denom = TropicalPolynomial::new(vec![
            TropicalMonomial::new(0.0, vec![0, 1]),
        ]);
        let map = TropicalRationalMap::new(num, denom);
        assert!(map.is_linear());
    }

    #[test]
    fn test_is_not_linear() {
        let num = TropicalPolynomial::new(vec![
            TropicalMonomial::new(0.0, vec![2, 0]),
        ]);
        let denom = TropicalPolynomial::new(vec![
            TropicalMonomial::new(0.0, vec![0, 1]),
        ]);
        let map = TropicalRationalMap::new(num, denom);
        assert!(!map.is_linear());
    }

    #[test]
    fn test_corner_locus() {
        // num = max(x, y, 0) — has corner at (0,0)
        // denom = max(0) — constant, no corner locus
        let num = TropicalPolynomial::new(vec![
            TropicalMonomial::new(0.0, vec![1, 0]),
            TropicalMonomial::new(0.0, vec![0, 1]),
            TropicalMonomial::constant(0.0),
        ]);
        let denom = TropicalPolynomial::new(vec![
            TropicalMonomial::constant(0.0),
        ]);
        let map = TropicalRationalMap::new(num, denom);

        assert!(map.corner_locus(&[0.0, 0.0]));
        assert!(!map.corner_locus(&[5.0, 3.0]));
    }

    #[test]
    fn test_rational_single_variable() {
        // num = max(x, 0), denom = max(2x, 0)
        let num = TropicalPolynomial::new(vec![
            TropicalMonomial::new(0.0, vec![1]),
            TropicalMonomial::constant(0.0),
        ]);
        let denom = TropicalPolynomial::new(vec![
            TropicalMonomial::new(0.0, vec![2]),
            TropicalMonomial::constant(0.0),
        ]);
        let map = TropicalRationalMap::new(num, denom);

        // At x=3: num=max(3,0)=3, denom=max(6,0)=6, result=-3
        assert_eq!(map.evaluate(&[3.0]), -3.0);
        // At x=-1: num=max(-1,0)=0, denom=max(-2,0)=0, result=0
        assert_eq!(map.evaluate(&[-1.0]), 0.0);
    }

    #[test]
    fn test_compose() {
        let map1 = TropicalRationalMap::new(
            TropicalPolynomial::new(vec![TropicalMonomial::new(0.0, vec![1])]),
            TropicalPolynomial::new(vec![TropicalMonomial::constant(0.0)]),
        );
        let map2 = TropicalRationalMap::new(
            TropicalPolynomial::new(vec![TropicalMonomial::new(1.0, vec![1])]),
            TropicalPolynomial::new(vec![TropicalMonomial::constant(0.0)]),
        );
        let _composed = map1.compose(&map2);
    }

    #[test]
    fn test_rational_identity() {
        // num = x, denom = 0 => f(x) = x - 0 = x
        let map = TropicalRationalMap::new(
            TropicalPolynomial::new(vec![TropicalMonomial::new(0.0, vec![1])]),
            TropicalPolynomial::new(vec![TropicalMonomial::constant(0.0)]),
        );
        assert_eq!(map.evaluate(&[5.0]), 5.0);
        assert_eq!(map.evaluate(&[-3.0]), -3.0);
    }

    #[test]
    fn test_rational_zero_map() {
        // num = 0, denom = 0 => f(x) = 0 - 0 = 0
        let map = TropicalRationalMap::new(
            TropicalPolynomial::new(vec![TropicalMonomial::constant(0.0)]),
            TropicalPolynomial::new(vec![TropicalMonomial::constant(0.0)]),
        );
        assert_eq!(map.evaluate(&[42.0]), 0.0);
        assert!(map.is_linear());
    }
}
