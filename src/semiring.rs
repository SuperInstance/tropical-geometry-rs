use serde::{Deserialize, Serialize};

/// The **max-plus tropical semiring** `(ℝ ∪ {−∞}, max, +)`.
///
/// Generic over any ordered type `T`; defaults to `f64`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TropicalSemiring<T = f64>(std::marker::PhantomData<T>);

impl<T> Default for TropicalSemiring<T> {
    fn default() -> Self {
        Self(std::marker::PhantomData)
    }
}

impl TropicalSemiring<f64> {
    /// Tropical additive identity: −∞.
    #[inline]
    pub fn zero() -> f64 {
        f64::NEG_INFINITY
    }

    /// Tropical multiplicative identity: 0.
    #[inline]
    pub fn one() -> f64 {
        0.0
    }

    /// Tropical addition: `max(a, b)`.
    #[inline]
    pub fn add(a: f64, b: f64) -> f64 {
        a.max(b)
    }

    /// Tropical multiplication: `a + b`.
    #[inline]
    pub fn mul(a: f64, b: f64) -> f64 {
        a + b
    }

    /// Tropical exponentiation: `n · a`.
    #[inline]
    pub fn pow(a: f64, n: u32) -> f64 {
        (n as f64) * a
    }

    /// Verify idempotency: `max(a, a) == a`.
    pub fn verify_idempotent(a: f64) -> bool {
        Self::add(a, a) == a
    }

    /// Verify distributivity: `max(a, b + c) == max(a + c, b + c)`.
    ///
    /// This checks that tropical multiplication distributes over tropical addition:
    /// `a ⊗ (b ⊕ c) = (a ⊗ b) ⊕ (a ⊗ c)`.
    pub fn verify_distributivity(a: f64, b: f64, c: f64) -> bool {
        let lhs = Self::add(Self::mul(a, b), Self::mul(a, c));
        let rhs = Self::mul(a, Self::add(b, c));
        (lhs - rhs).abs() < 1e-10
    }

    /// Verify associativity of tropical addition: `max(max(a,b),c) == max(a,max(b,c))`.
    pub fn verify_add_associative(a: f64, b: f64, c: f64) -> bool {
        Self::add(Self::add(a, b), c) == Self::add(a, Self::add(b, c))
    }

    /// Verify associativity of tropical multiplication: `(a+b)+c == a+(b+c)`.
    pub fn verify_mul_associative(a: f64, b: f64, c: f64) -> bool {
        Self::mul(Self::mul(a, b), c) == Self::mul(a, Self::mul(b, c))
    }

    /// Verify commutativity of tropical addition: `max(a,b) == max(b,a)`.
    pub fn verify_add_commutative(a: f64, b: f64) -> bool {
        Self::add(a, b) == Self::add(b, a)
    }

    /// Verify absorption: `max(a, a+b) == a` when b ≤ 0 (tropical zero).
    /// In general: `a ⊕ (a ⊗ b) = a` when b ≤ tropical one (0).
    pub fn verify_absorption(a: f64, b: f64) -> bool {
        let lhs = Self::add(a, Self::mul(a, b));
        // lhs = max(a, a+b), which equals a iff b ≤ 0
        if b <= 0.0 {
            (lhs - a).abs() < 1e-10
        } else {
            // not guaranteed
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_is_neg_inf() {
        assert_eq!(TropicalSemiring::zero(), f64::NEG_INFINITY);
    }

    #[test]
    fn test_one_is_zero() {
        assert_eq!(TropicalSemiring::one(), 0.0);
    }

    #[test]
    fn test_add_max() {
        assert_eq!(TropicalSemiring::add(1.0, 2.0), 2.0);
        assert_eq!(TropicalSemiring::add(-3.0, 5.0), 5.0);
        assert_eq!(TropicalSemiring::add(0.0, 0.0), 0.0);
    }

    #[test]
    fn test_mul_sum() {
        assert_eq!(TropicalSemiring::mul(1.0, 2.0), 3.0);
        assert_eq!(TropicalSemiring::mul(-1.0, 5.0), 4.0);
        assert_eq!(TropicalSemiring::mul(0.0, 100.0), 100.0);
    }

    #[test]
    fn test_pow() {
        assert_eq!(TropicalSemiring::pow(3.0, 4), 12.0);
        assert_eq!(TropicalSemiring::pow(-2.0, 3), -6.0);
        assert_eq!(TropicalSemiring::pow(1.5, 0), 0.0);
    }

    #[test]
    fn test_idempotency() {
        for a in [-100.0, -1.0, 0.0, 1.0, 42.0, 1e10] {
            assert!(TropicalSemiring::verify_idempotent(a), "failed for a={a}");
        }
    }

    #[test]
    fn test_distributivity() {
        for (a, b, c) in [
            (1.0, 2.0, 3.0),
            (-5.0, 0.0, 10.0),
            (3.5, -2.0, 1.0),
            (0.0, 0.0, 0.0),
        ] {
            assert!(
                TropicalSemiring::verify_distributivity(a, b, c),
                "failed for ({a}, {b}, {c})"
            );
        }
    }

    #[test]
    fn test_add_associative() {
        for (a, b, c) in [(1.0, 2.0, 3.0), (-5.0, 0.0, 10.0)] {
            assert!(TropicalSemiring::verify_add_associative(a, b, c));
        }
    }

    #[test]
    fn test_mul_associative() {
        for (a, b, c) in [(1.0, 2.0, 3.0), (-5.0, 0.0, 10.0)] {
            assert!(TropicalSemiring::verify_mul_associative(a, b, c));
        }
    }

    #[test]
    fn test_add_commutative() {
        assert!(TropicalSemiring::verify_add_commutative(3.0, 7.0));
        assert!(TropicalSemiring::verify_add_commutative(-1.0, 100.0));
    }

    #[test]
    fn test_add_with_zero() {
        // max(a, -∞) = a
        assert_eq!(TropicalSemiring::add(5.0, TropicalSemiring::zero()), 5.0);
        assert_eq!(TropicalSemiring::add(TropicalSemiring::zero(), -3.0), -3.0);
    }

    #[test]
    fn test_mul_with_one() {
        // a + 0 = a
        assert_eq!(TropicalSemiring::mul(5.0, TropicalSemiring::one()), 5.0);
        assert_eq!(TropicalSemiring::mul(TropicalSemiring::one(), -3.0), -3.0);
    }

    #[test]
    fn test_mul_with_zero() {
        // a + (-∞) = -∞
        assert_eq!(TropicalSemiring::mul(5.0, TropicalSemiring::zero()), f64::NEG_INFINITY);
    }
}
