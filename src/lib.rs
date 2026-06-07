#![deny(unsafe_code)]
//! # lau-tropical-geometry
//!
//! Research-grade tropical geometry over the min-plus and max-plus semirings.
//!
//! Tropical geometry replaces `+` with `max` (or `min`) and `×` with `+`.
//! The tropical semiring is `(ℝ ∪ {−∞}, max, +)`. A tropical polynomial in
//! *n* variables is a piecewise-linear convex function whose corner locus
//! (where ≥ 2 monomials simultaneously attain the maximum) is the **tropical
//! hypersurface** — a polyhedral complex of codimension 1.

pub mod semiring;
pub mod monomial;
pub mod polynomial;
pub mod polytope;
pub mod curve;
pub mod rational_map;
pub mod intersection;

pub use semiring::TropicalSemiring;
pub use monomial::TropicalMonomial;
pub use polynomial::TropicalPolynomial;
pub use polytope::{NewtonPolytope, Facet};
pub use curve::{TropicalCurve, TropicalEdge, TropicalVertex};
pub use rational_map::TropicalRationalMap;
pub use intersection::TropicalIntersection;
