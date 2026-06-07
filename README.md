# lau-tropical-geometry

Tropical geometry replaces addition with max and multiplication with addition. The result: polynomial root counting becomes piecewise-linear geometry, and algebraic curves become skeletons of polyhedral complexes.

Tropical geometry freezes the continuous into discrete — like a hermit crab's shell preserving the shape of every home it's ever had.

## The math in 60 seconds

The **tropical semiring** is (ℝ ∪ {-∞}, ⊕, ⊗) where ⊕ = max and ⊗ = +. A tropical polynomial like `max(x, y, x+y-3)` is a piecewise-linear convex function. The locus where two or more monomials tie — the **tropical hypersurface** — is a polyhedral complex of codimension 1.

Key results this crate implements:

- **Newton polytopes:** the convex hull of exponent vectors, with Pick's theorem for area
- **Tropical Bézout:** two tropical curves of degree d₁, d₂ intersect in d₁·d₂ points (counting multiplicity)
- **Baker's genus formula:** g = (d-1)(d-2)/2 for a smooth tropical curve of degree d
- **Stable intersection:** the limit of intersections under small perturbations

References: Maclagan & Sturmfels, *Introduction to Tropical Geometry* (2015)

## Quick start

```rust
use lau_tropical_geometry::{TropicalSemiring, TropicalPolynomial, NewtonPolytope};

// Create a tropical polynomial: max(x, y, x+y-3)
let trop = TropicalSemiring::max_plus();
let poly = TropicalPolynomial::from_terms(&[
    (1.0, vec![1, 0]),  // x
    (1.0, vec![0, 1]),  // y  
    (-3.0, vec![1, 1]), // x+y-3
]);

// Evaluate at a point
let val = poly.evaluate(&[2.0, 1.0]); // max(2, 1, 2+1-3) = 2

// Build Newton polytope
let newton = NewtonPolytope::from_polynomial(&poly);
let area = newton.area(); // Pick's theorem

// Find the tropical curve (where two monomials tie)
let curve = poly.tropical_curve();
let vertices = curve.vertices();
let edges = curve.edges();
```

## Key types

| Type | What it is |
|------|-----------|
| `TropicalSemiring` | The max-plus (or min-plus) semiring with identity elements |
| `TropicalPolynomial` | A piecewise-linear convex function from monomial terms |
| `NewtonPolytope` | Convex hull of exponent vectors with lattice point counting |
| `TropicalCurve` | The corner locus — polyhedral complex where monomials tie |
| `TropicalRationalMap` | Ratio of tropical polynomials (models ReLU networks) |
| `TropicalIntersection` | Stable intersection of tropical hypersurfaces |

## Contributing

[Open an issue](https://github.com/SuperInstance/lau-tropical-theory/issues) or PR. Interesting directions:

- Mixed volumes and Bernstein's theorem
- Tropical linear spaces and matroids
- Connections to ReLU network complexity
