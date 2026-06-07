# tropical-geometry

> Research-grade tropical geometry in Rust — min-plus/max-plus semirings, Newton polytopes, tropical curves, and stable intersection.

## What This Does

This crate implements tropical geometry over the max-plus semiring. It provides `TropicalSemiring` with verified algebraic laws, `TropicalPolynomial` for evaluating and analyzing piecewise-linear convex functions, `NewtonPolytope` for convex hulls of exponent vectors with volume computation, `TropicalCurve` for the 1-skeleton corner locus of bivariate polynomials, and `TropicalIntersection` for stable intersection and tropical Bézout verification. Tropical geometry turns algebraic problems into combinatorial ones — polynomial root counting becomes polyhedral counting, and curves become skeletons.

## Why It Matters

Classical algebraic geometry is continuous and fragile — small perturbations change root counts and intersection multiplicities. Tropical geometry freezes the continuous into discrete, preserving structural invariants under deformation. For AGI, this is a template for robust reasoning: instead of optimizing in a landscape that shifts with every hyperparameter, reason in a piecewise-linear skeleton that is stable by construction. Tropical methods also underpin modern optimization (max-plus linear control), scheduling (tropical matrix methods), and even deep learning (tropical neural networks).

## Quick Start

```bash
cargo add tropical-geometry
```

```rust
use tropical_geometry::{TropicalSemiring, TropicalPolynomial, TropicalMonomial};
use tropical_geometry::curve::TropicalCurve;
use tropical_geometry::intersection::TropicalIntersection;

fn main() {
    // Tropical polynomial: f(x,y) = max(x, y, x+y-3)
    let poly = TropicalPolynomial::new(vec![
        TropicalMonomial::new(1.0, vec![1, 0]),  // x
        TropicalMonomial::new(1.0, vec![0, 1]),  // y
        TropicalMonomial::new(-3.0, vec![1, 1]), // x+y-3
    ]);

    assert_eq!(poly.evaluate(&[2.0, 2.0]), 2.0); // max(2, 2, 1)
    assert!(poly.corner_locus(&[1.5, 1.5]));      // x = y = x+y-3 here

    // Tropical curve skeleton
    let curve = TropicalCurve::new(poly);
    println!("Curve has {} edges and {} vertices", curve.edges.len(), curve.curve_vertices.len());

    // Verify semiring laws
    assert!(TropicalSemiring::verify_distributivity(1.0, 2.0, 3.0));
}
```

## Architecture

| Module | Purpose |
|--------|---------|
| `semiring` | Max-plus tropical semiring with additive identity −∞, multiplicative identity 0, and verified algebraic laws |
| `monomial` | Tropical monomials `c + Σ aᵢxᵢ` with evaluation and degree |
| `polynomial` | Piecewise-linear convex functions as max of monomials; corner locus and smooth-point detection |
| `polytope` | Newton polytopes from exponent vectors, convex hull, facet enumeration, and lattice volume |
| `curve` | 1-skeleton of tropical curves: edges, vertices, valence, and weighted direction vectors |
| `rational_map` | Tropical rational maps (ratios of tropical polynomials) |
| `intersection` | Stable intersection, grid-based refinement, and tropical Bézout bound verification |

## API Tour

### `TropicalSemiring<T>`

The max-plus semiring: `⊕ = max`, `⊗ = +`.

```rust
impl TropicalSemiring<f64> {
    pub fn zero() -> f64;   // −∞
    pub fn one() -> f64;    // 0
    pub fn add(a: f64, b: f64) -> f64;  // max
    pub fn mul(a: f64, b: f64) -> f64;  // +
    pub fn pow(a: f64, n: u32) -> f64;  // n·a
    pub fn verify_distributivity(a: f64, b: f64, c: f64) -> bool;
    pub fn verify_idempotent(a: f64) -> bool;
}
```

### `TropicalPolynomial`

A tropical polynomial is the maximum of its monomials.

```rust
impl TropicalPolynomial {
    pub fn new(monomials: Vec<TropicalMonomial>) -> Self;
    pub fn evaluate(&self, point: &[f64]) -> f64;
    pub fn active_monomials(&self, point: &[f64]) -> Vec<usize>;
    pub fn corner_locus(&self, point: &[f64]) -> bool;   // ≥ 2 active
    pub fn is_smooth_point(&self, point: &[f64]) -> bool; // exactly 2 active
    pub fn degree(&self) -> u32;
    pub fn newton_polytope(&self) -> NewtonPolytope;
}
```

### `NewtonPolytope`

Convex hull of exponent vectors with facet enumeration.

```rust
impl NewtonPolytope {
    pub fn from_points(points: &[Vec<u32>]) -> Self;
    pub fn dimension(&self) -> usize;
    pub fn volume(&self) -> f64;  // Normalized lattice volume
    pub fn vertices: Vec<Vec<f64>>;
    pub fn facets: Vec<Facet>;
}
```

### `TropicalCurve`

The corner locus of a bivariate tropical polynomial.

```rust
impl TropicalCurve {
    pub fn new(poly: TropicalPolynomial) -> Self;
    pub fn from_monomials(monomials: Vec<TropicalMonomial>) -> Self;
    pub fn edges: Vec<TropicalEdge>;
    pub fn curve_vertices: Vec<TropicalVertex>;
}
```

### `TropicalIntersection`

Stable intersection and Bézout verification.

```rust
impl TropicalIntersection {
    pub fn stable_intersect(
        h1: &TropicalPolynomial,
        h2: &TropicalPolynomial,
        grid: &[Vec<f64>],
    ) -> Vec<Vec<f64>>;
    pub fn verify_bezout(
        h1: &TropicalPolynomial,
        h2: &TropicalPolynomial,
        grid: &[Vec<f64>],
    ) -> bool;
    pub fn refine_intersection(
        h1: &TropicalPolynomial,
        h2: &TropicalPolynomial,
        grid: &[Vec<f64>],
        iterations: usize,
    ) -> Vec<Vec<f64>>;
}
```

## Performance

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Semiring op | O(1) | Single `max` or `+` |
| Polynomial evaluate | O(m) | m = number of monomials |
| Corner locus test | O(m) | Find active monomials |
| Newton polytope | O(v²) | v = number of distinct exponent vectors |
| Curve skeleton | O(m³) | Triples of monomials for vertex detection |
| Stable intersection | O(g × m) | g = grid points, m = monomials per polynomial |

The curve skeleton computation uses a combinatorial approach over monomial triples. For high-degree polynomials, a half-plane intersection or divide-and-conquer approach would improve asymptotics.

## Ecosystem

- **[ga-core](https://github.com/SuperInstance/ga-core-rs)** — Embed tropical curves in conformal space for unified geometric processing
- **[conservation-law](https://github.com/SuperInstance/conservation-law-rs)** — Treat tropical potentials as Lagrangian energy landscapes
- **[spectral-fleet](https://github.com/SuperInstance/spectral-fleet-rs)** — Cluster polytope vertices by spectral embedding
- **[categorical-agents](https://github.com/SuperInstance/categorical-agents-rs)** — Compose tropical polynomial pipelines via monadic bind

## Ideas for Improvement

1. **Min-plus dual semiring** — Add the min-plus variant `(ℝ ∪ {+∞}, min, +)` for scheduling and shortest-path applications.
2. **Tropical matrix operations** — Implement tropical matrix multiplication and eigenvalue algorithms for max-plus linear systems.
3. **Subdivision algorithm** — Replace the triples-based curve skeleton with a subdivision method for O(m log m) complexity.
4. **Higher-dimensional hypersurfaces** — Extend curve computation to tropical surfaces and varieties in arbitrary dimension.
5. **Tropical neural network layer** — A max-plus linear layer with tropical backpropagation for interpretable deep learning.

## License

MIT OR Apache-2.0
