# INTEGRATION.md — tropical-geometry-rs × entropy-conservation-rs

**ReLU through the conservation lens**: tropical polynomials describe ReLU networks. Hodge decomposition of entropy flow reveals where conservation is violated.

## Synergy Map

```
tropical-geometry-rs          entropy-conservation-rs
┌─────────────────────┐      ┌──────────────────────────┐
│ TropicalSemiring     │      │ HodgeComponents           │
│ TropicalPolynomial   │◄────►│ decompose(entropy_matrix) │
│ TropicalMonomial     │      │ gradient_energy()         │
│ NewtonPolytope       │      │ curl_energy()             │
│ TropicalCurve        │      │ harmonic_energy()         │
│ optimal_voice_leading│      │ reconstruct()             │
└─────────────────────┘      └──────────────────────────┘
         │                               │
         └───────────────────────────────┘
                       ▼
          ReLU(x) = tropical_max(0, x)
          Neural network = tropical polynomial
          Entropy flow = Hodge decomposition
          Conservation violation = harmonic component
```

## Key Insight

ReLU activation `max(0, x)` is tropical addition. A ReLU network is a tropical polynomial: `max(bias + weight · x, 0)` is `tropical_add(tropical_mul(weight, x), bias)`. The tropical geometry of this polynomial — its Newton polytope, vertices, and edges — determines the network's expressiveness. Meanwhile, the entropy flow between layers decomposes via Hodge into gradient (conservative), curl (cyclic), and harmonic (wasteful) components. High harmonic energy = your network is fighting itself.

## Example 1: ReLU Layer as Tropical Polynomial

```rust
use tropical_geometry::semiring::TropicalSemiring;
use tropical_geometry::monomial::TropicalMonomial;
use tropical_geometry::polynomial::TropicalPolynomial;

/// Model a single ReLU neuron: max(w·x + b, 0)
/// This is tropical_max(tropical_mul(w, x) + b, 0)
fn relu_as_tropical(weights: &[f64], bias: f64, input: &[f64]) -> f64 {
    // w·x + b (tropical multiplication is ordinary addition)
    let weighted_sum: f64 = weights.iter()
        .zip(input.iter())
        .map(|(w, x)| w * x)
        .sum::<f64>() + bias;

    // ReLU: max(weighted_sum, 0) = tropical addition
    TropicalSemiring::add(weighted_sum, 0.0)
}

/// A two-layer ReLU network as a tropical polynomial.
/// Layer 1: f(x) = max(a1·x + b1, 0)
/// Layer 2: g(f) = max(a2·f + b2, 0)
/// Combined: tropical composition
fn two_layer_tropical(x: f64) -> f64 {
    // Layer 1
    let h1 = relu_as_tropical(&[2.0], -1.0, &[x]);  // max(2x-1, 0)
    let h2 = relu_as_tropical(&[-1.5], 0.5, &[x]);   // max(-1.5x+0.5, 0)

    // Layer 2: combine
    let out = relu_as_tropical(&[1.0], 0.0, &[h1]) + relu_as_tropical(&[0.5], 0.0, &[h2]);

    // Evaluate tropical polynomial: max of terms
    TropicalSemiring::add(out, 0.0)
}

fn main() {
    println!("x=0:  {:.3}", two_layer_tropical(0.0));
    println!("x=1:  {:.3}", two_layer_tropical(1.0));
    println!("x=2:  {:.3}", two_layer_tropical(2.0));
    println!("x=-1: {:.3}", two_layer_tropical(-1.0));

    // Verify tropical idempotency: max(a, a) = a
    assert!(TropicalSemiring::verify_idempotent(3.14));
}
```

## Example 2: Entropy Flow Hodge Decomposition

Decompose the entropy change between neural network layers:

```rust
use entropy_conservation_rs::hodge_decomposition::{self, HodgeComponents};

/// Compute entropy flow matrix between layers.
/// F[i][j] = entropy change from neuron i in layer L to neuron j in layer L+1
fn entropy_flow_matrix(layer_outputs: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = layer_outputs.len();
    let mut flow = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..n {
            // Entropy transferred: difference in activation magnitudes
            flow[i][j] = layer_outputs[i][j] - layer_outputs[j][i % layer_outputs[j].len()];
        }
    }
    flow
}

fn main() {
    // 4-neuron layer: activation magnitudes
    let activations = vec![
        vec![0.8, 0.2, 0.5, 0.1],
        vec![0.3, 0.9, 0.4, 0.6],
        vec![0.7, 0.1, 0.3, 0.8],
        vec![0.2, 0.6, 0.9, 0.4],
    ];

    // Build entropy flow matrix
    let flow = entropy_flow_matrix(&activations);

    // Hodge decomposition
    let decomp = hodge_decomposition::decompose(&flow);

    println!("=== Entropy Flow Hodge Decomposition ===");
    println!("Gradient energy (conservative): {:.4}", decomp.gradient_energy());
    println!("Curl energy (cyclic):           {:.4}", decomp.curl_energy());
    println!("Harmonic energy (wasteful):     {:.4}", decomp.harmonic_energy());

    let total = decomp.gradient_energy() + decomp.curl_energy() + decomp.harmonic_energy();
    if total > 0.0 {
        println!("Gradient fraction: {:.1}%", decomp.gradient_energy() / total * 100.0);
        println!("Harmonic fraction: {:.1}%", decomp.harmonic_energy() / total * 100.0);
        println!("→ High harmonic fraction means the network violates entropy conservation");
    }

    // Verify reconstruction
    let reconstructed = decomp.reconstruct();
    for i in 0..flow.len() {
        for j in 0..flow[0].len() {
            assert!((reconstructed[i][j] - flow[i][j]).abs() < 1e-6,
                "Reconstruction failed at ({},{})", i, j);
        }
    }
    println!("✓ Reconstruction verified: gradient + curl + harmonic = original");
}
```

## Example 3: Tropical Newton Polytope Entropy Analysis

Analyze the tropical geometry of a network and cross-reference with entropy conservation:

```rust
use tropical_geometry::semiring::TropicalSemiring;
use tropical_geometry::monomial::TropicalMonomial;
use tropical_geometry::polynomial::TropicalPolynomial;
use entropy_conservation_rs::hodge_decomposition;

/// Build a tropical polynomial representing a small ReLU network
/// and analyze both its tropical geometry and entropy properties.
fn analyze_network() {
    // ReLU network with 3 inputs:
    // f(x,y,z) = max(2x+y-1, x+3z-0.5, y+2z-2, 0)
    let p = TropicalPolynomial::new(vec![
        TropicalMonomial::new(1.0, vec![2, 1, 0]),  // coeff=1, exp=[2,1,0] → 1 + 2x + y
        TropicalMonomial::new(0.5, vec![1, 0, 3]),  // → 0.5 + x + 3z
        TropicalMonomial::new(2.0, vec![0, 1, 2]),   // → 2 + y + 2z
    ]);

    // Evaluate at several points
    let points = vec![
        vec![1.0, 1.0, 1.0],
        vec![0.0, 0.0, 0.0],
        vec![2.0, 1.0, 0.5],
    ];

    let mut outputs = Vec::new();
    println!("=== Tropical Polynomial Evaluation ===");
    for pt in &points {
        let val = p.evaluate(pt);
        outputs.push(val);
        println!("f({:?}) = {:.4}", pt, val);
    }

    // Build entropy flow matrix from outputs
    let n = outputs.len();
    let mut flow = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            flow[i][j] = outputs[i] - outputs[j];
        }
    }

    // Hodge decomposition of output entropy
    let decomp = hodge_decomposition::decompose(&flow);
    println!("\n=== Output Entropy Decomposition ===");
    println!("Conservative flow: {:.4}", decomp.gradient_energy());
    println!("Cyclic flow:       {:.4}", decomp.curl_energy());
    println!("Harmonic residual: {:.4}", decomp.harmonic_energy());

    // Tropical arithmetic check: ReLU = tropical addition with zero
    let relu_output = TropicalSemiring::add(outputs[0], 0.0);
    println!("\ntropical_max(f(x), 0) = {:.4}", relu_output);
    println!("tropical_mul(2, 3) = {:.1}", TropicalSemiring::mul(2.0, 3.0)); // 5.0
    println!("tropical_pow(2, 3) = {:.1}", TropicalSemiring::pow(2.0, 3));    // 6.0
}

fn main() {
    analyze_network();
}
```

## Data Flow

```
ReLU Network Weights/Biases
         │
         ▼
TropicalPolynomial (tropical_geometry)
    │          │
    ▼          ▼
evaluate()  NewtonPolytope
    │          │
    └────┬─────┘
         ▼
Layer Output Entropy Matrix
         │
         ▼
Hodge Decomposition (entropy_conservation)
    ┌────┼────┐
    ▼    ▼    ▼
gradient curl harmonic
    │    │    │
    │    │    └── conservation violations → network waste
    │    └─────── cyclic entropy flow → information recycling
    └──────────── conservative flow → useful computation
```

## When to Use This Combination

- **Neural network analysis**: understand the tropical geometry of ReLU networks and detect entropy conservation violations
- **Pruning**: high harmonic entropy neurons contribute to waste, not computation
- **Architecture search**: networks with low harmonic entropy are more efficient
- **Theoretical ML**: prove expressiveness bounds via Newton polytope degree
