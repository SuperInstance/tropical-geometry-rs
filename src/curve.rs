use serde::{Deserialize, Serialize};
use crate::polynomial::TropicalPolynomial;
use crate::monomial::TropicalMonomial;
use crate::polytope::NewtonPolytope;

/// A tropical edge in the 1-skeleton of a tropical curve.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TropicalEdge {
    /// Direction vector (primitive integer direction).
    pub direction: (f64, f64),
    /// Weight (multiplicity) of the edge.
    pub weight: u32,
    /// Connected vertices: (from_index, to_index). `None` means unbounded ray.
    pub vertices: (usize, Option<usize>),
}

/// A vertex of a tropical curve where ≥ 3 monomials meet.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TropicalVertex {
    /// The point where the vertex lies.
    pub point: (f64, f64),
    /// Number of incident edges.
    pub valence: usize,
    /// Indices of incident edges.
    pub incident_edges: Vec<usize>,
}

/// A tropical curve (the n=2 case): the corner locus of a tropical polynomial in 2 variables.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TropicalCurve {
    /// The underlying tropical polynomial.
    pub polynomial: TropicalPolynomial,
    /// Computed edges of the curve.
    pub edges: Vec<TropicalEdge>,
    /// Computed vertices of the curve.
    pub curve_vertices: Vec<TropicalVertex>,
}

impl TropicalCurve {
    /// Create a tropical curve from a polynomial in 2 variables.
    pub fn new(poly: TropicalPolynomial) -> Self {
        let mut curve = Self {
            polynomial: poly,
            edges: vec![],
            curve_vertices: vec![],
        };
        curve.compute_skeleton();
        curve
    }

    /// Convenience: build from monomials.
    pub fn from_monomials(monomials: Vec<TropicalMonomial>) -> Self {
        Self::new(TropicalPolynomial::new(monomials))
    }

    /// Compute the 1-skeleton (edges and vertices) of the tropical curve.
    fn compute_skeleton(&mut self) {
        // Strategy: For each pair of monomials, find where they are equal.
        // A tropical curve vertex occurs where ≥ 2 monomials are simultaneously maximal.
        //
        // For monomial i: f_i(x,y) = c_i + a_i*x + b_i*y
        // For monomial j: f_j(x,y) = c_j + a_j*x + b_j*y
        //
        // They are equal when: c_i + a_i*x + b_i*y = c_j + a_j*x + b_j*y
        // => (a_i - a_j)*x + (b_i - b_j)*y = c_j - c_i
        //
        // The tropical curve is the union of these lines restricted to where
        // both monomials are actually maximal.

        let n = self.polynomial.monomials.len();
        if n < 2 {
            return;
        }

        // For small cases, we find vertices by computing where ≥2 monomials meet
        // and are maximal, then connect them along the boundary lines.

        // Find candidate vertices: where ≥ 3 monomials are equal and maximal.
        // For each triple of monomials, solve the 2x2 system.
        let mut vertex_points: Vec<(f64, f64, Vec<usize>)> = Vec::new();

        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    if let Some(pt) = self.triple_intersection(i, j, k) {
                        // Verify all three are maximal at this point
                        let val = self.polynomial.monomials[i].evaluate(&[pt.0, pt.1]);
                        let poly_val = self.polynomial.evaluate(&[pt.0, pt.1]);
                        if (val - poly_val).abs() < 1e-9 {
                            vertex_points.push((pt.0, pt.1, vec![i, j, k]));
                        }
                    }
                }
            }
        }

        // Deduplicate vertices
        let mut deduped: Vec<(f64, f64, Vec<usize>)> = Vec::new();
        for vp in &vertex_points {
            let is_dup = deduped.iter().any(|(x, y, _)| {
                (x - vp.0).abs() < 1e-9 && (y - vp.1).abs() < 1e-9
            });
            if !is_dup {
                deduped.push(vp.clone());
            }
        }

        // Create TropicalVertex for each
        for (x, y, _mons) in deduped.iter() {
            // Count incident edges by looking at how many monomial pairs are active
            let active = self.polynomial.active_monomials(&[*x, *y]);
            self.curve_vertices.push(TropicalVertex {
                point: (*x, *y),
                valence: active.len(),
                incident_edges: vec![],
            });
            // Will fill incident_edges after edge computation
        }

        // For each pair of active monomials at a vertex, create an edge in the
        // direction perpendicular to the line connecting their exponent vectors.
        // The edge direction is (b_j - b_i, -(a_j - a_i)) normalized.
        let mut edge_list: Vec<TropicalEdge> = Vec::new();

        for (vi, (_x, _y, _mons)) in deduped.iter().enumerate() {
            let active = self.polynomial.active_monomials(&[*_x, *_y]);
            for ai in 0..active.len() {
                for aj in (ai + 1)..active.len() {
                    let i = active[ai];
                    let j = active[aj];
                    let mi = &self.polynomial.monomials[i];
                    let mj = &self.polynomial.monomials[j];

                    let da = mj.exponents[0] as f64 - mi.exponents[0] as f64;
                    let db = mj.exponents[1] as f64 - mi.exponents[1] as f64;

                    // Direction along which mi == mj
                    // Perpendicular to (da, db) => direction is (db, -da) or (-db, da)
                    let len = (da * da + db * db).sqrt().max(1e-12);
                    let dir = (db / len, -da / len);

                    // Weight = gcd of exponent differences
                    let weight = gcd(da.abs() as u32, db.abs() as u32).max(1);

                    let edge_idx = edge_list.len();
                    edge_list.push(TropicalEdge {
                        direction: dir,
                        weight,
                        vertices: (vi, None), // will be updated
                    });

                    self.curve_vertices[vi].incident_edges.push(edge_idx);
                }
            }
        }

        // Now try to connect vertices that share an edge (same pair of monomials)
        for _ei in 0..edge_list.len() {
            // Find if there's another vertex on this edge's line
            // For now, edges remain unbounded unless we find a second vertex
            // This is a simplification; a full implementation would trace the skeleton
        }

        self.edges = edge_list;
    }

    /// Solve where three monomials are simultaneously equal.
    /// Returns the intersection point if it exists.
    fn triple_intersection(&self, i: usize, j: usize, k: usize) -> Option<(f64, f64)> {
        let mi = &self.polynomial.monomials[i];
        let mj = &self.polynomial.monomials[j];
        let mk = &self.polynomial.monomials[k];

        // mi(x,y) = mj(x,y) and mi(x,y) = mk(x,y)
        // (ai - aj)*x + (bi - bj)*y = cj - ci
        // (ai - ak)*x + (bi - bk)*y = ck - ci
        let a11 = mi.exponents[0] as f64 - mj.exponents[0] as f64;
        let a12 = mi.exponents[1] as f64 - mj.exponents[1] as f64;
        let b1 = mj.coeff - mi.coeff;

        let a21 = mi.exponents[0] as f64 - mk.exponents[0] as f64;
        let a22 = mi.exponents[1] as f64 - mk.exponents[1] as f64;
        let b2 = mk.coeff - mi.coeff;

        let det = a11 * a22 - a12 * a21;
        if det.abs() < 1e-12 {
            return None;
        }

        let x = (b1 * a22 - b2 * a12) / det;
        let y = (a11 * b2 - a21 * b1) / det;
        Some((x, y))
    }

    /// The genus of the tropical curve (Baker's formula):
    /// number of interior lattice points of the Newton polygon.
    pub fn genus(&self) -> usize {
        let np = self.polynomial.newton_polytope();
        let (interior, _boundary) = np.pick_formula();
        interior as usize
    }

    /// Is the curve smooth? Every vertex should be 3-valent with primitive edge directions.
    pub fn is_smooth(&self) -> bool {
        self.curve_vertices.iter().all(|v| {
            if v.valence != 3 {
                return false;
            }
            // Check that all edge directions are primitive
            for &ei in &v.incident_edges {
                if ei < self.edges.len() && self.edges[ei].weight != 1 {
                    return false;
                }
            }
            true
        })
    }

    /// Check the balancing condition at a specific vertex.
    ///
    /// Σ outward primitive vectors weighted by edge weight = 0.
    pub fn balancing(&self, vertex: &TropicalVertex) -> bool {
        if vertex.incident_edges.is_empty() {
            return true;
        }

        let mut sum_x = 0.0;
        let mut sum_y = 0.0;

        for &ei in &vertex.incident_edges {
            if ei < self.edges.len() {
                let edge = &self.edges[ei];
                let w = edge.weight as f64;
                sum_x += edge.direction.0 * w;
                sum_y += edge.direction.1 * w;
            }
        }

        sum_x.abs() < 1e-9 && sum_y.abs() < 1e-9
    }

    /// Is the entire curve balanced (balancing condition at every vertex)?
    pub fn is_balanced(&self) -> bool {
        self.curve_vertices.iter().all(|v| self.balancing(v))
    }

    /// The Newton polygon of the underlying polynomial.
    pub fn newton_polygon(&self) -> NewtonPolytope {
        self.polynomial.newton_polytope()
    }
}

fn gcd(a: u32, b: u32) -> u32 {
    let mut a = a;
    let mut b = b;
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_max_xyz() -> TropicalCurve {
        // max(x, y, 0) — the simplest tropical curve
        TropicalCurve::from_monomials(vec![
            TropicalMonomial::new(0.0, vec![1, 0]),
            TropicalMonomial::new(0.0, vec![0, 1]),
            TropicalMonomial::constant(0.0),
        ])
    }

    #[test]
    fn test_curve_vertices_at_origin() {
        let curve = make_max_xyz();
        // At (0,0), all three monomials are equal: x=0, y=0, 0=0
        // So there should be a vertex at (0,0)
        let has_origin = curve.curve_vertices.iter().any(|v| {
            v.point.0.abs() < 1e-9 && v.point.1.abs() < 1e-9
        });
        assert!(has_origin, "should have vertex at origin, vertices: {:?}", curve.curve_vertices);
    }

    #[test]
    fn test_curve_edges() {
        let curve = make_max_xyz();
        // Should have 3 edges emanating from the origin vertex
        assert!(curve.edges.len() >= 3, "edges: {}", curve.edges.len());
    }

    #[test]
    fn test_genus_zero() {
        // max(x, y, 0) has Newton polygon = triangle {(1,0), (0,1), (0,0)}
        // Interior lattice points = 0, genus = 0
        let curve = make_max_xyz();
        assert_eq!(curve.genus(), 0);
    }

    #[test]
    fn test_genus_nonzero() {
        // Tropical curve with Newton polygon = {(3,0), (0,3), (0,0)}
        // Interior lattice point: (1,1), so genus = 1
        let curve = TropicalCurve::from_monomials(vec![
            TropicalMonomial::new(0.0, vec![3, 0]),
            TropicalMonomial::new(0.0, vec![0, 3]),
            TropicalMonomial::constant(0.0),
        ]);
        assert_eq!(curve.genus(), 1);
    }

    #[test]
    fn test_is_smooth_trivalent() {
        let curve = make_max_xyz();
        // The origin vertex has 3 monomials active, so valence 3
        let origin = curve.curve_vertices.iter().find(|v| {
            v.point.0.abs() < 1e-9 && v.point.1.abs() < 1e-9
        });
        if let Some(v) = origin {
            assert_eq!(v.valence, 3);
        }
    }

    #[test]
    fn test_newton_polygon() {
        let curve = make_max_xyz();
        let np = curve.newton_polygon();
        assert_eq!(np.vertices.len(), 3); // (0,0), (1,0), (0,1)
    }

    #[test]
    fn test_quadratic_curve() {
        // max(x², y², 0) — tropical conic
        let curve = TropicalCurve::from_monomials(vec![
            TropicalMonomial::new(0.0, vec![2, 0]),
            TropicalMonomial::new(0.0, vec![0, 2]),
            TropicalMonomial::constant(0.0),
        ]);
        // Should have a vertex at origin
        let has_origin = curve.curve_vertices.iter().any(|v| {
            v.point.0.abs() < 1e-9 && v.point.1.abs() < 1e-9
        });
        assert!(has_origin);
        // Newton polygon: {(2,0), (0,2), (0,0)}, boundary x+y=2
        // (1,1) is on the boundary (1+1=2), so interior = 0, genus = 0
        assert_eq!(curve.genus(), 0);
    }

    #[test]
    fn test_four_monomial_curve() {
        // max(x, y, x+y, 0) — 4 monomials
        let curve = TropicalCurve::from_monomials(vec![
            TropicalMonomial::new(0.0, vec![1, 0]),
            TropicalMonomial::new(0.0, vec![0, 1]),
            TropicalMonomial::new(0.0, vec![1, 1]),
            TropicalMonomial::constant(0.0),
        ]);
        // Should have multiple vertices
        assert!(curve.curve_vertices.len() >= 1);
    }

    #[test]
    fn test_tropical_edge_direction() {
        let edge = TropicalEdge {
            direction: (1.0, 0.0),
            weight: 1,
            vertices: (0, None),
        };
        assert_eq!(edge.direction, (1.0, 0.0));
    }

    #[test]
    fn test_tropical_vertex_valence() {
        let v = TropicalVertex {
            point: (0.0, 0.0),
            valence: 3,
            incident_edges: vec![0, 1, 2],
        };
        assert_eq!(v.valence, 3);
        assert_eq!(v.incident_edges.len(), 3);
    }
}
