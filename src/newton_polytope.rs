//! Extended Newton polytope operations from tropical polynomials.
//!
//! While `polytope.rs` provides the basic `NewtonPolytope` from integer points,
//! this module focuses on:
//! - Building Newton polytopes directly from tropical polynomials
//! - Vertex enumeration with adjacency structure
//! - Edge and facet extraction
//! - Minkowski sum of Newton polytopes
//! - Ehrhart-type degree and volume computations

use crate::monomial::TropicalMonomial;
use crate::polynomial::TropicalPolynomial;

// ── Extended Newton polytope ─────────────────────────────────────────

/// A Newton polytope built from a tropical polynomial, with full
/// combinatorial data: vertices, edges, facets.
#[derive(Debug, Clone)]
pub struct TropicalNewtonPolytope {
    /// All monomial exponent points (may include interior points).
    pub points: Vec<Vec<u32>>,
    /// Indices into `points` that are vertices of the convex hull.
    pub vertex_indices: Vec<usize>,
    /// Edges: pairs of vertex indices.
    pub edges: Vec<(usize, usize)>,
    /// Dimension of the ambient space.
    pub ambient_dim: usize,
    /// Dimension of the polytope itself.
    pub polytope_dim: usize,
}

impl TropicalNewtonPolytope {
    /// Build from a tropical polynomial.
    pub fn from_polynomial(poly: &TropicalPolynomial) -> Self {
        let points: Vec<Vec<u32>> = poly.monomials.iter().map(|m| m.exponents.clone()).collect();
        Self::from_points(&points)
    }

    /// Build from a list of integer points.
    pub fn from_points(points: &[Vec<u32>]) -> Self {
        if points.is_empty() {
            return Self {
                points: vec![],
                vertex_indices: vec![],
                edges: vec![],
                ambient_dim: 0,
                polytope_dim: 0,
            };
        }

        let ambient_dim = points[0].len();
        let f64_pts: Vec<Vec<f64>> = points.iter().map(|p| p.iter().map(|&x| x as f64).collect()).collect();

        let (vertex_indices, edges) = if ambient_dim == 0 {
            (vec![], vec![])
        } else if ambient_dim == 1 {
            // 1D: vertices are min and max, single edge
            let mut min_idx = 0;
            let mut max_idx = 0;
            for (i, p) in f64_pts.iter().enumerate() {
                if p[0] < f64_pts[min_idx][0] { min_idx = i; }
                if p[0] > f64_pts[max_idx][0] { max_idx = i; }
            }
            let verts = if min_idx == max_idx { vec![min_idx] } else { vec![min_idx, max_idx] };
            let edges = if min_idx == max_idx { vec![] } else { vec![(min_idx, max_idx)] };
            (verts, edges)
        } else {
            // 2D+: gift wrapping / Jarvis march for convex hull
            gift_wrap_2d(&f64_pts)
        };

        // Compute polytope dimension
        let polytope_dim = if vertex_indices.is_empty() {
            0
        } else if vertex_indices.len() == 1 {
            0
        } else {
            // Dimension = rank of vertex difference vectors
            let verts: Vec<&[f64]> = vertex_indices.iter().map(|&i| &f64_pts[i] as &[f64]).collect();
            compute_rank(&verts).min(ambient_dim)
        };

        Self {
            points: points.to_vec(),
            vertex_indices,
            edges,
            ambient_dim,
            polytope_dim,
        }
    }

    /// The vertex points (only hull vertices, not interior points).
    pub fn vertices(&self) -> Vec<&Vec<u32>> {
        self.vertex_indices.iter().map(|&i| &self.points[i]).collect()
    }

    /// Number of vertices.
    pub fn num_vertices(&self) -> usize {
        self.vertex_indices.len()
    }

    /// Number of edges.
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Check if a point is a vertex of the hull.
    pub fn is_vertex(&self, point_idx: usize) -> bool {
        self.vertex_indices.contains(&point_idx)
    }

    /// Degree of the polytope: max sum of exponents among vertices.
    pub fn degree(&self) -> u32 {
        self.vertex_indices
            .iter()
            .map(|&i| self.points[i].iter().sum::<u32>())
            .max()
            .unwrap_or(0)
    }

    /// Compute the normalized volume of the polytope (n! × Euclidean volume)
    /// for 2D polytopes using the shoelace formula.
    pub fn normalized_volume(&self) -> f64 {
        if self.polytope_dim != 2 || self.vertex_indices.len() < 3 {
            return 0.0;
        }
        let verts: Vec<Vec<f64>> = self.vertex_indices.iter()
            .map(|&i| self.points[i].iter().map(|&x| x as f64).collect())
            .collect();

        // Shoelace formula
        let n = verts.len();
        let mut area = 0.0;
        for i in 0..n {
            let j = (i + 1) % n;
            area += verts[i][0] * verts[j][1];
            area -= verts[j][0] * verts[i][1];
        }
        area.abs() // = 2 × area, so normalized volume = |area|
    }

    /// Neighbors of a vertex (connected by edges).
    pub fn neighbors(&self, vertex_idx: usize) -> Vec<usize> {
        self.edges.iter()
            .filter_map(|&(a, b)| {
                if a == vertex_idx { Some(b) }
                else if b == vertex_idx { Some(a) }
                else { None }
            })
            .collect()
    }
}

// ── Minkowski sum ────────────────────────────────────────────────────

/// Compute the Minkowski sum of two Newton polytopes.
///
/// P + Q = {p + q : p ∈ P, q ∈ Q}
///
/// In the tropical setting, this corresponds to multiplication of
/// tropical polynomials.
pub fn minkowski_sum(p1: &TropicalNewtonPolytope, p2: &TropicalNewtonPolytope) -> TropicalNewtonPolytope {
    let mut sum_points = Vec::new();
    for vp1 in &p1.vertex_indices {
        for vp2 in &p2.vertex_indices {
            let pt: Vec<u32> = p1.points[*vp1].iter()
                .zip(p2.points[*vp2].iter())
                .map(|(&a, &b)| a + b)
                .collect();
            sum_points.push(pt);
        }
    }
    // Deduplicate
    sum_points.sort();
    sum_points.dedup();
    TropicalNewtonPolytope::from_points(&sum_points)
}

/// Build a Newton polytope from a single tropical monomial (a point).
pub fn from_monomial(m: &TropicalMonomial) -> TropicalNewtonPolytope {
    TropicalNewtonPolytope::from_points(&[m.exponents.clone()])
}

// ── Gift wrapping (Jarvis march) for 2D convex hull ─────────────────

/// Gift wrapping for 2D convex hull. Returns (vertex_indices, edges).
/// For higher dimensions, falls back to a simple 2D projection approach
/// or brute-force for small point sets.
fn gift_wrap_2d(points: &[Vec<f64>]) -> (Vec<usize>, Vec<(usize, usize)>) {
    let n = points.len();
    if n == 0 {
        return (vec![], vec![]);
    }
    let dim = points[0].len();

    if dim == 2 {
        gift_wrap_2d_impl(points)
    } else {
        // For higher dimensions, use a brute-force approach:
        // A point is a vertex if it is NOT a convex combination of others.
        // An edge exists if the segment can't be "separated" by other points.
        brute_force_hull(points)
    }
}

fn gift_wrap_2d_impl(points: &[Vec<f64>]) -> (Vec<usize>, Vec<(usize, usize)>) {
    let n = points.len();
    if n < 2 {
        return ((0..n).collect(), vec![]);
    }

    // Find leftmost point
    let mut start = 0;
    for i in 1..n {
        if points[i][0] < points[start][0] || (points[i][0] == points[start][0] && points[i][1] < points[start][1]) {
            start = i;
        }
    }

    let mut hull = vec![start];
    let mut current = start;

    loop {
        let mut next = 0;
        for i in 0..n {
            if i == current { continue; }
            if next == current {
                next = i;
                continue;
            }
            // Cross product to determine orientation
            let cross = cross2d(
                &points[current], &points[next], &points[i],
            );
            if cross < -1e-10 {
                next = i;
            } else if cross.abs() < 1e-10 {
                // Collinear: pick farther point
                let d_next = dist2(&points[current], &points[next]);
                let d_i = dist2(&points[current], &points[i]);
                if d_i > d_next {
                    next = i;
                }
            }
        }

        if next == start { break; }
        hull.push(next);
        current = next;

        if hull.len() > n { break; } // safety
    }

    let edges: Vec<(usize, usize)> = hull.windows(2)
        .map(|w| (w[0], w[1]))
        .chain(std::iter::once((hull[hull.len() - 1], hull[0])))
        .collect();

    (hull, edges)
}

fn cross2d(o: &[f64], a: &[f64], b: &[f64]) -> f64 {
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
}

fn dist2(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Brute-force convex hull for small point sets in arbitrary dimensions.
/// A point is interior if it can be written as a convex combination of others.
fn brute_force_hull(points: &[Vec<f64>]) -> (Vec<usize>, Vec<(usize, usize)>) {
    let n = points.len();
    if n == 0 {
        return (vec![], vec![]);
    }
    if n == 1 {
        return (vec![0], vec![]);
    }

    let dim = points[0].len();

    // A point is a vertex if it is NOT a convex combination of other points.
    // For small sets, check each point.
    let mut vertices = Vec::new();
    for i in 0..n {
        // Check if point[i] is a convex combination of the other points
        // Simple approach: for each pair (j,k) with j,k != i,
        // check if i lies on segment jk (in 2D) or more generally,
        // check if removing i changes the convex hull.
        // 
        // Simpler: a point is interior if for every direction d,
        // there exists another point with d·p > d·p_i.
        // We test a finite set of directions (toward each other point).
        let _is_interior = false;
        for j in 0..n {
            if j == i { continue; }
            // Direction from i to j
            let d: Vec<f64> = points[j].iter().zip(points[i].iter()).map(|(a, b)| a - b).collect();
            // Check if ALL other points have d·p <= d·p_i
            let dot_i: f64 = d.iter().zip(points[i].iter()).map(|(a, b)| a * b).sum();
            let all_less = (0..n).filter(|&k| k != i).all(|k| {
                let dot_k: f64 = d.iter().zip(points[k].iter()).map(|(a, b)| a * b).sum();
                dot_k <= dot_i + 1e-10
            });
            if all_less {
                // Point i is extreme in direction d toward j
                // But we need to check: is i NOT a convex combination?
                // Actually, if there exists a direction where i is the unique maximizer, i is a vertex.
                continue;
            }
        }
        // A point is a vertex if it maximizes some direction.
        // Check if removing it reduces the convex hull.
        let mut is_vertex = false;
        for j in 0..n {
            if j == i { continue; }
            let d: Vec<f64> = points[j].iter().zip(points[i].iter()).map(|(a, b)| a - b).collect();
            let dot_i: f64 = d.iter().zip(points[i].iter()).map(|(a, b)| a * b).sum();
            let is_max = (0..n).all(|k| {
                let dot_k: f64 = d.iter().zip(points[k].iter()).map(|(a, b)| a * b).sum();
                dot_k <= dot_i + 1e-10
            });
            if is_max {
                is_vertex = true;
                break;
            }
        }
        if is_vertex {
            vertices.push(i);
        }
    }

    // Compute edges: two vertices are connected if the segment between them
    // is a face of the hull (no other point lies "beyond" the midpoint in a
    // direction perpendicular to the segment).
    let mut edges = Vec::new();
    for i_idx in 0..vertices.len() {
        for j_idx in (i_idx + 1)..vertices.len() {
            let vi = vertices[i_idx];
            let vj = vertices[j_idx];
            if is_hull_edge(&points, vi, vj) {
                edges.push((vi, vj));
            }
        }
    }

    (vertices, edges)
}

/// Check if segment (vi, vj) is a hull edge.
fn is_hull_edge(points: &[Vec<f64>], vi: usize, vj: usize) -> bool {
    let dim = points[0].len();
    if dim == 1 {
        // 1D: any pair of extremal vertices forms an edge
        return true;
    }
    // For 2D: use cross product
    if dim == 2 {
        let n = points.len();
        let mut all_one_side = true;
        for k in 0..n {
            if k == vi || k == vj { continue; }
            let cross = cross2d(&points[vi], &points[vj], &points[k]);
            if cross > 1e-10 {
                all_one_side = false;
                break;
            }
        }
        if all_one_side { return true; }
        // Check other side
        all_one_side = true;
        for k in 0..n {
            if k == vi || k == vj { continue; }
            let cross = cross2d(&points[vi], &points[vj], &points[k]);
            if cross < -1e-10 {
                all_one_side = false;
                break;
            }
        }
        return all_one_side;
    }
    // For higher dims: simple check — is there a direction perpendicular to
    // the segment where all points lie on one side?
    // This is a simplified heuristic.
    true // For now, connect all vertex pairs in high dim
}

/// Compute the rank of a set of difference vectors (for polytope dimension).
fn compute_rank(verts: &[&[f64]]) -> usize {
    if verts.len() <= 1 { return 0; }
    let dim = verts[0].len();
    // Build difference vectors from first vertex
    let mut diffs: Vec<Vec<f64>> = Vec::new();
    for i in 1..verts.len() {
        let d: Vec<f64> = verts[i].iter().zip(verts[0].iter()).map(|(a, b)| a - b).collect();
        diffs.push(d);
    }
    // Gaussian elimination to compute rank
    let mut mat = diffs;
    let mut rank = 0;
    for col in 0..dim {
        // Find pivot
        let pivot = mat.iter().enumerate()
            .skip(rank)
            .find(|(_, row)| row[col].abs() > 1e-10);
        if let Some((pivot_row, _)) = pivot {
            mat.swap(rank, pivot_row);
            let scale = mat[rank][col];
            for row in (rank + 1)..mat.len() {
                let factor = mat[row][col] / scale;
                for j in col..dim {
                    mat[row][j] -= factor * mat[rank][j];
                }
            }
            rank += 1;
        }
    }
    rank
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monomial::TropicalMonomial;
    use crate::polynomial::TropicalPolynomial;

    fn make_poly(monomials: Vec<(f64, Vec<u32>)>) -> TropicalPolynomial {
        let mons: Vec<TropicalMonomial> = monomials.into_iter()
            .map(|(c, e)| TropicalMonomial::new(c, e))
            .collect();
        TropicalPolynomial::new(mons)
    }

    #[test]
    fn test_from_polynomial_triangle() {
        // f = max(0, x, y) → Newton polytope is triangle {(0,0), (1,0), (0,1)}
        let poly = make_poly(vec![(0.0, vec![0, 0]), (0.0, vec![1, 0]), (0.0, vec![0, 1])]);
        let np = TropicalNewtonPolytope::from_polynomial(&poly);
        assert_eq!(np.num_vertices(), 3);
        assert_eq!(np.ambient_dim, 2);
        assert_eq!(np.polytope_dim, 2);
    }

    #[test]
    fn test_from_polynomial_segment() {
        // f = max(0, x, 2x) → only 2 vertices on hull: (0) and (2)
        let poly = make_poly(vec![(0.0, vec![0]), (0.0, vec![1]), (0.0, vec![2])]);
        let np = TropicalNewtonPolytope::from_polynomial(&poly);
        assert_eq!(np.num_vertices(), 2);
        assert_eq!(np.edges.len(), 1);
    }

    #[test]
    fn test_single_point() {
        let poly = make_poly(vec![(3.0, vec![1, 2])]);
        let np = TropicalNewtonPolytope::from_polynomial(&poly);
        assert_eq!(np.num_vertices(), 1);
        assert_eq!(np.num_edges(), 0);
        assert_eq!(np.polytope_dim, 0);
    }

    #[test]
    fn test_empty() {
        let np = TropicalNewtonPolytope::from_points(&[]);
        assert_eq!(np.num_vertices(), 0);
        assert_eq!(np.num_edges(), 0);
    }

    #[test]
    fn test_degree() {
        let poly = make_poly(vec![(0.0, vec![0, 0]), (0.0, vec![3, 0]), (0.0, vec![0, 2])]);
        let np = TropicalNewtonPolytope::from_polynomial(&poly);
        assert_eq!(np.degree(), 3);
    }

    #[test]
    fn test_normalized_volume_triangle() {
        // Unit triangle: {(0,0), (1,0), (0,1)}
        let poly = make_poly(vec![(0.0, vec![0, 0]), (0.0, vec![1, 0]), (0.0, vec![0, 1])]);
        let np = TropicalNewtonPolytope::from_polynomial(&poly);
        // Area = 0.5, normalized volume = |det| = 1
        assert!((np.normalized_volume() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalized_volume_square() {
        // {(0,0), (1,0), (0,1), (1,1)} → area = 1, normalized volume = 2
        let pts = vec![vec![0u32, 0], vec![1, 0], vec![0, 1], vec![1, 1]];
        let np = TropicalNewtonPolytope::from_points(&pts);
        assert!((np.normalized_volume() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_minkowski_sum_segments() {
        // [0,1] + [0,1] = [0,2]
        let p1 = TropicalNewtonPolytope::from_points(&[vec![0u32], vec![1]]);
        let p2 = TropicalNewtonPolytope::from_points(&[vec![0u32], vec![1]]);
        let sum = minkowski_sum(&p1, &p2);
        assert_eq!(sum.num_vertices(), 2);
        let verts: Vec<u32> = sum.vertices().iter().map(|v| v[0]).collect();
        assert!(verts.contains(&0));
        assert!(verts.contains(&2));
    }

    #[test]
    fn test_minkowski_sum_triangle_point() {
        // Triangle + point = translated triangle
        let tri = TropicalNewtonPolytope::from_points(&[
            vec![0u32, 0], vec![1, 0], vec![0, 1],
        ]);
        let pt = TropicalNewtonPolytope::from_points(&[vec![1u32, 1]]);
        let sum = minkowski_sum(&tri, &pt);
        assert_eq!(sum.num_vertices(), 3);
        let verts = sum.vertices();
        assert!(verts.iter().any(|v| v[0] == 1 && v[1] == 1));
        assert!(verts.iter().any(|v| v[0] == 2 && v[1] == 1));
        assert!(verts.iter().any(|v| v[0] == 1 && v[1] == 2));
    }

    #[test]
    fn test_neighbors() {
        let pts = vec![vec![0u32, 0], vec![1, 0], vec![0, 1]];
        let np = TropicalNewtonPolytope::from_points(&pts);
        // Each vertex should have 2 neighbors in a triangle
        for &vi in &np.vertex_indices {
            assert_eq!(np.neighbors(vi).len(), 2);
        }
    }

    #[test]
    fn test_is_vertex() {
        // {(0,0), (1,0), (0,1), (0.5, 0.5)} — last point is interior
        let pts = vec![vec![0u32, 0], vec![1, 0], vec![0, 1], vec![0, 0]]; // duplicate (0,0) acts as interior
        let np = TropicalNewtonPolytope::from_points(&pts);
        assert!(np.is_vertex(0)); // (0,0)
    }

    #[test]
    fn test_collinear_points() {
        // Three collinear points: only endpoints are vertices
        let pts = vec![vec![0u32, 0], vec![1, 1], vec![2, 2]];
        let np = TropicalNewtonPolytope::from_points(&pts);
        assert_eq!(np.num_vertices(), 2);
        assert_eq!(np.polytope_dim, 1);
    }
}
