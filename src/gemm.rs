use std::ops::{Add, AddAssign, Mul, MulAssign, Neg};
use num_traits::{One, Zero};
use crate::shared::{Layout, Op};
use crate::utils::transform_matrix;

/// `gemm` computes a scalar-matrix-matrix product and adds the result to a scalar-matrix product.
///
/// This operation is defined as:
///
/// `C <- α * op(A) * op(B) + β * C`
///
/// Where:
///
/// * `op` is one of noop, transpose, or hermitian.
/// * `A`, `B`, `C` are matrices.
/// * `α`, `β` are scalars.
/// * `ldx` is the leading dimension of matrix `x`.
#[allow(clippy::min_ident_chars)]
#[allow(clippy::too_many_arguments)]
pub fn gemm<T>(
    layout: &Layout,
    op_a: &Op,
    op_b: &Op,
    alpha: &T,
    a: &[T],
    lda: usize,
    b: &[T],
    ldb: usize,
    beta: &T,
    c: &mut [T],
    ldc: usize,
) where
    T: Mul<Output = T>
        + MulAssign
        + Add<Output = T>
        + AddAssign
        + Copy
        + Neg<Output = T>
        + Zero
        + One
        + PartialEq
        + Default
        + std::fmt::Debug,
{
    if a.is_empty() || b.is_empty() || c.is_empty() {
        // Early return, we can do no work here.
        return;
    }

    // Early return, alpha == zero => we can avoid the matrix multiply step
    if alpha.is_zero() {
        if beta.is_zero() {
            for ci in c.iter_mut() {
                *ci = T::zero();
            }
        } else if beta.is_one() {
            // No need to multiply by one, just continue
        } else {
            for ci in c.iter_mut() {
                *ci *= *beta;
            }
        }
        return;
    }

    // Transform matrices based on layout and operations
    let a_t: Vec<T> = transform_matrix(layout, a, op_a, lda);
    let b_t: Vec<T> = transform_matrix(layout, b, op_b, ldb);

    // Pass slices of the vectors to gemm_calc
    gemm_calc(layout, alpha, &a_t, lda, &b_t, ldb, beta, c, ldc);
}

fn gemm_calc<T>(
    layout: &Layout,
    alpha: &T,
    a: &[T],
    lda: usize,
    b: &[T],
    ldb: usize,
    beta: &T,
    c: &mut [T],
    ldc: usize,
) where
    T: Mul<Output = T> 
    + Add<Output = T> 
    + Copy 
    + Zero 
    + One 
    + PartialEq 
    + std::fmt::Debug,
{
    match layout {
        Layout::RowMajor => {
            // Determine dimensions of matrices A, B, and C for row-major layout
            let m = a.len() / lda;
            let k_a = lda;
            let k_b = b.len() / ldb;
            let n = ldb;
            let m_c = c.len() / ldc;
            let n_c = ldc;

            // Debug prints for dimensions
            println!("m: {}, k_a: {}, k_b: {}, n: {}, m_c: {}, n_c: {}", m, k_a, k_b, n, m_c, n_c);

            // Ensure dimensions match for matrix multiplication
            assert!(k_a == k_b, "Inner dimensions of A and B must match.");
            assert!(m == m_c, "Row dimensions of A and C must match.");
            assert!(n == n_c, "Column dimensions of B and C must match.");

            for i in 0..m {
                for j in 0..n {
                    let mut sum = T::zero();
                    for l in 0..k_a {
                        let a_index = i * lda + l; // Access A[i, l] in row-major
                        let b_index = l * ldb + j; // Access B[l, j] in row-major

                        // Ensure indices are within bounds
                        if a_index >= a.len() || b_index >= b.len() {
                            panic!("Index out of bounds: a_index = {}, b_index = {}", a_index, b_index);
                        }

                        // Debug prints for indices and values
                        println!("a[{}]: {:?}, b[{}]: {:?}", a_index, a[a_index], b_index, b[b_index]);

                        // Accumulate the product into sum
                        sum = sum + (*alpha) * a[a_index] * b[b_index];
                    }

                    let c_index = i * ldc + j; // Row-major indexing for C[i, j]

                    // Debug prints for C update
                    println!("c[{}] before: {:?}, sum: {:?}", c_index, c[c_index], sum);

                    // Update matrix C
                    c[c_index] = (*beta) * c[c_index] + sum;

                    // Debug prints for C after update
                    println!("c[{}] after: {:?}", c_index, c[c_index]);
                }
            }
        }
        Layout::ColumnMajor => {
            // Determine dimensions of matrices A, B, and C for column-major layout
            let m = lda;
            let k_a = a.len() / lda;
            let k_b = ldb;
            let n = b.len() / ldb;
            let m_c = ldc;
            let n_c = c.len() / ldc;

            // Debug prints for dimensions
            println!("m: {}, k_a: {}, k_b: {}, n: {}, m_c: {}, n_c: {}", m, k_a, k_b, n, m_c, n_c);

            // Ensure dimensions match for matrix multiplication
            assert!(k_a == k_b, "Inner dimensions of A and B must match.");
            assert!(m == m_c, "Row dimensions of A and C must match.");
            assert!(n == n_c, "Column dimensions of B and C must match.");

            for i in 0..m {
                for j in 0..n {
                    let mut sum = T::zero();
                    for l in 0..k_a {
                        let a_index = l * lda + i; // Access A[l, i] in column-major
                        let b_index = j * ldb + l; // Access B[j, l] in column-major

                        // Ensure indices are within bounds
                        if a_index >= a.len() || b_index >= b.len() {
                            panic!("Index out of bounds: a_index = {}, b_index = {}", a_index, b_index);
                        }

                        // Debug prints for indices and values
                        println!("a[{}]: {:?}, b[{}]: {:?}", a_index, a[a_index], b_index, b[b_index]);

                        // Accumulate the product into sum
                        sum = sum + (*alpha) * a[a_index] * b[b_index];
                    }

                    let c_index = j * ldc + i; // Column-major indexing for C[j, i]

                    // Debug prints for C update
                    println!("c[{}] before: {:?}, sum: {:?}", c_index, c[c_index], sum);

                    // Update matrix C
                    c[c_index] = (*beta) * c[c_index] + sum;

                    // Debug prints for C after update
                    println!("c[{}] after: {:?}", c_index, c[c_index]);
                }
            }
        }
    }
}