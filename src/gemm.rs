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
        + Default,
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
    T: Mul<Output = T> + Add<Output = T> + Copy + Zero + One + PartialEq,
{
    // Determine dimensions based on layout
    let (m, n, k) = match layout {
        Layout::RowMajor => (c.len() / ldc, b.len() / ldb, a.len() / lda),
        Layout::ColumnMajor => (ldc, ldb, lda),
    };

    // Iterate through the matrix dimensions
    for i in 0..m {
        for j in 0..n {
            let mut sum = T::zero();
            for l in 0..k {
                // Use match to handle indexing based on the layout
                let (a_index, b_index) = match layout {
                    Layout::RowMajor => (
                        i * lda + l,       // Row-major indexing for `a`
                        l * ldb + j,       // Row-major indexing for `b`
                    ),
                    Layout::ColumnMajor => (
                        l * lda + i,       // Column-major indexing for `a`
                        j * ldb + l,       // Column-major indexing for `b`
                    ),
                };

                // Dereference alpha, a, and b to use the values in operations
                sum = sum + (*alpha) * a[a_index] * b[b_index];
            }

            // Determine the index for `c` and update the value
            let c_index = match layout {
                Layout::RowMajor => i * ldc + j,
                Layout::ColumnMajor => j * ldc + i,
            };

            // Update the result matrix `c`, dereferencing beta
            c[c_index] = (*beta) * c[c_index] + sum;
        }
    }
}