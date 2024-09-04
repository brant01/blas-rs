use std::ops::{Add, AddAssign, Mul, MulAssign, Neg};
use num_traits::{One, Zero};
use crate::shared::{Layout, Op};

pub fn transform_matrix<T>(
    layout: &Layout,
    matrix: &[T],
    op: &Op,
    ldx: usize,
) -> Vec<T>
where
    T: Mul<Output = T>
        + MulAssign
        + Add<Output = T>
        + AddAssign
        + Zero
        + One
        + PartialEq
        + Copy
        + Default
        + Neg<Output = T>,
{
    match (op, layout) {
        (Op::NoOp, _) => matrix.to_vec(),
        (Op::Transpose, Layout::RowMajor) => row_transpose(matrix, ldx),
        (Op::Transpose, Layout::ColumnMajor) => col_transpose(matrix, ldx),
        (Op::Hermitian, Layout::RowMajor) => row_hermitian(matrix, ldx),
        (Op::Hermitian, Layout::ColumnMajor) => col_hermitian(matrix, ldx),
    }
}

fn row_transpose<T>(matrix: &[T], ldx: usize) -> Vec<T>
where
    T: Copy + Default,
{
    let n = matrix.len() / ldx;
    let mut transposed = vec![T::default(); matrix.len()];

    for i in 0..ldx {
        for j in 0..n {
            transposed[j * ldx + i] = matrix[i * n + j];
        }
    }

    transposed
}

fn row_hermitian<T>(matrix: &[T], ldx: usize) -> Vec<T>
where
    T: Copy + Default + Neg<Output = T>,
{
    let n = matrix.len() / ldx;
    let mut hermitian = vec![T::default(); matrix.len()];

    for i in 0..ldx {
        for j in 0..n {
            hermitian[j * ldx + i] = -matrix[i * n + j]; // Negate the value directly
        }
    }

    hermitian
}

fn col_transpose<T>(matrix: &[T], ldx: usize) -> Vec<T>
where
    T: Copy + Default,
{
    let n = matrix.len() / ldx;
    let mut transposed = vec![T::default(); matrix.len()];

    for i in 0..ldx {
        for j in 0..n {
            transposed[i * n + j] = matrix[j * ldx + i];
        }
    }

    transposed
}

fn col_hermitian<T>(matrix: &[T], ldx: usize) -> Vec<T>
where
    T: Copy + Default + Neg<Output = T>,
{
    let n = matrix.len() / ldx;
    let mut hermitian = vec![T::default(); matrix.len()];

    for i in 0..ldx {
        for j in 0..n {
            hermitian[i * n + j] = -matrix[j * ldx + i]; // Negate the value directly
        }
    }

    hermitian
}