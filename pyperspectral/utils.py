import logging

import numpy as np


def unfold_cube(cube: np.ndarray) -> np.ndarray:
    """
    Unfolds a cube of shape (m, n, c) into a matrix of shape (m*n, c)
    @param cube:
    @return:
    """
    if not np.ndim(cube) == 3:
        raise AttributeError("Input array does not have 3 dimensions.")
    rows, cols, bins = cube.shape
    matrix_unfolded = cube.reshape(rows * cols, bins)
    return matrix_unfolded


def fold_matrix(matrix: np.ndarray, cube_dims: tuple) -> np.ndarray:
    """

    @param matrix:
    @param cube_dims: a len 2 or len3 tuple. if there is no no 3rd dimension, the second matrix dimension will be used.
    @return:
    """
    assert np.ndim(matrix) == 2
    m, n, *c = cube_dims

    # If 3rd dim is missing, use matrix width.
    if not c:
        c = [matrix.shape[-1]]
        logging.info(f"No 3rd dimension provided. Defaulting to {c}.")
    cube = matrix.reshape((m, n, *c))
    return cube


if __name__ == '__main__':
    mat = np.array([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [1, 2, 3, 3],
                    [5, 5, 4, 3]])
    shape = (2, 2)
    fold_matrix(mat, shape)