#!/usr/bin/env python
# coding: utf-8

"""
This module is used for Matrix-Vector operation
"""

__author__ = "Phuc"

import copy

# -------- Matrix initialization --------
def zeros(m,n):
    """
    Create a zero matrix of size mxn
    """
    return [[0 for i in range(n)] for j in range(m)]

def identity(n):
    """
    Create an identity matrix of size nxn
    """
    C = zeros(n,n)
    for i in range(n):
        C[i][i] = 1
    return C

# -------- Matrix identity --------
def is_column(a):
    """
    Check if vector a is a column vector
    """
    m, n = dim(a)
    return m > 1 and n == 1

def is_row(a):
    """
    Check if vector a is a row vector
    """
    m, n = dim(a)
    return m == 1 and n > 1

def is_vector(a):
    """
    Check if a is a vector
    """
    return is_row(a) or is_column(a)

def is_scalar(a):
    """
    Check if a is a scalar
    """
    m, n = dim(a)
    return m == 1 and n == 1

def is_equal(A, B):
    """
    Check if matrix A and B are equal
    """
    if not dim(A) == dim(B):
        return False
    epsilon = 10**-7
    for i in range(len(A)):
        for j in range(len(A[0])):
            if abs(A[i][j] - B[i][j]) > epsilon:
                return False
    return True

def is_zero(A):
    """
    Check if A is a zero matrix
    """
    return is_equal(A, zeros(dim(A)[0], dim(A)[1]))

# -------- Matrix property --------
def dim(A):
    """
    Return the size of matrix A
    """
    if isinstance(A, int) or isinstance(A, float):
        return 1,1
    # row vector
    elif isinstance(A, list) and not isinstance(A[0], list):
        return 1, len(A)
    elif isinstance(A[0], list) and not isinstance(A, list):
        return len(A), 1
    else:
        return len(A), len(A[0])
    
def mat_get_row(A,i):
    """
    Return the row at index i of matrix A
    """
    return [A[i]]

def mat_get_col(A,j):
    """
    Return the column at index j of matrix A
    """
    return [[A[i][j]] for i in range(len(A))]

def mat_get_sub(A, i1, j1, i2, j2):
    """
    Return submatrix of A from (i1,j1) to (i2,j2)
    """
    C = zeros(i2-i1+1, j2-j1+1)
    for i in range(i1, i2+1):
        for j in range(j1, j2+1):
            C[i-i1][j-j1] = A[i][j]
    return C

def tr(A):
    """
    Calculate the trace of matrix A
    """
    if not dim(A)[0] == dim(A)[1]:
        return f"Not a square matrix {dim(A)}"
    return sum(A[i][i] for i in range(len(A)))

# -------- Basic matrix operation --------
def transpose(A):
    """
    Transpose matrix A
    """
    n_C, m_C = dim(A)
    C = zeros(m_C, n_C)
    if is_scalar(A):
        C = [A]
    elif is_row(A):
        for i in range(len(A[0])):
            C[i][0] = A[0][i]
    elif is_column(A):
        for i in range(len(A)):
            C[0][i] = A[i][0]
    else:
        for i in range (m_C):
            for j in range (n_C):
                C[i][j] = A[j][i]
    return C

def mat_augmented(A,B):
    """
    Combine A and B to an augmented matrix
    """
    m_A, n_A = dim(A)
    m_B, n_B = dim(B)
    if not m_A == m_B:
        return f"Invalid size matrices {dim(A)} and {dim(B)}"
    C = zeros(m_A, n_A+n_B)
    for i in range(m_A):
        for j in range(n_A):
            C[i][j] = A[i][j]
        for j in range(n_B):
            C[i][j+n_A] = B[i][j]
    return C
    
def mat_add(A,B):
    """
    Add entry-wise matrix B to A
    """
    C = copy.deepcopy(A)
    if not dim(A) == dim(B):
        return "Unequal size A and B"
    
    for i in range(len(A)):
        for j in range(len(A[0])):
            C[i][j] += B[i][j]
    return C

def mat_sub(A,B):
    """
    Subtract entry-wise matrix B from A
    """
    C = copy.deepcopy(A)
    if not dim(A) == dim(B):
        return "Unequal size A and B"
    
    for i in range(len(A)):
        for j in range(len(A[0])):
            C[i][j] -= B[i][j]
    return C

def mat_scal_mul(A, alpha):
    """
    Multiply entry-wise matrix A with scalar alpha
    """
    C = copy.deepcopy(A)
    for i in range(len(A)):
        for j in range(len(A[0])):
            C[i][j] *= alpha
    return C

def dot(a,b):
    """
    Calculate dot product between vector a and b
    """
    if not (is_vector(a) and is_vector(b)):
        return f"Invalid vector size {dim(a)} and {dim(b)}"
    if not is_row(a):
        a = transpose(a)
    if not is_column(b):
        b = transpose(b)
    if not dim(a)[1] == dim(b)[0]:
        return f"Invalid vector size {dim(a)} and {dim(b)}"
    return sum(a[0][i]*b[i][0] for i in range(len(a[0]))) 

def mat_mat_mul(A,B):
    """
    Implement matrix-matrix multiplication
    """
    if not dim(A)[1] == dim(B)[0]:
        return f"Invalid matrix size {dim(A)} and {dim(B)}"
    
    C = zeros(dim(A)[0], dim(B)[1])
    for i in range(len(A)):
        for j in range(len(B[0])):
            C[i][j] = dot(mat_get_row(A,i), mat_get_col(B,j))
    return C

def inverse(A):
    """
    Return the inverse of matrix A
    """
    m_A, n_A = dim(A)
    if not dim(A)[0] == dim(A)[1]:
        return f"Can not inverse a non-square matrix {dim(A)}"
    I = identity(dim(A)[0])
    C = mat_augmented(A,I)
    C = backward_helper(forward_helper(C, 0, 0, m_A-1, n_A-1), 0, 0, m_A-1, n_A-1)
    inv = mat_get_col(C, n_A)
    for j in range(n_A+1, n_A+n_A):
        inv = mat_augmented(inv, mat_get_col(C, j))
    if is_zero(mat_get_sub(C, len(C)-1,0,len(C)-1, m_A-1)):
        return "This matrix has no inverse"
    return inv

# -------- Elementary row operation --------
def row_interchange(A, row1, row2):
    """
    Interchange row1 and row2 of matrix A
    """
    if max(row1, row2) >= len(A):
        return f"There is no such pair ({row1}, {row2})"
    I = identity(dim(A)[0])
    I[row1][row1] = 0
    I[row1][row2] = 1
    I[row2][row2] = 0
    I[row2][row1] = 1

    return mat_mat_mul(I,A)

def row_add(A, alpha, row1, row2):
    """
    row2 := alpha*row1 + row2
    """
    if max(row1, row2) >= len(A):
        return f"There is no such pair ({row1}, {row2})"
    I = identity(dim(A)[0])
    I[row2][row1] = alpha
    return mat_mat_mul(I,A)
    
def row_mul(A, alpha, row):
    """
    row := alpha*row
    """
    if row >= len(A):
        return f"Row index exceeds the number of row {len(A)}"
    I = identity(dim(A)[0])
    I[row][row] = alpha
    return mat_mat_mul(I,A)

# -------- Gaussian-Jordan elimination --------
def forward_helper(A, start_row, start_column, end_row, end_column):
    """
    Recursive helper method for forward(A)
    with elementary row operation on submatrix from start_row and start_column to end_row and end_column
    """
    row = start_row
    column = start_column
    
    # base case
    if (row == end_row+1):
        return A
    else:
        # 5. move zero row to the last row of matrix
        if is_zero(mat_get_sub(mat_get_row(A, row), 0, column, 0, end_column)):
            if row == end_row:
                return A
            temp_row = end_row
            while is_zero(mat_get_sub(mat_get_row(A, temp_row), 0, column, 0, end_column)):
                temp_row -= 1
            if temp_row <= row:
                return A
            else:
                A = row_interchange(A, row, temp_row)
        # 1. locate the leftmost non-zero column
        while(column < end_column+1 and is_zero(mat_get_sub(mat_get_col(A, column), row, 0, end_row, 0))): # move to the next column if current column is zero-column
            column += 1
        if column == end_column+1:
            return A
        for i in range(row, end_row+1):
            if not A[i][column] == 0:
                A = row_interchange(A, row, i) # 2. interchange    
                break
        # 3. multiply the top row by 1/a
        alpha = A[row][column]
        A = row_mul(A, 1/alpha, row)
        # 4. multiply the top row with a suitable number
        if not(row == end_row):
            for i in range (row+1, end_row+1):
                if not A[i][column] == 0:
                    alpha = A[i][column]
                    A = row_add(A, -alpha, row, i)
            
        # 6. Recursion        
        return forward_helper(A, row+1, column+1, end_row, end_column)
    
def backward_helper(A, start_row, start_column, end_row, end_column):
    """
    Recursive helper method for backward(A)
    """
    row = end_row
    # base case
    if row == start_row:
        return A
    
    # recursive case
    if is_zero(mat_get_row(A, row)):
        return backward_helper(A, start_row, start_column, row-1, end_column)
    else:
        column = -1
        for i in range(start_column, end_column+1):
            if A[row][i] == 1:
                column = i
                break
                
        for i in range(row-1, start_row-1, -1):
            alpha = A[i][column]
            if not alpha == 0: # ignore 0 entry
                A = row_add(A, -alpha, row, i)

        return backward_helper(A, start_row, start_column, row-1, end_column)
    
def reduced_echelon_form(A):
    """
    Calculate the reduced echelon form of matrix A using Gaussian-Jordan elimination method
    """
    def forward(A):
        return forward_helper(A,0,0, len(A)-1, len(A[0])-1)

    def backward(A):
        return backward_helper(A,0,0, len(A)-1, len(A[0])-1)
    return backward(forward(A))