import numpy as np
np.set_printoptions(precision=3) # set numpy matrix display of floating point not exceed 3 decimal points

def forward_helper(A, row, column):
    """
    Recursive helper method for forward(A)
    """
    # base case
    if (row == len(A)):
#         print("\nDONE\n", A)
        return A
    else:
        # 5. move zero row to the last row of matrix
        if is_zero_row(A, row):
            if row == len(A) - 1:
                return A
            temp_row = len(A)-1
            while is_zero_row(A, temp_row):
                temp_row -= 1
            if temp_row == row:
                return A
            else:
                A = row_interchange(A, row, temp_row)
        # 1. locate the leftmost non-zero column
        while(is_zero_column(A, row, column)): # move to the next column if current column is zero-column
            column += 1
        if column == len(A[0]-1):
            return A
        for i in range(row, len(A)):
            if not A[i][column] == 0:
                A = row_interchange(A, row, i) # 2. interchange
                
#                 print(f"Interchange row {row} with row {i} \n", A)
                
                break
        # 3. multiply the top row by 1/a
        alpha = A[row][column]
        A = row_scalar_multiply(A, row, 1/alpha)
        
#         print(f"Multiply row {row} with 1/{alpha} \n", A)
        
        # 4. multiply the top row with a suitable number
        if not(row == len(A)-1):
            for i in range (row+1, len(A)):
                if not A[i][column] == 0:
                    alpha = A[i][column]
                    B = row_addition(row_scalar_multiply(A, row, -1*alpha), row, i)
                    A = row_scalar_multiply(B, row, -1/alpha)
                    
#             print(f"Multiply the row {row} with a suitable number and addd it to other", \
#             "below rows so that all other elements of the current column is 0 \n", A) 
            
        # 6. Recursion        
        return forward_helper(A, row+1, column+1)

def backward_helper(A, row):
    """
    Recursive helper method for backward(A)
    """
    # base case
    if row == 0:
        return A
    
    # recursive case
    if is_zero_row(A, row):
        return backward_helper(A, row-1)
    else:
        column = -1
        for i in range(len(A[0])):
            if A[row][i] == 1:
                column = i
                break
                
        for i in range(row-1, -1, -1):
            alpha = A[i][column]
            if not alpha == 0: # ignore 0 entry
                B = row_addition(row_scalar_multiply(A, row, -1*alpha), row, i)
                A = row_scalar_multiply(B, row, -1/alpha)
#         print(f"Row {row} \n", A)
        return backward_helper(A, row-1)

def forward(A):
    """
    Forward step of Gaussian-Jordan elimination
    @Param numpy matrix A
    @Return row echelon form of matrix A
    """
    return forward_helper(A,0,0)

def backward(A):
    """
    Backward step of Gaussian-Jordan elimination
    @Param echelon form of matrix A
    @Return reduced row echelon form of matrix A
    """
    return backward_helper(A, len(A)-1)

# Set of matrix row operation routines
def row_interchange(A, row1, row2):
    """
    Interchange 2 rows (row1 and row2) of matrix A
    @Return the updated matrix
    """
    A[[row1, row2]] = A[[row2, row1]]
    return A

def row_scalar_multiply(A, row, alpha):
    """
    Multiply all entries in row of matrix A with scalar alpha
    @Return the updated matrix
    """
    A[row] = A[row]*alpha
    return A
    
def row_addition(A, row1, row2):
    """
    Add entry-wisely row1 to row2 of matrix A
    @Return the updated matrix
    """
    A[row2] = A[row1] + A[row2]
    return A
    
def is_zero_row(A, row):
    """
    Check if row of matrix A is a zero-row. 
    @Return True if it is, False otherwise
    """
    return np.sum(A[row]**2) == 0

def is_zero_column(A, row, column):
    """
    Check if sub column below row of matrix A is a zero sub column
    @Return True if it is, False otherwise
    """
    return np.sum(A[row:,column]**2) == 0