# def solution(A):
#     # Implement your solution here
#     a_set = set(A)
#     i = 1
#     while True:
#         if i not in a_set:
#             return i
#         i = i + 1
#         print(i)
#
#
# # A = [1, 3, 6, 4, 1, 2]
# A = [-1, -3]
# solution(A)

# 1
# def solution(S):
#     # Implement your solution here
#     pass

# 2:
# def solution(A, B):
#     N = len(A)
#     dp = [0] * N
#     dp[0] = A[0]
#     for i in range(1, N):
#         dp[i] = max(dp[i - 1], A[i])
#     dp[0] = max(dp[0], B[0])
#     for i in range(1, N):
#         dp[i] = min(max(dp[i], B[i]), max(dp[i - 1], B[i]))
#     return dp[-1]

# def solution(A, B):
#     N = len(A)
#     # We need to keep track of the maximum value encountered so far for each path.
#     dp = [0] * N  # dp[i] will hold the max value for the optimal path ending at column i
#
#     # The max value for the first cell is the value of the first cell itself
#     dp[0] = A[0]
#
#     # Fill the dp array for the first row (A)
#     for i in range(1, N):
#         dp[i] = max(dp[i-1], A[i])
#
#     # Now, we need to consider the bottom row (B)
#     # The only way to get to B[0] is from A[0], so we compare the current max with B[0]
#     dp[0] = max(dp[0], B[0])
#
#     # Iterate over the remaining cells in the bottom row
#     for i in range(1, N):
#         # The max value for the cell in B row is the max between the current cell in B,
#         # the previous cell in B, and the cell above it in A
#         dp[i] = min(max(dp[i], B[i]), max(dp[i-1], B[i]))
#
#     # The last cell of dp will have the max value for the optimal path ending at B[N-1]
#     return dp[-1]
#
# # Test the function with the given example
# A = [3, 4, 6]
# B = [6, 5, 4]
# solution(A, B)


# def solution(A, B):
#     l_a = len(A)
#     gird = [0] * l_a
#     gird[0] = A[0]
#
#     for i in range(1, l_a):
#         gird[i] = max(gird[i - 1], A[i])
#
#
#     gird[0] = max(gird[0], B[0])
#     for i in range(1, l_a):
#         gird[i] = min(max(gird[i], B[i]), max(gird[i - 1], B[i]))
#
#     return gird[-1]
#
#
# # Test the function with the given example
# # A = [3, 4, 6]
# # B = [6, 5, 4]
#
# A = [-5, -1, -3]
# B = [-5, 5, -2]
# print(solution(A, B))


def solution(S):
    minimum_swaps = float('inf')
    for i in range(len(S) + 1):
        red_right = S[i:].count('R')
        white_left = S[:i].count('W')
        swaps = red_right + white_left
        minimum_swaps = min(minimum_swaps, swaps)
    return minimum_swaps if minimum_swaps <= 10 ** 9 else -1


example = "WWW"
print(solution(example))


