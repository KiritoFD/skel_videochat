#include <bits/stdc++.h>
using namespace std;

// 检查在 (i,j) 填 c 是否合法
bool isValid(const vector<string>& mat, int i, int j, char c) {
    for (int i2 = 0; i2 < i; ++i2) {
        for (int j2 = 0; j2 < j; ++j2) {
            if (mat[i2][j2] == mat[i2][j] &&
                mat[i2][j] == mat[i][j2] &&
                mat[i][j2] == c) {
                return false;
            }
        }
    }
    return true;
}

// DFS：pos = i * m + j
bool dfs(vector<string>& mat, int n, int m, int pos) {
    if (pos == n * m) return true;
    int i = pos / m, j = pos % m;
    // 优先尝试 '0'（保证字典序最小）
    for (char c : {'0', '1'}) {
        if (isValid(mat, i, j, c)) {
            mat[i][j] = c;
            if (dfs(mat, n, m, pos + 1)) 
                return true;
        }
    }
    return false;
}

int main() {
    cout << "=== 打表：n 行 m 列，无单色矩形 01 矩阵 ===\n\n";
    
    // 设置打表范围
    const int MAX_N = 10;
    const int MAX_M = 10;

    for (int n = 1; n <= MAX_N; ++n) {
        for (int m = 1; m <= MAX_M; ++m) {
            cout << "n=" << n << ", m=" << m << ": ";
            vector<string> mat(n, string(m, ' '));
            if (dfs(mat, n, m, 0)) {
                cout << "✅\n";
                for (int i = 0; i < n; ++i) {
                    cout << "  " << mat[i] << "\n";
                }
            } else {
                cout << "❌ No solution\n";
            }
            cout << "\n";
        }
    }

    return 0;
}