package io.leavesfly.tinyai.nnet.init;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

/**
 * 正交初始化器
 * <p>
 * 生成正交矩阵，常用于RNN的权重初始化。
 * 实现思路参考 PyTorch：对随机矩阵做正交化（Gram-Schmidt），
 * 根据行列大小选择 Q 或 Q^T，最后缩放 gain。
 */
public class OrthogonalInitializer implements Initializer {

    private static final float EPS = 1e-8f;
    private final float gain;

    /**
     * 构造函数
     *
     * @param gain 增益系数（默认为1.0）
     */
    public OrthogonalInitializer(float gain) {
        this.gain = gain;
    }

    /**
     * 默认构造函数（gain=1.0）
     */
    public OrthogonalInitializer() {
        this(1.0f);
    }

    @Override
    public void initialize(NdArray tensor) {
        int[] dims = tensor.getShape().getShapeDims();
        int rows = dims[0];
        int cols = 1;
        for (int i = 1; i < dims.length; i++) {
            cols *= dims[i];
        }

        int dim = Math.max(rows, cols);
        float[][] random = NdArray.randn(Shape.of(dim, dim)).getMatrix();
        float[][] q = gramSchmidt(random);

        // 如果行数小于列数，使用 Q^T 以保证输出形状正确
        if (rows < cols) {
            q = transpose2D(q);
        }

        float[] data = tensor.getArray();
        int idx = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[idx++] = q[i][j] * gain;
            }
        }
    }

    private float[][] gramSchmidt(float[][] a) {
        int m = a.length;
        int n = a[0].length;
        float[][] q = new float[m][n];

        for (int j = 0; j < n; j++) {
            float[] v = new float[m];
            for (int i = 0; i < m; i++) {
                v[i] = a[i][j];
            }

            for (int k = 0; k < j; k++) {
                float dot = 0f;
                for (int i = 0; i < m; i++) {
                    dot += q[i][k] * v[i];
                }
                for (int i = 0; i < m; i++) {
                    v[i] -= dot * q[i][k];
                }
            }

            float norm = 0f;
            for (float value : v) {
                norm += value * value;
            }
            norm = (float) Math.sqrt(norm);
            if (norm < EPS) {
                norm = EPS;
            }

            for (int i = 0; i < m; i++) {
                q[i][j] = v[i] / norm;
            }
        }
        return q;
    }

    private float[][] transpose2D(float[][] src) {
        int m = src.length;
        int n = src[0].length;
        float[][] t = new float[n][m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                t[j][i] = src[i][j];
            }
        }
        return t;
    }
}
