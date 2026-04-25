package io.leavesfly.tinyai.banana.encoder;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.core.Module;
import io.leavesfly.tinyai.nnet.core.Parameter;

/**
 * 2D 可学习位置编码（行列分解实现）。
 *
 * <p>为图像 patches 提供真正的二维空间位置信息。与朴素的"展平 1D 可学习嵌入"相比，
 * 本实现显式地将位置分解为 <b>行嵌入</b> 和 <b>列嵌入</b> 两组参数：</p>
 * <pre>
 *   rowEmbedding : [1, numRows, H]
 *   colEmbedding : [1, numCols, H]
 *   pos[r, c, :] = rowEmbedding[0, r, :] + colEmbedding[0, c, :]
 * </pre>
 *
 * <p>优势：</p>
 * <ul>
 *   <li><b>参数更少</b>：{@code (numRows + numCols) * H} 远小于 {@code numRows * numCols * H}；</li>
 *   <li><b>空间先验明确</b>：同一行 / 同一列的 patch 共享一部分位置信息，泛化更好；</li>
 *   <li><b>对 imageSize 更友好</b>：行列独立调整，便于后续支持变分辨率。</li>
 * </ul>
 *
 * <p>与 1D 位置编码的区别：</p>
 * <ul>
 *   <li>1D：序列位置 {@code [0, 1, ..., seq_len-1]}</li>
 *   <li>2D：网格位置 {@code [(0,0), (0,1), ..., (h-1, w-1)]}</li>
 * </ul>
 *
 * <p>输入：Patch 序列 {@code [batch, numPatches, H]}；
 * 输出：位置编码 {@code [1, numPatches, H]}，可通过广播叠加到任意 batch。</p>
 *
 * @author leavesfly
 * @version 2.0
 */
public class Position2D extends Module {

    private final int numRows;      // 行数（等于 imageSize / patchSize）
    private final int numCols;      // 列数
    private final int numPatches;   // numRows × numCols
    private final int hiddenSize;   // 嵌入维度

    /** 行方向可学习嵌入 {@code [1, numRows, H]}。 */
    private final Parameter rowEmbedding;

    /** 列方向可学习嵌入 {@code [1, numCols, H]}。 */
    private final Parameter colEmbedding;

    /**
     * 便捷构造：由 {@code numPatches} 推导行列，要求 {@code numPatches} 为完全平方数。
     *
     * @param name       模块名称
     * @param numPatches Patch 总数（必须是完全平方数，如 64、256、576 等）
     * @param hiddenSize 嵌入维度
     * @throws IllegalArgumentException 当 {@code numPatches} 不是完全平方数时
     */
    public Position2D(String name, int numPatches, int hiddenSize) {
        this(name, inferSideLength(numPatches), inferSideLength(numPatches), hiddenSize);
    }

    /**
     * 显式指定行列的构造函数。
     *
     * @param name       模块名称
     * @param numRows    行数
     * @param numCols    列数
     * @param hiddenSize 嵌入维度
     */
    public Position2D(String name, int numRows, int numCols, int hiddenSize) {
        super(name);
        if (numRows <= 0 || numCols <= 0 || hiddenSize <= 0) {
            throw new IllegalArgumentException(String.format(
                    "非法参数: numRows=%d, numCols=%d, hiddenSize=%d（必须均为正）",
                    numRows, numCols, hiddenSize));
        }
        this.numRows = numRows;
        this.numCols = numCols;
        this.numPatches = numRows * numCols;
        this.hiddenSize = hiddenSize;

        // 行嵌入 [1, numRows, H]
        NdArray rowData = NdArray.of(Shape.of(1, numRows, hiddenSize));
        this.rowEmbedding = registerParameter("row_emb", new Parameter(rowData));

        // 列嵌入 [1, numCols, H]
        NdArray colData = NdArray.of(Shape.of(1, numCols, hiddenSize));
        this.colEmbedding = registerParameter("col_emb", new Parameter(colData));

        init();
    }

    /**
     * 推导正方形网格的边长；要求 {@code numPatches} 是完全平方数。
     */
    private static int inferSideLength(int numPatches) {
        int side = (int) Math.round(Math.sqrt(numPatches));
        if (side * side != numPatches) {
            throw new IllegalArgumentException(String.format(
                    "numPatches=%d 不是完全平方数，无法自动推导为正方形网格；" +
                            "请使用 Position2D(name, numRows, numCols, hiddenSize) 显式指定行列。",
                    numPatches));
        }
        return side;
    }

    @Override
    public void resetParameters() {
        // 使用小方差正态分布初始化，std=0.02 使位置编码在训练初期不会主导特征
        rowEmbedding.setData(NdArray.randn(rowEmbedding.data().getShape()).mulNum(0.02f));
        colEmbedding.setData(NdArray.randn(colEmbedding.data().getShape()).mulNum(0.02f));
    }

    /**
     * 前向传播：动态合成 {@code [1, numPatches, H]} 的位置编码。
     *
     * <p>合成步骤：</p>
     * <ol>
     *   <li>{@code rowEmbedding}: {@code [1, R, H]} → unsqueeze → {@code [1, R, 1, H]} → broadcast → {@code [1, R, C, H]}；</li>
     *   <li>{@code colEmbedding}: {@code [1, C, H]} → unsqueeze → {@code [1, 1, C, H]} → broadcast → {@code [1, R, C, H]}；</li>
     *   <li>两者相加后 reshape 为 {@code [1, R*C, H]}。</li>
     * </ol>
     *
     * @param inputs 可忽略，位置编码独立于输入
     * @return 位置编码 {@code [1, numPatches, H]}，可通过广播叠加到任意 batch
     */
    @Override
    public Variable forward(Variable... inputs) {
        return buildCombinedEmbedding();
    }

    /**
     * 动态合成完整的 2D 位置编码 {@code [1, numPatches, H]}，可参与反向传播。
     */
    public Variable buildCombinedEmbedding() {
        Shape target4D = Shape.of(1, numRows, numCols, hiddenSize);

        // row: [1, R, H] → [1, R, 1, H] → [1, R, C, H]
        Variable rowExpanded = rowEmbedding.unsqueeze(2).broadcastTo(target4D);
        // col: [1, C, H] → [1, 1, C, H] → [1, R, C, H]
        Variable colExpanded = colEmbedding.unsqueeze(1).broadcastTo(target4D);

        // 相加后展平为序列形式
        return rowExpanded.add(colExpanded)
                .reshape(Shape.of(1, numPatches, hiddenSize));
    }

    /**
     * 获取指定 patch 索引处的位置编码（按行优先顺序）。
     *
     * @param patchIndex Patch 索引 {@code [0, numPatches)}
     * @return 该位置的编码向量 {@code [1, 1, H]}
     */
    public Variable getPositionAt(int patchIndex) {
        if (patchIndex < 0 || patchIndex >= numPatches) {
            throw new IllegalArgumentException(
                    "Patch索引越界: " + patchIndex + ", 有效范围[0, " + numPatches + ")"
            );
        }
        int row = patchIndex / numCols;
        int col = patchIndex % numCols;
        return getPositionAt2D(row, col);
    }

    /**
     * 按 2D 坐标获取位置编码（推荐入口）。
     *
     * <p>等价于 {@code rowEmbedding[0, row, :] + colEmbedding[0, col, :]}，
     * 结果形状为 {@code [1, 1, H]}。</p>
     *
     * @param row 行索引 {@code [0, numRows)}
     * @param col 列索引 {@code [0, numCols)}
     * @return 位置编码 {@code [1, 1, H]}
     */
    public Variable getPositionAt2D(int row, int col) {
        if (row < 0 || row >= numRows || col < 0 || col >= numCols) {
            throw new IllegalArgumentException(String.format(
                    "2D 坐标越界: (row=%d, col=%d), 有效范围 row∈[0,%d), col∈[0,%d)",
                    row, col, numRows, numCols));
        }
        // rowEmbedding[:, row:row+1, :]: [1, 1, H]
        Variable r = rowEmbedding.sliceRange(1, row, row + 1);
        Variable c = colEmbedding.sliceRange(1, col, col + 1);
        return r.add(c);
    }

    /**
     * 兼容旧签名的 2D 坐标访问。此方法保留 {@code numPatchesPerRow} 参数仅为兼容历史调用，
     * 实际计算使用对象内的 {@link #numCols}；当传入值与 {@code numCols} 不一致时会抛异常，
     * 以避免静默错位。
     *
     * @param row              行索引
     * @param col              列索引
     * @param numPatchesPerRow 每行的 patch 数量（必须与 {@link #numCols} 一致）
     * @return 位置编码 {@code [1, 1, H]}
     */
    public Variable getPositionAt2D(int row, int col, int numPatchesPerRow) {
        if (numPatchesPerRow != numCols) {
            throw new IllegalArgumentException(String.format(
                    "numPatchesPerRow=%d 与当前配置的 numCols=%d 不一致",
                    numPatchesPerRow, numCols));
        }
        return getPositionAt2D(row, col);
    }

    // ==================== Getter方法 ====================

    public int getNumRows() {
        return numRows;
    }

    public int getNumCols() {
        return numCols;
    }

    public int getNumPatches() {
        return numPatches;
    }

    public int getHiddenSize() {
        return hiddenSize;
    }

    /** 行方向可学习位置嵌入 {@code [1, numRows, H]}。 */
    public Parameter getRowEmbedding() {
        return rowEmbedding;
    }

    /** 列方向可学习位置嵌入 {@code [1, numCols, H]}。 */
    public Parameter getColEmbedding() {
        return colEmbedding;
    }

    @Override
    public String toString() {
        return String.format(
                "Position2D{numRows=%d, numCols=%d, numPatches=%d, hiddenSize=%d}",
                numRows, numCols, numPatches, hiddenSize
        );
    }
}
