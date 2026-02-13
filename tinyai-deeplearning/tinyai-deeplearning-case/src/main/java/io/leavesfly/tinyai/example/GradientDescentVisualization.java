package io.leavesfly.tinyai.example;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

/**
 * 梯度下降可视化演示 - 理解优化算法的内部机制
 *
 * <p>本演示通过可视化展示:
 * <ul>
 *   <li>损失函数的等高线图（2D可视化）</li>
 *   <li>梯度下降的路径轨迹</li>
 *   <li>不同学习率的收敛效果对比</li>
 *   <li>损失值的变化曲线</li>
 * </ul>
 *
 * <p><b>运行方式:</b>
 * <pre>
 * mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.example.GradientDescentVisualization" \
 *   -pl tinyai-deeplearning-case
 * </pre>
 *
 * <p><b>学习目标:</b>
 * 1. 直观理解梯度下降的工作原理
 * 2. 观察学习率对收敛的影响
 * 3. 理解局部最优和全局最优的概念
 *
 * @author TinyAI Team
 */
public class GradientDescentVisualization {

    // 目标函数: f(x,y) = x² + 2y² （一个简单的二次函数，有全局最小值在(0,0)）
    // 梯度: ∇f = [2x, 4y]

    private static final int MAX_ITERATIONS = 50;
    private static final float CONVERGENCE_THRESHOLD = 0.001f;

    public static void main(String[] args) {
        System.out.println("==========================================");
        System.out.println("    梯度下降可视化演示 - 理解优化过程    ");
        System.out.println("==========================================\n");

        explainObjectiveFunction();

        // 演示1: 标准梯度下降
        System.out.println("\n【演示1】标准梯度下降（学习率=0.1）");
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        demonstrateGradientDescent(0.1f, "标准");

        // 演示2: 不同学习率对比
        System.out.println("\n\n【演示2】不同学习率对比");
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        compareLearningRates();

        // 演示3: 可视化等高线和路径
        System.out.println("\n\n【演示3】梯度下降路径可视化");
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        visualizeDescentPath();

        // 演示4: 学习率过大的情况
        System.out.println("\n\n【演示4】学习率过大的影响");
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        demonstrateLargeLearningRate();

        showSummary();
    }

    /**
     * 解释目标函数
     */
    private static void explainObjectiveFunction() {
        System.out.println("【目标函数】");
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        System.out.println();
        System.out.println("我们使用一个简单的二次函数作为演示:");
        System.out.println();
        System.out.println("    f(x, y) = x² + 2y²");
        System.out.println();
        System.out.println("特性:");
        System.out.println("  • 这是一个凸函数，有唯一的全局最小值");
        System.out.println("  • 最小值点在 (0, 0)，此时 f(0,0) = 0");
        System.out.println("  • 梯度: ∇f = [∂f/∂x, ∂f/∂y] = [2x, 4y]");
        System.out.println();
        System.out.println("等高线示意图:");
        System.out.println("        y");
        System.out.println("        ↑");
        System.out.println("      2 │    ≈8    ≈4    ≈2");
        System.out.println("        │       ╭──╮");
        System.out.println("      1 │    ≈4  │≈2│  ≈1");
        System.out.println("        │      ╭─┼─┼─╮");
        System.out.println("    ────┼─────╭──┼0┼──╮────→ x");
        System.out.println("        │      ╰─┼─┼─╯");
        System.out.println("     -1 │    ≈4  │≈2│  ≈1");
        System.out.println("        │       ╰──╯");
        System.out.println("     -2 │    ≈8    ≈4    ≈2");
        System.out.println();
    }

    /**
     * 演示梯度下降过程
     */
    private static void demonstrateGradientDescent(float learningRate, String label) {
        // 初始点
        float x = 3.0f;
        float y = 2.0f;

        System.out.printf("初始点: (%.2f, %.2f), 损失值: %.4f%n", x, y, objectiveFunction(x, y));
        System.out.println();
        System.out.println("迭代过程:");
        System.out.println("  迭代  │    x    │    y    │  损失值  │  梯度范数");
        System.out.println("  ──────┼─────────┼─────────┼──────────┼──────────");

        for (int i = 0; i < MAX_ITERATIONS; i++) {
            float loss = objectiveFunction(x, y);
            float[] grad = computeGradient(x, y);
            float gradNorm = (float) Math.sqrt(grad[0] * grad[0] + grad[1] * grad[1]);

            System.out.printf("  %3d   │ %7.4f │ %7.4f │ %8.4f │ %8.4f%n",
                i, x, y, loss, gradNorm);

            // 更新参数
            x -= learningRate * grad[0];
            y -= learningRate * grad[1];

            // 检查收敛
            if (gradNorm < CONVERGENCE_THRESHOLD) {
                System.out.printf("  → 收敛于第%d次迭代%n", i + 1);
                break;
            }

            // 每10次迭代显示一次
            if (i > 0 && i % 10 == 0 && i < MAX_ITERATIONS - 1) {
                System.out.println("  ...   │   ...   │   ...   │   ...    │   ...");
            }
        }

        System.out.printf("%n最终点: (%.4f, %.4f)%n", x, y);
        System.out.printf("最终损失值: %.6f%n", objectiveFunction(x, y));
    }

    /**
     * 对比不同学习率
     */
    private static void compareLearningRates() {
        float[] learningRates = {0.01f, 0.1f, 0.3f, 0.5f};
        float initialX = 3.0f;
        float initialY = 2.0f;

        System.out.println("从初始点 (3.0, 2.0) 开始，对比不同学习率的收敛情况:");
        System.out.println();
        System.out.println("学习率  │ 迭代次数 │ 最终损失值 │ 收敛状态");
        System.out.println("────────┼──────────┼────────────┼──────────");

        for (float lr : learningRates) {
            Result result = runGradientDescent(initialX, initialY, lr, MAX_ITERATIONS);
            String status = result.converged ? "✓ 收敛" : "✗ 未收敛";
            if (result.diverged) status = "✗ 发散";

            System.out.printf(" %.2f   │    %2d    │  %8.6f  │ %s%n",
                lr, result.iterations, result.finalLoss, status);
        }

        System.out.println();
        System.out.println("观察结果:");
        System.out.println("  • 学习率太小(0.01): 收敛慢，需要更多迭代");
        System.out.println("  • 学习率适中(0.1): 收敛快且稳定");
        System.out.println("  • 学习率较大(0.3, 0.5): 可能震荡或发散");
    }

    /**
     * 可视化下降路径
     */
    private static void visualizeDescentPath() {
        float learningRate = 0.15f;
        float x = 3.0f;
        float y = 2.0f;

        System.out.println("梯度下降路径（ASCII可视化）:");
        System.out.println();

        // 创建简单的网格可视化
        int gridSize = 15;
        char[][] grid = new char[gridSize][gridSize];

        // 初始化网格
        for (int i = 0; i < gridSize; i++) {
            for (int j = 0; j < gridSize; j++) {
                grid[i][j] = '·';
            }
        }

        // 标记中心点（最小值）
        grid[gridSize/2][gridSize/2] = '★';

        // 记录路径
        int prevGridX = -1, prevGridY = -1;

        for (int iter = 0; iter < 20; iter++) {
            // 映射到网格坐标
            int gridX = (int) ((x + 4) / 8 * (gridSize - 1));
            int gridY = (int) ((4 - y) / 8 * (gridSize - 1));  // Y轴翻转

            // 确保在范围内
            gridX = Math.max(0, Math.min(gridSize - 1, gridX));
            gridY = Math.max(0, Math.min(gridSize - 1, gridY));

            // 标记路径
            if (grid[gridY][gridX] == '·' || grid[gridY][gridX] == '★') {
                if (iter == 0) {
                    grid[gridY][gridX] = 'S';  // 起点
                } else {
                    grid[gridY][gridX] = (char) ('0' + Math.min(iter, 9));
                }
            }

            // 更新参数
            float[] grad = computeGradient(x, y);
            x -= learningRate * grad[0];
            y -= learningRate * grad[1];

            // 检查收敛
            float gradNorm = (float) Math.sqrt(grad[0] * grad[0] + grad[1] * grad[1]);
            if (gradNorm < 0.01f) break;
        }

        // 打印网格
        System.out.println("      y");
        System.out.println("      ↑");
        for (int i = 0; i < gridSize; i++) {
            System.out.printf("  %3d │ ", (int)(4 - i * 8.0 / (gridSize - 1)));
            for (int j = 0; j < gridSize; j++) {
                System.out.print(grid[i][j] + " ");
            }
            System.out.println();
        }
        System.out.println("      └" + "─".repeat(gridSize * 2));
        System.out.print("        ");
        for (int j = 0; j < gridSize; j++) {
            System.out.printf("%d ", (int)(-4 + j * 8.0 / (gridSize - 1)));
        }
        System.out.println("→ x");
        System.out.println();
        System.out.println("图例: S=起点, ★=最小值(0,0), 数字=迭代次数");
    }

    /**
     * 演示学习率过大的情况
     */
    private static void demonstrateLargeLearningRate() {
        float learningRate = 0.6f;  // 过大的学习率
        float x = 3.0f;
        float y = 2.0f;

        System.out.printf("学习率 = %.1f（过大）%n", learningRate);
        System.out.println();
        System.out.println("迭代过程:");
        System.out.println("  迭代  │    x    │    y    │  损失值  ");
        System.out.println("  ──────┼─────────┼─────────┼──────────");

        for (int i = 0; i < 10; i++) {
            float loss = objectiveFunction(x, y);
            System.out.printf("  %3d   │ %7.2f │ %7.2f │ %8.2f%n", i, x, y, loss);

            float[] grad = computeGradient(x, y);
            x -= learningRate * grad[0];
            y -= learningRate * grad[1];

            // 检查发散
            if (Math.abs(x) > 100 || Math.abs(y) > 100) {
                System.out.println("  → 参数值过大，算法发散！");
                break;
            }
        }

        System.out.println();
        System.out.println("💡 原因分析:");
        System.out.println("  学习率过大导致每次更新的步长太大，");
        System.out.println("  越过了最小值点，甚至使损失值增大。");
    }

    /**
     * 目标函数: f(x,y) = x² + 2y²
     */
    private static float objectiveFunction(float x, float y) {
        return x * x + 2 * y * y;
    }

    /**
     * 计算梯度: ∇f = [2x, 4y]
     */
    private static float[] computeGradient(float x, float y) {
        return new float[]{2 * x, 4 * y};
    }

    /**
     * 运行梯度下降并返回结果
     */
    private static Result runGradientDescent(float x, float y, float lr, int maxIter) {
        float finalX = x, finalY = y;
        int iterations = 0;
        boolean converged = false;
        boolean diverged = false;

        for (int i = 0; i < maxIter; i++) {
            float[] grad = computeGradient(finalX, finalY);
            float gradNorm = (float) Math.sqrt(grad[0] * grad[0] + grad[1] * grad[1]);

            finalX -= lr * grad[0];
            finalY -= lr * grad[1];
            iterations++;

            if (gradNorm < CONVERGENCE_THRESHOLD) {
                converged = true;
                break;
            }

            if (Math.abs(finalX) > 100 || Math.abs(finalY) > 100) {
                diverged = true;
                break;
            }
        }

        return new Result(iterations, objectiveFunction(finalX, finalY), converged, diverged);
    }

    /**
     * 结果显示类
     */
    private static class Result {
        int iterations;
        float finalLoss;
        boolean converged;
        boolean diverged;

        Result(int iterations, float finalLoss, boolean converged, boolean diverged) {
            this.iterations = iterations;
            this.finalLoss = finalLoss;
            this.converged = converged;
            this.diverged = diverged;
        }
    }

    /**
     * 显示总结
     */
    private static void showSummary() {
        System.out.println("\n\n【总结】");
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        System.out.println();
        System.out.println("梯度下降算法要点:");
        System.out.println();
        System.out.println("1. 核心思想");
        System.out.println("   沿着梯度的反方向更新参数，逐步逼近最小值");
        System.out.println("   θ_new = θ_old - learning_rate × ∇J(θ)");
        System.out.println();
        System.out.println("2. 学习率选择");
        System.out.println("   • 太小: 收敛慢，容易陷入局部最优");
        System.out.println("   • 适中: 收敛快且稳定");
        System.out.println("   • 太大: 震荡或发散");
        System.out.println();
        System.out.println("3. 收敛判断");
        System.out.println("   • 梯度范数接近0");
        System.out.println("   • 损失值变化很小");
        System.out.println("   • 达到最大迭代次数");
        System.out.println();
        System.out.println("4. 实际应用技巧");
        System.out.println("   • 使用学习率衰减（逐渐减小学习率）");
        System.out.println("   • 使用动量（Momentum）加速收敛");
        System.out.println("   • 使用自适应学习率优化器（Adam, RMSprop）");
        System.out.println();
        System.out.println("==========================================");
        System.out.println("         演示完成！继续探索深度学习吧！    ");
        System.out.println("==========================================");
    }
}
