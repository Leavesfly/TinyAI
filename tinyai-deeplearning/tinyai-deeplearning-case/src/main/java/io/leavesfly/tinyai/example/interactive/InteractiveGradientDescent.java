package io.leavesfly.tinyai.example.interactive;

import io.leavesfly.tinyai.example.interactive.base.InteractiveDemo;
import io.leavesfly.tinyai.example.interactive.util.ConsoleVisualizer;

import java.util.ArrayList;
import java.util.List;

/**
 * 交互式梯度下降 Playground
 *
 * <p>让用户能够实时调整参数，观察梯度下降的行为：
 * <ul>
 *   <li>实时调整学习率</li>
 *   <li>切换不同优化器</li>
 *   <li>选择不同目标函数</li>
 *   <li>可视化收敛过程</li>
 * </ul>
 *
 * <p><b>运行方式:</b>
 * <pre>
 * mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.example.interactive.InteractiveGradientDescent" \
 *   -pl tinyai-deeplearning-case
 * </pre>
 *
 * @author TinyAI Team
 */
public class InteractiveGradientDescent extends InteractiveDemo {

    // 目标函数定义
    private enum ObjectiveFunction {
        QUADRATIC("二次函数 (x² + 2y²)", "凸函数，有全局最小值"),
        ROSENBROCK("Rosenbrock函数", "经典测试函数，有狭长山谷"),
        SADDLE("鞍点函数 (x² - y²)", "测试鞍点逃逸能力"),
        RIPPLE("波纹函数", "多局部最优，测试跳出能力");

        final String name;
        final String description;

        ObjectiveFunction(String name, String description) {
            this.name = name;
            this.description = description;
        }
    }

    // 优化器类型
    private enum Optimizer {
        SGD("SGD", "标准梯度下降"),
        MOMENTUM("Momentum", "带动量的梯度下降"),
        ADAM("Adam", "自适应学习率");

        final String name;
        final String description;

        Optimizer(String name, String description) {
            this.name = name;
            this.description = description;
        }
    }

    private ObjectiveFunction currentFunction = ObjectiveFunction.QUADRATIC;
    private Optimizer currentOptimizer = Optimizer.SGD;
    private float learningRate = 0.1f;
    private float momentum = 0.9f;
    private float x = 3.0f;
    private float y = 2.0f;
    private float vx = 0.0f;  // 速度（用于Momentum）
    private float vy = 0.0f;

    private final List<Float> lossHistory = new ArrayList<>();
    private final List<Float> xHistory = new ArrayList<>();
    private final List<Float> yHistory = new ArrayList<>();

    private boolean paused = false;
    private int iteration = 0;
    private final int maxIterations = 100;

    public InteractiveGradientDescent() {
        super("交互式梯度下降 Playground");
    }

    @Override
    protected void run() {
        while (isRunning()) {
            showMainMenu();
            int choice = getChoiceInput("请选择操作", 1, 6);

            switch (choice) {
                case 1:
                    configureParameters();
                    break;
                case 2:
                    runGradientDescent();
                    break;
                case 3:
                    visualizeFunction();
                    break;
                case 4:
                    compareOptimizers();
                    break;
                case 5:
                    resetParameters();
                    break;
                case 6:
                    stop();
                    return;
            }
        }
    }

    private void showMainMenu() {
        clearScreen();
        showSection("主菜单");

        visualizer.println("当前配置:", ConsoleVisualizer.CYAN);
        visualizer.println("  目标函数: " + currentFunction.name + " - " + currentFunction.description, ConsoleVisualizer.RESET);
        visualizer.println("  优化器: " + currentOptimizer.name + " - " + currentOptimizer.description, ConsoleVisualizer.RESET);
        visualizer.println("  学习率: " + learningRate, ConsoleVisualizer.RESET);
        visualizer.println("  初始点: (" + String.format("%.2f", x) + ", " + String.format("%.2f", y) + ")", ConsoleVisualizer.RESET);
        visualizer.println("", ConsoleVisualizer.RESET);

        showMenu("操作选项", new String[]{
            "配置参数（函数、优化器、学习率）",
            "运行梯度下降",
            "可视化目标函数",
            "对比不同优化器",
            "重置参数",
            "退出"
        });
    }

    private void configureParameters() {
        showSection("参数配置");

        // 选择目标函数
        visualizer.println("选择目标函数:", ConsoleVisualizer.BOLD);
        for (int i = 0; i < ObjectiveFunction.values().length; i++) {
            ObjectiveFunction func = ObjectiveFunction.values()[i];
            visualizer.println("  [" + (i + 1) + "] " + func.name + " - " + func.description, ConsoleVisualizer.RESET);
        }
        int funcChoice = getChoiceInput("请选择函数", 1, ObjectiveFunction.values().length);
        currentFunction = ObjectiveFunction.values()[funcChoice - 1];

        // 选择优化器
        visualizer.println("", ConsoleVisualizer.RESET);
        visualizer.println("选择优化器:", ConsoleVisualizer.BOLD);
        for (int i = 0; i < Optimizer.values().length; i++) {
            Optimizer opt = Optimizer.values()[i];
            visualizer.println("  [" + (i + 1) + "] " + opt.name + " - " + opt.description, ConsoleVisualizer.RESET);
        }
        int optChoice = getChoiceInput("请选择优化器", 1, Optimizer.values().length);
        currentOptimizer = Optimizer.values()[optChoice - 1];

        // 设置学习率
        visualizer.println("", ConsoleVisualizer.RESET);
        learningRate = getFloatInput("设置学习率", 0.001f, 1.0f, learningRate);

        // 设置初始点
        visualizer.println("", ConsoleVisualizer.RESET);
        x = getFloatInput("设置初始点 x", -5.0f, 5.0f, x);
        y = getFloatInput("设置初始点 y", -5.0f, 5.0f, y);

        showSuccess("参数配置完成！");
        pause();
    }

    private void runGradientDescent() {
        showSection("梯度下降运行中");

        // 重置历史
        lossHistory.clear();
        xHistory.clear();
        yHistory.clear();
        vx = 0.0f;
        vy = 0.0f;
        iteration = 0;

        visualizer.println("目标函数: " + currentFunction.name, ConsoleVisualizer.CYAN);
        visualizer.println("优化器: " + currentOptimizer.name, ConsoleVisualizer.CYAN);
        visualizer.println("学习率: " + learningRate, ConsoleVisualizer.CYAN);
        visualizer.println("", ConsoleVisualizer.RESET);

        visualizer.println("按 [Enter] 暂停/继续, [Q] 退出", ConsoleVisualizer.YELLOW);
        visualizer.println("", ConsoleVisualizer.RESET);

        // 运行梯度下降
        for (iteration = 0; iteration < maxIterations && isRunning(); iteration++) {
            // 检查暂停
            if (paused) {
                handlePause();
                if (!isRunning()) break;
            }

            // 计算损失和梯度
            float loss = computeLoss(x, y);
            float[] grad = computeGradient(x, y);

            // 记录历史
            lossHistory.add(loss);
            xHistory.add(x);
            yHistory.add(y);

            // 更新参数
            updateParameters(grad);

            // 显示进度
            if (iteration % 5 == 0 || iteration < 10) {
                displayProgress(iteration, loss, grad);
            }

            // 检查收敛
            float gradNorm = (float) Math.sqrt(grad[0] * grad[0] + grad[1] * grad[1]);
            if (gradNorm < 0.001f) {
                visualizer.println("", ConsoleVisualizer.RESET);
                showSuccess("收敛于第 " + (iteration + 1) + " 次迭代！");
                break;
            }

            // 短暂延迟以便观察
            sleep(100);
        }

        // 显示最终结果
        visualizer.println("", ConsoleVisualizer.RESET);
        showSection("最终结果");
        visualizer.println("最终点: (" + String.format("%.6f", x) + ", " + String.format("%.6f", y) + ")", ConsoleVisualizer.GREEN);
        visualizer.println("最终损失: " + String.format("%.6f", computeLoss(x, y)), ConsoleVisualizer.GREEN);
        visualizer.println("迭代次数: " + (iteration + 1), ConsoleVisualizer.GREEN);

        // 显示损失曲线
        visualizer.println("", ConsoleVisualizer.RESET);
        showLossCurve();

        // 显示轨迹
        visualizer.println("", ConsoleVisualizer.RESET);
        showTrajectory();

        pause();
    }

    private void displayProgress(int iter, float loss, float[] grad) {
        float gradNorm = (float) Math.sqrt(grad[0] * grad[0] + grad[1] * grad[1]);
        String progress = String.format("Iter %3d: x=%7.4f, y=%7.4f, Loss=%8.4f, |grad|=%7.4f",
            iter, x, y, loss, gradNorm);
        visualizer.println(progress, ConsoleVisualizer.CYAN);
    }

    private void handlePause() {
        visualizer.println("\n[已暂停] 按 [Enter] 继续, [Q] 退出, [R] 重置", ConsoleVisualizer.YELLOW);
        String input = getStringInput("");
        if (input.equalsIgnoreCase("q")) {
            stop();
        } else if (input.equalsIgnoreCase("r")) {
            resetParameters();
        }
        paused = false;
    }

    private void updateParameters(float[] grad) {
        switch (currentOptimizer) {
            case SGD:
                x -= learningRate * grad[0];
                y -= learningRate * grad[1];
                break;
            case MOMENTUM:
                vx = momentum * vx - learningRate * grad[0];
                vy = momentum * vy - learningRate * grad[1];
                x += vx;
                y += vy;
                break;
            case ADAM:
                // 简化的Adam实现
                float beta1 = 0.9f;
                float beta2 = 0.999f;
                float epsilon = 1e-8f;

                vx = beta1 * vx + (1 - beta1) * grad[0];
                vy = beta1 * vy + (1 - beta1) * grad[1];

                float vxCorr = vx / (1 - (float) Math.pow(beta1, iteration + 1));
                float vyCorr = vy / (1 - (float) Math.pow(beta1, iteration + 1));

                x -= learningRate * vxCorr;
                y -= learningRate * vyCorr;
                break;
        }
    }

    private float computeLoss(float x, float y) {
        switch (currentFunction) {
            case QUADRATIC:
                return x * x + 2 * y * y;
            case ROSENBROCK:
                float a = 1.0f;
                float b = 100.0f;
                return (a - x) * (a - x) + b * (y - x * x) * (y - x * x);
            case SADDLE:
                return x * x - y * y;
            case RIPPLE:
                return (float) (Math.sin(x * x + y * y) / (x * x + y * y + 0.1));
            default:
                return x * x + y * y;
        }
    }

    private float[] computeGradient(float x, float y) {
        float eps = 1e-5f;
        float loss = computeLoss(x, y);
        float lossX = computeLoss(x + eps, y);
        float lossY = computeLoss(x, y + eps);

        return new float[]{
            (lossX - loss) / eps,
            (lossY - loss) / eps
        };
    }

    private void visualizeFunction() {
        showSection("目标函数可视化");

        visualizer.println("函数: " + currentFunction.name, ConsoleVisualizer.BOLD);
        visualizer.println("", ConsoleVisualizer.RESET);

        // 简单的ASCII等高线图
        int size = 20;
        float minVal = -3.0f;
        float maxVal = 3.0f;

        visualizer.println("等高线图 (x: " + minVal + " to " + maxVal + ", y: " + minVal + " to " + maxVal + ")", ConsoleVisualizer.CYAN);
        visualizer.println("", ConsoleVisualizer.RESET);

        // 打印Y轴标签
        visualizer.print("      ", ConsoleVisualizer.RESET);
        for (int i = 0; i < size; i++) {
            if (i % 5 == 0) {
                float xVal = minVal + (maxVal - minVal) * i / (size - 1);
                visualizer.print(String.format("%5.1f", xVal), ConsoleVisualizer.YELLOW);
            } else {
                visualizer.print("     ", ConsoleVisualizer.RESET);
            }
        }
        visualizer.println("", ConsoleVisualizer.RESET);
        visualizer.println("      " + "─".repeat(size * 5), ConsoleVisualizer.CYAN);

        for (int j = 0; j < size; j++) {
            float yVal = maxVal - (maxVal - minVal) * j / (size - 1);

            if (j % 5 == 0) {
                visualizer.print(String.format("%5.1f │", yVal), ConsoleVisualizer.YELLOW);
            } else {
                visualizer.print("      │", ConsoleVisualizer.RESET);
            }

            for (int i = 0; i < size; i++) {
                float xVal = minVal + (maxVal - minVal) * i / (size - 1);
                float loss = computeLoss(xVal, yVal);

                // 根据损失值选择字符
                char c;
                String color;
                if (loss < 1.0f) {
                    c = '★';
                    color = ConsoleVisualizer.GREEN;
                } else if (loss < 5.0f) {
                    c = '·';
                    color = ConsoleVisualizer.CYAN;
                } else if (loss < 20.0f) {
                    c = ':';
                    color = ConsoleVisualizer.YELLOW;
                } else {
                    c = ' ';
                    color = ConsoleVisualizer.RESET;
                }

                visualizer.print("  " + c + "  ", color);
            }
            visualizer.println("", ConsoleVisualizer.RESET);
        }

        visualizer.println("", ConsoleVisualizer.RESET);
        visualizer.println("图例: ★=低损失 ·=中低损失 :=中高损失", ConsoleVisualizer.CYAN);

        pause();
    }

    private void compareOptimizers() {
        showSection("优化器对比");

        float initialX = x;
        float initialY = y;
        float initialLR = learningRate;

        visualizer.println("对比配置:", ConsoleVisualizer.BOLD);
        visualizer.println("  初始点: (" + initialX + ", " + initialY + ")", ConsoleVisualizer.RESET);
        visualizer.println("  学习率: " + initialLR, ConsoleVisualizer.RESET);
        visualizer.println("  目标函数: " + currentFunction.name, ConsoleVisualizer.RESET);
        visualizer.println("", ConsoleVisualizer.RESET);

        String[] headers = {"优化器", "迭代次数", "最终损失", "收敛状态"};
        int[] widths = {12, 12, 15, 12};

        visualizer.printTable(headers, new String[][]{}, widths);

        for (Optimizer opt : Optimizer.values()) {
            // 重置参数
            x = initialX;
            y = initialY;
            vx = 0.0f;
            vy = 0.0f;
            currentOptimizer = opt;

            int iters = 0;
            float finalLoss = 0.0f;
            boolean converged = false;

            for (iters = 0; iters < maxIterations; iters++) {
                float[] grad = computeGradient(x, y);
                updateParameters(grad);
                finalLoss = computeLoss(x, y);

                float gradNorm = (float) Math.sqrt(grad[0] * grad[0] + grad[1] * grad[1]);
                if (gradNorm < 0.001f) {
                    converged = true;
                    break;
                }
            }

            String status = converged ? "✓ 收敛" : "✗ 未收敛";
            String[] row = {
                opt.name,
                String.valueOf(iters),
                String.format("%.6f", finalLoss),
                status
            };

            visualizer.printTable(headers, new String[][]{row}, widths);
        }

        // 恢复原始参数
        x = initialX;
        y = initialY;
        currentOptimizer = Optimizer.SGD;

        pause();
    }

    private void showLossCurve() {
        visualizer.println("损失值变化曲线:", ConsoleVisualizer.BOLD);
        visualizer.printLineChart(lossHistory, 60, 15, null);
    }

    private void showTrajectory() {
        visualizer.println("优化轨迹 (前20步):", ConsoleVisualizer.BOLD);
        int steps = Math.min(20, xHistory.size());
        for (int i = 0; i < steps; i++) {
            visualizer.println(String.format("  Step %2d: (%.4f, %.4f) -> Loss: %.4f",
                i, xHistory.get(i), yHistory.get(i), lossHistory.get(i)), ConsoleVisualizer.RESET);
        }
    }

    private void resetParameters() {
        x = 3.0f;
        y = 2.0f;
        learningRate = 0.1f;
        currentOptimizer = Optimizer.SGD;
        currentFunction = ObjectiveFunction.QUADRATIC;
        vx = 0.0f;
        vy = 0.0f;
        lossHistory.clear();
        xHistory.clear();
        yHistory.clear();
        showSuccess("参数已重置为默认值！");
    }

    public static void main(String[] args) {
        InteractiveGradientDescent demo = new InteractiveGradientDescent();
        demo.start();
    }
}
