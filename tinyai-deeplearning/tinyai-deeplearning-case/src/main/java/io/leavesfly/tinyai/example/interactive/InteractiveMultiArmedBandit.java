package io.leavesfly.tinyai.example.interactive;

import io.leavesfly.tinyai.example.interactive.base.InteractiveDemo;
import io.leavesfly.tinyai.example.interactive.util.ConsoleVisualizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * 交互式多臂老虎机演示
 *
 * <p>让用户能够体验不同的探索-利用策略：
 * <ul>
 *   <li>Epsilon-Greedy 策略</li>
 *   <li>UCB (Upper Confidence Bound) 策略</li>
 *   <li>Thompson Sampling 策略</li>
 * </ul>
 *
 * <p><b>运行方式:</b>
 * <pre>
 * mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.example.interactive.InteractiveMultiArmedBandit" \
 *   -pl tinyai-deeplearning-case
 * </pre>
 *
 * @author TinyAI Team
 */
public class InteractiveMultiArmedBandit extends InteractiveDemo {

    // 老虎机类
    private static class SlotMachine {
        final int id;
        final double trueMean;  // 真实的平均奖励
        int pulls;
        double totalReward;

        SlotMachine(int id, double trueMean) {
            this.id = id;
            this.trueMean = trueMean;
            this.pulls = 0;
            this.totalReward = 0;
        }

        double pull(Random random) {
            // 模拟拉动老虎机，返回奖励（高斯分布）
            double reward = trueMean + random.nextGaussian() * 0.5;
            pulls++;
            totalReward += reward;
            return reward;
        }

        double getEstimatedMean() {
            return pulls == 0 ? 0 : totalReward / pulls;
        }
    }

    // 策略接口
    private interface Strategy {
        int selectMachine(List<SlotMachine> machines, int totalPulls, Random random);
        String getName();
        String getDescription();
    }

    // Epsilon-Greedy 策略
    private static class EpsilonGreedy implements Strategy {
        private final double epsilon;

        EpsilonGreedy(double epsilon) {
            this.epsilon = epsilon;
        }

        @Override
        public int selectMachine(List<SlotMachine> machines, int totalPulls, Random random) {
            if (random.nextDouble() < epsilon) {
                // 探索：随机选择
                return random.nextInt(machines.size());
            } else {
                // 利用：选择当前估计奖励最高的
                int bestMachine = 0;
                double bestMean = machines.get(0).getEstimatedMean();
                for (int i = 1; i < machines.size(); i++) {
                    double mean = machines.get(i).getEstimatedMean();
                    if (mean > bestMean) {
                        bestMean = mean;
                        bestMachine = i;
                    }
                }
                return bestMachine;
            }
        }

        @Override
        public String getName() {
            return "Epsilon-Greedy (ε=" + epsilon + ")";
        }

        @Override
        public String getDescription() {
            return "以概率 " + epsilon + " 随机探索，否则选择当前最优";
        }
    }

    // UCB 策略
    private static class UCB implements Strategy {
        private final double c;

        UCB(double c) {
            this.c = c;
        }

        @Override
        public int selectMachine(List<SlotMachine> machines, int totalPulls, Random random) {
            int bestMachine = 0;
            double bestUCB = Double.NEGATIVE_INFINITY;

            for (int i = 0; i < machines.size(); i++) {
                SlotMachine machine = machines.get(i);
                double ucb;

                if (machine.pulls == 0) {
                    ucb = Double.POSITIVE_INFINITY;  // 优先尝试未探索的
                } else {
                    double mean = machine.getEstimatedMean();
                    double uncertainty = c * Math.sqrt(Math.log(totalPulls + 1) / machine.pulls);
                    ucb = mean + uncertainty;
                }

                if (ucb > bestUCB) {
                    bestUCB = ucb;
                    bestMachine = i;
                }
            }

            return bestMachine;
        }

        @Override
        public String getName() {
            return "UCB (c=" + c + ")";
        }

        @Override
        public String getDescription() {
            return "平衡探索和利用，考虑不确定性";
        }
    }

    // Thompson Sampling 策略
    private static class ThompsonSampling implements Strategy {
        @Override
        public int selectMachine(List<SlotMachine> machines, int totalPulls, Random random) {
            int bestMachine = 0;
            double bestSample = Double.NEGATIVE_INFINITY;

            for (int i = 0; i < machines.size(); i++) {
                SlotMachine machine = machines.get(i);
                // 使用Beta分布采样（简化为高斯分布）
                double mean = machine.getEstimatedMean();
                double std = machine.pulls == 0 ? 1.0 : 1.0 / Math.sqrt(machine.pulls);
                double sample = mean + random.nextGaussian() * std;

                if (sample > bestSample) {
                    bestSample = sample;
                    bestMachine = i;
                }
            }

            return bestMachine;
        }

        @Override
        public String getName() {
            return "Thompson Sampling";
        }

        @Override
        public String getDescription() {
            return "从后验分布采样，自然平衡探索和利用";
        }
    }

    private List<SlotMachine> machines;
    private final Random random = new Random();
    private int numMachines = 5;
    private int numPulls = 1000;

    public InteractiveMultiArmedBandit() {
        super("交互式多臂老虎机");
    }

    @Override
    protected void run() {
        while (isRunning()) {
            showMainMenu();
            int choice = getChoiceInput("请选择操作", 1, 5);

            switch (choice) {
                case 1:
                    configureEnvironment();
                    break;
                case 2:
                    runSingleStrategy();
                    break;
                case 3:
                    compareStrategies();
                    break;
                case 4:
                    showEnvironmentInfo();
                    break;
                case 5:
                    stop();
                    return;
            }
        }
    }

    private void showMainMenu() {
        clearScreen();
        showSection("主菜单");

        visualizer.println("当前配置:", ConsoleVisualizer.CYAN);
        visualizer.println("  老虎机数量: " + numMachines, ConsoleVisualizer.RESET);
        visualizer.println("  拉动次数: " + numPulls, ConsoleVisualizer.RESET);
        visualizer.println("", ConsoleVisualizer.RESET);

        showMenu("操作选项", new String[]{
            "配置环境（老虎机数量、拉动次数）",
            "运行单一策略",
            "对比不同策略",
            "查看环境信息",
            "退出"
        });
    }

    private void configureEnvironment() {
        showSection("环境配置");

        numMachines = getIntInput("设置老虎机数量", 3, 20, numMachines);
        numPulls = getIntInput("设置拉动次数", 100, 10000, numPulls);

        // 初始化老虎机
        initializeMachines();

        showSuccess("环境配置完成！");
        pause();
    }

    private void initializeMachines() {
        machines = new ArrayList<>();
        Random initRandom = new Random();

        // 创建具有不同真实奖励分布的老虎机
        for (int i = 0; i < numMachines; i++) {
            // 真实奖励在 0 到 2 之间，有些老虎机更好
            double trueMean = initRandom.nextDouble() * 2.0;
            machines.add(new SlotMachine(i, trueMean));
        }

        // 确保至少有一个好老虎机
        int bestMachine = initRandom.nextInt(numMachines);
        machines.set(bestMachine, new SlotMachine(bestMachine, 1.8));
    }

    private void runSingleStrategy() {
        if (machines == null) {
            initializeMachines();
        }

        showSection("选择策略");

        visualizer.println("可用策略:", ConsoleVisualizer.BOLD);
        visualizer.println("  [1] Epsilon-Greedy - 以概率ε随机探索", ConsoleVisualizer.RESET);
        visualizer.println("  [2] UCB - 上置信界算法", ConsoleVisualizer.RESET);
        visualizer.println("  [3] Thompson Sampling - 汤普森采样", ConsoleVisualizer.RESET);

        int strategyChoice = getChoiceInput("请选择策略", 1, 3);

        Strategy strategy;
        switch (strategyChoice) {
            case 1:
                double epsilon = getFloatInput("设置Epsilon (0-1)", 0.0f, 1.0f, 0.1f);
                strategy = new EpsilonGreedy(epsilon);
                break;
            case 2:
                double c = getFloatInput("设置UCB参数c", 0.1f, 5.0f, 1.0f);
                strategy = new UCB(c);
                break;
            case 3:
                strategy = new ThompsonSampling();
                break;
            default:
                strategy = new EpsilonGreedy(0.1);
        }

        // 重置老虎机状态
        for (SlotMachine machine : machines) {
            machine.pulls = 0;
            machine.totalReward = 0;
        }

        runStrategy(strategy);
    }

    private void runStrategy(Strategy strategy) {
        showSection("运行策略: " + strategy.getName());
        visualizer.println(strategy.getDescription(), ConsoleVisualizer.CYAN);
        visualizer.println("", ConsoleVisualizer.RESET);

        List<Double> cumulativeRewards = new ArrayList<>();
        double totalReward = 0;

        // 运行策略
        for (int pull = 0; pull < numPulls; pull++) {
            int selectedMachine = strategy.selectMachine(machines, pull, random);
            double reward = machines.get(selectedMachine).pull(random);
            totalReward += reward;
            cumulativeRewards.add(totalReward);

            // 显示进度
            if ((pull + 1) % 100 == 0 || pull == numPulls - 1) {
                showProgress(pull + 1, numPulls, "拉动进度");
            }
        }

        visualizer.println("", ConsoleVisualizer.RESET);
        visualizer.println("", ConsoleVisualizer.RESET);

        // 显示结果
        showResults(strategy.getName(), totalReward, cumulativeRewards);
    }

    private void showResults(String strategyName, double totalReward, List<Double> cumulativeRewards) {
        showSection("结果: " + strategyName);

        // 总体统计
        visualizer.println("总体统计:", ConsoleVisualizer.BOLD);
        visualizer.println("  总奖励: " + String.format("%.2f", totalReward), ConsoleVisualizer.GREEN);
        visualizer.println("  平均奖励: " + String.format("%.4f", totalReward / numPulls), ConsoleVisualizer.GREEN);
        visualizer.println("", ConsoleVisualizer.RESET);

        // 每台老虎机的统计
        visualizer.println("每台老虎机统计:", ConsoleVisualizer.BOLD);
        String[] headers = {"老虎机", "真实均值", "拉动次数", "估计均值", "占比"};
        int[] widths = {10, 12, 12, 12, 10};

        visualizer.printTable(headers, new String[][]{}, widths);

        for (SlotMachine machine : machines) {
            double percentage = (double) machine.pulls / numPulls * 100;
            String[] row = {
                String.valueOf(machine.id),
                String.format("%.2f", machine.trueMean),
                String.valueOf(machine.pulls),
                String.format("%.4f", machine.getEstimatedMean()),
                String.format("%.1f%%", percentage)
            };
            visualizer.printTable(headers, new String[][]{row}, widths);
        }

        // 累积奖励曲线
        visualizer.println("", ConsoleVisualizer.RESET);
        visualizer.println("累积奖励曲线 (前200步):", ConsoleVisualizer.BOLD);

        List<Float> displayRewards = new ArrayList<>();
        for (int i = 0; i < Math.min(200, cumulativeRewards.size()); i += 5) {
            displayRewards.add(cumulativeRewards.get(i).floatValue());
        }
        visualizer.printLineChart(displayRewards, 60, 10, null);

        pause();
    }

    private void compareStrategies() {
        if (machines == null) {
            initializeMachines();
        }

        showSection("策略对比");

        // 保存原始状态
        List<Double> trueMeans = new ArrayList<>();
        for (SlotMachine machine : machines) {
            trueMeans.add(machine.trueMean);
        }

        // 要对比的策略
        Strategy[] strategies = {
            new EpsilonGreedy(0.0),   // 纯利用
            new EpsilonGreedy(0.1),   // 标准Epsilon-Greedy
            new EpsilonGreedy(0.3),   // 高探索
            new UCB(1.0),
            new ThompsonSampling()
        };

        String[] headers = {"策略", "总奖励", "平均奖励", "最优机器选择率"};
        int[] widths = {25, 12, 12, 16};

        visualizer.printTable(headers, new String[][]{}, widths);

        int bestMachineId = 0;
        double bestMean = trueMeans.get(0);
        for (int i = 1; i < trueMeans.size(); i++) {
            if (trueMeans.get(i) > bestMean) {
                bestMean = trueMeans.get(i);
                bestMachineId = i;
            }
        }

        for (Strategy strategy : strategies) {
            // 重置老虎机状态
            machines.clear();
            for (int i = 0; i < numMachines; i++) {
                machines.add(new SlotMachine(i, trueMeans.get(i)));
            }

            double totalReward = 0;
            int bestMachinePulls = 0;

            for (int pull = 0; pull < numPulls; pull++) {
                int selectedMachine = strategy.selectMachine(machines, pull, random);
                if (selectedMachine == bestMachineId) {
                    bestMachinePulls++;
                }
                double reward = machines.get(selectedMachine).pull(random);
                totalReward += reward;
            }

            double avgReward = totalReward / numPulls;
            double bestMachineRate = (double) bestMachinePulls / numPulls * 100;

            String[] row = {
                strategy.getName(),
                String.format("%.2f", totalReward),
                String.format("%.4f", avgReward),
                String.format("%.1f%%", bestMachineRate)
            };

            visualizer.printTable(headers, new String[][]{row}, widths);
        }

        visualizer.println("", ConsoleVisualizer.RESET);
        visualizer.println("观察:", ConsoleVisualizer.BOLD);
        visualizer.println("  • 纯利用(Epsilon=0): 可能陷入局部最优", ConsoleVisualizer.RESET);
        visualizer.println("  • 适度探索(Epsilon=0.1): 通常表现最好", ConsoleVisualizer.RESET);
        visualizer.println("  • UCB和Thompson Sampling: 理论上最优", ConsoleVisualizer.RESET);

        pause();
    }

    private void showEnvironmentInfo() {
        if (machines == null) {
            initializeMachines();
        }

        showSection("环境信息");

        visualizer.println("老虎机配置:", ConsoleVisualizer.BOLD);
        visualizer.println("  数量: " + numMachines, ConsoleVisualizer.RESET);
        visualizer.println("", ConsoleVisualizer.RESET);

        visualizer.println("真实奖励分布:", ConsoleVisualizer.BOLD);

        // 找到最优机器
        int bestMachine = 0;
        double bestMean = machines.get(0).trueMean;
        for (int i = 1; i < machines.size(); i++) {
            if (machines.get(i).trueMean > bestMean) {
                bestMean = machines.get(i).trueMean;
                bestMachine = i;
            }
        }

        for (SlotMachine machine : machines) {
            String marker = (machine.id == bestMachine) ? " ★最优" : "";
            visualizer.println(String.format("  老虎机 %d: 真实均值 = %.2f%s",
                machine.id, machine.trueMean, marker), ConsoleVisualizer.RESET);
        }

        visualizer.println("", ConsoleVisualizer.RESET);
        visualizer.println("说明:", ConsoleVisualizer.BOLD);
        visualizer.println("  ★ 标记的是最优老虎机", ConsoleVisualizer.YELLOW);
        visualizer.println("  每次拉动的奖励 = 真实均值 + 随机噪声", ConsoleVisualizer.RESET);
        visualizer.println("  目标是尽快找到并主要拉动最优老虎机", ConsoleVisualizer.RESET);

        pause();
    }

    public static void main(String[] args) {
        InteractiveMultiArmedBandit demo = new InteractiveMultiArmedBandit();
        demo.start();
    }
}
