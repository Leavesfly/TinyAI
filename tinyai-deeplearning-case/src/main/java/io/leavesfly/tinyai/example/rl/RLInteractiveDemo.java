package io.leavesfly.tinyai.example.rl;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.rl.Agent;
import io.leavesfly.tinyai.rl.Environment;
import io.leavesfly.tinyai.rl.Experience;
import io.leavesfly.tinyai.rl.agent.*;
import io.leavesfly.tinyai.rl.environment.*;

import java.util.Scanner;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;

/**
 * 交互式强化学习演示
 * 
 * 这个演示允许用户：
 * 1. 选择不同的强化学习算法
 * 2. 选择不同的环境
 * 3. 实时观察学习过程
 * 4. 调整超参数
 * 5. 比较不同算法的表现
 * 
 * @author 山泽
 */
public class RLInteractiveDemo {
    
    private Scanner scanner;
    private boolean running;
    
    // 算法选项
    private static final String[] ALGORITHM_OPTIONS = {
        "DQN (Deep Q-Network)",
        "REINFORCE (Policy Gradient)", 
        "ε-贪心多臂老虎机",
        "UCB多臂老虎机",
        "汤普森采样多臂老虎机"
    };
    
    // 环境选项
    private static final String[] ENVIRONMENT_OPTIONS = {
        "CartPole 倒立摆",
        "GridWorld 网格世界",
        "MultiArmedBandit 多臂老虎机"
    };
    
    public RLInteractiveDemo() {
        this.scanner = new Scanner(System.in);
        this.running = true;
    }
    
    public static void main(String[] args) {
        RLInteractiveDemo demo = new RLInteractiveDemo();
        demo.run();
    }
    
    /**
     * 运行交互式演示
     */
    public void run() {
        printWelcomeMessage();
        
        while (running) {
            printMainMenu();
            int choice = getIntInput("请选择操作: ", 1, 5);
            
            switch (choice) {
                case 1:
                    runSingleAlgorithmDemo();
                    break;
                case 2:
                    runAlgorithmComparison();
                    break;
                case 3:
                    runParameterTuning();
                    break;
                case 4:
                    runLearningProcessVisualization();
                    break;
                case 5:
                    running = false;
                    System.out.println("感谢使用TinyAI强化学习演示系统！");
                    break;
            }
        }
        
        scanner.close();
    }
    
    /**
     * 打印欢迎信息
     */
    private void printWelcomeMessage() {
        System.out.println("=========================================");
        System.out.println("      TinyAI 交互式强化学习演示系统        ");
        System.out.println("=========================================");
        System.out.println("欢迎使用TinyAI强化学习模块演示系统！");
        System.out.println("本系统支持多种强化学习算法和环境。");
        System.out.println();
    }
    
    /**
     * 打印主菜单
     */
    private void printMainMenu() {
        System.out.println("\n========== 主菜单 ==========");
        System.out.println("1. 单算法演示");
        System.out.println("2. 算法性能比较");
        System.out.println("3. 参数调优演示");
        System.out.println("4. 学习过程可视化");
        System.out.println("5. 退出系统");
        System.out.println("============================");
    }
    
    /**
     * 运行单算法演示
     */
    private void runSingleAlgorithmDemo() {
        System.out.println("\n========== 单算法演示 ==========");
        
        // 选择算法
        int algorithmChoice = selectAlgorithm();
        
        // 选择环境
        int environmentChoice = selectEnvironment();
        
        // 创建算法和环境
        Agent agent = createAgent(algorithmChoice);
        Environment environment = createEnvironment(environmentChoice);
        
        if (agent == null || environment == null) {
            System.out.println("算法或环境创建失败，返回主菜单。");
            return;
        }
        
        // 设置训练参数
        int episodes = getIntInput("请输入训练回合数 (1-1000): ", 1, 1000);
        boolean showDetails = getBooleanInput("是否显示详细过程? (y/n): ");
        
        // 开始训练
        System.out.println("\n开始训练...");
        runTraining(agent, environment, episodes, showDetails);
        
        // 显示训练结果
        showTrainingResults(agent, environment);
        
        // 测试trained模型
        if (getBooleanInput("是否测试训练后的模型? (y/n): ")) {
            testTrainedModel(agent, environment);
        }
    }
    
    /**
     * 运行算法比较
     */
    private void runAlgorithmComparison() {
        System.out.println("\n========== 算法性能比较 ==========");
        
        // 选择环境
        int environmentChoice = selectEnvironment();
        Environment environment = createEnvironment(environmentChoice);
        
        if (environment == null) {
            System.out.println("环境创建失败，返回主菜单。");
            return;
        }
        
        // 选择要比较的算法
        List<Integer> selectedAlgorithms = selectMultipleAlgorithms();
        
        if (selectedAlgorithms.isEmpty()) {
            System.out.println("未选择任何算法，返回主菜单。");
            return;
        }
        
        // 设置比较参数
        int episodes = getIntInput("请输入每个算法的训练回合数 (1-1000): ", 1, 1000);
        int runs = getIntInput("请输入独立运行次数 (1-20): ", 1, 20);
        
        // 运行比较实验
        runComparisonExperiment(selectedAlgorithms, environment, episodes, runs);
    }
    
    /**
     * 运行参数调优演示
     */
    private void runParameterTuning() {
        System.out.println("\n========== 参数调优演示 ==========");
        
        // 选择算法
        int algorithmChoice = selectAlgorithm();
        
        // 选择环境
        int environmentChoice = selectEnvironment();
        
        if (algorithmChoice == 0) { // DQN
            runDQNParameterTuning(environmentChoice);
        } else if (algorithmChoice == 2) { // ε-贪心
            runEpsilonGreedyParameterTuning(environmentChoice);
        } else {
            System.out.println("该算法暂不支持参数调优演示。");
        }
    }
    
    /**
     * 运行学习过程可视化
     */
    private void runLearningProcessVisualization() {
        System.out.println("\n========== 学习过程可视化 ==========");
        
        // 选择算法和环境
        int algorithmChoice = selectAlgorithm();
        int environmentChoice = selectEnvironment();
        
        Agent agent = createAgent(algorithmChoice);
        Environment environment = createEnvironment(environmentChoice);
        
        if (agent == null || environment == null) {
            System.out.println("算法或环境创建失败，返回主菜单。");
            return;
        }
        
        // 实时展示学习过程
        visualizeLearningProcess(agent, environment);
    }
    
    /**
     * 选择算法
     */
    private int selectAlgorithm() {
        System.out.println("\n可用算法:");
        for (int i = 0; i < ALGORITHM_OPTIONS.length; i++) {
            System.out.println((i + 1) + ". " + ALGORITHM_OPTIONS[i]);
        }
        
        return getIntInput("请选择算法 (1-" + ALGORITHM_OPTIONS.length + "): ", 
                          1, ALGORITHM_OPTIONS.length) - 1;
    }
    
    /**
     * 选择环境
     */
    private int selectEnvironment() {
        System.out.println("\n可用环境:");
        for (int i = 0; i < ENVIRONMENT_OPTIONS.length; i++) {
            System.out.println((i + 1) + ". " + ENVIRONMENT_OPTIONS[i]);
        }
        
        return getIntInput("请选择环境 (1-" + ENVIRONMENT_OPTIONS.length + "): ", 
                          1, ENVIRONMENT_OPTIONS.length) - 1;
    }
    
    /**
     * 选择多个算法
     */
    private List<Integer> selectMultipleAlgorithms() {
        System.out.println("\n请选择要比较的算法 (输入算法编号，用空格分隔，如: 1 3 5):");
        for (int i = 0; i < ALGORITHM_OPTIONS.length; i++) {
            System.out.println((i + 1) + ". " + ALGORITHM_OPTIONS[i]);
        }
        
        System.out.print("选择: ");
        String input = scanner.nextLine().trim();
        
        List<Integer> selected = new ArrayList<>();
        for (String part : input.split("\\s+")) {
            try {
                int choice = Integer.parseInt(part);
                if (choice >= 1 && choice <= ALGORITHM_OPTIONS.length) {
                    selected.add(choice - 1);
                }
            } catch (NumberFormatException e) {
                // 忽略无效输入
            }
        }
        
        return selected;
    }
    
    /**
     * 创建智能体
     */
    private Agent createAgent(int algorithmChoice) {
        switch (algorithmChoice) {
            case 0: // DQN
                return new DQNAgent("DQN", 4, 2, new int[]{64, 64}, 
                                  0.001f, 0.1f, 0.99f, 32, 10000, 100);
            case 1: // REINFORCE
                return new REINFORCEAgent("REINFORCE", 4, 2, new int[]{64, 64}, 
                                        0.001f, 0.99f, false);
            case 2: // ε-贪心多臂老虎机
                return new EpsilonGreedyBanditAgent("ε-贪心", 10, 0.1f);
            case 3: // UCB多臂老虎机
                return new UCBBanditAgent("UCB", 10);
            case 4: // 汤普森采样多臂老虎机
                return new ThompsonSamplingBanditAgent("汤普森采样", 10);
            default:
                return null;
        }
    }
    
    /**
     * 创建环境
     */
    private Environment createEnvironment(int environmentChoice) {
        switch (environmentChoice) {
            case 0: // CartPole
                return new CartPoleEnvironment(500);
            case 1: // GridWorld
                return new GridWorldEnvironment(5, 5);
            case 2: // MultiArmedBandit
                float[] rewards = {0.1f, 0.3f, 0.8f, 0.2f, 0.6f, 0.4f, 0.9f, 0.5f, 0.7f, 0.35f};
                return new MultiArmedBanditEnvironment(rewards, 1000);
            default:
                return null;
        }
    }
    
    /**
     * 运行训练
     */
    private void runTraining(Agent agent, Environment environment, int episodes, boolean showDetails) {
        float totalReward = 0.0f;
        int successfulEpisodes = 0;
        
        for (int episode = 0; episode < episodes; episode++) {
            Variable state = environment.reset();
            float episodeReward = 0.0f;
            int steps = 0;
            
            while (!environment.isDone() && steps < 1000) {
                Variable action = agent.selectAction(state);
                Environment.StepResult result = environment.step(action);
                
                Experience experience = new Experience(
                    state, action, result.getReward(), 
                    result.getNextState(), result.isDone(), steps
                );
                
                agent.learn(experience);
                
                state = result.getNextState();
                episodeReward += result.getReward();
                steps++;
            }
            
            totalReward += episodeReward;
            if (episodeReward > 0) successfulEpisodes++;
            
            if (showDetails && (episode % 50 == 0 || episode < 10)) {
                System.out.println(String.format("回合 %d: 奖励 = %.2f, 步数 = %d", 
                                 episode + 1, episodeReward, steps));
            }
        }
        
        System.out.println(String.format("\n训练完成！平均奖励: %.2f, 成功率: %.1f%%", 
                         totalReward / episodes, (float) successfulEpisodes / episodes * 100));
    }
    
    /**
     * 显示训练结果
     */
    private void showTrainingResults(Agent agent, Environment environment) {
        System.out.println("\n========== 训练结果 ==========");
        System.out.println("智能体: " + agent.getName());
        System.out.println("训练步数: " + agent.getTrainingStep());
        
        if (agent instanceof DQNAgent) {
            DQNAgent dqnAgent = (DQNAgent) agent;
            System.out.println("平均损失: " + String.format("%.6f", dqnAgent.getAverageLoss()));
            System.out.println("当前探索率: " + String.format("%.4f", dqnAgent.getCurrentEpsilon()));
        } else if (agent instanceof BanditAgent) {
            BanditAgent banditAgent = (BanditAgent) agent;
            banditAgent.printStatus();
        }
    }
    
    /**
     * 测试训练后的模型
     */
    private void testTrainedModel(Agent agent, Environment environment) {
        System.out.println("\n========== 模型测试 ==========");
        agent.setTraining(false);
        
        int testEpisodes = 10;
        float totalReward = 0.0f;
        
        for (int episode = 0; episode < testEpisodes; episode++) {
            Variable state = environment.reset();
            float episodeReward = 0.0f;
            int steps = 0;
            
            System.out.println("测试回合 " + (episode + 1) + ":");
            
            while (!environment.isDone() && steps < 1000) {
                Variable action = agent.selectAction(state);
                Environment.StepResult result = environment.step(action);
                
                state = result.getNextState();
                episodeReward += result.getReward();
                steps++;
                
                if (steps % 100 == 0) {
                    System.out.println("  步数 " + steps + ", 累积奖励: " + String.format("%.2f", episodeReward));
                }
            }
            
            totalReward += episodeReward;
            System.out.println("  回合结束，总奖励: " + String.format("%.2f", episodeReward) + ", 总步数: " + steps);
        }
        
        System.out.println("\n测试完成！平均奖励: " + String.format("%.2f", totalReward / testEpisodes));
    }
    
    /**
     * 运行比较实验
     */
    private void runComparisonExperiment(List<Integer> algorithmIndices, Environment environment, 
                                       int episodes, int runs) {
        System.out.println("\n开始算法比较实验...");
        
        Map<String, List<Float>> results = new HashMap<>();
        
        for (int algorithmIndex : algorithmIndices) {
            String algorithmName = ALGORITHM_OPTIONS[algorithmIndex];
            System.out.println("\n测试算法: " + algorithmName);
            
            List<Float> algorithmResults = new ArrayList<>();
            
            for (int run = 0; run < runs; run++) {
                Agent agent = createAgent(algorithmIndex);
                if (agent == null) continue;
                
                float totalReward = 0.0f;
                
                for (int episode = 0; episode < episodes; episode++) {
                    Variable state = environment.reset();
                    float episodeReward = 0.0f;
                    int steps = 0;
                    
                    while (!environment.isDone() && steps < 1000) {
                        Variable action = agent.selectAction(state);
                        Environment.StepResult result = environment.step(action);
                        
                        Experience experience = new Experience(
                            state, action, result.getReward(), 
                            result.getNextState(), result.isDone(), steps
                        );
                        
                        agent.learn(experience);
                        
                        state = result.getNextState();
                        episodeReward += result.getReward();
                        steps++;
                    }
                    
                    totalReward += episodeReward;
                }
                
                float averageReward = totalReward / episodes;
                algorithmResults.add(averageReward);
                
                System.out.print(".");
                if ((run + 1) % 10 == 0) {
                    System.out.println(" " + (run + 1) + "/" + runs);
                }
            }
            
            results.put(algorithmName, algorithmResults);
        }
        
        // 显示比较结果
        displayComparisonResults(results);
    }
    
    /**
     * 显示比较结果
     */
    private void displayComparisonResults(Map<String, List<Float>> results) {
        System.out.println("\n========== 算法比较结果 ==========");
        System.out.printf("%-25s | %12s | %12s | %12s%n", "算法", "平均奖励", "标准差", "最佳表现");
        System.out.println("--------------------------|-------------|-------------|-------------");
        
        for (Map.Entry<String, List<Float>> entry : results.entrySet()) {
            String algorithmName = entry.getKey();
            List<Float> rewards = entry.getValue();
            
            double mean = rewards.stream().mapToDouble(Double::valueOf).average().orElse(0.0);
            double variance = rewards.stream().mapToDouble(r -> Math.pow(r - mean, 2)).average().orElse(0.0);
            double stdDev = Math.sqrt(variance);
            double max = rewards.stream().mapToDouble(Double::valueOf).max().orElse(0.0);
            
            System.out.printf("%-25s | %12.4f | %12.4f | %12.4f%n", 
                             algorithmName, mean, stdDev, max);
        }
    }
    
    /**
     * DQN参数调优
     */
    private void runDQNParameterTuning(int environmentChoice) {
        System.out.println("\n========== DQN参数调优 ==========");
        
        float[] learningRates = {0.1f, 0.01f, 0.001f, 0.0001f};
        float[] epsilons = {0.01f, 0.05f, 0.1f, 0.2f};
        
        Environment environment = createEnvironment(environmentChoice);
        if (environment == null) return;
        
        System.out.println("正在测试不同的学习率和探索率组合...");
        
        float bestReward = Float.NEGATIVE_INFINITY;
        float bestLR = 0, bestEpsilon = 0;
        
        for (float lr : learningRates) {
            for (float eps : epsilons) {
                System.out.println("测试 LR=" + lr + ", ε=" + eps);
                
                DQNAgent agent = new DQNAgent("DQN-Test", 4, 2, new int[]{64, 64}, 
                                            lr, eps, 0.99f, 32, 10000, 100);
                
                float totalReward = 0.0f;
                int episodes = 100;
                
                for (int episode = 0; episode < episodes; episode++) {
                    Variable state = environment.reset();
                    float episodeReward = 0.0f;
                    int steps = 0;
                    
                    while (!environment.isDone() && steps < 500) {
                        Variable action = agent.selectAction(state);
                        Environment.StepResult result = environment.step(action);
                        
                        Experience experience = new Experience(
                            state, action, result.getReward(), 
                            result.getNextState(), result.isDone(), steps
                        );
                        
                        agent.learn(experience);
                        
                        state = result.getNextState();
                        episodeReward += result.getReward();
                        steps++;
                    }
                    
                    totalReward += episodeReward;
                }
                
                float averageReward = totalReward / episodes;
                System.out.println("  平均奖励: " + String.format("%.4f", averageReward));
                
                if (averageReward > bestReward) {
                    bestReward = averageReward;
                    bestLR = lr;
                    bestEpsilon = eps;
                }
            }
        }
        
        System.out.println("\n最佳参数组合:");
        System.out.println("学习率: " + bestLR);
        System.out.println("探索率: " + bestEpsilon);
        System.out.println("最佳平均奖励: " + String.format("%.4f", bestReward));
    }
    
    /**
     * ε-贪心参数调优
     */
    private void runEpsilonGreedyParameterTuning(int environmentChoice) {
        System.out.println("\n========== ε-贪心参数调优 ==========");
        
        if (environmentChoice != 2) {
            System.out.println("ε-贪心参数调优仅支持多臂老虎机环境。");
            return;
        }
        
        float[] epsilons = {0.01f, 0.05f, 0.1f, 0.15f, 0.2f, 0.3f};
        
        MultiArmedBanditEnvironment environment = 
            (MultiArmedBanditEnvironment) createEnvironment(environmentChoice);
        
        System.out.println("正在测试不同的探索率...");
        
        for (float eps : epsilons) {
            System.out.println("测试 ε=" + eps);
            
            EpsilonGreedyBanditAgent agent = new EpsilonGreedyBanditAgent("ε-贪心", 10, eps);
            
            float totalReward = 0.0f;
            int steps = 1000;
            
            environment.reset();
            
            for (int step = 0; step < steps; step++) {
                Variable action = agent.selectAction(environment.getCurrentState());
                Environment.StepResult result = environment.step(action);
                
                Experience experience = new Experience(
                    environment.getCurrentState(), action, result.getReward(), 
                    result.getNextState(), result.isDone(), step
                );
                
                agent.learn(experience);
                totalReward += result.getReward();
            }
            
            float averageReward = totalReward / steps;
            System.out.println("  平均奖励: " + String.format("%.4f", averageReward));
            System.out.println("  最优选择率: " + String.format("%.2f%%", agent.getOptimalActionRate() * 100));
        }
    }
    
    /**
     * 可视化学习过程
     */
    private void visualizeLearningProcess(Agent agent, Environment environment) {
        System.out.println("\n开始实时展示学习过程...");
        System.out.println("(按Enter键继续下一步，输入'skip'跳过详细过程，输入'quit'退出)");
        
        int episode = 0;
        boolean skipDetails = false;
        
        while (episode < 50 && !skipDetails) {
            Variable state = environment.reset();
            float episodeReward = 0.0f;
            int steps = 0;
            
            System.out.println("\n=== 回合 " + (episode + 1) + " ===");
            
            while (!environment.isDone() && steps < 100) {
                Variable action = agent.selectAction(state);
                Environment.StepResult result = environment.step(action);
                
                Experience experience = new Experience(
                    state, action, result.getReward(), 
                    result.getNextState(), result.isDone(), steps
                );
                
                agent.learn(experience);
                
                state = result.getNextState();
                episodeReward += result.getReward();
                steps++;
                
                if (steps % 10 == 0) {
                    System.out.println("步数: " + steps + ", 累积奖励: " + String.format("%.2f", episodeReward));
                    
                    System.out.print("继续? (Enter/skip/quit): ");
                    String input = scanner.nextLine().trim();
                    
                    if ("quit".equalsIgnoreCase(input)) {
                        return;
                    } else if ("skip".equalsIgnoreCase(input)) {
                        skipDetails = true;
                        break;
                    }
                }
            }
            
            System.out.println("回合结束，总奖励: " + String.format("%.2f", episodeReward) + ", 总步数: " + steps);
            episode++;
        }
        
        System.out.println("\n学习过程展示完成！");
    }
    
    /**
     * 获取整数输入
     */
    private int getIntInput(String prompt, int min, int max) {
        while (true) {
            System.out.print(prompt);
            try {
                int value = Integer.parseInt(scanner.nextLine().trim());
                if (value >= min && value <= max) {
                    return value;
                } else {
                    System.out.println("请输入 " + min + " 到 " + max + " 之间的数字。");
                }
            } catch (NumberFormatException e) {
                System.out.println("请输入有效的数字。");
            }
        }
    }
    
    /**
     * 获取布尔输入
     */
    private boolean getBooleanInput(String prompt) {
        while (true) {
            System.out.print(prompt);
            String input = scanner.nextLine().trim().toLowerCase();
            if ("y".equals(input) || "yes".equals(input)) {
                return true;
            } else if ("n".equals(input) || "no".equals(input)) {
                return false;
            } else {
                System.out.println("请输入 y/yes 或 n/no。");
            }
        }
    }
}