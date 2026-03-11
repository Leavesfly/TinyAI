package io.leavesfly.tinyai.rl.demo;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.rl.Experience;
import io.leavesfly.tinyai.rl.agent.EpsilonGreedyBanditAgent;
import io.leavesfly.tinyai.rl.environment.MultiArmedBanditEnvironment;
import io.leavesfly.tinyai.rl.Environment;

import java.util.Scanner;

/**
 * 交互式学习演示 - 让用户调整参数观察学习效果
 *
 * 本演示允许用户:
 * - 调整探索率(ε)观察探索-利用权衡
 * - 设置不同的奖励分布
 * - 实时观察学习过程
 * - 对比不同参数的效果
 *
 * 运行方式:
 *   mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.rl.demo.InteractiveLearningDemo" -pl tinyai-deeplearning-rl
 */
public class InteractiveLearningDemo {

    private static final Scanner SCANNER = new Scanner(System.in);
    
    public static void main(String[] args) {
        System.out.println("==========================================");
        System.out.println("    交互式强化学习演示                   ");
        System.out.println("==========================================\n");
        
        System.out.println("欢迎来到交互式学习实验室!");
        System.out.println("在这里，您可以调整参数，观察智能体如何学习。\n");
        
        boolean continueExperiment = true;
        
        while (continueExperiment) {
            runExperiment();
            
            System.out.print("\n是否进行新的实验? (y/n): ");
            String choice = SCANNER.nextLine().trim().toLowerCase();
            continueExperiment = choice.equals("y") || choice.equals("yes");
        }
        
        System.out.println("\n感谢使用交互式学习演示!");
    }

    /**
     * 运行一次实验
     */
    private static void runExperiment() {
        System.out.println("\n【实验配置】");
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        // 1. 选择场景
        int scenario = selectScenario();
        float[] rewards = getScenarioRewards(scenario);
        
        // 2. 配置参数
        System.out.println("\n配置智能体参数:");
        
        float epsilon = getFloatInput("探索率 ε (0.0-1.0, 推荐0.1-0.3): ", 0.0f, 1.0f, 0.1f);
        int maxSteps = getIntInput("训练步数 (10-1000, 推荐100): ", 10, 1000, 100);
        
        // 3. 运行实验
        System.out.println("\n开始实验...");
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        ExperimentResult result = runBanditExperiment(rewards, epsilon, maxSteps);
        
        // 4. 显示结果
        displayResults(result, rewards, epsilon, maxSteps);
    }

    /**
     * 选择实验场景
     */
    private static int selectScenario() {
        System.out.println("请选择实验场景:");
        System.out.println("  1. 简单场景 - 明显的最优选择 (奖励: 0.2, 0.8, 0.4)");
        System.out.println("  2. 困难场景 - 接近的奖励 (奖励: 0.45, 0.55, 0.50)");
        System.out.println("  3. 陷阱场景 - 次优选择初期表现好 (奖励: 0.7, 0.3, 0.6)");
        System.out.println("  4. 自定义场景 - 自己设置奖励");
        
        return getIntInput("选择 (1-4): ", 1, 4, 1);
    }

    /**
     * 获取场景的奖励分布
     */
    private static float[] getScenarioRewards(int scenario) {
        switch (scenario) {
            case 1:
                return new float[]{0.2f, 0.8f, 0.4f};
            case 2:
                return new float[]{0.45f, 0.55f, 0.50f};
            case 3:
                return new float[]{0.7f, 0.3f, 0.6f};
            case 4:
                return getCustomRewards();
            default:
                return new float[]{0.2f, 0.8f, 0.4f};
        }
    }

    /**
     * 获取自定义奖励
     */
    private static float[] getCustomRewards() {
        System.out.println("\n设置自定义奖励 (0.0-1.0):");
        float[] rewards = new float[3];
        for (int i = 0; i < 3; i++) {
            rewards[i] = getFloatInput("老虎机 " + (char)('A' + i) + " 的奖励概率: ", 0.0f, 1.0f, 0.5f);
        }
        return rewards;
    }

    /**
     * 运行多臂老虎机实验
     */
    private static ExperimentResult runBanditExperiment(float[] rewards, float epsilon, int maxSteps) {
        MultiArmedBanditEnvironment env = new MultiArmedBanditEnvironment(rewards, maxSteps);
        EpsilonGreedyBanditAgent agent = new EpsilonGreedyBanditAgent("学习者", 3, epsilon);
        
        Variable state = env.reset();
        int[] actionCounts = new int[3];
        float totalReward = 0;
        int optimalActionCount = 0;
        
        // 找出最优动作
        int optimalAction = 0;
        for (int i = 1; i < rewards.length; i++) {
            if (rewards[i] > rewards[optimalAction]) {
                optimalAction = i;
            }
        }
        
        // 学习过程
        for (int step = 0; step < maxSteps; step++) {
            Variable action = agent.selectAction(state);
            int selectedArm = (int) action.getValue().getNumber().floatValue();
            actionCounts[selectedArm]++;
            
            if (selectedArm == optimalAction) {
                optimalActionCount++;
            }
            
            Environment.StepResult result = env.step(action);
            float reward = result.getReward();
            totalReward += reward;
            
            Experience experience = new Experience(
                state, action, reward, result.getNextState(), result.isDone(), step
            );
            agent.learn(experience);
            
            // 每20步显示进度
            if ((step + 1) % 20 == 0) {
                showProgress(step + 1, maxSteps, actionCounts, totalReward);
            }
            
            state = result.getNextState();
        }
        
        return new ExperimentResult(actionCounts, totalReward, optimalActionCount, optimalAction, rewards);
    }

    /**
     * 显示学习进度
     */
    private static void showProgress(int currentStep, int totalSteps, int[] actionCounts, float totalReward) {
        float progress = (float) currentStep / totalSteps * 100;
        System.out.printf("\r进度: %.0f%% | 步数: %d/%d | 总奖励: %.0f | 选择分布: [%d, %d, %d]",
            progress, currentStep, totalSteps, totalReward,
            actionCounts[0], actionCounts[1], actionCounts[2]);
    }

    /**
     * 显示实验结果
     */
    private static void displayResults(ExperimentResult result, float[] rewards, float epsilon, int maxSteps) {
        System.out.println("\n\n【实验结果】");
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        System.out.println("\n1. 实验参数:");
        System.out.printf("   探索率 ε: %.2f%n", epsilon);
        System.out.printf("   训练步数: %d%n", maxSteps);
        
        System.out.println("\n2. 真实奖励分布:");
        for (int i = 0; i < rewards.length; i++) {
            System.out.printf("   老虎机 %c: %.2f %s%n", 
                (char)('A' + i), rewards[i], 
                i == result.optimalAction ? "(最优)" : "");
        }
        
        System.out.println("\n3. 智能体选择分布:");
        int totalActions = result.actionCounts[0] + result.actionCounts[1] + result.actionCounts[2];
        for (int i = 0; i < result.actionCounts.length; i++) {
            float percentage = (float) result.actionCounts[i] / totalActions * 100;
            int barLength = (int) (percentage / 2);
            String bar = "█".repeat(barLength);
            System.out.printf("   老虎机 %c: %3d次 (%5.1f%%) %s%n", 
                (char)('A' + i), result.actionCounts[i], percentage, bar);
        }
        
        System.out.println("\n4. 学习效果评估:");
        float optimalRate = (float) result.optimalActionCount / maxSteps * 100;
        System.out.printf("   最优选择次数: %d/%d (%.1f%%)%n", 
            result.optimalActionCount, maxSteps, optimalRate);
        System.out.printf("   总奖励: %.0f%n", result.totalReward);
        System.out.printf("   平均奖励: %.2f%n", result.totalReward / maxSteps);
        
        // 理论最优奖励
        float theoreticalOptimal = maxSteps * rewards[result.optimalAction];
        float regret = theoreticalOptimal - result.totalReward;
        System.out.printf("   遗憾(Regret): %.1f (理论最优: %.1f)%n", regret, theoreticalOptimal);
        
        System.out.println("\n5. 学习洞察:");
        if (optimalRate > 80) {
            System.out.println("   ✓ 优秀! 智能体成功找到了最优策略");
        } else if (optimalRate > 50) {
            System.out.println("   ✓ 良好! 智能体有一定学习效果");
        } else {
            System.out.println("   ⚠ 需要改进，可能需要:");
            System.out.println("     • 增加训练步数");
            System.out.println("     • 调整探索率");
            System.out.println("     • 检查奖励分布设置");
        }
        
        // 探索-利用分析
        System.out.println("\n6. 探索-利用分析:");
        if (epsilon < 0.05) {
            System.out.println("   探索率过低，可能陷入局部最优");
        } else if (epsilon > 0.5) {
            System.out.println("   探索率过高，学习不够稳定");
        } else {
            System.out.println("   探索率适中，平衡较好");
        }
    }

    /**
     * 获取整数输入
     */
    private static int getIntInput(String prompt, int min, int max, int defaultValue) {
        while (true) {
            System.out.print(prompt);
            String input = SCANNER.nextLine().trim();
            
            if (input.isEmpty()) {
                return defaultValue;
            }
            
            try {
                int value = Integer.parseInt(input);
                if (value >= min && value <= max) {
                    return value;
                }
                System.out.printf("请输入 %d 到 %d 之间的数字%n", min, max);
            } catch (NumberFormatException e) {
                System.out.println("请输入有效的数字");
            }
        }
    }

    /**
     * 获取浮点数输入
     */
    private static float getFloatInput(String prompt, float min, float max, float defaultValue) {
        while (true) {
            System.out.print(prompt);
            String input = SCANNER.nextLine().trim();
            
            if (input.isEmpty()) {
                return defaultValue;
            }
            
            try {
                float value = Float.parseFloat(input);
                if (value >= min && value <= max) {
                    return value;
                }
                System.out.printf("请输入 %.1f 到 %.1f 之间的数字%n", min, max);
            } catch (NumberFormatException e) {
                System.out.println("请输入有效的数字");
            }
        }
    }

    /**
     * 实验结果类
     */
    private static class ExperimentResult {
        final int[] actionCounts;
        final float totalReward;
        final int optimalActionCount;
        final int optimalAction;
        final float[] trueRewards;
        
        ExperimentResult(int[] actionCounts, float totalReward, int optimalActionCount, 
                        int optimalAction, float[] trueRewards) {
            this.actionCounts = actionCounts;
            this.totalReward = totalReward;
            this.optimalActionCount = optimalActionCount;
            this.optimalAction = optimalAction;
            this.trueRewards = trueRewards;
        }
    }
}
