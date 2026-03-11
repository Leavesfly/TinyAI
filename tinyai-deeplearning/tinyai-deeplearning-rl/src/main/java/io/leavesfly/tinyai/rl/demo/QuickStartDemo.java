package io.leavesfly.tinyai.rl.demo;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.rl.Experience;
import io.leavesfly.tinyai.rl.agent.EpsilonGreedyBanditAgent;
import io.leavesfly.tinyai.rl.environment.MultiArmedBanditEnvironment;
import io.leavesfly.tinyai.rl.Environment;

/**
 * 快速入门演示 - 3分钟了解强化学习
 *
 * 本演示通过一个简单的多臂老虎机问题,帮助您快速理解:
 * - 什么是强化学习
 * - 智能体-环境交互模式
 * - 如何使用TinyAI RL模块
 *
 * 运行方式:
 *   mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.rl.demo.QuickStartDemo" -pl tinyai-deeplearning-rl
 */
public class QuickStartDemo {

    public static void main(String[] args) {
        System.out.println("==========================================");
        System.out.println("    TinyAI 强化学习 - 3分钟快速入门      ");
        System.out.println("==========================================\n");

        // ========== 第1步: 理解问题 ==========
        System.out.println("【第1步】理解问题");
        System.out.println("想象您面前有3台老虎机,每台中奖概率不同:");
        System.out.println("  老虎机A: 中奖概率 30%");
        System.out.println("  老虎机B: 中奖概率 70% (最佳选择)");
        System.out.println("  老虎机C: 中奖概率 50%");
        System.out.println("目标: 通过学习,找出中奖概率最高的老虎机\n");

        // ========== 第2步: 创建环境 ==========
        System.out.println("【第2步】创建强化学习环境");
        float[] rewards = {0.3f, 0.7f, 0.5f}; // 每台老虎机的真实中奖概率
        int maxSteps = 100; // 允许尝试100次
        
        MultiArmedBanditEnvironment env = new MultiArmedBanditEnvironment(rewards, maxSteps);
        System.out.println("✓ 环境创建完成: 3台老虎机,共" + maxSteps + "次尝试机会\n");

        // ========== 第3步: 创建智能体 ==========
        System.out.println("【第3步】创建智能决策者(智能体)");
        System.out.println("使用ε-贪心策略: 90%选择当前最优,10%随机探索");
        
        EpsilonGreedyBanditAgent agent = new EpsilonGreedyBanditAgent(
            "学习者",    // 智能体名称
            3,          // 3个选择(3台老虎机)
            0.1f        // 探索率 = 10%
        );
        System.out.println("✓ 智能体创建完成\n");

        // ========== 第4步: 学习过程 ==========
        System.out.println("【第4步】开始学习(智能体与环境交互)");
        System.out.println("前10次尝试的详细过程:");
        System.out.println("尝试次数 | 选择老虎机 | 是否中奖 | 奖励");
        System.out.println("---------|-----------|---------|------");

        Variable state = env.reset();
        int totalReward = 0;
        int[] actionCounts = new int[3];

        for (int step = 0; step < maxSteps; step++) {
            // 智能体选择动作
            Variable action = agent.selectAction(state);
            int selectedArm = (int) action.getValue().getNumber().floatValue();
            actionCounts[selectedArm]++;

            // 环境执行动作,返回结果
            Environment.StepResult result = env.step(action);
            float reward = result.getReward();
            totalReward += (int) reward;

            // 智能体学习
            Experience experience = new Experience(
                state, action, reward, 
                result.getNextState(), result.isDone(), step
            );
            agent.learn(experience);

            // 显示前10次详细过程
            if (step < 10) {
                System.out.printf("   %2d    |     %s      |   %s    | %.0f\n",
                    step + 1,
                    (char)('A' + selectedArm),
                    reward > 0 ? "中奖" : "未中",
                    reward
                );
            }

            state = result.getNextState();
        }

        System.out.println("...(中间过程省略)...\n");

        // ========== 第5步: 查看学习结果 ==========
        System.out.println("【第5步】学习结果分析");
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        System.out.println("\n各老虎机的尝试次数:");
        for (int i = 0; i < 3; i++) {
            double percentage = (double) actionCounts[i] / maxSteps * 100;
            String bar = "█".repeat(actionCounts[i] / 2);
            System.out.printf("  老虎机%c: %2d次 (%.1f%%) %s\n", 
                (char)('A' + i), actionCounts[i], percentage, bar);
        }

        System.out.println("\n智能体学到的经验:");
        agent.printStatus();

        System.out.println("\n总中奖次数: " + totalReward + " / " + maxSteps);
        System.out.println("中奖率: " + String.format("%.1f%%", (double) totalReward / maxSteps * 100));

        // ========== 总结 ==========
        System.out.println("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        System.out.println("【学习总结】");
        
        int bestArm = 0;
        for (int i = 1; i < 3; i++) {
            if (actionCounts[i] > actionCounts[bestArm]) {
                bestArm = i;
            }
        }
        
        char bestArmChar = (char)('A' + bestArm);
        System.out.println("✓ 智能体成功学会: 老虎机" + bestArmChar + " 是最优选择");
        
        if (bestArm == 1) {
            System.out.println("✓ 学习正确! 确实老虎机B的中奖概率最高(70%)");
        } else {
            System.out.println("⚠ 可能需要更多尝试,或调整探索率");
        }

        System.out.println("\n==========================================");
        System.out.println("            恭喜完成快速入门!             ");
        System.out.println("==========================================");
        System.out.println("💡 下一步:");
        System.out.println("  • 查看 BasicConceptsDemo 深入理解RL核心概念");
        System.out.println("  • 查看 BanditAlgorithmsDemo 对比不同算法");
        System.out.println("  • 查看 DQNCartPoleDemo 尝试深度强化学习");
    }
}
