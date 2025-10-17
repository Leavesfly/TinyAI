package io.leavesfly.tinyai.agent.embodied;

import io.leavesfly.tinyai.agent.embodied.env.EnvironmentConfig;
import io.leavesfly.tinyai.agent.embodied.model.Episode;
import io.leavesfly.tinyai.agent.embodied.model.StepResult;

/**
 * 完整智能体演示
 * 展示感知-决策-执行完整闭环
 *
 * @author TinyAI Team
 */
public class AgentDemo {
    public static void main(String[] args) {
        System.out.println("=== TinyAI 具身智能体完整演示 ===\n");
        
        // 1. 创建配置
        EnvironmentConfig config = EnvironmentConfig.createTestConfig();
        System.out.println("场景配置: " + config);
        System.out.println();
        
        // 2. 创建智能体
        EmbodiedAgent agent = new EmbodiedAgent(config);
        System.out.println("智能体创建完成");
        
        // 3. 运行单步模式
        System.out.println("\n=== 单步运行模式 ===");
        agent.reset();
        
        for (int step = 0; step < 20; step++) {
            StepResult result = agent.step();
            
            if (step % 5 == 0) {
                System.out.println(String.format("步骤 %d: 奖励=%.3f, 总奖励=%.2f, 完成=%b",
                    step, result.getReward(), agent.getTotalReward(), result.isDone()));
            }
            
            if (result.isDone()) {
                System.out.println("情景终止于步骤 " + step);
                break;
            }
        }
        
        // 4. 运行完整情景
        System.out.println("\n=== 完整情景运行模式 ===");
        Episode episode = agent.runEpisode(100);
        
        System.out.println("情景统计:");
        System.out.println("  - 情景ID: " + episode.getEpisodeId());
        System.out.println("  - 场景类型: " + episode.getScenarioType().getName());
        System.out.println("  - 情景长度: " + episode.getLength() + " 步");
        System.out.println("  - 总奖励: " + String.format("%.2f", episode.getTotalReward()));
        System.out.println("  - 平均奖励: " + String.format("%.3f", episode.getAverageReward()));
        System.out.println("  - 持续时间: " + episode.getDuration() + " ms");
        System.out.println("  - 关键事件: " + episode.getCriticalEvents().size() + " 个");
        
        // 5. 清理
        agent.close();
        
        System.out.println("\n=== 演示完成 ===");
        System.out.println("\n✅ 具身智能体运行成功!");
        System.out.println("已实现完整的感知-决策-执行闭环");
    }
}
