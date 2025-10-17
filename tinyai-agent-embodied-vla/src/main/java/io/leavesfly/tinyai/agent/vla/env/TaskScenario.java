package io.leavesfly.tinyai.agent.vla.env;

/**
 * 任务场景定义
 * 定义不同难度的具身智能任务
 * 
 * @author TinyAI
 */
public enum TaskScenario {
    
    /** 拾取并放置任务 */
    PICK_AND_PLACE("PickAndPlace", 2, "拾取物体并放置到目标位置"),
    
    /** 堆叠方块任务 */
    STACK_BLOCKS("StackBlocks", 3, "堆叠多个方块"),
    
    /** 打开抽屉任务 */
    OPEN_DRAWER("OpenDrawer", 3, "打开抽屉"),
    
    /** 倒水任务 */
    POUR_WATER("PourWater", 4, "倒水任务"),
    
    /** 组装零件任务 */
    ASSEMBLE_PARTS("AssembleParts", 5, "组装零件");
    
    private final String name;
    private final int difficulty; // 1-5星难度
    private final String description;
    
    TaskScenario(String name, int difficulty, String description) {
        this.name = name;
        this.difficulty = difficulty;
        this.description = description;
    }
    
    public String getName() {
        return name;
    }
    
    public int getDifficulty() {
        return difficulty;
    }
    
    public String getDescription() {
        return description;
    }
    
    public String getDifficultyStars() {
        return "⭐".repeat(difficulty);
    }
}
