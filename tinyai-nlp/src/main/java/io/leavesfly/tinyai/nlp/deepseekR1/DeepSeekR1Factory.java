package io.leavesfly.tinyai.nlp.deepseekR1;

/**
 * DeepSeek-R1 模型工厂类
 * 
 * 这个工厂类提供了创建DeepSeek-R1模型实例的各种方法，
 * 支持不同规模和用途的模型配置。
 * 
 * 工厂方法包括：
 * 1. 基于配置对象创建模型
 * 2. 预定义模型规模（tiny, medium, large）
 * 3. 专用场景模型（推理、训练、调试）
 * 4. 自定义参数模型
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekR1Factory {
    
    /**
     * 基于配置对象创建模型
     * 
     * @param name 模型名称
     * @param config 配置对象
     * @return DeepSeek-R1模型实例
     */
    public static DeepSeekR1Model createModel(String name, DeepSeekR1Config config) {
        if (config == null) {
            throw new IllegalArgumentException("配置对象不能为空");
        }
        
        // 验证配置
        config.validate();
        
        // 创建模型
        DeepSeekR1Model model = new DeepSeekR1Model(
            name,
            config.getVocabSize(),
            config.getDModel(),
            config.getNumLayers(),
            config.getNumHeads(),
            config.getDFF(),
            config.getMaxSeqLength(),
            config.getMaxReasoningSteps(),
            config.getDropoutRate(),
            config.getReasoningThreshold()
        );
        
        // 设置推理模式
        model.setReasoningMode(config.isEnableReasoning());
        
        return model;
    }
    
    /**
     * 创建小型模型
     * 适用于快速原型验证、资源受限环境或教学演示
     * 
     * @param name 模型名称
     * @param vocabSize 词汇表大小
     * @return 小型DeepSeek-R1模型
     */
    public static DeepSeekR1Model createTinyModel(String name, int vocabSize) {
        DeepSeekR1Config config = DeepSeekR1Config.createTinyConfig();
        config.setVocabSize(vocabSize);
        return createModel(name, config);
    }
    
    /**
     * 创建小型模型（默认词汇表）
     */
    public static DeepSeekR1Model createTinyModel(String name) {
        return createTinyModel(name, 50257);
    }
    
    /**
     * 创建中型模型
     * 平衡性能和资源消耗，适用于大多数应用场景
     * 
     * @param name 模型名称
     * @param vocabSize 词汇表大小
     * @return 中型DeepSeek-R1模型
     */
    public static DeepSeekR1Model createMediumModel(String name, int vocabSize) {
        DeepSeekR1Config config = DeepSeekR1Config.createMediumConfig();
        config.setVocabSize(vocabSize);
        return createModel(name, config);
    }
    
    /**
     * 创建中型模型（默认词汇表）
     */
    public static DeepSeekR1Model createMediumModel(String name) {
        return createMediumModel(name, 50257);
    }
    
    /**
     * 创建大型模型
     * 追求最佳性能，适用于高性能计算环境
     * 
     * @param name 模型名称
     * @param vocabSize 词汇表大小
     * @return 大型DeepSeek-R1模型
     */
    public static DeepSeekR1Model createLargeModel(String name, int vocabSize) {
        DeepSeekR1Config config = DeepSeekR1Config.createLargeConfig();
        config.setVocabSize(vocabSize);
        return createModel(name, config);
    }
    
    /**
     * 创建大型模型（默认词汇表）
     */
    public static DeepSeekR1Model createLargeModel(String name) {
        return createLargeModel(name, 50257);
    }
    
    /**
     * 创建推理优化模型
     * 专门针对推理任务优化，提升推理能力和效率
     * 
     * @param name 模型名称
     * @param vocabSize 词汇表大小
     * @param maxReasoningSteps 最大推理步骤数
     * @return 推理优化的DeepSeek-R1模型
     */
    public static DeepSeekR1Model createReasoningModel(String name, int vocabSize, int maxReasoningSteps) {
        DeepSeekR1Config config = DeepSeekR1Config.createReasoningConfig();
        config.setVocabSize(vocabSize);
        config.setMaxReasoningSteps(maxReasoningSteps);
        return createModel(name, config);
    }
    
    /**
     * 创建推理优化模型（默认参数）
     */
    public static DeepSeekR1Model createReasoningModel(String name) {
        return createReasoningModel(name, 50257, 15);
    }
    
    /**
     * 创建训练优化模型
     * 专门针对训练效率优化，提升训练速度和稳定性
     * 
     * @param name 模型名称
     * @param vocabSize 词汇表大小
     * @param maxSeqLength 最大序列长度
     * @return 训练优化的DeepSeek-R1模型
     */
    public static DeepSeekR1Model createTrainingModel(String name, int vocabSize, int maxSeqLength) {
        DeepSeekR1Config config = DeepSeekR1Config.createTrainingConfig();
        config.setVocabSize(vocabSize);
        config.setMaxSeqLength(maxSeqLength);
        return createModel(name, config);
    }
    
    /**
     * 创建训练优化模型（默认参数）
     */
    public static DeepSeekR1Model createTrainingModel(String name) {
        return createTrainingModel(name, 50257, 1024);
    }
    
    /**
     * 创建调试模型
     * 用于开发和调试的小规模快速配置
     * 
     * @param name 模型名称
     * @return 调试用DeepSeek-R1模型
     */
    public static DeepSeekR1Model createDebugModel(String name) {
        DeepSeekR1Config config = DeepSeekR1Config.createDebugConfig();
        return createModel(name, config);
    }
    
    /**
     * 创建自定义模型
     * 允许完全自定义所有关键参数
     * 
     * @param name 模型名称
     * @param vocabSize 词汇表大小
     * @param dModel 模型维度
     * @param numLayers 层数
     * @param numHeads 注意力头数
     * @param maxSeqLength 最大序列长度
     * @param maxReasoningSteps 最大推理步骤数
     * @param reasoningThreshold 推理置信度阈值
     * @return 自定义配置的DeepSeek-R1模型
     */
    public static DeepSeekR1Model createCustomModel(String name, int vocabSize, int dModel, 
                                                   int numLayers, int numHeads, int maxSeqLength,
                                                   int maxReasoningSteps, double reasoningThreshold) {
        DeepSeekR1Config config = new DeepSeekR1Config(
            vocabSize, dModel, numLayers, numHeads,
            dModel * 4, maxSeqLength, maxReasoningSteps,
            reasoningThreshold, 0.1
        );
        return createModel(name, config);
    }
    
    /**
     * 创建数学推理模型
     * 专门针对数学问题优化的配置
     * 
     * @param name 模型名称
     * @return 数学推理优化的DeepSeek-R1模型
     */
    public static DeepSeekR1Model createMathReasoningModel(String name) {
        DeepSeekR1Config config = DeepSeekR1Config.createReasoningConfig();
        // 数学推理需要更多步骤和更高置信度
        config.setMaxReasoningSteps(20);
        config.setReasoningThreshold(0.85);
        config.setAdaptiveThreshold(0.03);
        config.setEnableExploration(true);
        config.setExplorationRate(0.2);
        return createModel(name, config);
    }
    
    /**
     * 创建代码推理模型
     * 专门针对代码生成和理解优化的配置
     * 
     * @param name 模型名称
     * @return 代码推理优化的DeepSeek-R1模型
     */
    public static DeepSeekR1Model createCodeReasoningModel(String name) {
        DeepSeekR1Config config = DeepSeekR1Config.createReasoningConfig();
        // 代码推理需要更长的序列和适中的推理步骤
        config.setMaxSeqLength(2048);
        config.setMaxReasoningSteps(12);
        config.setReasoningThreshold(0.75);
        config.setEnableEarlyStopping(false); // 代码生成通常需要完整过程
        return createModel(name, config);
    }
    
    /**
     * 创建对话推理模型
     * 专门针对对话系统优化的配置
     * 
     * @param name 模型名称
     * @return 对话推理优化的DeepSeek-R1模型
     */
    public static DeepSeekR1Model createDialogueReasoningModel(String name) {
        DeepSeekR1Config config = DeepSeekR1Config.createReasoningConfig();
        // 对话系统需要快速响应和适度推理
        config.setMaxReasoningSteps(8);
        config.setReasoningThreshold(0.7);
        config.setEnableEarlyStopping(true);
        config.setMaxSeqLength(1024);
        return createModel(name, config);
    }
    
    /**
     * 从现有模型创建新配置的模型
     * 
     * @param existingModel 现有模型
     * @param newName 新模型名称
     * @param newConfig 新配置
     * @return 新配置的模型实例
     */
    public static DeepSeekR1Model createFromExisting(DeepSeekR1Model existingModel, 
                                                    String newName, DeepSeekR1Config newConfig) {
        if (existingModel == null) {
            throw new IllegalArgumentException("现有模型不能为空");
        }
        
        // 创建新模型但保留某些设置
        DeepSeekR1Model newModel = createModel(newName, newConfig);
        
        // 复制推理状态设置
        newModel.setReasoningMode(existingModel.isReasoningMode());
        
        return newModel;
    }
    
    /**
     * 创建模型集合
     * 创建多个不同配置的模型用于对比实验
     * 
     * @param baseName 基础名称
     * @param vocabSize 词汇表大小
     * @return 包含不同配置的模型数组
     */
    public static DeepSeekR1Model[] createModelEnsemble(String baseName, int vocabSize) {
        DeepSeekR1Model[] models = new DeepSeekR1Model[4];
        
        models[0] = createTinyModel(baseName + "_tiny", vocabSize);
        models[1] = createMediumModel(baseName + "_medium", vocabSize);
        models[2] = createReasoningModel(baseName + "_reasoning", vocabSize, 15);
        models[3] = createTrainingModel(baseName + "_training", vocabSize, 1024);
        
        return models;
    }
    
    /**
     * 获取推荐配置
     * 根据使用场景推荐最佳配置
     * 
     * @param useCase 使用场景（"research", "production", "education", "debugging"）
     * @param resourceLevel 资源级别（"low", "medium", "high"）
     * @return 推荐的配置对象
     */
    public static DeepSeekR1Config getRecommendedConfig(String useCase, String resourceLevel) {
        DeepSeekR1Config config;
        
        // 根据资源级别选择基础配置
        switch (resourceLevel.toLowerCase()) {
            case "low":
                config = DeepSeekR1Config.createTinyConfig();
                break;
            case "high":
                config = DeepSeekR1Config.createLargeConfig();
                break;
            case "medium":
            default:
                config = DeepSeekR1Config.createMediumConfig();
                break;
        }
        
        // 根据使用场景调整配置
        switch (useCase.toLowerCase()) {
            case "research":
                config = DeepSeekR1Config.createReasoningConfig();
                break;
            case "production":
                config = DeepSeekR1Config.createTrainingConfig();
                config.setEnableEarlyStopping(true);
                break;
            case "education":
                config = DeepSeekR1Config.createTinyConfig();
                config.setMaxReasoningSteps(5);
                break;
            case "debugging":
                config = DeepSeekR1Config.createDebugConfig();
                break;
        }
        
        return config;
    }
    
    /**
     * 打印所有预定义配置的摘要
     */
    public static void printConfigurationSummary() {
        System.out.println("\n=== DeepSeek-R1 Factory - Configuration Summary ===");
        
        System.out.println("\n1. Tiny Configuration:");
        DeepSeekR1Config.createTinyConfig().printConfig();
        
        System.out.println("\n2. Medium Configuration:");
        DeepSeekR1Config.createMediumConfig().printConfig();
        
        System.out.println("\n3. Large Configuration:");
        DeepSeekR1Config.createLargeConfig().printConfig();
        
        System.out.println("\n4. Reasoning Configuration:");
        DeepSeekR1Config.createReasoningConfig().printConfig();
        
        System.out.println("\n5. Training Configuration:");
        DeepSeekR1Config.createTrainingConfig().printConfig();
        
        System.out.println("\n6. Debug Configuration:");
        DeepSeekR1Config.createDebugConfig().printConfig();
        
        System.out.println("================================================\n");
    }
}