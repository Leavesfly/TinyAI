package io.leavesfly.tinyai.nlp.deepseekV3;

/**
 * DeepSeek-V3 模型工厂类
 * 
 * 提供便捷的方法来创建不同配置的DeepSeek-V3模型实例。
 * 包含预定义的模型配置和自定义配置选项。
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekV3Factory {
    
    /**
     * 创建迷你版DeepSeek-V3模型（用于测试和开发）
     * 
     * 配置特点：
     * - 词汇表: 1K
     * - 模型维度: 256
     * - 层数: 4
     * - 专家数: 4
     * - 序列长度: 128
     * 
     * @param name 模型名称
     * @return DeepSeek-V3模型实例
     */
    public static DeepSeekV3Model createTinyModel(String name) {
        DeepSeekV3Config config = DeepSeekV3Config.createTinyConfig();
        DeepSeekV3Model model = new DeepSeekV3Model(name, config);
        
        System.out.println("Created Tiny DeepSeek-V3 model: " + name);
        return model;
    }
    
    /**
     * 创建小型DeepSeek-V3模型
     * 
     * 配置特点：
     * - 词汇表: 32K
     * - 模型维度: 768
     * - 层数: 12
     * - 专家数: 16
     * - 序列长度: 2K
     * 
     * @param name 模型名称
     * @return DeepSeek-V3模型实例
     */
    public static DeepSeekV3Model createSmallModel(String name) {
        DeepSeekV3Config config = DeepSeekV3Config.createSmallConfig();
        DeepSeekV3Model model = new DeepSeekV3Model(name, config);
        
        System.out.println("Created Small DeepSeek-V3 model: " + name);
        return model;
    }
    
    /**
     * 创建标准DeepSeek-V3模型
     * 
     * 配置特点：
     * - 词汇表: 102K
     * - 模型维度: 2048
     * - 层数: 28
     * - 专家数: 64
     * - 序列长度: 4K
     * 
     * @param name 模型名称
     * @return DeepSeek-V3模型实例
     */
    public static DeepSeekV3Model createStandardModel(String name) {
        DeepSeekV3Config config = DeepSeekV3Config.createStandardConfig();
        DeepSeekV3Model model = new DeepSeekV3Model(name, config);
        
        System.out.println("Created Standard DeepSeek-V3 model: " + name);
        return model;
    }
    
    /**
     * 使用自定义配置创建DeepSeek-V3模型
     * 
     * @param name 模型名称
     * @param config 自定义配置
     * @return DeepSeek-V3模型实例
     */
    public static DeepSeekV3Model createModelWithConfig(String name, DeepSeekV3Config config) {
        config.validate(); // 验证配置有效性
        DeepSeekV3Model model = new DeepSeekV3Model(name, config);
        
        System.out.println("Created Custom DeepSeek-V3 model: " + name);
        return model;
    }
    
    /**
     * 创建针对推理优化的DeepSeek-V3模型
     * 
     * @param name 模型名称
     * @param baseConfig 基础配置
     * @return 推理优化的DeepSeek-V3模型实例
     */
    public static DeepSeekV3Model createInferenceOptimizedModel(String name, DeepSeekV3Config baseConfig) {
        // 复制并优化配置
        DeepSeekV3Config optimizedConfig = baseConfig; // 直接使用原配置
        
        // 推理优化设置
        optimizedConfig.setUseFlashAttention(true);
        optimizedConfig.setUseKVCache(true);
        optimizedConfig.setUseMLACache(true);
        optimizedConfig.setDropoutRate(0.0); // 推理时关闭dropout
        optimizedConfig.setUseExpertDropout(false); // 推理时关闭专家dropout
        
        DeepSeekV3Model model = new DeepSeekV3Model(name, optimizedConfig);
        
        // 启用推理优化
        model.enableKVCache();
        model.setTraining(false);
        
        System.out.println("Created Inference-Optimized DeepSeek-V3 model: " + name);
        return model;
    }
    
    /**
     * 创建针对训练优化的DeepSeek-V3模型
     * 
     * @param name 模型名称
     * @param baseConfig 基础配置
     * @return 训练优化的DeepSeek-V3模型实例
     */
    public static DeepSeekV3Model createTrainingOptimizedModel(String name, DeepSeekV3Config baseConfig) {
        // 复制并优化配置
        DeepSeekV3Config optimizedConfig = baseConfig; // 直接使用原配置
        
        // 训练优化设置
        optimizedConfig.setUseGradientCheckpointing(true);
        optimizedConfig.setUseFP8Training(true);
        optimizedConfig.setUseAuxiliaryLoss(true); // 启用负载均衡损失
        optimizedConfig.setExpertLoadBalanceWeight(0.001);
        
        DeepSeekV3Model model = new DeepSeekV3Model(name, optimizedConfig);
        
        // 启用训练模式
        model.setTraining(true);
        model.disableKVCache(); // 训练时通常不使用KV缓存
        
        System.out.println("Created Training-Optimized DeepSeek-V3 model: " + name);
        return model;
    }
    
    /**
     * 创建用于研究的DeepSeek-V3模型
     * 
     * @param name 模型名称
     * @param researchType 研究类型 ("attention", "moe", "scaling")
     * @return 研究用DeepSeek-V3模型实例
     */
    public static DeepSeekV3Model createResearchModel(String name, String researchType) {
        DeepSeekV3Config config;
        
        switch (researchType.toLowerCase()) {
            case "attention":
                // 注意力机制研究配置
                config = DeepSeekV3Config.createSmallConfig();
                config.setDMLA(128); // 较小的MLA维度便于分析
                config.setQkNormDim(64);
                config.setUseMLACache(true);
                break;
                
            case "moe":
                // MoE机制研究配置
                config = DeepSeekV3Config.createSmallConfig();
                config.setNumExperts(8); // 适中的专家数量
                config.setTopK(2);
                config.setUseSharedExperts(true);
                config.setNumSharedExperts(1);
                break;
                
            case "scaling":
                // 扩展性研究配置
                config = DeepSeekV3Config.createTinyConfig();
                config.setNumLayers(6); // 较多层数
                config.setNumExperts(16); // 较多专家
                break;
                
            default:
                // 默认研究配置
                config = DeepSeekV3Config.createSmallConfig();
                break;
        }
        
        DeepSeekV3Model model = new DeepSeekV3Model(name, config);
        
        System.out.println("Created Research DeepSeek-V3 model (" + researchType + "): " + name);
        return model;
    }
    
    /**
     * 批量创建不同规模的DeepSeek-V3模型
     * 
     * @param baseName 基础名称
     * @return 包含不同规模模型的数组
     */
    public static DeepSeekV3Model[] createModelSuite(String baseName) {
        DeepSeekV3Model[] models = new DeepSeekV3Model[3];
        
        models[0] = createTinyModel(baseName + "_tiny");
        models[1] = createSmallModel(baseName + "_small");
        models[2] = createStandardModel(baseName + "_standard");
        
        System.out.println("Created DeepSeek-V3 model suite: " + baseName);
        return models;
    }
    
    /**
     * 创建基于参数约束的DeepSeek-V3模型
     * 
     * @param name 模型名称
     * @param maxParameters 最大参数数量
     * @param maxSeqLength 最大序列长度
     * @return 符合约束的DeepSeek-V3模型实例
     */
    public static DeepSeekV3Model createConstrainedModel(String name, long maxParameters, int maxSeqLength) {
        DeepSeekV3Config config;
        
        // 根据参数约束选择合适的配置
        if (maxParameters <= 1_000_000L) { // 1M以下
            config = DeepSeekV3Config.createTinyConfig();
        } else if (maxParameters <= 100_000_000L) { // 100M以下
            config = DeepSeekV3Config.createSmallConfig();
        } else {
            config = DeepSeekV3Config.createStandardConfig();
        }
        
        // 调整序列长度
        config.setMaxSeqLength(Math.min(maxSeqLength, config.getMaxSeqLength()));
        
        // 如果参数数量仍然过大，进一步调整
        while (config.getTotalParameterCount() > maxParameters && config.getNumLayers() > 1) {
            config.setNumLayers(config.getNumLayers() - 1);
            if (config.getNumExperts() > 4) {
                config.setNumExperts(config.getNumExperts() / 2);
            }
        }
        
        DeepSeekV3Model model = new DeepSeekV3Model(name, config);
        
        System.out.printf("Created Constrained DeepSeek-V3 model: %s (%.1fM params)%n", 
                         name, config.getTotalParameterCount() / 1_000_000.0);
        return model;
    }
    
    /**
     * 获取预定义配置的信息
     * 
     * @return 配置信息字符串
     */
    public static String getPresetConfigsInfo() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== DeepSeek-V3 Preset Configurations ===\n");
        
        DeepSeekV3Config tiny = DeepSeekV3Config.createTinyConfig();
        sb.append("TINY: ").append(tiny.getSummary()).append("\n\n");
        
        DeepSeekV3Config small = DeepSeekV3Config.createSmallConfig();
        sb.append("SMALL: ").append(small.getSummary()).append("\n\n");
        
        DeepSeekV3Config standard = DeepSeekV3Config.createStandardConfig();
        sb.append("STANDARD: ").append(standard.getSummary()).append("\n");
        
        sb.append("==========================================");
        return sb.toString();
    }
    
    /**
     * 打印所有预定义配置信息
     */
    public static void printPresetConfigsInfo() {
        System.out.println(getPresetConfigsInfo());
    }
    
    /**
     * 验证模型创建是否成功
     * 
     * @param model 要验证的模型
     * @return 验证是否通过
     */
    public static boolean validateModel(DeepSeekV3Model model) {
        try {
            // 基本验证
            if (model == null) {
                System.err.println("Model is null");
                return false;
            }
            
            if (model.getConfig() == null) {
                System.err.println("Model config is null");
                return false;
            }
            
            // 验证组件
            if (model.getTokenEmbedding() == null) {
                System.err.println("Token embedding is null");
                return false;
            }
            
            if (model.getTransformerBlocks() == null || model.getTransformerBlocks().isEmpty()) {
                System.err.println("Transformer blocks are null or empty");
                return false;
            }
            
            if (model.getFinalLayerNorm() == null) {
                System.err.println("Final layer norm is null");
                return false;
            }
            
            if (model.getOutputHead() == null) {
                System.err.println("Output head is null");
                return false;
            }
            
            System.out.println("Model validation passed: " + "model_" + System.currentTimeMillis());
            return true;
            
        } catch (Exception e) {
            System.err.println("Model validation failed: " + e.getMessage());
            return false;
        }
    }
}