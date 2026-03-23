package io.leavesfly.tinyai.minimind.cli;

import io.leavesfly.tinyai.minimind.model.MiniMindConfig;
import io.leavesfly.tinyai.minimind.model.MiniMindModel;
import io.leavesfly.tinyai.minimind.tokenizer.MiniMindTokenizer;

import java.io.File;

/**
 * 模型加载工具类
 * 
 * 负责加载 MiniMind 模型和分词器，提供统一的模型加载接口
 * 
 * @author TinyAI Team
 */
public class ModelLoader {
    
    /**
     * 模型加载结果
     */
    public static class ModelLoadResult {
        private final MiniMindModel model;
        private final MiniMindTokenizer tokenizer;
        
        public ModelLoadResult(MiniMindModel model, MiniMindTokenizer tokenizer) {
            this.model = model;
            this.tokenizer = tokenizer;
        }
        
        public MiniMindModel getModel() {
            return model;
        }
        
        public MiniMindTokenizer getTokenizer() {
            return tokenizer;
        }
    }
    
    /**
     * 加载模型
     * 
     * @param modelPath 模型文件路径
     * @param modelName 模型名称（用于标识）
     * @return 模型加载结果
     * @throws Exception 加载失败时抛出异常
     */
    public static ModelLoadResult loadModel(String modelPath, String modelName) throws Exception {
        MiniMindModel model;
        MiniMindTokenizer tokenizer;
        
        if (new File(modelPath).exists()) {
            System.out.println("正在加载模型: " + modelPath);
            // TODO: 实现模型加载逻辑
            // model = MiniMindModel.load(modelPath);
            // tokenizer = MiniMindTokenizer.load(modelPath + "/tokenizer.json");
            System.out.println("[注意] 模型加载功能开发中,使用默认配置");
            
            MiniMindConfig config = MiniMindConfig.createSmallConfig();
            model = new MiniMindModel(modelName, config);
            tokenizer = MiniMindTokenizer.createCharLevelTokenizer(
                config.getVocabSize(), config.getMaxSeqLen()
            );
        } else {
            System.out.println("模型文件不存在,使用默认配置");
            MiniMindConfig config = MiniMindConfig.createSmallConfig();
            model = new MiniMindModel(modelName, config);
            tokenizer = MiniMindTokenizer.createCharLevelTokenizer(
                config.getVocabSize(), config.getMaxSeqLen()
            );
        }
        
        return new ModelLoadResult(model, tokenizer);
    }
    
    /**
     * 加载模型（使用默认模型名称）
     * 
     * @param modelPath 模型文件路径
     * @return 模型加载结果
     * @throws Exception 加载失败时抛出异常
     */
    public static ModelLoadResult loadModel(String modelPath) throws Exception {
        return loadModel(modelPath, "minimind-model");
    }
}
