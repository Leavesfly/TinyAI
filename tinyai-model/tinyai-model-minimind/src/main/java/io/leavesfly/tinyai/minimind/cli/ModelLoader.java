package io.leavesfly.tinyai.minimind.cli;

import io.leavesfly.tinyai.minimind.model.MiniMindConfig;
import io.leavesfly.tinyai.minimind.model.MiniMindModel;
import io.leavesfly.tinyai.minimind.training.demo.DemoConfig;
import io.leavesfly.tinyai.minimind.tokenizer.MiniMindTokenizer;
import io.leavesfly.tinyai.ml.model.Model;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * 模型加载与保存工具类
 * 
 * 负责 MiniMind 模型和分词器的加载与保存，提供统一的模型持久化接口。
 * 
 * 模型保存目录结构（位于 CHECKPOINT_DIR/model/ 下）：
 * <pre>
 * checkpoints/minimind/
 * ├── checkpoint_epoch0_step500.model   ← 训练快照（断点续训）
 * ├── checkpoint_epoch1_step1000.model
 * └── model/                            ← 最终模型（推理部署）
 *     └── {modelName}/
 *         ├── model.bin                 ← 序列化的模型参数
 *         ├── config.json               ← 模型配置
 *         └── tokenizer/                ← 分词器目录
 * </pre>
 * 
 * @author TinyAI Team
 */
public class ModelLoader {

    /**
     * 模型保存子目录名称，位于 CHECKPOINT_DIR 下
     */
    private static final String MODEL_SUBDIR = "model";
    
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

    // ==================== 模型保存 ====================
    
    /**
     * 保存完整模型到 CHECKPOINT_DIR/model/{modelName}/ 目录
     * <p>
     * 保存内容包括：模型参数（model.bin）、模型配置（config.json）、分词器
     *
     * @param model     训练好的模型
     * @param tokenizer 分词器
     * @param modelName 模型名称（用于子目录命名）
     * @throws IOException 保存失败时抛出
     */
    public static void saveModel(MiniMindModel model, MiniMindTokenizer tokenizer,
                                  String modelName) throws IOException {
        String modelDir = getModelDir(modelName);
        ensureDirectoryExists(modelDir);

        // 保存模型参数
        String modelFilePath = Paths.get(modelDir, "model.bin").toString();
        model.saveModel(modelFilePath);
        System.out.println("✅ 模型参数已保存: " + modelFilePath);

        // 保存模型配置
        String configFilePath = Paths.get(modelDir, "config.json").toString();
        saveConfig(model.getConfig(), configFilePath);
        System.out.println("✅ 模型配置已保存: " + configFilePath);

        // 保存分词器
        String tokenizerDir = Paths.get(modelDir, "tokenizer").toString();
        ensureDirectoryExists(tokenizerDir);
        tokenizer.save(tokenizerDir);
        System.out.println("✅ 分词器已保存: " + tokenizerDir);

        System.out.println("模型完整保存到: " + modelDir);
    }

    /**
     * 保存完整模型（使用模型自身名称作为目录名）
     *
     * @param model     训练好的模型
     * @param tokenizer 分词器
     * @throws IOException 保存失败时抛出
     */
    public static void saveModel(MiniMindModel model, MiniMindTokenizer tokenizer) 
            throws IOException {
        saveModel(model, tokenizer, model.getName());
    }

    // ==================== 模型加载 ====================
    
    /**
     * 加载模型
     * <p>
     * 优先从 CHECKPOINT_DIR/model/{modelName}/ 标准目录加载，
     * 若不存在则尝试从指定路径加载，最后回退到默认配置。
     * 
     * @param modelPath 模型文件路径
     * @param modelName 模型名称（用于标识）
     * @return 模型加载结果
     * @throws Exception 加载失败时抛出异常
     */
    public static ModelLoadResult loadModel(String modelPath, String modelName) throws Exception {
        // 优先尝试从标准 model 目录加载
        String standardModelDir = getModelDir(modelName);
        String standardModelFile = Paths.get(standardModelDir, "model.bin").toString();

        if (new File(standardModelFile).exists()) {
            System.out.println("从标准模型目录加载: " + standardModelDir);
            return loadFromModelDir(standardModelDir, modelName);
        }

        // 尝试从指定路径加载模型文件
        if (modelPath != null && !modelPath.isEmpty() && new File(modelPath).exists()) {
            System.out.println("正在从文件加载模型: " + modelPath);
            File modelFile = new File(modelPath);

            if (modelFile.isDirectory()) {
                // modelPath 是目录，按标准模型目录结构加载
                return loadFromModelDir(modelPath, modelName);
            }

            // modelPath 是单个模型文件，直接反序列化
            Model baseModel = Model.loadModel(modelPath);
            if (!(baseModel instanceof MiniMindModel)) {
                throw new IllegalStateException("加载的模型类型不匹配,期望 MiniMindModel,实际为: " 
                    + baseModel.getClass().getSimpleName());
            }
            MiniMindModel model = (MiniMindModel) baseModel;

            // 尝试从同目录下加载分词器
            MiniMindTokenizer tokenizer;
            String parentDir = modelFile.getParent();
            String tokenizerDir = Paths.get(parentDir, "tokenizer").toString();
            if (new File(tokenizerDir).exists()) {
                tokenizer = MiniMindTokenizer.load(tokenizerDir);
            } else {
                MiniMindConfig config = model.getConfig();
                tokenizer = MiniMindTokenizer.createCharLevelTokenizer(
                    config.getVocabSize(), config.getMaxSeqLen()
                );
                System.out.println("⚠️ 分词器目录不存在,使用默认字符级分词器");
            }

            System.out.println("✅ 模型加载完成: " + modelPath);
            return new ModelLoadResult(model, tokenizer);
        }

        // 回退到默认配置
        System.out.println("模型文件不存在,使用默认配置");
        MiniMindConfig config = MiniMindConfig.createSmallConfig();
        MiniMindModel model = new MiniMindModel(modelName, config);
        MiniMindTokenizer tokenizer = MiniMindTokenizer.createCharLevelTokenizer(
            config.getVocabSize(), config.getMaxSeqLen()
        );
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

    // ==================== 内部方法 ====================

    /**
     * 从标准模型目录加载模型、配置和分词器
     */
    private static ModelLoadResult loadFromModelDir(String modelDir, String modelName) 
            throws Exception {
        // 加载模型参数
        String modelFilePath = Paths.get(modelDir, "model.bin").toString();
        Model baseModel = Model.loadModel(modelFilePath);
        
        if (!(baseModel instanceof MiniMindModel)) {
            throw new IllegalStateException("加载的模型类型不匹配,期望 MiniMindModel");
        }
        MiniMindModel model = (MiniMindModel) baseModel;

        // 加载分词器
        MiniMindTokenizer tokenizer;
        String tokenizerDir = Paths.get(modelDir, "tokenizer").toString();
        if (new File(tokenizerDir).exists()) {
            tokenizer = MiniMindTokenizer.load(tokenizerDir);
            System.out.println("✅ 分词器已加载: " + tokenizerDir);
        } else {
            MiniMindConfig config = model.getConfig();
            tokenizer = MiniMindTokenizer.createCharLevelTokenizer(
                config.getVocabSize(), config.getMaxSeqLen()
            );
            System.out.println("⚠️ 分词器目录不存在,使用默认字符级分词器");
        }

        System.out.println("✅ 模型加载完成: " + modelDir);
        return new ModelLoadResult(model, tokenizer);
    }

    /**
     * 获取模型保存目录路径：CHECKPOINT_DIR/model/{modelName}/
     */
    private static String getModelDir(String modelName) {
        return Paths.get(DemoConfig.CHECKPOINT_DIR, MODEL_SUBDIR, modelName).toString();
    }

    /**
     * 确保目录存在，不存在则创建
     */
    private static void ensureDirectoryExists(String dirPath) throws IOException {
        Path path = Paths.get(dirPath);
        if (!Files.exists(path)) {
            Files.createDirectories(path);
        }
    }

    /**
     * 保存模型配置到 JSON 文件
     */
    private static void saveConfig(MiniMindConfig config, String filePath) throws IOException {
        try (FileWriter writer = new FileWriter(filePath)) {
            StringBuilder json = new StringBuilder();
            json.append("{\n");
            json.append("  \"vocabSize\": ").append(config.getVocabSize()).append(",\n");
            json.append("  \"hiddenSize\": ").append(config.getHiddenSize()).append(",\n");
            json.append("  \"numLayers\": ").append(config.getNumLayers()).append(",\n");
            json.append("  \"numHeads\": ").append(config.getNumHeads()).append(",\n");
            json.append("  \"maxSeqLen\": ").append(config.getMaxSeqLen()).append(",\n");
            json.append("  \"ffnHiddenSize\": ").append(config.getFfnHiddenSize()).append(",\n");
            json.append("  \"dropout\": ").append(config.getDropout()).append(",\n");
            json.append("  \"useMoE\": ").append(config.isUseMoE()).append(",\n");
            json.append("  \"numExperts\": ").append(config.getNumExperts()).append(",\n");
            json.append("  \"numExpertsPerToken\": ").append(config.getNumExpertsPerToken()).append("\n");
            json.append("}");
            writer.write(json.toString());
        }
    }
}
