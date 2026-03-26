package io.leavesfly.tinyai.minimind.api;

import io.leavesfly.tinyai.minimind.cli.ModelLoader;
import io.leavesfly.tinyai.minimind.model.MiniMindModel;
import io.leavesfly.tinyai.minimind.tokenizer.MiniMindTokenizer;

/**
 * 共享模型持有者
 * 
 * 为 CompletionHandler 和 ChatCompletionHandler 提供统一的模型实例，
 * 避免重复加载模型，节省内存开销。
 * 
 * 使用懒加载 + 双重检查锁定保证线程安全的单例初始化。
 * 
 * @author leavesfly
 * @since 2024
 */
public class SharedModelHolder {

    private static volatile MiniMindModel model;
    private static volatile MiniMindTokenizer tokenizer;
    private static volatile boolean initialized = false;

    /** 默认模型文件路径，可通过 {@link #initialize(String)} 覆盖 */
    private static final String DEFAULT_MODEL_PATH = "models/minimind";

    private SharedModelHolder() {
        // 工具类禁止实例化
    }

    /**
     * 使用默认路径初始化模型（懒加载）
     */
    public static void ensureInitialized() {
        if (!initialized) {
            synchronized (SharedModelHolder.class) {
                if (!initialized) {
                    initialize(DEFAULT_MODEL_PATH);
                }
            }
        }
    }

    /**
     * 使用指定的模型文件路径初始化模型
     * 
     * @param modelPath 模型文件路径
     */
    public static synchronized void initialize(String modelPath) {
        try {
            System.out.println("正在初始化共享模型实例...");
            ModelLoader.ModelLoadResult result = ModelLoader.loadModel(modelPath, "minimind-api");
            model = result.getModel();
            tokenizer = result.getTokenizer();
            model.setTraining(false);
            initialized = true;
            System.out.println("共享模型实例初始化完成");
        } catch (Exception e) {
            // 初始化失败时清理部分初始化的状态
            model = null;
            tokenizer = null;
            initialized = false;
            System.err.println("共享模型初始化失败: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * 获取共享的模型实例
     * 
     * @return 模型实例，未初始化时返回 null
     */
    public static MiniMindModel getModel() {
        ensureInitialized();
        return model;
    }

    /**
     * 获取共享的分词器实例
     * 
     * @return 分词器实例，未初始化时返回 null
     */
    public static MiniMindTokenizer getTokenizer() {
        ensureInitialized();
        return tokenizer;
    }

    /**
     * 模型是否已初始化
     */
    public static boolean isInitialized() {
        return initialized;
    }
}