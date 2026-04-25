package io.leavesfly.tinyai.banana.demo;

import io.leavesfly.tinyai.banana.config.BananaConfig;
import io.leavesfly.tinyai.banana.config.TaskType;
import io.leavesfly.tinyai.banana.model.BananaModel;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

/**
 * Banana模型演示程序
 * 
 * 展示如何创建和使用Banana多模态图像生成模型
 * 
 * @author leavesfly
 * @version 1.0
 */
public class BananaDemo {
    
    public static void main(String[] args) {
        System.out.println("=".repeat(80));
        System.out.println("Banana多模态图像生成模型演示");
        System.out.println("=".repeat(80));
        System.out.println();
        
        // 演示1: 创建Tiny配置模型
        demonstrateTinyModel();
        
        System.out.println();
        
        // 演示2: 创建Small配置模型
        demonstrateSmallModel();
        
        System.out.println();
        
        // 演示3: 创建Base配置模型
        demonstrateBaseModel();
        
        System.out.println();
        
        // 演示4: 自定义配置
        demonstrateCustomConfig();
        
        System.out.println();
        
        // 演示5: 测试图像编码功能
        demonstrateImageEncoding();
        
        System.out.println();
        System.out.println("=".repeat(80));
        System.out.println("演示完成!");
        System.out.println("=".repeat(80));
    }
    
    /**
     * 演示Tiny模型(教学用)
     */
    private static void demonstrateTinyModel() {
        System.out.println("【演示1】创建Tiny配置模型(教学用)");
        System.out.println("-".repeat(80));
        
        // 创建Tiny模型
        BananaModel model = BananaModel.create("banana_tiny", "tiny");
        
        // 打印模型信息
        System.out.println("模型名称: " + model.getName());
        System.out.println("模型: " + model);
        System.out.println();
        
        // 打印配置摘要
        System.out.println(model.getConfigSummary());
        
        // 打印完整模型信息
        // model.printModelInfo();
    }
    
    /**
     * 演示Small模型(实验用)
     */
    private static void demonstrateSmallModel() {
        System.out.println("【演示2】创建Small配置模型(实验用)");
        System.out.println("-".repeat(80));
        
        BananaModel model = BananaModel.create("banana_small", "small");
        
        System.out.println("模型: " + model);
        System.out.println("参数量: " + model.getConfig().formatParameters());
        System.out.println("图像尺寸: " + model.getConfig().getImageSize() + "x" + 
                          model.getConfig().getImageSize());
        System.out.println("Patch数量: " + model.getConfig().getNumPatches());
    }
    
    /**
     * 演示Base模型(标准规模)
     */
    private static void demonstrateBaseModel() {
        System.out.println("【演示3】创建Base配置模型(标准规模)");
        System.out.println("-".repeat(80));
        
        BananaModel model = BananaModel.create("banana_base", "base");
        
        System.out.println("模型: " + model);
        System.out.println("参数量: " + model.getConfig().formatParameters());
        System.out.println("隐藏维度: " + model.getConfig().getHiddenSize());
        System.out.println("层数: " + model.getConfig().getNumLayers());
    }
    
    /**
     * 演示自定义配置
     */
    private static void demonstrateCustomConfig() {
        System.out.println("【演示4】自定义配置");
        System.out.println("-".repeat(80));
        
        // 创建自定义配置
        // 注意：imageSize / patchSize 必须是 2 的幂（ImageDecoder 的上采样链路是 ×2 级联）。
        // 这里选 256/16 = 16 = 2^4，合法。
        BananaConfig config = new BananaConfig();
        config.setHiddenSize(384);
        config.setNumLayers(6);
        config.setNumHeads(6);
        config.setFfnHiddenSize(1536);
        config.setImageSize(256);
        config.setPatchSize(16);
        config.updateNumPatches();
        
        // 验证配置
        try {
            config.validate();
            System.out.println("✓ 配置验证通过");
        } catch (Exception e) {
            System.out.println("✗ 配置验证失败: " + e.getMessage());
            return;
        }
        
        // 创建模型
        BananaModel model = new BananaModel("banana_custom", config);
        
        System.out.println("自定义模型: " + model);
        System.out.println("配置详情:");
        System.out.println(config);
    }
    
    /**
     * 演示图像编码功能
     */
    private static void demonstrateImageEncoding() {
        System.out.println("【演示5】测试图像编码功能");
        System.out.println("-".repeat(80));
        
        // 创建Tiny模型
        BananaModel model = BananaModel.create("banana_test", "tiny");
        System.out.println("使用模型: " + model.getName());
        
        // 创建模拟图像数据 [batch=2, channels=3, height=256, width=256]
        int batch = 2;
        int channels = 3;
        int imageSize = 256;
        NdArray imageData = NdArray.of(Shape.of(batch, channels, imageSize, imageSize));
        
        // 随机初始化图像数据
        float[] array = imageData.getArray();
        for (int i = 0; i < array.length; i++) {
            array[i] = (float) Math.random();
        }
        
        Variable imageInput = new Variable(imageData);
        System.out.println("\n输入图像 shape: " + imageInput.getValue().getShape());
        
        // 测试图像编码
        try {
            long startTime = System.currentTimeMillis();
            Variable imageFeatures = model.encodeImage(imageInput);
            long endTime = System.currentTimeMillis();
            
            System.out.println("图像特征 shape: " + imageFeatures.getValue().getShape());
            System.out.println("\n✓ 图像编码成功!");
            
            // 显示特征信息
            int[] shape = imageFeatures.getValue().getShape().getShapeDims();
            System.out.println("\n特征详情:");
            System.out.println("  Batch Size: " + shape[0]);
            System.out.println("  Num Patches: " + shape[1]);
            System.out.println("  Hidden Size: " + shape[2]);
            System.out.println("  总特征数: " + (shape[0] * shape[1] * shape[2]));
            System.out.println("  编码耗时: " + (endTime - startTime) + "ms");
        } catch (Exception e) {
            System.err.println("\n✗ 图像编码失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
