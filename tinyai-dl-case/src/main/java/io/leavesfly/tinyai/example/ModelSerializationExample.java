package io.leavesfly.tinyai.example;


import io.leavesfly.tinyai.ml.Model;
import io.leavesfly.tinyai.ml.ModelInfo;
import io.leavesfly.tinyai.ml.ModelInfoExporter;
import io.leavesfly.tinyai.ml.ParameterManager;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.Parameter;
import io.leavesfly.tinyai.nnet.block.MlpBlock;
import io.leavesfly.tinyai.util.Config;

import java.io.File;
import java.util.Map;

/**
 * 模型序列化示例
 * 
 * @author leavesfly
 * @version 0.01
 * 
 * 演示完整的模型保存和加载功能，包括：
 * 1. 创建示例模型
 * 2. 测试各种保存格式（完整模型、压缩模型、参数文件、检查点）
 * 3. 测试模型加载功能
 * 4. 测试参数操作功能
 * 5. 测试JSON导出功能
 * 6. 测试检查点功能
 */
public class ModelSerializationExample {
    
    /**
     * 主函数，执行所有序列化测试
     * 
     * @param args 命令行参数
     */
    public static void main(String[] args) {
        System.out.println("=== TinyDL 模型序列化示例 ===");
        
        try {
            // 1. 创建一个简单的MLP模型
            System.out.println("\n1. 创建模型...");
            Model originalModel = createSampleModel();
            
            // 设置模型信息
            setupModelInfo(originalModel);
            
            // 显示原始模型信息
            System.out.println("\n原始模型信息:");
            originalModel.printModelInfo();
            
            // 2. 测试各种保存格式
            testModelSaving(originalModel);
            
            // 3. 测试模型加载
            testModelLoading();
            
            // 4. 测试参数操作
            testParameterOperations(originalModel);
            
            // 5. 测试JSON导出
            testJsonExport(originalModel);
            
            // 6. 测试检查点功能
            testCheckpointFunctionality(originalModel);
            
            System.out.println("\n=== 所有测试完成! ===");
            
        } catch (Exception e) {
            System.err.println("测试过程中出现错误: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 创建示例模型
     * 
     * @return 创建的模型实例
     */
    private static Model createSampleModel() {
        // 创建一个简单的MLP网络
        // MlpBlock的构造函数参数: (名称, 批次大小, 激活函数, 层大小...)
        int batchSize = 1;
        MlpBlock mlpBlock = new MlpBlock("sampleMLP", batchSize, Config.ActiveFunc.ReLU, 10, 64, 32, 1);
        mlpBlock.init();
        
        Model model = new Model("SampleModel", mlpBlock);
        
        // 初始化一些随机参数
        initializeRandomParameters(model);
        
        return model;
    }
    
    /**
     * 初始化随机参数
     * 
     * @param model 要初始化参数的模型
     */
    private static void initializeRandomParameters(Model model) {
        Map<String, Parameter> params = model.getAllParams();
        for (Parameter param : params.values()) {
            // 用随机值初始化参数
            NdArray randomArray = NdArray.likeRandomN(param.getValue().getShape());
            param.setValue(randomArray);
        }
    }
    
    /**
     * 设置模型信息
     * 
     * @param model 要设置信息的模型
     */
    private static void setupModelInfo(Model model) {
        ModelInfo info = model.getModelInfo();
        info.setDescription("这是一个用于演示序列化功能的示例MLP模型");
        info.setModelVersion("1.0.0");
        info.setArchitectureType("MLP");
        
        // 设置训练信息
        model.updateTrainingInfo(100, 0.025, "SGD", 0.01);
        
        // 添加性能指标
        model.addMetric("accuracy", 0.95);
        model.addMetric("precision", 0.93);
        model.addMetric("recall", 0.97);
        
        info.addCustomProperty("dataset", "synthetic_data");
        info.addCustomProperty("author", "TinyDL Example");
    }
    
    /**
     * 测试模型保存
     * 
     * @param model 要保存的模型
     */
    private static void testModelSaving(Model model) {
        System.out.println("\n2. 测试模型保存...");
        
        // 创建输出目录
        new File("models").mkdirs();
        
        // 保存完整模型
        System.out.println("  保存完整模型...");
        model.saveModel("models/sample_model.model");
        
        // 保存压缩模型
        System.out.println("  保存压缩模型...");
        model.saveModelCompressed("models/sample_model_compressed.model");
        
        // 仅保存参数
        System.out.println("  保存模型参数...");
        model.saveParameters("models/sample_model.params");
        
        // 保存检查点
        System.out.println("  保存训练检查点...");
        model.saveCheckpoint("models/sample_model_epoch100.ckpt", 100, 0.025);
        
        // 显示文件大小
        showFileInfo("models/sample_model.model", "完整模型");
        showFileInfo("models/sample_model_compressed.model", "压缩模型");
        showFileInfo("models/sample_model.params", "参数文件");
        showFileInfo("models/sample_model_epoch100.ckpt", "检查点文件");
    }
    
    /**
     * 测试模型加载
     */
    private static void testModelLoading() {
        System.out.println("\n3. 测试模型加载...");
        
        try {
            // 加载完整模型
            System.out.println("  加载完整模型...");
            Model loadedModel = Model.loadModel("models/sample_model.model");
            System.out.println("  ✓ 成功加载模型: " + loadedModel.getName());
            
            // 验证模型信息
            ModelInfo info = loadedModel.getModelInfo();
            System.out.println("  模型描述: " + info.getDescription());
            System.out.println("  总参数数量: " + info.getTotalParameters());
            
            // 加载压缩模型
            System.out.println("  加载压缩模型...");
            Model compressedModel = Model.loadModel("models/sample_model_compressed.model");
            System.out.println("  ✓ 成功加载压缩模型: " + compressedModel.getName());
            
        } catch (Exception e) {
            System.err.println("  ✗ 加载模型失败: " + e.getMessage());
        }
    }
    
    /**
     * 测试参数操作
     * 
     * @param originalModel 原始模型
     */
    private static void testParameterOperations(Model originalModel) {
        System.out.println("\n4. 测试参数操作...");
        
        try {
            // 创建新模型
            Model newModel = createSampleModel();
            
            // 比较参数（应该不同）
            boolean paramsEqual = ParameterManager.compareParameters(originalModel, newModel);
            System.out.println("  新模型参数是否与原模型相同: " + paramsEqual);
            
            // 复制参数
            int copiedCount = ParameterManager.copyParameters(originalModel, newModel);
            System.out.println("  复制的参数数量: " + copiedCount);
            
            // 再次比较参数（应该相同）
            paramsEqual = ParameterManager.compareParameters(originalModel, newModel);
            System.out.println("  复制后参数是否相同: " + paramsEqual);
            
            // 获取参数统计
            Map<String, Parameter> params = originalModel.getAllParams();
            ParameterManager.ParameterStats stats = ParameterManager.getParameterStats(params);
            System.out.println("  参数统计: " + stats);
            
            // 保存参数统计到文件
            ParameterManager.saveParameterStats(params, "models/parameter_stats.txt");
            System.out.println("  ✓ 参数统计已保存到 models/parameter_stats.txt");
            
        } catch (Exception e) {
            System.err.println("  ✗ 参数操作失败: " + e.getMessage());
        }
    }
    
    /**
     * 测试JSON导出
     * 
     * @param model 要导出的模型
     */
    private static void testJsonExport(Model model) {
        System.out.println("\n5. 测试JSON导出...");
        
        try {
            // 导出完整模型信息
            System.out.println("  导出完整模型信息到JSON...");
            ModelInfoExporter.exportToJson(model, "models/model_info_full.json", true);
            
            // 导出简化信息
            System.out.println("  导出简化模型信息到JSON...");
            ModelInfoExporter.exportToJson(model, "models/model_info_simple.json", false);
            
            // 导出简单报告
            System.out.println("  导出简单报告...");
            ModelInfoExporter.exportSimpleReport(model, "models/simple_report.json");
            
            System.out.println("  ✓ JSON文件导出完成");
            
            showFileInfo("models/model_info_full.json", "完整JSON信息");
            showFileInfo("models/model_info_simple.json", "简化JSON信息");
            showFileInfo("models/simple_report.json", "简单报告");
            
        } catch (Exception e) {
            System.err.println("  ✗ JSON导出失败: " + e.getMessage());
        }
    }
    
    /**
     * 测试检查点功能
     * 
     * @param originalModel 原始模型
     */
    private static void testCheckpointFunctionality(Model originalModel) {
        System.out.println("\n6. 测试检查点功能...");
        
        try {
            // 模拟训练过程中保存多个检查点
            System.out.println("  模拟训练过程，保存多个检查点...");
            
            for (int epoch = 10; epoch <= 50; epoch += 10) {
                double loss = 1.0 / epoch; // 模拟损失递减
                String checkpointPath = "models/checkpoint_epoch" + epoch + ".ckpt";
                originalModel.saveCheckpoint(checkpointPath, epoch, loss);
                System.out.println("    ✓ 保存检查点: epoch=" + epoch + ", loss=" + String.format("%.4f", loss));
            }
            
            // 从检查点恢复
            System.out.println("  从检查点恢复模型...");
            Model restoredModel = Model.resumeFromCheckpoint("models/checkpoint_epoch50.ckpt");
            
            // 验证恢复的模型
            ModelInfo info = restoredModel.getModelInfo();
            System.out.println("  恢复的模型训练轮次: " + info.getTrainedEpochs());
            System.out.println("  恢复的模型最终损失: " + String.format("%.4f", info.getFinalLoss()));
            
        } catch (Exception e) {
            System.err.println("  ✗ 检查点功能测试失败: " + e.getMessage());
        }
    }
    
    /**
     * 显示文件信息
     * 
     * @param filePath 文件路径
     * @param description 文件描述
     */
    private static void showFileInfo(String filePath, String description) {
        File file = new File(filePath);
        if (file.exists()) {
            long sizeKB = file.length() / 1024;
            System.out.println("    " + description + ": " + sizeKB + " KB");
        }
    }
}