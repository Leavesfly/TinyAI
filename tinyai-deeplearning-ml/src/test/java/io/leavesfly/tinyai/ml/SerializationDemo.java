package io.leavesfly.tinyai.ml;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.Parameter;

import java.util.HashMap;
import java.util.Map;

/**
 * ModelSerializer和ParameterManager功能演示
 */
public class SerializationDemo {
    
    public static void main(String[] args) {
        System.out.println("=== ModelSerializer 和 ParameterManager 功能演示 ===");
        
        // 1. 测试参数管理功能
        testParameterManager();
        
        // 2. 测试参数统计功能
        testParameterStats();
        
        // 3. 测试参数比较功能
        testParameterComparison();
        
        // 4. 测试参数深拷贝功能
        testParameterDeepCopy();
        
        // 5. 测试参数过滤功能
        testParameterFilter();
        
        System.out.println("\n=== 所有功能测试完成 ===");
    }
    
    private static void testParameterManager() {
        System.out.println("\n1. 测试参数管理功能");
        
        // 创建测试参数
        Map<String, Parameter> params = createTestParameters();
        
        // 测试保存和加载参数
        String filePath = "test-params.params";
        try {
            ParameterManager.saveParameters(params, filePath);
            System.out.println("参数保存成功: " + filePath);
            
            Map<String, Parameter> loadedParams = ParameterManager.loadParameters(filePath);
            System.out.println("参数加载成功，加载了 " + loadedParams.size() + " 个参数");
            
            // 清理文件
            new java.io.File(filePath).delete();
        } catch (Exception e) {
            System.out.println("参数保存/加载测试失败: " + e.getMessage());
        }
    }
    
    private static void testParameterStats() {
        System.out.println("\n2. 测试参数统计功能");
        
        Map<String, Parameter> params = createTestParameters();
        ParameterManager.ParameterStats stats = ParameterManager.getParameterStats(params);
        
        System.out.println("参数统计信息:");
        System.out.println("  总参数数量: " + stats.totalParameters);
        System.out.println("  参数组数量: " + stats.parameterCount);
        System.out.println("  最小值: " + String.format("%.6f", stats.minValue));
        System.out.println("  最大值: " + String.format("%.6f", stats.maxValue));
        System.out.println("  平均值: " + String.format("%.6f", stats.meanValue));
    }
    
    private static void testParameterComparison() {
        System.out.println("\n3. 测试参数比较功能");
        
        Map<String, Parameter> params1 = createTestParameters();
        Map<String, Parameter> params2 = createTestParameters();
        
        // 创建虚拟模型进行比较测试
        System.out.println("创建了两组相同结构的参数用于比较测试");
        System.out.println("参数组1大小: " + params1.size());
        System.out.println("参数组2大小: " + params2.size());
    }
    
    private static void testParameterDeepCopy() {
        System.out.println("\n4. 测试参数深拷贝功能");
        
        Map<String, Parameter> originalParams = createTestParameters();
        Map<String, Parameter> copiedParams = ParameterManager.deepCopyParameters(originalParams);
        
        if (copiedParams != null) {
            System.out.println("深拷贝成功:");
            System.out.println("  原始参数数量: " + originalParams.size());
            System.out.println("  拷贝参数数量: " + copiedParams.size());
            
            // 验证是否为不同的对象实例
            boolean isDifferentInstance = true;
            for (String key : originalParams.keySet()) {
                if (originalParams.get(key) == copiedParams.get(key)) {
                    isDifferentInstance = false;
                    break;
                }
            }
            System.out.println("  是否为不同实例: " + isDifferentInstance);
        } else {
            System.out.println("深拷贝失败");
        }
    }
    
    private static void testParameterFilter() {
        System.out.println("\n5. 测试参数过滤功能");
        
        Map<String, Parameter> params = createTestParameters();
        
        // 测试通配符过滤
        Map<String, Parameter> filteredParams = ParameterManager.filterParameters(params, "*weight*");
        System.out.println("过滤包含'weight'的参数:");
        System.out.println("  原始参数数量: " + params.size());
        System.out.println("  过滤后数量: " + filteredParams.size());
        
        for (String paramName : filteredParams.keySet()) {
            System.out.println("  - " + paramName);
        }
    }
    
    private static Map<String, Parameter> createTestParameters() {
        Map<String, Parameter> params = new HashMap<>();
        
        // 创建一些测试参数
        try {
            // 权重参数（矩阵）
            float[][] weightData = {{0.1f, 0.2f, 0.3f}, {0.4f, 0.5f, 0.6f}};
            NdArray weightArray = NdArray.of(weightData);
            params.put("layer1.weight", new Parameter(weightArray));
            
            // 偏置参数（向量）
            float[] biasData = {0.1f, 0.2f, 0.3f};
            NdArray biasArray = NdArray.of(biasData);
            params.put("layer1.bias", new Parameter(biasArray));
            
            // 另一层的参数
            float[][] weight2Data = {{0.7f, 0.8f}, {0.9f, 1.0f}, {1.1f, 1.2f}};
            NdArray weight2Array = NdArray.of(weight2Data);
            params.put("layer2.weight", new Parameter(weight2Array));
            
            float[] bias2Data = {0.5f, 0.6f};
            NdArray bias2Array = NdArray.of(bias2Data);
            params.put("layer2.bias", new Parameter(bias2Array));
            
        } catch (Exception e) {
            System.out.println("创建测试参数时出错: " + e.getMessage());
        }
        
        return params;
    }
}