package io.leavesfly.tinyai.ndarr;

/**
 * 验证NdArray TODO功能完善的简单测试
 * 
 * @author TinyDL
 */
public class NdArrayTodoVerification {
    
    public static void main(String[] args) {
        System.out.println("=== NdArray TODO功能完善验证 ===");
        
        try {
            // 创建测试矩阵
            NdArray testMatrix = new NdArray(new float[][]{
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
            });
            
            System.out.println("测试矩阵:");
            printMatrix(testMatrix);
            
            // 测试max(axis=0) - 按列查找最大值
            System.out.println("\n测试 max(axis=0) - 按列查找最大值:");
            NdArray maxAxis0 = testMatrix.max(0);
            printMatrix(maxAxis0);
            System.out.println("期望结果: [7, 8, 9]");
            
            // 测试max(axis=1) - 按行查找最大值  
            System.out.println("\n测试 max(axis=1) - 按行查找最大值:");
            NdArray maxAxis1 = testMatrix.max(1);
            printMatrix(maxAxis1);
            System.out.println("期望结果: [[3], [6], [9]]");
            
            // 测试min(axis=0) - 按列查找最小值
            System.out.println("\n测试 min(axis=0) - 按列查找最小值:");
            NdArray minAxis0 = testMatrix.min(0);
            printMatrix(minAxis0);
            System.out.println("期望结果: [1, 2, 3]");
            
            // 测试min(axis=1) - 按行查找最小值
            System.out.println("\n测试 min(axis=1) - 按行查找最小值:");
            NdArray minAxis1 = testMatrix.min(1);
            printMatrix(minAxis1);
            System.out.println("期望结果: [[1], [4], [7]]");
            
            // 测试异常处理改进
            System.out.println("\n测试异常处理改进:");
            
            // 测试无效轴参数
            try {
                testMatrix.max(2);
                System.out.println("❌ 错误：应该抛出异常!");
            } catch (IllegalArgumentException e) {
                System.out.println("✅ 正确捕获异常: " + e.getMessage());
            }
            
            // 测试矩阵乘法维度不匹配
            try {
                NdArray a = new NdArray(new float[][]{{1, 2, 3}});
                NdArray b = new NdArray(new float[][]{{1, 2}});
                a.dot(b);
                System.out.println("❌ 错误：应该抛出异常!");
            } catch (IllegalArgumentException e) {
                System.out.println("✅ 正确捕获异常: " + e.getMessage());
            }
            
            // 测试更复杂的数据
            System.out.println("\n测试复杂数据:");
            NdArray complexMatrix = new NdArray(new float[][]{
                {-1.5f, 0.0f, 3.5f},
                {2.5f, -2.5f, 1.0f},
                {10.0f, 8.5f, -1.0f}
            });
            
            System.out.println("复杂矩阵:");
            printMatrix(complexMatrix);
            
            System.out.println("max(axis=0): ");
            printMatrix(complexMatrix.max(0));
            
            System.out.println("min(axis=0): ");
            printMatrix(complexMatrix.min(0));
            
            System.out.println("\n=== 所有测试完成！✅ ===");
            
        } catch (Exception e) {
            System.err.println("❌ 测试失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void printMatrix(NdArray array) {
        try {
            if (array.getShape().isMatrix()) {
                float[][] matrix = array.getMatrix();
                for (float[] row : matrix) {
                    System.out.print("[");
                    for (int i = 0; i < row.length; i++) {
                        System.out.printf("%.1f", row[i]);
                        if (i < row.length - 1) System.out.print(", ");
                    }
                    System.out.println("]");
                }
            } else {
                System.out.println("形状: " + array.getShape());
                System.out.println("第一个元素: " + array.getNumber());
            }
        } catch (Exception e) {
            System.out.println("打印矩阵时出错: " + e.getMessage());
        }
    }
}