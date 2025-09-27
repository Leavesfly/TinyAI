import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.layer.transformer.LayerNorm;

public class TestLayerNormForward {
    public static void main(String[] args) {
        // 创建LayerNorm层
        LayerNorm layerNorm = new LayerNorm("test", Shape.of(2, 10));
        
        // 创建测试输入
        float[][] inputData = {{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f},
                               {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f}};
        NdArray input = NdArray.of(inputData);
        Variable inputVar = new Variable(input);
        
        System.out.println("输入: " + input);
        System.out.println("输入形状: " + input.getShape());
        
        try {
            // 执行前向传播
            Variable output = layerNorm.layerForward(inputVar);
            System.out.println("输出: " + output.getValue());
            System.out.println("输出形状: " + output.getValue().getShape());
        } catch (Exception e) {
            System.out.println("前向传播出错: " + e.getMessage());
            e.printStackTrace();
        }
    }
}