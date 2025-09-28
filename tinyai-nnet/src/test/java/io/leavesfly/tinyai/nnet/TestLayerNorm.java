package io.leavesfly.tinyai.nnet;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.layer.transformer.LayerNorm;

public class TestLayerNorm {
    public static void main(String[] args) {
        // 创建LayerNorm层
        LayerNorm layerNorm = new LayerNorm("test", Shape.of(2, 10));
        
        // 创建测试输入
        float[][] inputData = {{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f},
                               {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f}};
        NdArray input = NdArray.of(inputData);
        Variable inputVar = new Variable(input);
        
        System.out.println("输入形状: " + input.getShape());
        System.out.println("Gamma参数形状: " + layerNorm.getParamBy("gamma").getValue().getShape());
        System.out.println("Beta参数形状: " + layerNorm.getParamBy("beta").getValue().getShape());
        
        try {
            // 尝试广播gamma参数
            NdArray gammaBroadcasted = layerNorm.getParamBy("gamma").getValue().broadcastTo(input.getShape());
            System.out.println("Gamma广播后形状: " + gammaBroadcasted.getShape());
        } catch (Exception e) {
            System.out.println("Gamma广播出错: " + e.getMessage());
            e.printStackTrace();
        }
        
        try {
            // 尝试广播beta参数
            NdArray betaBroadcasted = layerNorm.getParamBy("beta").getValue().broadcastTo(input.getShape());
            System.out.println("Beta广播后形状: " + betaBroadcasted.getShape());
        } catch (Exception e) {
            System.out.println("Beta广播出错: " + e.getMessage());
            e.printStackTrace();
        }
    }
}