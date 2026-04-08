package io.leavesfly.tinyai.nnet.examples;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.core.Module;
import io.leavesfly.tinyai.nnet.layer.dnn.LazyLinear;
import io.leavesfly.tinyai.nnet.layer.conv.LazyConv2d;
import io.leavesfly.tinyai.nnet.layer.activation.ReLU;
import io.leavesfly.tinyai.nnet.layer.conv.MaxPool2d;


/**
 * 示例2: 延迟初始化的使用
 * 
 * 本示例展示如何:
 * 1. 使用LazyLinear自动推断输入维度
 * 2. 使用LazyConv2d自动推断输入通道数
 * 3. 延迟初始化的优势和注意事项
 */
public class LazyInitialization {

    /**
     * 使用LazyLinear的简单网络
     */
    static class LazyNet extends Module {
        private final LazyLinear fc1;
        private final ReLU relu;
        private final LazyLinear fc2;

        public LazyNet(String name) {
            super(name);
            
            // 使用LazyLinear，不需要指定输入维度
            // 输入维度将在首次forward时自动推断
            fc1 = new LazyLinear("fc1", 128, true);
            relu = new ReLU("relu");
            fc2 = new LazyLinear("fc2", 10, true);
            
            registerModule("fc1", fc1);
            registerModule("relu", relu);
            registerModule("fc2", fc2);
        }

        @Override
        public Variable forward(Variable... inputs) {
            Variable x = inputs[0];
            x = fc1.forward(x);
            x = relu.forward(x);
            x = fc2.forward(x);
            return x;
        }
    }

    /**
     * 使用LazyConv2d的卷积网络
     */
    static class LazyCNN extends Module {
        private final LazyConv2d conv1;
        private final ReLU relu1;
        private final MaxPool2d pool1;
        private final LazyConv2d conv2;
        private final ReLU relu2;
        private final MaxPool2d pool2;

        public LazyCNN(String name) {
            super(name);
            
            // 使用LazyConv2d，不需要指定输入通道数
            // 输入通道数将在首次forward时自动推断
            conv1 = new LazyConv2d("conv1", 32, 3, 3, 1, 1, true);
            relu1 = new ReLU("relu1");
            pool1 = new MaxPool2d("pool1", 2, 2,2);
            
            conv2 = new LazyConv2d("conv2", 64, 3, 3, 1, 1, true);
            relu2 = new ReLU("relu2");
            pool2 = new MaxPool2d("pool2", 2, 2,2);
            
            registerModule("conv1", conv1);
            registerModule("relu1", relu1);
            registerModule("pool1", pool1);
            registerModule("conv2", conv2);
            registerModule("relu2", relu2);
            registerModule("pool2", pool2);
        }

        @Override
        public Variable forward(Variable... inputs) {
            Variable x = inputs[0];
            
            x = conv1.forward(x);
            x = relu1.forward(x);
            x = pool1.forward(x);
            
            x = conv2.forward(x);
            x = relu2.forward(x);
            x = pool2.forward(x);
            
            return x;
        }
    }

    public static void main(String[] args) {
        System.out.println("=== 延迟初始化示例 ===\n");

        // 示例1: LazyLinear的使用
        System.out.println("示例1: LazyLinear");
        System.out.println("----------------------------------------");
        
        LazyNet lazyNet = new LazyNet("lazy_net");
        
        System.out.println("1. 创建模型后（初始化前）:");
        System.out.println("   参数数量: " + lazyNet.namedParameters().size());
        System.out.println("   注: 此时参数尚未创建，因为输入维度未知");
        System.out.println();

        // 创建输入数据 (batch_size=2, features=784)
        float[] inputData = new float[2 * 784];
        for (int i = 0; i < inputData.length; i++) {
            inputData[i] = (float) (Math.random() * 0.1);
        }
        NdArray inputArray = NdArray.of(inputData, Shape.of(2, 784));
        Variable input = new Variable(inputArray);
        
        System.out.println("2. 首次前向传播:");
        System.out.println("   输入形状: [2, 784]");
        Variable output = lazyNet.forward(input);
        System.out.println("   输出形状: " + shapeToString(output.getValue().getShape()));
        System.out.println();

        System.out.println("3. 初始化后:");
        System.out.println("   参数数量: " + lazyNet.namedParameters().size());
        for (String name : lazyNet.namedParameters().keySet()) {
            Shape shape = lazyNet.namedParameters().get(name).data().getShape();
            System.out.println("   - " + name + ": " + shapeToString(shape));
        }
        System.out.println();

        // 示例2: LazyConv2d的使用
        System.out.println("\n示例2: LazyConv2d");
        System.out.println("----------------------------------------");
        
        LazyCNN lazyCNN = new LazyCNN("lazy_cnn");
        
        System.out.println("1. 创建模型后（初始化前）:");
        System.out.println("   参数数量: " + lazyCNN.namedParameters().size());
        System.out.println();

        // 创建输入数据 (batch_size=2, channels=3, height=32, width=32)
        float[] imageData = new float[2 * 3 * 32 * 32];
        for (int i = 0; i < imageData.length; i++) {
            imageData[i] = (float) (Math.random() * 0.1);
        }
        NdArray imageArray = NdArray.of(imageData, Shape.of(2, 3, 32, 32));
        Variable imageInput = new Variable(imageArray);
        
        System.out.println("2. 首次前向传播:");
        System.out.println("   输入形状: [2, 3, 32, 32] (batch, channels, height, width)");
        Variable imageOutput = lazyCNN.forward(imageInput);
        System.out.println("   输出形状: " + shapeToString(imageOutput.getValue().getShape()));
        System.out.println();

        System.out.println("3. 初始化后:");
        System.out.println("   参数数量: " + lazyCNN.namedParameters().size());
        for (String name : lazyCNN.namedParameters().keySet()) {
            Shape shape = lazyCNN.namedParameters().get(name).data().getShape();
            System.out.println("   - " + name + ": " + shapeToString(shape));
        }
        System.out.println();

        // 延迟初始化的优势
        System.out.println("\n延迟初始化的优势:");
        System.out.println("----------------------------------------");
        System.out.println("1. 简化模型定义: 不需要手动计算中间层的输入维度");
        System.out.println("2. 提高灵活性: 同一模型可以处理不同维度的输入");
        System.out.println("3. 减少错误: 避免手动计算维度时的错误");
        System.out.println();

        // 注意事项
        System.out.println("注意事项:");
        System.out.println("----------------------------------------");
        System.out.println("1. 必须先调用forward()才能访问参数");
        System.out.println("2. 首次forward()会触发参数初始化，可能稍慢");
        System.out.println("3. 初始化后，输入维度不应该改变");
        System.out.println();

        System.out.println("=== 示例完成 ===");
    }

    private static String shapeToString(Shape shape) {
        int[] dims = shape.getShapeDims();
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < dims.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(dims[i]);
        }
        sb.append("]");
        return sb.toString();
    }
}
