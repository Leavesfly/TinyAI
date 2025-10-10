package io.leavesfly.tinyai.cv.model.cnn;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class VGGBlockTest {
    
    @Test
    public void testVGGBlockConstruction() {
        // 测试VGG块构造（2个卷积层）
        int numConvs = 2;
        int inChannels = 64;
        int outChannels = 128;
        VGGBlock vggBlock = new VGGBlock(numConvs, inChannels, outChannels);
        
        // 验证VGG块不为null
        assertNotNull(vggBlock);
        
        // 验证层数
        assertEquals(7, vggBlock.getLayers().length); // 2 conv + 2 bn + 2 activation + 1 pool = 7
        
        // 验证参数数量大于0
        assertTrue(vggBlock.getParameterCount() > 0);
    }
    
    @Test
    public void testVGGBlockForward() {
        // 创建VGG块（2个卷积层）
        int numConvs = 2;
        int inChannels = 64;
        int outChannels = 128;
        VGGBlock vggBlock = new VGGBlock(numConvs, inChannels, outChannels);
        
        // 创建输入特征图
        // 形状: (batch_size=1, channels=64, height=112, width=112)
        float[][][][] inputData = new float[1][64][112][112];
        // 填充一些示例数据
        for (int c = 0; c < 64; c++) {
            for (int h = 0; h < 112; h++) {
                for (int w = 0; w < 112; w++) {
                    inputData[0][c][h][w] = (float) (c * 112 * 112 + h * 112 + w) / (64 * 112 * 112);
                }
            }
        }
        
        NdArray input = NdArray.of(inputData);
        Variable inputVar = new Variable(input);
        
        // 执行前向传播
        Variable output = vggBlock.forward(inputVar);
        
        // 验证输出形状
        Shape outputShape = output.getValue().getShape();
        assertEquals(4, outputShape.getDimNum()); // 应该是4D (batch_size, channels, height, width)
        assertEquals(1, outputShape.getDimension(0)); // batch_size = 1
        assertEquals(128, outputShape.getDimension(1)); // channels = 128
        assertEquals(56, outputShape.getDimension(2)); // height = 56 (112/2 due to pooling)
        assertEquals(56, outputShape.getDimension(3)); // width = 56 (112/2 due to pooling)
    }
    
    @Test
    public void testVGGBlockWithSingleConv() {
        // 测试VGG块（1个卷积层）
        int numConvs = 1;
        int inChannels = 64;
        int outChannels = 128;
        VGGBlock vggBlock = new VGGBlock(numConvs, inChannels, outChannels);
        
        // 验证层数
        assertEquals(4, vggBlock.getLayers().length); // 1 conv + 1 bn + 1 activation + 1 pool = 4
        
        // 创建输入特征图
        float[][][][] inputData = new float[1][64][112][112];
        NdArray input = NdArray.of(inputData);
        Variable inputVar = new Variable(input);
        
        // 执行前向传播
        Variable output = vggBlock.forward(inputVar);
        
        // 验证输出形状
        Shape outputShape = output.getValue().getShape();
        assertEquals(4, outputShape.getDimNum());
        assertEquals(1, outputShape.getDimension(0));
        assertEquals(128, outputShape.getDimension(1));
        assertEquals(56, outputShape.getDimension(2));
        assertEquals(56, outputShape.getDimension(3));
    }
    
    @Test
    public void testVGGBlockParameterCount() {
        // 创建VGG块（2个卷积层）
        int numConvs = 2;
        int inChannels = 64;
        int outChannels = 128;
        VGGBlock vggBlock = new VGGBlock(numConvs, inChannels, outChannels);
        
        // 验证参数数量大于0
        assertTrue(vggBlock.getParameterCount() > 0);
    }
}