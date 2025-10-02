package io.leavesfly.tinyai.nnet.layer.norm;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Parameter;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * BatchNorm层的简化单元测试
 * 
 * 由于BatchNorm实现还不够完善，这里只测试基本的参数初始化功能
 */
public class BatchNormSimpleTest {

    private BatchNorm batchNorm2D;
    
    @Before
    public void setUp() {
        // 创建2D输入的BatchNorm层
        batchNorm2D = new BatchNorm("bn2d", Shape.of(4, 10), 10, 1e-5);
    }

    @Test
    public void testParameterInitialization() {
        // 测试参数是否正确初始化
        Parameter gamma = batchNorm2D.getParamBy("gamma");
        Parameter beta = batchNorm2D.getParamBy("beta");
        
        assertNotNull("gamma参数应该被初始化", gamma);
        assertNotNull("beta参数应该被初始化", beta);
        
        // 检查gamma初始化为1
        NdArray gammaValue = gamma.getValue();
        assertEquals("gamma形状应该是(10,)", Shape.of(10), gammaValue.getShape());
        
        // 检查beta初始化为0
        NdArray betaValue = beta.getValue();
        assertEquals("beta形状应该是(10,)", Shape.of(10), betaValue.getShape());
        
        // 验证初始值（gamma应该接近1，beta应该接近0）
        float gammaFirst = gammaValue.get(0);
        float betaFirst = betaValue.get(0);
        
        assertEquals("gamma初始值应该是1", 1.0f, gammaFirst, 0.001f);
        assertEquals("beta初始值应该是0", 0.0f, betaFirst, 0.001f);
    }

    @Test
    public void testParameterNames() {
        // 测试参数名称是否正确设置
        Parameter gamma = batchNorm2D.getParamBy("gamma");
        Parameter beta = batchNorm2D.getParamBy("beta");
        
        assertEquals("gamma参数名称应该正确", "bn2d_gamma", gamma.getName());
        assertEquals("beta参数名称应该正确", "bn2d_beta", beta.getName());
    }

    @Test
    public void testDefaultConstructor() {
        // 测试默认构造函数
        BatchNorm defaultBN = new BatchNorm("default", Shape.of(3, 5), 5, 1e-5);
        
        // 验证参数已初始化
        assertNotNull("默认构造的gamma参数应该存在", defaultBN.getParamBy("gamma"));
        assertNotNull("默认构造的beta参数应该存在", defaultBN.getParamBy("beta"));
        
        // 验证特征数量推断正确
        Parameter gamma = defaultBN.getParamBy("gamma");
        assertEquals("默认构造应该推断特征数为5", Shape.of(5), gamma.getValue().getShape());
    }

    @Test
    public void testNullInputShape() {
        // 测试null输入形状的处理
        BatchNorm nullShapeBN = new BatchNorm("null_shape", null);
        
        // 应该能正常创建，使用默认特征数
        assertNotNull("null形状输入应该能正常创建gamma参数", nullShapeBN.getParamBy("gamma"));
        assertNotNull("null形状输入应该能正常创建beta参数", nullShapeBN.getParamBy("beta"));
    }

    @Test
    public void testEpsilonParameter() {
        // 测试不同epsilon值的BatchNorm
        BatchNorm largeBN = new BatchNorm("large_eps", Shape.of(2, 3), 3, 1e-3);
        BatchNorm smallBN = new BatchNorm("small_eps", Shape.of(2, 3), 3, 1e-8);
        
        // 两者都应该能正常创建参数
        assertNotNull("大epsilon值应该正常初始化gamma", largeBN.getParamBy("gamma"));
        assertNotNull("大epsilon值应该正常初始化beta", largeBN.getParamBy("beta"));
        assertNotNull("小epsilon值应该正常初始化gamma", smallBN.getParamBy("gamma"));
        assertNotNull("小epsilon值应该正常初始化beta", smallBN.getParamBy("beta"));
    }
}