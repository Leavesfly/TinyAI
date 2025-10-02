package io.leavesfly.tinyai.nlp;

import org.junit.Test;

import static org.junit.Assert.*;

/**
 * MoE-GPT模型简单测试
 * 用于隔离复杂测试中的问题
 *
 * @author leavesfly
 * @version 1.0
 */
public class MoEGPTModelSimpleTest {

    @Test
    public void testModelConstructionBasic() {
        // 测试最基本的模型构造
        try {
            MoEGPTModel model = new MoEGPTModel(
                    "test_simple",
                    100,  // vocabSize (小一点)
                    32,   // dModel (小一点)
                    1,    // numLayers (只有1层)
                    2,    // numHeads
                    2,    // numExperts (只有2个专家)
                    2,    // topK
                    8     // maxSeqLength (很短)
            );

            // 基本验证
            assertNotNull("模型不应为null", model);
            assertEquals("词汇表大小应该正确", 100, model.getVocabSize());
            assertEquals("模型维度应该正确", 32, model.getDModel());
            assertEquals("层数应该正确", 1, model.getNumLayers());
            assertEquals("注意力头数应该正确", 2, model.getNumHeads());
            assertEquals("专家数量应该正确", 2, model.getNumExperts());
            assertEquals("topK应该正确", 2, model.getTopK());
            assertEquals("最大序列长度应该正确", 8, model.getMaxSeqLength());

            System.out.println("基本构造测试通过");

        } catch (Exception e) {
            e.printStackTrace();
            fail("模型构造失败: " + e.getMessage());
        }
    }

    @Test
    public void testComponentInitialization() {
        try {
            MoEGPTModel model = new MoEGPTModel(
                    "test_components",
                    50,   // vocabSize
                    16,   // dModel
                    1,    // numLayers
                    1,    // numHeads
                    1,    // numExperts
                    1,    // topK
                    4     // maxSeqLength
            );

            // 验证组件初始化
            assertNotNull("Token嵌入层不应为null", model.getTokenEmbedding());
            assertNotNull("MoE Transformer块列表不应为null", model.getMoeTransformerBlocks());
            assertNotNull("最终层归一化不应为null", model.getFinalLayerNorm());
            assertNotNull("输出头不应为null", model.getOutputHead());

            assertEquals("MoE Transformer块数量应该正确", 1, model.getMoeTransformerBlocks().size());

            System.out.println("组件初始化测试通过");

        } catch (Exception e) {
            e.printStackTrace();
            fail("组件初始化失败: " + e.getMessage());
        }
    }

    @Test
    public void testInvalidParameters() {
        // 测试无效参数
        try {
            new MoEGPTModel(
                    "invalid_model",
                    10,  // vocabSize
                    5,   // dModel = 5，不能被numHeads=2整除
                    1,   // numLayers
                    2,   // numHeads
                    2,   // numExperts
                    1,   // topK
                    4    // maxSeqLength
            );
            fail("应该抛出异常因为dModel不能被numHeads整除");
        } catch (IllegalArgumentException e) {
            assertTrue("异常消息应该包含相关错误信息",
                    e.getMessage().contains("dModel") || e.getMessage().contains("numHeads"));
            System.out.println("无效参数测试通过: " + e.getMessage());
        }
    }
}