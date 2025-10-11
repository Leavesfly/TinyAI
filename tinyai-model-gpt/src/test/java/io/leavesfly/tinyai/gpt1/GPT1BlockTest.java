package io.leavesfly.tinyai.gpt1;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertTrue;

public class GPT1BlockTest {

    @Test
    public void testSampleFromLogits() {
        // 创建一个小型GPT1Block用于测试
        GPT1Block gpt1Block = new GPT1Block("test_gpt1", 100, 32);
        
        // 创建测试logits数据 (batchSize=1, sequenceLength=1, vocabSize=100)
        float[][][] logitsData = new float[1][1][100];
        Random random = new Random(42);
        
        // 填充一些随机logits值
        for (int i = 0; i < 100; i++) {
            logitsData[0][0][i] = random.nextFloat() * 10 - 5; // 范围在[-5, 5]
        }
        
        Variable logits = new Variable(NdArray.of(logitsData));
        
        // 测试不同的温度值
        int tokenId1 = gpt1Block.sampleFromLogits(logits, 1.0);
        int tokenId2 = gpt1Block.sampleFromLogits(logits, 0.5);
        int tokenId3 = gpt1Block.sampleFromLogits(logits, 2.0);
        
        // 验证返回的token ID在有效范围内
        assertTrue("Token ID should be in valid range", tokenId1 >= 0 && tokenId1 < 100);
        assertTrue("Token ID should be in valid range", tokenId2 >= 0 && tokenId2 < 100);
        assertTrue("Token ID should be in valid range", tokenId3 >= 0 && tokenId3 < 100);
        
        System.out.println("Sampled token IDs: " + tokenId1 + ", " + tokenId2 + ", " + tokenId3);
    }
    
    @Test
    public void testSampleFromLogitsWithSequence() {
        // 创建一个小型GPT1Block用于测试
        GPT1Block gpt1Block = new GPT1Block("test_gpt1_seq", 50, 16);
        
        // 创建测试logits数据 (batchSize=1, sequenceLength=5, vocabSize=50)
        float[][][] logitsData = new float[1][5][50];
        Random random = new Random(123);
        
        // 填充一些随机logits值
        for (int s = 0; s < 5; s++) {
            for (int i = 0; i < 50; i++) {
                logitsData[0][s][i] = random.nextFloat() * 8 - 4; // 范围在[-4, 4]
            }
        }
        
        Variable logits = new Variable(NdArray.of(logitsData));
        
        // 测试采样
        int tokenId = gpt1Block.sampleFromLogits(logits, 1.0);
        
        // 验证返回的token ID在有效范围内
        assertTrue("Token ID should be in valid range", tokenId >= 0 && tokenId < 50);
        
        System.out.println("Sampled token ID from sequence: " + tokenId);
    }
}