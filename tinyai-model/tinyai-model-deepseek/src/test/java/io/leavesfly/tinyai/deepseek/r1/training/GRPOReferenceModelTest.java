package io.leavesfly.tinyai.deepseek.r1.training;

import io.leavesfly.tinyai.deepseek.r1.DeepSeekR1Config;
import io.leavesfly.tinyai.deepseek.r1.DeepSeekR1Model;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.core.Parameter;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Collection;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * GRPOReferenceModel 单元测试
 *
 * <p>重点验证 P0-3 GRPO 重构中参考模型的三个关键不变式：
 * <ol>
 *   <li><b>Freeze</b>：构造后所有参数 {@code requireGrad=false}</li>
 *   <li><b>Detach</b>：forwardLogits 返回的 Variable 不连接 policy 的计算图</li>
 *   <li><b>Sync</b>：syncFrom 后仍然保持 frozen 状态</li>
 * </ol>
 *
 * @author leavesfly
 */
public class GRPOReferenceModelTest {

    private DeepSeekR1Config config;
    private DeepSeekR1Model policy;

    @BeforeEach
    public void setUp() {
        // 使用极小配置加速测试（< 1s）
        config = DeepSeekR1Config.createTinyConfig();
        config.setVocabSize(100);
        config.setNEmbd(32);
        config.setNLayer(1);
        config.setNHead(2);
        config.setNInner(64);
        config.setNPositions(16);
        config.setNumExperts(2);
        config.setTopK(1);
        config.setExpertHiddenDim(64);
        policy = new DeepSeekR1Model("grpo-test-policy", config);
    }

    /**
     * 不变式 1：参考模型构造后所有参数必须 requireGrad=false
     */
    @Test
    public void testConstructor_freezesAllParameters() {
        GRPOReferenceModel refModel = new GRPOReferenceModel(policy, config);
        Collection<Parameter> params = refModel.getReferenceModel().getModule().parameters(true);
        assertFalse(params.isEmpty(), "参考模型必须有参数");
        for (Parameter p : params) {
            if (p != null) {
                assertFalse(p.isRequireGrad(),
                        "参考模型所有参数必须 requireGrad=false，违反项: " + p.getName());
            }
        }
    }

    /**
     * 不变式 2：forwardLogits 返回的 Variable 必须已 detach（不连 policy 计算图）
     */
    @Test
    public void testForwardLogits_returnsDetachedVariable() {
        GRPOReferenceModel refModel = new GRPOReferenceModel(policy, config);
        // 构造合法输入 [batchSize=1, seqLen=4]
        float[][] tokenData = {{1f, 2f, 3f, 4f}};
        Variable inputIds = new Variable(NdArray.of(tokenData));

        Variable logits = refModel.forwardLogits(inputIds);

        assertNotNull(logits, "forwardLogits 不得返回 null");
        assertNotNull(logits.getValue(), "返回 Variable 的 value 不得为 null");
        // detach 后 creator 应为空
        assertTrue(logits.getCreator() == null,
                "forwardLogits 返回的 Variable 必须已 detach（creator 为 null）");
    }

    /**
     * 不变式 3：syncFrom 后参考模型依然保持 frozen
     */
    @Test
    public void testSyncFrom_keepsFrozen() {
        GRPOReferenceModel refModel = new GRPOReferenceModel(policy, config);

        // 模拟训练一段时间后再同步
        refModel.syncFrom(policy);

        Collection<Parameter> params = refModel.getReferenceModel().getModule().parameters(true);
        for (Parameter p : params) {
            if (p != null) {
                assertFalse(p.isRequireGrad(),
                        "syncFrom 后必须重新 freeze，违反项: " + p.getName());
            }
        }
    }

    /**
     * 显式 freeze 方法的幂等性
     */
    @Test
    public void testFreeze_isIdempotent() {
        GRPOReferenceModel refModel = new GRPOReferenceModel(policy, config);
        refModel.freeze();
        refModel.freeze();
        refModel.freeze();
        for (Parameter p : refModel.getReferenceModel().getModule().parameters(true)) {
            if (p != null) {
                assertFalse(p.isRequireGrad());
            }
        }
    }

    @Test
    public void testConstructor_rejectsNullPolicy() {
        assertThrows(IllegalArgumentException.class,
                () -> new GRPOReferenceModel(null, config));
    }

    @Test
    public void testConstructor_rejectsNullConfig() {
        assertThrows(IllegalArgumentException.class,
                () -> new GRPOReferenceModel(policy, null));
    }

    @Test
    public void testSyncFrom_rejectsNullPolicy() {
        GRPOReferenceModel refModel = new GRPOReferenceModel(policy, config);
        assertThrows(IllegalArgumentException.class, () -> refModel.syncFrom(null));
    }

    @Test
    public void testForwardLogits_rejectsNullInput() {
        GRPOReferenceModel refModel = new GRPOReferenceModel(policy, config);
        assertThrows(IllegalArgumentException.class, () -> refModel.forwardLogits(null));
    }
}
