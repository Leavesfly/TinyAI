package io.leavesfly.tinyai.nnet.core;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.util.GradientChecker;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Module基类功能测试
 */
public class ModuleTest {

    /**
     * 一个简化的测试模块
     */
    static class IdentityModule extends Module {
        private final Parameter weight;
        private final Parameter bias;

        IdentityModule(String name, boolean withParams, boolean withBuffer) {
            super(name);
            if (withParams) {
                this.weight = registerParameter("weight",
                    new Parameter(NdArray.of(new float[]{1.0f}, Shape.of(1))));
                this.bias = registerParameter("bias",
                    new Parameter(NdArray.of(new float[]{0.0f}, Shape.of(1))));
            } else {
                this.weight = null;
                this.bias = null;
            }
            if (withBuffer) {
                registerBuffer("running", NdArray.of(new float[]{0.0f}, Shape.of(1)));
            }
        }

        @Override
        public Variable forward(Variable... inputs) {
            Variable out = inputs[0];
            if (weight != null) {
                out = out.mul(weight);
            }
            if (bias != null) {
                out = out.add(bias);
            }
            return out;
        }
    }

    @Test
    public void testRegisterConflicts() {
        IdentityModule module = new IdentityModule("m", false, false);

        module.registerParameter("p", null);
        module.registerBuffer("buf", null);

        assertThrows(IllegalArgumentException.class,
            () -> module.registerParameter("p", new Parameter(NdArray.of(1.0f))));
        assertThrows(IllegalArgumentException.class,
            () -> module.registerBuffer("p", NdArray.of(Shape.of(1))));
        assertThrows(IllegalArgumentException.class,
            () -> module.registerModule("buf", new IdentityModule("child", false, false)));
    }

    @Test
    public void testNamedCollections() {
        IdentityModule parent = new IdentityModule("parent", true, true);
        IdentityModule child = new IdentityModule("child", true, true);
        parent.registerModule("child", child);

        Map<String, Parameter> params = parent.namedParameters();
        assertEquals(4, params.size());
        assertTrue(params.containsKey("weight"));
        assertTrue(params.containsKey("bias"));
        assertTrue(params.containsKey("child.weight"));
        assertTrue(params.containsKey("child.bias"));

        Map<String, NdArray> buffers = parent.namedBuffers();
        assertEquals(2, buffers.size());
        assertTrue(buffers.containsKey("running"));
        assertTrue(buffers.containsKey("child.running"));

        Map<String, Module> modules = parent.namedModules();
        assertEquals(1, modules.size());
        assertTrue(modules.containsKey("child"));
    }

    @Test
    public void testTrainEvalPropagation() {
        IdentityModule parent = new IdentityModule("parent", false, false);
        IdentityModule child = new IdentityModule("child", false, false);
        parent.registerModule("child", child);

        parent.eval();
        assertFalse(parent.isTraining());
        assertFalse(child.isTraining());

        parent.train();
        assertTrue(parent.isTraining());
        assertTrue(child.isTraining());
    }

    @Test
    public void testApplyRecursively() {
        IdentityModule parent = new IdentityModule("parent", false, false);
        IdentityModule child = new IdentityModule("child", false, false);
        parent.registerModule("child", child);

        parent.apply(module -> module.setName(module.getName() + "_applied"));

        assertEquals("parent_applied", parent.getName());
        assertEquals("child_applied", child.getName());
    }

    @Test
    public void testStateDictRoundTripForParamsAndBuffers() {
        IdentityModule origin = new IdentityModule("origin", true, true);
        IdentityModule originChild = new IdentityModule("child", true, true);
        origin.registerModule("child", originChild);

        // 写入可重复的参数与缓冲区数值
        origin.getParameter("weight").setData(NdArray.of(new float[]{2.0f}, Shape.of(1)));
        origin.getParameter("bias").setData(NdArray.of(new float[]{-1.0f}, Shape.of(1)));
        originChild.getParameter("weight").setData(NdArray.of(new float[]{3.0f}, Shape.of(1)));
        originChild.getParameter("bias").setData(NdArray.of(new float[]{0.5f}, Shape.of(1)));
        origin.getBuffer("running").getArray()[0] = 7.0f;
        originChild.getBuffer("running").getArray()[0] = 5.0f;

        Map<String, NdArray> state = origin.stateDict();

        IdentityModule restored = new IdentityModule("origin", true, true);
        IdentityModule restoredChild = new IdentityModule("child", true, true);
        restored.registerModule("child", restoredChild);

        restored.loadStateDict(state, true);

        assertArrayEquals(origin.getParameter("weight").data().getArray(),
            restored.getParameter("weight").data().getArray(), 1e-6f);
        assertArrayEquals(origin.getParameter("bias").data().getArray(),
            restored.getParameter("bias").data().getArray(), 1e-6f);
        assertArrayEquals(originChild.getParameter("weight").data().getArray(),
            restoredChild.getParameter("weight").data().getArray(), 1e-6f);
        assertArrayEquals(originChild.getParameter("bias").data().getArray(),
            restoredChild.getParameter("bias").data().getArray(), 1e-6f);

        assertEquals(origin.getBuffer("running").getArray()[0],
            restored.getBuffer("running").getArray()[0], 1e-6f);
        assertEquals(originChild.getBuffer("running").getArray()[0],
            restoredChild.getBuffer("running").getArray()[0], 1e-6f);
    }

    @Test
    public void testIdentityModuleGradientCheck() {
        IdentityModule module = new IdentityModule("test", true, false);
        NdArray inputData = NdArray.randn(Shape.of(32, 1));
        Variable input = new Variable(inputData);
        
        // 使用 GradientChecker 检查计算图连通性
        GradientChecker.checkGraphConnectivity(module, input);
    }
}

