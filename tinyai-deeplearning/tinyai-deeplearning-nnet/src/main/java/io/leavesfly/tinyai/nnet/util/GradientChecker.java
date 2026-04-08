package io.leavesfly.tinyai.nnet.util;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.core.Module;
import io.leavesfly.tinyai.nnet.core.Parameter;

import java.util.Map;

/**
 * 自动微分连通性检查工具
 * <p>
 * 用于在开发和测试阶段诊断 Module 的前向传播实现是否正确构建了计算图，
 * 确保梯度能够正确地回传到所有可训练参数。
 * </p>
 */
public class GradientChecker {

    /**
     * 检查 Module 的 forward 是否正确构建了计算图
     * <p>
     * 该方法执行以下检查：
     * 1. 输出 Variable 是否具有 Creator（除非直接返回输入或参数）。
     * 2. 执行反向传播后，所有 requiresGrad=true 的参数是否都获得了非空梯度。
     * </p>
     *
     * @param module 待检查的模块
     * @param inputs 模拟输入
     * @throws IllegalStateException 如果检查失败
     */
    public static void checkGraphConnectivity(Module module, Variable... inputs) {
        // 1. 确保进入训练模式
        boolean originalMode = module.isTraining();
        module.train();
        
        try {
            // 2. 清理旧梯度
            module.clearGrads();
            
            // 3. 执行前向传播
            Variable output = module.forward(inputs);
            
            if (output == null) {
                throw new IllegalStateException("Module forward returned null.");
            }

            // 检查点 A: 输出必须有 Creator 
            // (除非它是直接返回了某个 Param 或 Input，但通常会有计算，或者是叶子节点)
            if (output.getCreator() == null) {
                // 特例：如果模块就是一个 Identity，可能直接返回输入；或者直接返回参数
                boolean isInputOrParam = false;
                for (Variable in : inputs) {
                    if (in == output) isInputOrParam = true;
                }
                // 检查是否直接返回了某个参数
                if (!isInputOrParam) {
                    Map<String, Parameter> params = module.namedParameters();
                    for (Parameter param : params.values()) {
                        if (param == output) {
                            isInputOrParam = true;
                            break;
                        }
                    }
                }

                if (!isInputOrParam) {
                    throw new IllegalStateException(
                        String.format("【计算图断裂】Module [%s] 的输出 Variable 没有 Creator。\n" +
                            "原因：可能在 forward 中使用了 new Variable(ndarray) 或纯 NdArray 操作，导致自动微分链条断开。", 
                            module.getClass().getSimpleName())
                    );
                }
            }

            // 4. 执行反向传播（构造一个伪 Loss）
            // 如果输出是标量，直接backward；如果是张量，先求和变标量
            Variable loss = output.sum(); 
            loss.backward();

            // 检查点 B: 所有 requireGrad=true 的参数必须有梯度
            Map<String, Parameter> params = module.namedParameters();
            for (Map.Entry<String, Parameter> entry : params.entrySet()) {
                String name = entry.getKey();
                Parameter param = entry.getValue();
                
                if (param.requiresGrad()) {
                    if (param.getGrad() == null) {
                        throw new IllegalStateException(
                            String.format("【梯度丢失】参数 [%s] 在反向传播后梯度仍为 null。\n" +
                                "原因：该参数未参与计算图构建，或者计算路径中断。", name)
                        );
                    }
                    
                    // 检查梯度是否全 0 (只是警告，有时全0是数学上可能的，例如 ReLU 处于死区)
                    if (Math.abs(param.getGrad().abs().max()) < 1e-9f) {
                        System.err.printf("警告: Module [%s] 参数 [%s] 的梯度全为 0，请确认是否符合预期。%n", 
                            module.getClass().getSimpleName(), name);
                    }
                }
            }
            
            System.out.printf("✅ Module [%s] 计算图连通性检查通过。%n", module.getClass().getSimpleName());
            
        } finally {
            // 恢复原始模式
            module.train(originalMode);
        }
    }
}

