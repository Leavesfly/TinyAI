package io.leavesfly.tinyai.util;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.*;

/**
 * 逐步反向传播可视化工具
 * 
 * 该类提供了逐步展示反向传播过程的功能，能够详细显示每一步的梯度计算过程，
 * 包括当前处理的变量、函数、梯度值的变化等信息。
 * 
 * @author 山泽
 * @version 1.0
 */
public class StepByStepVisualizer {
    
    /**
     * 显示反向传播的逐步过程
     * 
     * @param rootVariable 开始反向传播的根变量
     */
    public static void showBackpropagation(Variable rootVariable) {
        System.out.println("🚀 开始逐步反向传播演示");
        System.out.println("==============================");
        
        if (!rootVariable.isRequireGrad()) {
            System.out.println("⚠️  警告: 根变量不需要梯度计算，反向传播将被跳过");
            return;
        }
        
        // 初始化根变量的梯度
        if (rootVariable.getGrad() == null) {
            rootVariable.setGrad(NdArray.ones(rootVariable.getValue().getShape()));
            System.out.printf("📍 步骤 0: 初始化根变量梯度%n");
            displayVariableInfo(rootVariable, "根变量");
            System.out.println();
        }
        
        // 使用栈来实现迭代的反向传播并记录每一步
        Stack<BackpropStep> stack = new Stack<>();
        Set<Variable> processedVariables = new HashSet<>();
        int stepCounter = 1;
        
        // 添加根变量到处理栈
        stack.push(new BackpropStep(rootVariable, "开始反向传播"));
        
        while (!stack.isEmpty()) {
            BackpropStep currentStep = stack.pop();
            Variable currentVar = currentStep.variable;
            String stepDescription = currentStep.description;
            
            // 避免重复处理同一个变量
            if (processedVariables.contains(currentVar)) {
                continue;
            }
            processedVariables.add(currentVar);
            
            System.out.printf("📍 步骤 %d: %s%n", stepCounter++, stepDescription);
            displayVariableInfo(currentVar, "当前变量");
            
            Function creator = currentVar.getCreator();
            if (creator == null) {
                System.out.println("   🔸 这是叶子节点变量，无需继续反向传播");
                System.out.println();
                continue;
            }
            
            // 显示函数信息
            displayFunctionInfo(creator);
            
            Variable[] inputs = creator.getInputs();
            if (inputs == null || inputs.length == 0) {
                System.out.println("   🔸 函数没有输入变量");
                System.out.println();
                continue;
            }
            
            try {
                // 计算梯度
                System.out.println("   🧮 计算输入变量的梯度...");
                List<NdArray> inputGrads = creator.backward(currentVar.getGrad());
                
                if (inputs.length != inputGrads.size()) {
                    System.out.printf("   ❌ 错误: 输入变量数量(%d)与梯度数量(%d)不匹配%n", 
                        inputs.length, inputGrads.size());
                    continue;
                }
                
                // 更新输入变量的梯度并添加到处理栈
                for (int i = 0; i < inputs.length; i++) {
                    Variable input = inputs[i];
                    NdArray grad = inputGrads.get(i);
                    
                    if (!input.isRequireGrad()) {
                        System.out.printf("   🔸 输入变量 %d (%s) 不需要梯度，跳过%n", 
                            i, getVariableName(input));
                        continue;
                    }
                    
                    // 累加梯度
                    if (input.getGrad() != null) {
                        NdArray oldGrad = input.getGrad();
                        NdArray newGrad = oldGrad.add(grad);
                        input.setGrad(newGrad);
                        System.out.printf("   ✅ 输入变量 %d (%s): 梯度累加%n", 
                            i, getVariableName(input));
                        System.out.printf("      旧梯度: %s%n", formatGradient(oldGrad));
                        System.out.printf("      新增梯度: %s%n", formatGradient(grad));
                        System.out.printf("      累加后: %s%n", formatGradient(newGrad));
                    } else {
                        input.setGrad(grad);
                        System.out.printf("   ✅ 输入变量 %d (%s): 设置梯度%n", 
                            i, getVariableName(input));
                        System.out.printf("      梯度值: %s%n", formatGradient(grad));
                    }
                    
                    // 如果输入变量有创建者，添加到栈中继续处理
                    if (input.getCreator() != null) {
                        String nextDescription = String.format("处理变量 %s 的反向传播", getVariableName(input));
                        stack.push(new BackpropStep(input, nextDescription));
                    }
                }
                
            } catch (Exception e) {
                System.out.printf("   ❌ 梯度计算出错: %s%n", e.getMessage());
            }
            
            System.out.println();
        }
        
        System.out.println("🎉 反向传播完成！");
        System.out.println("==============================");
        
        // 显示最终的梯度结果
        showFinalGradients(rootVariable, new HashSet<>());
    }
    
    /**
     * 显示变量信息
     */
    private static void displayVariableInfo(Variable variable, String label) {
        String name = getVariableName(variable);
        NdArray value = variable.getValue();
        NdArray grad = variable.getGrad();
        
        System.out.printf("   📊 %s: %s%n", label, name);
        System.out.printf("      形状: %s%n", value.getShape());
        System.out.printf("      数值: %s%n", formatValue(value));
        System.out.printf("      梯度: %s%n", grad != null ? formatGradient(grad) : "未设置");
        System.out.printf("      需要梯度: %s%n", variable.isRequireGrad() ? "是" : "否");
    }
    
    /**
     * 显示函数信息
     */
    private static void displayFunctionInfo(Function function) {
        String funcName = function.getClass().getSimpleName();
        Variable[] inputs = function.getInputs();
        
        System.out.printf("   🔧 函数: %s%n", funcName);
        if (inputs != null && inputs.length > 0) {
            System.out.printf("      输入数量: %d%n", inputs.length);
            for (int i = 0; i < inputs.length; i++) {
                System.out.printf("      输入 %d: %s [形状: %s]%n", 
                    i, getVariableName(inputs[i]), inputs[i].getValue().getShape());
            }
        }
    }
    
    /**
     * 显示最终的梯度结果
     */
    private static void showFinalGradients(Variable variable, Set<Variable> visited) {
        if (variable == null || visited.contains(variable)) {
            return;
        }
        
        visited.add(variable);
        
        if (variable.isRequireGrad() && variable.getGrad() != null) {
            System.out.printf("📈 %s 最终梯度: %s%n", 
                getVariableName(variable), formatGradient(variable.getGrad()));
        }
        
        Function creator = variable.getCreator();
        if (creator != null && creator.getInputs() != null) {
            for (Variable input : creator.getInputs()) {
                showFinalGradients(input, visited);
            }
        }
    }
    
    /**
     * 获取变量名称
     */
    private static String getVariableName(Variable variable) {
        return variable.getName() != null ? variable.getName() : "未命名变量";
    }
    
    /**
     * 格式化数值显示
     */
    private static String formatValue(NdArray value) {
        if (value == null) {
            return "null";
        }
        
        // 转换为具体实现类来访问数据
        float[] data = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) value).buffer;
        if (data.length == 0) {
            return "[]";
        } else if (data.length == 1) {
            return String.format("%.4f", data[0]);
        } else if (data.length <= 3) {
            StringBuilder sb = new StringBuilder("[");
            for (int i = 0; i < data.length; i++) {
                if (i > 0) sb.append(", ");
                sb.append(String.format("%.4f", data[i]));
            }
            sb.append("]");
            return sb.toString();
        } else {
            return String.format("[%.4f, %.4f, ..., %.4f] (%d个元素)", 
                data[0], data[1], data[data.length-1], data.length);
        }
    }
    
    /**
     * 格式化梯度显示
     */
    private static String formatGradient(NdArray grad) {
        return formatValue(grad);
    }
    
    /**
     * 反向传播步骤的包装类
     */
    private static class BackpropStep {
        final Variable variable;
        final String description;
        
        BackpropStep(Variable variable, String description) {
            this.variable = variable;
            this.description = description;
        }
    }
}