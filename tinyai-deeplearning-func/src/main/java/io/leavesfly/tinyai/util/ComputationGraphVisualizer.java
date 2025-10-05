package io.leavesfly.tinyai.util;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.*;

/**
 * 计算图可视化工具
 * 
 * 该类提供了计算图的可视化功能，能够以文本形式展示变量之间的依赖关系，
 * 包括变量的名称、形状、数值以及函数之间的连接关系。
 * 
 * @author 山泽
 * @version 1.0
 */
public class ComputationGraphVisualizer {
    
    /**
     * 显示以指定变量为根节点的计算图
     * 
     * @param rootVariable 计算图的根节点变量
     */
    public static void display(Variable rootVariable) {
        System.out.println("=== 计算图结构 ===");
        System.out.println();
        
        // 收集所有变量和函数
        Set<Variable> allVariables = new LinkedHashSet<>();
        Set<Function> allFunctions = new LinkedHashSet<>();
        Map<Variable, Integer> variableIds = new HashMap<>();
        Map<Function, Integer> functionIds = new HashMap<>();
        
        // 深度优先遍历收集节点
        collectNodes(rootVariable, allVariables, allFunctions, new HashSet<>());
        
        // 为变量和函数分配ID
        assignIds(allVariables, allFunctions, variableIds, functionIds);
        
        // 显示变量信息
        displayVariables(allVariables, variableIds);
        
        // 显示函数信息
        displayFunctions(allFunctions, functionIds, variableIds);
        
        // 显示计算图结构
        displayGraphStructure(rootVariable, variableIds, functionIds, new HashSet<>());
        
        System.out.println("=== 计算图结束 ===");
    }
    
    /**
     * 递归收集计算图中的所有变量和函数
     */
    private static void collectNodes(Variable variable, Set<Variable> variables, 
                                   Set<Function> functions, Set<Variable> visited) {
        if (variable == null || visited.contains(variable)) {
            return;
        }
        
        visited.add(variable);
        variables.add(variable);
        
        Function creator = variable.getCreator();
        if (creator != null) {
            functions.add(creator);
            Variable[] inputs = creator.getInputs();
            if (inputs != null) {
                for (Variable input : inputs) {
                    collectNodes(input, variables, functions, visited);
                }
            }
        }
    }
    
    /**
     * 为变量和函数分配唯一ID
     */
    private static void assignIds(Set<Variable> variables, Set<Function> functions,
                                Map<Variable, Integer> variableIds, Map<Function, Integer> functionIds) {
        int varId = 0;
        for (Variable var : variables) {
            variableIds.put(var, varId++);
        }
        
        int funcId = 0;
        for (Function func : functions) {
            functionIds.put(func, funcId++);
        }
    }
    
    /**
     * 显示所有变量的信息
     */
    private static void displayVariables(Set<Variable> variables, Map<Variable, Integer> variableIds) {
        System.out.println("📊 变量列表:");
        for (Variable var : variables) {
            int id = variableIds.get(var);
            String name = var.getName() != null ? var.getName() : "unnamed";
            NdArray value = var.getValue();
            String shape = value.getShape().toString();
            String valueStr = formatValue(value);
            
            System.out.printf("  V%d: %s [形状: %s] [值: %s]%s%n", 
                id, name, shape, valueStr, 
                var.isRequireGrad() ? " [需要梯度]" : "");
        }
        System.out.println();
    }
    
    /**
     * 显示所有函数的信息
     */
    private static void displayFunctions(Set<Function> functions, Map<Function, Integer> functionIds,
                                       Map<Variable, Integer> variableIds) {
        System.out.println("🔧 函数列表:");
        for (Function func : functions) {
            int id = functionIds.get(func);
            String funcName = func.getClass().getSimpleName();
            
            StringBuilder inputsStr = new StringBuilder();
            Variable[] inputs = func.getInputs();
            if (inputs != null) {
                for (int i = 0; i < inputs.length; i++) {
                    if (i > 0) inputsStr.append(", ");
                    inputsStr.append("V").append(variableIds.get(inputs[i]));
                }
            }
            
            Variable output = func.getOutput();
            String outputStr = output != null ? "V" + variableIds.get(output) : "null";
            
            System.out.printf("  F%d: %s [输入: %s] [输出: %s]%n", 
                id, funcName, inputsStr.toString(), outputStr);
        }
        System.out.println();
    }
    
    /**
     * 显示计算图的树形结构
     */
    private static void displayGraphStructure(Variable variable, Map<Variable, Integer> variableIds,
                                            Map<Function, Integer> functionIds, Set<Variable> visited) {
        if (variable == null || visited.contains(variable)) {
            return;
        }
        
        visited.add(variable);
        
        System.out.println("🌳 计算图结构 (从输出到输入):");
        displayNode(variable, variableIds, functionIds, "", true);
    }
    
    /**
     * 递归显示节点结构
     */
    private static void displayNode(Variable variable, Map<Variable, Integer> variableIds,
                                  Map<Function, Integer> functionIds, String prefix, boolean isLast) {
        if (variable == null) {
            return;
        }
        
        // 显示变量节点
        int varId = variableIds.get(variable);
        String name = variable.getName() != null ? variable.getName() : "unnamed";
        String connector = isLast ? "└── " : "├── ";
        System.out.printf("%s%sV%d (%s)%n", prefix, connector, varId, name);
        
        Function creator = variable.getCreator();
        if (creator != null) {
            // 显示函数节点
            int funcId = functionIds.get(creator);
            String funcName = creator.getClass().getSimpleName();
            String newPrefix = prefix + (isLast ? "    " : "│   ");
            System.out.printf("%s└── F%d (%s)%n", newPrefix, funcId, funcName);
            
            // 显示输入变量
            Variable[] inputs = creator.getInputs();
            if (inputs != null && inputs.length > 0) {
                String inputPrefix = newPrefix + "    ";
                for (int i = 0; i < inputs.length; i++) {
                    boolean isLastInput = (i == inputs.length - 1);
                    displayNode(inputs[i], variableIds, functionIds, inputPrefix, isLastInput);
                }
            }
        }
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
        } else if (data.length <= 5) {
            StringBuilder sb = new StringBuilder("[");
            for (int i = 0; i < data.length; i++) {
                if (i > 0) sb.append(", ");
                sb.append(String.format("%.4f", data[i]));
            }
            sb.append("]");
            return sb.toString();
        } else {
            return String.format("[%.4f, %.4f, ..., %.4f] (%d elements)", 
                data[0], data[1], data[data.length-1], data.length);
        }
    }
}