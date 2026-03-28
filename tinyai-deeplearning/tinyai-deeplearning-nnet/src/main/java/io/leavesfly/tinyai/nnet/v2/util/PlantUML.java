package io.leavesfly.tinyai.nnet.v2.util;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.core.Parameter;
import io.leavesfly.tinyai.nnet.v2.layer.transformer.MultiHeadAttention;

import java.util.*;

/**
 * 增强版 PlantUML 计算图可视化工具
 * <p>
 * 在 UmlPrinter 的基础上增强，支持在一张图中同时展示：
 * <ul>
 *   <li>算子级计算图：Variable → Function → Variable 的数据流</li>
 *   <li>Module 维度信息：当 Function 是 Module 时，用 package 嵌套展示模块层级、参数、子模块</li>
 * </ul>
 * <p>
 * 生成的 PlantUML 可在 <a href="http://www.plantuml.com/plantuml/uml">PlantUML Online</a> 渲染。
 *
 * @author leavesfly
 * @version 2.0
 */
public class PlantUML {

    /**
     * 生成包含 Module 维度信息的增强计算图
     * <p>
     * 由于 Module.forward() 内部调用子模块时通常使用 forward() 而非 call()，
     * 计算图中的 creator 是底层算子而非 Module 本身。因此需要显式传入 Module，
     * 将 Module 层级结构与算子计算图融合在一张图中展示。
     *
     * @param outputVariable 计算图的输出变量（通常是 loss 或模型输出）
     * @param rootModule     根模块（如 MultiHeadAttention、Sequential 等）
     * @return PlantUML 格式的字符串
     */
    public static String generateGraph(Variable outputVariable, Module rootModule) {
        StringBuilder declarations = new StringBuilder();
        StringBuilder edges = new StringBuilder();
        Set<Integer> visited = new HashSet<>();

        // 收集计算图中的所有节点和边
        collectNodes(outputVariable, declarations, edges, visited, null);

        // 生成 Module 层级结构
        StringBuilder modulePackages = new StringBuilder();
        Set<Integer> visitedModules = new HashSet<>();
        generateModulePackage(rootModule, modulePackages, visitedModules);

        StringBuilder result = new StringBuilder();
        result.append("@startuml\n");
        result.append("left to right direction\n");
        result.append("skinparam packageStyle rectangle\n");
        result.append("skinparam packageBorderColor #4A90D9\n");
        result.append("skinparam packageBackgroundColor #F0F8FF\n\n");

        // Module 层级结构
        result.append("' === Module 层级结构 ===\n");
        result.append(modulePackages);
        result.append("\n");

        // 计算图节点和边
        result.append("' === 计算图节点 ===\n");
        result.append(declarations);
        result.append("\n' === 计算图数据流 ===\n");
        result.append(edges);

        result.append("@enduml");
        return result.toString();
    }


    /**
     * 生成基础计算图（与 UmlPrinter 兼容的简单模式）
     *
     * @param outputVariable 计算图的输出变量
     * @return PlantUML 格式的字符串
     */
    public static String generateSimpleGraph(Variable outputVariable) {
        StringBuilder declarations = new StringBuilder();
        StringBuilder edges = new StringBuilder();
        Set<Integer> visited = new HashSet<>();
        collectNodes(outputVariable, declarations, edges, visited, null);
        return "@startuml\nleft to right direction\n" + declarations + "\n" + edges + "@enduml";
    }

    /**
     * 生成 Module 结构图（仅展示模块层级，不含计算图）
     *
     * @param module 根模块
     * @return PlantUML 格式的字符串
     */
    public static String generateModuleGraph(Module module) {
        StringBuilder content = new StringBuilder();
        content.append("@startuml\n");
        content.append("top to bottom direction\n");
        content.append("skinparam packageStyle rectangle\n");
        content.append("skinparam packageBorderColor #4A90D9\n");
        content.append("skinparam packageBackgroundColor #F0F8FF\n\n");

        Set<Integer> visitedModules = new HashSet<>();
        generateModulePackage(module, content, visitedModules);

        content.append("\n@enduml");
        return content.toString();
    }

    // ==================== 计算图遍历 ====================

    /**
     * 递归收集计算图中的节点声明和边关系
     *
     * @param variableNode     当前变量节点
     * @param declarations     节点声明构建器
     * @param edges            边关系构建器
     * @param visited          已访问节点集合
     * @param functionToModule Module 收集器（可为 null 表示不收集）
     */
    private static void collectNodes(Variable variableNode, StringBuilder declarations,
                                     StringBuilder edges, Set<Integer> visited,
                                     Map<Integer, Module> functionToModule) {
        if (variableNode == null) {
            return;
        }

        int varId = System.identityHashCode(variableNode);
        if (visited.contains(varId)) {
            return;
        }
        visited.add(varId);

        // 生成变量节点声明
        declarations.append(buildVariableDeclaration(variableNode));

        Function functionNode = variableNode.getCreator();
        if (functionNode == null) {
            return;
        }

        int funcId = System.identityHashCode(functionNode);
        if (!visited.contains(funcId)) {
            visited.add(funcId);
            declarations.append(buildFunctionDeclaration(functionNode));

            // 如果 Function 是 Module，记录映射关系
            if (functionToModule != null && functionNode instanceof Module) {
                functionToModule.put(funcId, (Module) functionNode);
            }
        }

        // 边: Function --> 输出 Variable
        edges.append(String.format("F%d --> V%d\n", funcId, varId));

        // 边: 输入 Variable --> Function
        Variable[] inputs = functionNode.getInputs();
        if (inputs != null) {
            for (Variable input : inputs) {
                int inputId = System.identityHashCode(input);
                edges.append(String.format("V%d --> F%d\n", inputId, funcId));
                collectNodes(input, declarations, edges, visited, functionToModule);
            }
        }
    }

    // ==================== 节点声明生成 ====================

    /**
     * 生成变量节点的 PlantUML 声明
     * <p>
     * 包含变量名称和 Shape 信息。Parameter 类型使用绿色标识。
     */
    private static String buildVariableDeclaration(Variable node) {
        int varId = System.identityHashCode(node);
        String label = buildVariableLabel(node);
        String color;
        if (node instanceof Parameter) {
            color = "#90EE90";
        } else if ("input".equals(node.getName())) {
            color = "#DDA0DD";
        } else {
            color = "#FFA500";
        }
        return String.format("card \"%s\" as V%d %s\n", label, varId, color);
    }

    /**
     * 构建变量节点的显示标签
     */
    private static String buildVariableLabel(Variable node) {
        StringBuilder label = new StringBuilder();

        // 名称
        if (node.getName() != null) {
            label.append(node.getName());
        } else if (node.getCreator() != null) {
            label.append("out_").append(node.getCreator().getClass().getSimpleName());
        } else {
            label.append("var");
        }

        // Shape 信息
        if (node.getValue() != null) {
            Shape shape = node.getValue().getShape();
            if (shape != null) {
                label.append("\\n").append(formatShape(shape));
            }
        }

        // Parameter 额外信息
        if (node instanceof Parameter) {
            Parameter param = (Parameter) node;
            label.append("\\n[param");
            label.append(param.requiresGrad() ? ", grad" : ", frozen");
            label.append("]");
        }

        return label.toString();
    }

    /**
     * 生成函数节点的 PlantUML 声明
     * <p>
     * 普通 Function 使用浅蓝色椭圆形，Module 类型使用深蓝色并附带额外信息。
     */
    private static String buildFunctionDeclaration(Function node) {
        int funcId = System.identityHashCode(node);

        if (node instanceof Module) {
            return buildModuleFunctionDeclaration((Module) node, funcId);
        }

        String funcName = node.getClass().getSimpleName();
        // MatMul 算子使用红色字体突出显示
        if ("MatMul".equals(funcName)) {
            return String.format("usecase \"<color:red>%s</color>\" as F%d #LightBlue\n", funcName, funcId);
        }
        return String.format("usecase \"%s\" as F%d #LightBlue\n", funcName, funcId);
    }

    /**
     * 生成 Module 类型 Function 节点的声明
     * <p>
     * 展示模块类型、名称、参数数量等信息。
     */
    private static String buildModuleFunctionDeclaration(Module module, int funcId) {
        String moduleType = module.getClass().getSimpleName();
        boolean isMatMul = "MatMul".equals(moduleType);

        StringBuilder label = new StringBuilder();

        // 模块类型（MatMul 标红）
        if (isMatMul) {
            label.append("<color:red>").append(moduleType).append("</color>");
        } else {
            label.append(moduleType);
        }

        // 模块名称
        if (module.getName() != null && !module.getName().isEmpty()) {
            label.append("\\n'").append(module.getName()).append("'");
        }

        // 参数数量
        long paramCount = module.numParameters();
        if (paramCount > 0) {
            label.append("\\n").append(formatNumber(paramCount)).append(" params");
        }

        // 训练/推理模式
        label.append("\\n[").append(module.isTraining() ? "train" : "eval").append("]");

        return String.format("usecase \"%s\" as F%d #87CEEB\n", label, funcId);
    }

    // ==================== Module 层级结构 ====================

    /**
     * 递归生成 Module 的 package 嵌套结构
     * <p>
     * 展示模块层级、参数列表、缓冲区、子模块等信息。
     */
    private static void generateModulePackage(Module module, StringBuilder content,
                                              Set<Integer> visitedModules) {
        int moduleId = System.identityHashCode(module);
        if (visitedModules.contains(moduleId)) {
            return;
        }
        visitedModules.add(moduleId);

        String moduleName = module.getName() != null ? module.getName() : "unnamed";
        String moduleType = module.getClass().getSimpleName();

        // 打开 package
        content.append(String.format("package \"%s : %s\" as M%d #F0F8FF {\n",
                moduleName, moduleType, moduleId));

        // 参数信息（使用 note 语法，PlantUML card 不支持花括号子内容）
        Map<String, Parameter> directParams = module.namedParameters("", false);
        if (!directParams.isEmpty()) {
            StringBuilder paramLabel = new StringBuilder();
            for (Map.Entry<String, Parameter> entry : directParams.entrySet()) {
                Parameter param = entry.getValue();
                if (param != null && param.data() != null) {
                    String shapeStr = formatShape(param.data().getShape());
                    String gradStr = param.requiresGrad() ? "grad" : "frozen";
                    if (paramLabel.length() > 0) {
                        paramLabel.append("\\n");
                    }
                    paramLabel.append(entry.getKey()).append(": ")
                            .append(shapeStr).append(" [").append(gradStr).append("]");
                }
            }
            content.append(String.format("  card \"%s\" as M%d_params #FFFACD\n",
                    paramLabel, moduleId));
        }

        // 缓冲区信息
        Map<String, NdArray> directBuffers = module.namedBuffers("", false);
        if (!directBuffers.isEmpty()) {
            StringBuilder bufferLabel = new StringBuilder();
            for (Map.Entry<String, NdArray> entry : directBuffers.entrySet()) {
                NdArray buffer = entry.getValue();
                if (buffer != null) {
                    if (bufferLabel.length() > 0) {
                        bufferLabel.append("\\n");
                    }
                    bufferLabel.append(entry.getKey()).append(": ")
                            .append(formatShape(buffer.getShape()));
                }
            }
            content.append(String.format("  card \"%s\" as M%d_buffers #E0E0E0\n",
                    bufferLabel, moduleId));
        }

        // 递归处理子模块，并收集子模块 ID 用于生成垂直排列的隐藏连接
        List<String> childIds = new ArrayList<>();
        Map<String, Module> childModules = module.namedModules("", false);
        for (Map.Entry<String, Module> entry : childModules.entrySet()) {
            if (entry.getValue() != module) {
                int childId = System.identityHashCode(entry.getValue());
                childIds.add("M" + childId);
                generateModulePackage(entry.getValue(), content, visitedModules);
            }
        }

        // 关闭 package
        content.append("}\n");

        // 添加隐藏连接，强制子模块上下排列
        for (int i = 0; i < childIds.size() - 1; i++) {
            content.append(String.format("%s -[hidden]down-> %s\n", childIds.get(i), childIds.get(i + 1)));
        }
    }

    // ==================== 格式化工具方法 ====================

    /**
     * 格式化 Shape 为可读字符串
     */
    private static String formatShape(Shape shape) {
        if (shape == null) {
            return "?";
        }
        int[] dims = shape.getShapeDims();
        if (dims == null || dims.length == 0) {
            return "scalar";
        }
        StringBuilder sb = new StringBuilder("(");
        for (int i = 0; i < dims.length; i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(dims[i]);
        }
        sb.append(")");
        return sb.toString();
    }

    /**
     * 格式化数字为可读字符串（带千分位）
     */
    private static String formatNumber(long number) {
        if (number >= 1_000_000) {
            return String.format("%.2fM", number / 1_000_000.0);
        } else if (number >= 1_000) {
            return String.format("%.1fK", number / 1_000.0);
        }
        return String.valueOf(number);
    }

    public static void main(String[] args) {
        // 构建 MultiHeadAttention 自注意力计算图 demo
        // 参数: d_model=8, num_heads=2, 输入序列 batch=1, seq_len=3
        MultiHeadAttention mha = new MultiHeadAttention("self_attn", 8, 2, 0.0f);
        mha.eval(); // 推理模式，避免 dropout 随机性

        // 创建输入: (batch=1, seq_len=3, d_model=8)
        float[] inputData = new float[1 * 3 * 8];
        for (int i = 0; i < inputData.length; i++) {
            inputData[i] = (float) (Math.random() * 0.5);
        }
        Variable input = new Variable(NdArray.of(inputData, Shape.of(1, 3, 8)), "input");

        // 生成因果掩码 (seq_len=3)
        Variable causalMask = MultiHeadAttention.generateCausalMaskBatched(3);
        causalMask.setName("causal_mask");

        // 前向传播: 自注意力 (Q=K=V=input)
        Variable output = mha.forward(input, causalMask);
        output.setName("attn_output");

        // === 1. 增强计算图（Module 维度 + 算子计算图） ===
        System.out.println("=== 增强计算图（含 Module 维度信息） ===");
        System.out.println(PlantUML.generateGraph(output, mha));

        System.out.println();

//        // === 2. Module 结构图（仅模块层级） ===
//        System.out.println("=== Module 结构图（仅模块层级） ===");
//        System.out.println(PlantUML.generateModuleGraph(mha));

        System.out.println();
        System.out.println("=== 模型参数摘要 ===");
        System.out.println(mha.parameterSummary());
        System.out.println("=== 可复制到 http://www.plantuml.com/plantuml/uml 渲染查看 ===");
    }
}
