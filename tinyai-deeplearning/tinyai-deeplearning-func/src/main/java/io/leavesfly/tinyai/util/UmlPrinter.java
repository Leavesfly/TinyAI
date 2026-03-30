package io.leavesfly.tinyai.util;


import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.HashSet;
import java.util.Objects;
import java.util.Set;

/**
 * UML绘图工具类，用于绘制计算图
 *
 * @author leavesfly
 * @version 0.01
 * <p>
 * Uml类提供了将计算图转换为PlantUML格式的功能，
 * 可以可视化展示变量和函数之间的依赖关系。
 */
public class UmlPrinter {

    /**
     * 在线验证：
     *  http://www.plantuml.com/plantuml/uml/SyfFKj2rKt3CoKnELR1Io4ZDoSa70000
     *
     */

    /**
     * 获取变量节点的PlantUML图表示
     *
     * @param variableNode 变量节点
     * @return PlantUML格式的图表示字符串
     */
    public static String getDotGraph(Variable variableNode) {
        StringBuilder declarations = new StringBuilder();
        StringBuilder edges = new StringBuilder();
        Set<Integer> visited = new HashSet<>();
        getDotNode(variableNode, declarations, edges, visited);
        return "@startuml\nleft to right direction\n" + declarations + "\n" + edges + "@enduml";
    }

    /**
     * 递归获取变量节点的PlantUML表示
     *
     * @param variableNode 变量节点
     * @param declarations 节点声明构建器
     * @param edges        边关系构建器
     * @param visited      已访问节点集合（防止重复）
     */
    private static void getDotNode(Variable variableNode, StringBuilder declarations,
                                   StringBuilder edges, Set<Integer> visited) {
        if (Objects.isNull(variableNode)) {
            return;
        }

        int varId = System.identityHashCode(variableNode);
        if (visited.contains(varId)) {
            return;
        }
        visited.add(varId);

        declarations.append(getVarDeclaration(variableNode));

        Function functionNode = variableNode.getCreator();
        if (!Objects.isNull(functionNode)) {
            int funcId = System.identityHashCode(functionNode);
            if (!visited.contains(funcId)) {
                visited.add(funcId);
                declarations.append(getFuncDeclaration(functionNode));
            }

            // 边: 函数 --> 输出变量
            edges.append(String.format("F%d --> V%d\n", funcId, varId));

            // 边: 输入变量 --> 函数
            Variable[] inputs = functionNode.getInputs();
            if (!Objects.isNull(inputs)) {
                for (Variable input : inputs) {
                    int inputId = System.identityHashCode(input);
                    edges.append(String.format("V%d --> F%d\n", inputId, funcId));
                    getDotNode(input, declarations, edges, visited);
                }
            }
        }
    }

    /**
     * 获取变量节点的PlantUML声明
     *
     * @param node 变量节点
     * @return PlantUML格式的节点声明字符串
     */
    private static String getVarDeclaration(Variable node) {
        String label;
        if (node.getName() != null) {
            label = node.getName();
        } else if (node.getCreator() != null) {
            label = "out_" + node.getCreator().getClass().getSimpleName();
        } else {
            label = "var";
        }
        int varId = System.identityHashCode(node);
        String color = "input".equals(label) ? "#LightGreen" : "#Orange";
        return String.format("card \"%s\" as V%d %s\n", label, varId, color);
    }

    /**
     * 获取函数节点的PlantUML声明
     *
     * @param node 函数节点
     * @return PlantUML格式的函数节点声明字符串
     */
    private static String getFuncDeclaration(Function node) {
        int funcId = System.identityHashCode(node);
        String funcName = node.getClass().getSimpleName();
        String label = funcName.equals("MatMul")
                ? "<color:red>" + funcName + "</color>"
                : funcName;
        return String.format("usecase \"%s\" as F%d #LightBlue\n", label, funcId);
    }

    public static void main(String[] args) {
        // 构建一个复杂的计算图示例: 模拟两层神经网络的前向传播
        // loss = relu(x @ w1 + b1) @ w2 + b2 - target 的平方损失
        // 其中 @ 表示矩阵乘法，relu 是激活函数

        // 输入层: 特征 x (2维输入)
        Variable x = new Variable(NdArray.of(new float[]{1.0f, 2.0f}, Shape.of(1, 2)), "x");

        // 第一层权重和偏置
        Variable w1 = new Variable(NdArray.of(new float[]{0.5f, 0.3f, -0.2f, 0.4f}, Shape.of(2, 2)), "w1");
        Variable b1 = new Variable(NdArray.of(new float[]{0.1f, 0.2f}, Shape.of(1, 2)), "b1");

        // 第二层权重和偏置
        Variable w2 = new Variable(NdArray.of(new float[]{0.6f, -0.1f}, Shape.of(2, 1)), "w2");
        Variable b2 = new Variable(NdArray.of(new float[]{0.05f}, Shape.of(1, 1)), "b2");

        // 目标值
        Variable target = new Variable(NdArray.of(1.0f), "target");

        // 第一层: h1 = x @ w1 + b1 (线性变换)
        Variable xw1 = x.matMul(w1);
        Variable h1 = xw1.add(b1);

        // 激活函数: a1 = relu(h1)
        Variable a1 = h1.relu();

        // 第二层: h2 = a1 @ w2 + b2 (线性变换)
        Variable a1w2 = a1.matMul(w2);
        Variable h2 = a1w2.add(b2);

        // 输出层: y = h2 (预测值)
        Variable y = h2;
        y.setName("y_pred");

        // 计算损失: diff = y - target, loss = diff^2 (平方损失)
        Variable diff = y.sub(target);
        Variable loss = diff.squ();
        loss.setName("loss");

        // 输出 PlantUML 格式的计算图
        System.out.println("=== 两层神经网络计算图 PlantUML 表示 ===");
        System.out.println(getDotGraph(loss));
        System.out.println("=== 计算图结构说明 ===");
        System.out.println("输入层: x (2维特征)");
        System.out.println("隐藏层: h1 = x @ w1 + b1, a1 = relu(h1)");
        System.out.println("输出层: h2 = a1 @ w2 + b2, y = h2");
        System.out.println("损失函数: loss = (y - target)^2");
        System.out.println("=== 可复制到 http://www.plantuml.com/plantuml/uml 验证 ===");
    }
}