package io.leavesfly.tinyai.nnet.core;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.Shape;

/**
 * V2版本的延迟初始化模块基类
 * <p>
 * LazyModule支持根据实际输入动态推断参数形状，
 * 在首次前向传播时触发参数初始化。
 * <p>
 * 使用场景：
 * - 输入维度未知的网络构建
 * - 动态网络架构
 * - 简化网络定义代码
 *
 * @author leavesfly
 * @version 2.0
 */
public abstract class LazyModule extends Module {

    /**
     * 初始化锁，保证多线程环境下延迟初始化的线程安全
     */
    private final Object initLock = new Object();

    /**
     * 构造函数
     *
     * @param name 模块名称
     */
    public LazyModule(String name) {
        super(name);
    }

    /**
     * 默认构造函数
     */
    public LazyModule() {
        super();
    }

    /**
     * 根据输入形状初始化参数
     * <p>
     * 子类必须实现此方法，根据输入形状创建并注册参数
     *
     * @param inputShapes 输入张量的形状数组
     */
    protected abstract void initialize(Shape... inputShapes);

    /**
     * 检查并触发延迟初始化（线程安全）
     * <p>
     * 在forward方法开头调用，如果参数未初始化则触发初始化。
     * 使用双重检查锁定保证多线程环境下只初始化一次。
     *
     * @param inputs 输入变量数组
     */
    protected void checkLazyInitialization(Variable... inputs) {
        if (_initialized) {
            return;  // 快速路径：已初始化，直接返回
        }

        synchronized (initLock) {
            if (_initialized) {
                return;  // 双重检查：另一个线程可能已完成初始化
            }

            // 提取输入形状并校验
            Shape[] inputShapes = new Shape[inputs.length];
            for (int i = 0; i < inputs.length; i++) {
                if (inputs[i] == null) {
                    throw new IllegalArgumentException(
                            "Input at index " + i + " is null during lazy initialization of module: " + name);
                }
                inputShapes[i] = inputs[i].getShape();
            }

            // 调用子类实现的initialize方法
            try {
                initialize(inputShapes);
                _initialized = true;

                // 调用resetParameters初始化参数值
                resetParameters();
            } catch (Exception e) {
                throw new RuntimeException("Failed to initialize lazy module: " + name, e);
            }
        }
    }

    /**
     * 判断是否已完成初始化
     *
     * @return true表示尚未初始化
     */
    public boolean hasUnInitializedParams() {
        return !_initialized;
    }

    /**
     * 重写init方法，延迟模块不在此处初始化
     */
    @Override
    public void init() {
        // LazyModule在首次forward时初始化，这里不做任何操作
    }
}
