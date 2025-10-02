package io.leavesfly.tinyai.ml.inference;

import io.leavesfly.tinyai.ndarr.NdArray;

/**
 * 数据转换器接口
 * 
 * 该接口定义了输入数据和输出数据与NdArray之间的转换方法，
 * 用于模型推理时的数据预处理和后处理。
 * 
 * @param <I> 输入数据类型
 * @param <O> 输出数据类型
 * 
 * @author TinyDL
 * @version 1.0
 */
public interface Translator<I, O> {

    /**
     * 将输入数据转换为NdArray
     * @param input 输入数据
     * @return NdArray表示
     */
    NdArray input2NdArray(I input);

    /**
     * 将NdArray转换为输出数据
     * @param ndArray NdArray对象
     * @return 输出数据
     */
    O ndArray2Output(NdArray ndArray);

}