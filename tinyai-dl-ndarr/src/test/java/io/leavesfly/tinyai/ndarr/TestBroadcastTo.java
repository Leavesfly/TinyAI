package io.leavesfly.tinyai.ndarr;


public class TestBroadcastTo {
    public static void main(String[] args) {
        try {
            // 创建一个1D数组
            NdArray array1d = NdArray.of(new float[]{1.0f, 2.0f, 3.0f});
            System.out.println("1D数组: " + array1d);
            System.out.println("1D数组形状: " + array1d.getShape());

            // 尝试广播到2D形状
            NdArray broadcasted2d = array1d.broadcastTo(Shape.of(2, 3));
            System.out.println("广播到2D: " + broadcasted2d);
            System.out.println("广播后形状: " + broadcasted2d.getShape());

            // 创建一个2D数组
            float[][] data2d = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
            NdArray array2d = NdArray.of(data2d);
            System.out.println("2D数组: " + array2d);
            System.out.println("2D数组形状: " + array2d.getShape());

            // 尝试广播到3D形状
            NdArray broadcasted3d = array2d.broadcastTo(Shape.of(2, 2, 3));
            System.out.println("广播到3D: " + broadcasted3d);
            System.out.println("广播后形状: " + broadcasted3d.getShape());

        } catch (Exception e) {
            System.out.println("出错: " + e.getMessage());
            e.printStackTrace();
        }
    }
}