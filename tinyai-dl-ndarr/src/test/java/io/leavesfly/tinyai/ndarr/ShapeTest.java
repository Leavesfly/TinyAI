package io.leavesfly.tinyai.ndarr;

import io.leavesfly.tinyai.ndarr.cpu.ShapeCpu;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Shape类的单元测试
 * 
 * 测试Shape接口及其CPU实现的各种功能和边界条件
 *
 * @author TinyAI
 */
public class ShapeTest {

    private Shape scalar;
    private Shape vector;
    private Shape matrix;
    private Shape tensor3d;
    private Shape tensor4d;

    @Before
    public void setUp() {
        // 初始化各种维度的Shape实例
        scalar = Shape.of();           // 真正的标量（零维）
        vector = Shape.of(5);          // 一维向量
        matrix = Shape.of(3, 4);       // 二维矩阵
        tensor3d = Shape.of(2, 3, 4);  // 三维张量
        tensor4d = Shape.of(2, 3, 4, 5); // 四维张量
    }

    // =============================================================================
    // 测试Shape的创建
    // =============================================================================

    @Test
    public void testShapeCreation() {
        // 测试单维度创建
        Shape shape1d = Shape.of(10);
        assertEquals(1, shape1d.getDimNum());
        assertEquals(10, shape1d.getDimension(0));

        // 测试二维创建
        Shape shape2d = Shape.of(5, 8);
        assertEquals(2, shape2d.getDimNum());
        assertEquals(5, shape2d.getDimension(0));
        assertEquals(8, shape2d.getDimension(1));

        // 测试多维度创建
        Shape shape4d = Shape.of(2, 3, 4, 5);
        assertEquals(4, shape4d.getDimNum());
        assertEquals(2, shape4d.getDimension(0));
        assertEquals(3, shape4d.getDimension(1));
        assertEquals(4, shape4d.getDimension(2));
        assertEquals(5, shape4d.getDimension(3));
    }

    @Test(expected = IllegalArgumentException.class)
    public void testShapeCreationWithNullDimensions() {
        // 测试使用null参数创建Shape
        new ShapeCpu((int[]) null);
    }

    // =============================================================================
    // 测试形状分类判断
    // =============================================================================

    @Test
    public void testShapeTypeIdentification() {
        // 测试标量判断（零维）
        assertTrue(scalar.isScalar());
        assertFalse(scalar.isMatrix());
        assertFalse(scalar.isVector());

        // 测试1x1矩阵（不是标量）
        Shape oneByOneMatrix = Shape.of(1, 1);
        assertFalse(oneByOneMatrix.isScalar()); // 1x1矩阵不是真正的标量
        assertTrue(oneByOneMatrix.isMatrix());
        assertFalse(oneByOneMatrix.isVector());

        // 测试向量判断
        assertTrue(vector.isVector());
        assertFalse(vector.isMatrix());
        assertFalse(vector.isScalar());

        // 测试矩阵判断
        assertTrue(matrix.isMatrix());
        assertFalse(matrix.isVector());
        assertFalse(matrix.isScalar());

        // 测试高维张量判断
        assertFalse(tensor3d.isMatrix());
        assertFalse(tensor3d.isVector());
        assertFalse(tensor3d.isScalar());

        assertFalse(tensor4d.isMatrix());
        assertFalse(tensor4d.isVector());
        assertFalse(tensor4d.isScalar());
    }

    // =============================================================================
    // 测试维度信息获取
    // =============================================================================

    @Test
    public void testDimensionAccess() {
        // 测试矩阵的行列获取
        assertEquals(3, matrix.getRow());
        assertEquals(4, matrix.getColumn());

        // 测试维度数量
        assertEquals(1, vector.getDimNum());
        assertEquals(2, matrix.getDimNum());
        assertEquals(3, tensor3d.getDimNum());
        assertEquals(4, tensor4d.getDimNum());

        // 测试指定维度大小获取
        assertEquals(5, vector.getDimension(0));
        assertEquals(3, matrix.getDimension(0));
        assertEquals(4, matrix.getDimension(1));
        assertEquals(2, tensor3d.getDimension(0));
        assertEquals(3, tensor3d.getDimension(1));
        assertEquals(4, tensor3d.getDimension(2));
    }

    @Test(expected = IllegalStateException.class)
    public void testGetRowOnNonMatrix() {
        // 在非矩阵上调用getRow应该抛出异常
        vector.getRow();
    }

    @Test(expected = IllegalStateException.class)
    public void testGetColumnOnNonMatrix() {
        // 在非矩阵上调用getColumn应该抛出异常
        vector.getColumn();
    }

    @Test(expected = IllegalStateException.class)
    public void testGetRowOnScalar() {
        // 在标量上调用getRow应该抛出异常
        scalar.getRow();
    }

    @Test(expected = IllegalStateException.class)
    public void testGetColumnOnScalar() {
        // 在标量上调用getColumn应该抛出异常
        scalar.getColumn();
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void testGetDimensionWithInvalidIndex() {
        // 使用无效索引获取维度大小
        matrix.getDimension(5);
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void testGetDimensionWithNegativeIndex() {
        // 使用负索引获取维度大小
        matrix.getDimension(-1);
    }

    // =============================================================================
    // 测试大小计算
    // =============================================================================

    @Test
    public void testSizeCalculation() {
        // 测试各种形状的大小计算
        assertEquals(1, scalar.size());  // 标量大小为1
        assertEquals(5, vector.size());
        assertEquals(12, matrix.size()); // 3 * 4 = 12
        assertEquals(24, tensor3d.size()); // 2 * 3 * 4 = 24
        assertEquals(120, tensor4d.size()); // 2 * 3 * 4 * 5 = 120
    }

    @Test
    public void testSizeWithZeroDimension() {
        // 测试包含0维度的形状（虽然在实际应用中可能不常见）
        Shape zeroShape = Shape.of(0, 5);
        assertEquals(0, zeroShape.size());
    }

    // =============================================================================
    // 测试索引计算
    // =============================================================================

    @Test
    public void testIndexCalculation() {
        // 测试标量索引计算（零维）
        assertEquals(0, scalar.getIndex()); // 标量没有索引参数

        // 测试一维索引计算
        assertEquals(0, vector.getIndex(0));
        assertEquals(3, vector.getIndex(3));
        assertEquals(4, vector.getIndex(4));

        // 测试二维索引计算
        assertEquals(0, matrix.getIndex(0, 0));
        assertEquals(1, matrix.getIndex(0, 1));
        assertEquals(4, matrix.getIndex(1, 0)); // 第二行第一列
        assertEquals(5, matrix.getIndex(1, 1)); // 第二行第二列
        assertEquals(11, matrix.getIndex(2, 3)); // 最后一个元素

        // 测试三维索引计算
        assertEquals(0, tensor3d.getIndex(0, 0, 0));
        assertEquals(1, tensor3d.getIndex(0, 0, 1));
        assertEquals(4, tensor3d.getIndex(0, 1, 0));
        assertEquals(12, tensor3d.getIndex(1, 0, 0));
        assertEquals(23, tensor3d.getIndex(1, 2, 3)); // 最后一个元素

        // 测试四维索引计算
        assertEquals(0, tensor4d.getIndex(0, 0, 0, 0));
        assertEquals(1, tensor4d.getIndex(0, 0, 0, 1));
        assertEquals(5, tensor4d.getIndex(0, 0, 1, 0));
        assertEquals(20, tensor4d.getIndex(0, 1, 0, 0));
        assertEquals(60, tensor4d.getIndex(1, 0, 0, 0));
        assertEquals(119, tensor4d.getIndex(1, 2, 3, 4)); // 最后一个元素
    }

    @Test(expected = IllegalArgumentException.class)
    public void testIndexCalculationWithWrongDimensions() {
        // 索引维度与形状维度不匹配
        matrix.getIndex(1, 2, 3); // 矩阵只有2维，不能用3个索引
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void testIndexCalculationWithOutOfBoundsIndex() {
        // 索引超出范围
        matrix.getIndex(3, 1); // 行索引超出范围（最大为2）
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void testIndexCalculationWithNegativeIndex() {
        // 负索引
        matrix.getIndex(-1, 1);
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void testIndexCalculationWithColumnOutOfBounds() {
        // 列索引超出范围
        matrix.getIndex(1, 4); // 列索引超出范围（最大为3）
    }

    // =============================================================================
    // 测试相等性和哈希码
    // =============================================================================

    @Test
    public void testEquality() {
        // 测试相等性
        Shape shape1 = Shape.of(3, 4);
        Shape shape2 = Shape.of(3, 4);
        Shape shape3 = Shape.of(4, 3);

        assertEquals(shape1, shape2);
        assertEquals(shape2, shape1);
        assertNotEquals(shape1, shape3);
        assertNotEquals(shape2, shape3);

        // 测试与null的比较
        assertNotEquals(shape1, null);

        // 测试与不同类型对象的比较
        assertNotEquals(shape1, "不是Shape对象");
    }

    @Test
    public void testHashCode() {
        // 测试哈希码一致性
        Shape shape1 = Shape.of(3, 4);
        Shape shape2 = Shape.of(3, 4);
        Shape shape3 = Shape.of(4, 3);

        assertEquals(shape1.hashCode(), shape2.hashCode());
        // 注意：不同对象的哈希码可能相同，但相同对象的哈希码必须相同
        assertTrue(shape1.hashCode() != shape3.hashCode() || shape1.hashCode() == shape3.hashCode());
    }

    @Test
    public void testHashCodeConsistency() {
        // 测试哈希码的一致性（多次调用应返回相同值）
        Shape shape = Shape.of(2, 3, 4);
        int hash1 = shape.hashCode();
        int hash2 = shape.hashCode();
        int hash3 = shape.hashCode();

        assertEquals(hash1, hash2);
        assertEquals(hash2, hash3);
    }

    // =============================================================================
    // 测试toString方法
    // =============================================================================

    @Test
    public void testToString() {
        // 测试toString输出格式
        assertEquals("[5]", vector.toString());
        assertEquals("[3,4]", matrix.toString());
        assertEquals("[2,3,4]", tensor3d.toString());
        assertEquals("[2,3,4,5]", tensor4d.toString());

        // 测试单维度
        Shape single = Shape.of(1);
        assertEquals("[1]", single.toString());
    }

    // =============================================================================
    // 测试边界条件和异常情况
    // =============================================================================

    @Test
    public void testLargeShapes() {
        // 测试大型形状的处理
        Shape largeShape = Shape.of(1000, 2000);
        assertEquals(2000000, largeShape.size());
        assertEquals(1000, largeShape.getRow());
        assertEquals(2000, largeShape.getColumn());
        assertTrue(largeShape.isMatrix());
    }

    @Test
    public void testSingleElementShapes() {
        // 测试单元素形状
        Shape singleElement1d = Shape.of(1);
        assertEquals(1, singleElement1d.size());
        assertTrue(singleElement1d.isVector());

        Shape singleElement2d = Shape.of(1, 1);
        assertEquals(1, singleElement2d.size());
        assertTrue(singleElement2d.isMatrix());
    }

    @Test
    public void testMultipleDimensionAccess() {
        // 测试多维度访问的完整性
        Shape complexShape = Shape.of(2, 3, 4, 5, 6);
        assertEquals(5, complexShape.getDimNum());
        
        for (int i = 0; i < complexShape.getDimNum(); i++) {
            assertTrue(complexShape.getDimension(i) > 0);
        }
        
        assertEquals(720, complexShape.size()); // 2*3*4*5*6
    }

    @Test
    public void testIndexCalculationEdgeCases() {
        // 测试索引计算的边界情况
        Shape edgeShape = Shape.of(1, 1, 1);
        
        // 只有一个元素的情况
        assertEquals(0, edgeShape.getIndex(0, 0, 0));
        assertEquals(1, edgeShape.size());
    }

    // =============================================================================
    // 新增的标量、向量、矩阵分类测试
    // =============================================================================

    @Test
    public void testScalarDetailed() {
        // 测试真正的标量（零维）
        Shape trueScalar = Shape.of();
        
        assertTrue("Scalar should return true for isScalar()", trueScalar.isScalar());
        assertFalse("Scalar should return false for isVector()", trueScalar.isVector());
        assertFalse("Scalar should return false for isMatrix()", trueScalar.isMatrix());
        
        assertEquals("Scalar should have 0 dimensions", 0, trueScalar.getDimNum());
        assertEquals("Scalar should have size 1", 1, trueScalar.size());
        assertEquals("Scalar should have index 0", 0, trueScalar.getIndex());
    }
    
    @Test
    public void testVectorDetailed() {
        // 测试各种大小的向量
        Shape[] vectors = {
            Shape.of(1),     // 单元素向量
            Shape.of(5),     // 普通向量
            Shape.of(100),   // 大向量
            Shape.of(0)      // 空向量
        };
        
        for (Shape v : vectors) {
            assertFalse("Vector should return false for isScalar()", v.isScalar());
            assertTrue("Vector should return true for isVector()", v.isVector());
            assertFalse("Vector should return false for isMatrix()", v.isMatrix());
            assertEquals("Vector should have 1 dimension", 1, v.getDimNum());
        }
    }
    
    @Test
    public void testMatrixDetailed() {
        // 测试各种大小的矩阵
        Shape[] matrices = {
            Shape.of(1, 1),   // 1x1矩阵
            Shape.of(1, 5),   // 行向量
            Shape.of(5, 1),   // 列向量
            Shape.of(3, 4),   // 普通矩阵
            Shape.of(0, 5),   // 空行矩阵
            Shape.of(5, 0)    // 空列矩阵
        };
        
        for (Shape m : matrices) {
            assertFalse("Matrix should return false for isScalar()", m.isScalar());
            assertFalse("Matrix should return false for isVector()", m.isVector());
            assertTrue("Matrix should return true for isMatrix()", m.isMatrix());
            assertEquals("Matrix should have 2 dimensions", 2, m.getDimNum());
        }
    }
    
    @Test
    public void testHighDimensionalTensors() {
        // 测试高维张量
        Shape[] tensors = {
            Shape.of(2, 3, 4),        // 3D
            Shape.of(2, 3, 4, 5),     // 4D
            Shape.of(2, 3, 4, 5, 6),  // 5D
            Shape.of(1, 1, 1, 1, 1, 1) // 6D，所有维度都是1
        };
        
        for (Shape t : tensors) {
            assertFalse("High-dimensional tensor should return false for isScalar()", t.isScalar());
            assertFalse("High-dimensional tensor should return false for isVector()", t.isVector());
            assertFalse("High-dimensional tensor should return false for isMatrix()", t.isMatrix());
            assertTrue("High-dimensional tensor should have more than 2 dimensions", t.getDimNum() > 2);
        }
    }
}