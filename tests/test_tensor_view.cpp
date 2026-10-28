#include "test_harness.hpp"
#include "kinesis/core/arena_allocator.hpp"
#include "kinesis/core/tensor.hpp"
#include "kinesis/compute/elementwise.hpp"
#include <vector>

using namespace kinesis::core;
using namespace kinesis::compute;

KINESIS_TEST_CASE(TensorView_ShapeAndStrides) {
    Shape shape{2, 3, 4};
    ASSERT_EQ(shape.rank(), 3);
    ASSERT_EQ(shape.total_elements(), 24);

    Strides strides = Strides::from_shape_row_major(shape);
    // (2, 3, 4) in row-major: strides should be (12, 4, 1)
    ASSERT_EQ(strides[0], 12);
    ASSERT_EQ(strides[1], 4);
    ASSERT_EQ(strides[2], 1);
    ASSERT_TRUE(is_c_contiguous(shape, strides));
}

KINESIS_TEST_CASE(TensorView_IndexingAndZeroCopy) {
    std::vector<float> raw(12);
    for (size_t i = 0; i < 12; ++i) {
        raw[i] = static_cast<float>(i);
    }

    // Wrap raw vector into 2D view (3, 4)
    TensorView<float> view(raw.data(), Shape{3, 4});
    ASSERT_EQ(view.size(0), 3);
    ASSERT_EQ(view.size(1), 4);
    ASSERT_EQ(view(0, 0), 0.0f);
    ASSERT_EQ(view(1, 2), 6.0f);
    ASSERT_EQ(view(2, 3), 11.0f);

    // Modify via view, verifies underlying memory is modified
    view(1, 2) = 42.0f;
    ASSERT_EQ(raw[6], 42.0f);
}

KINESIS_TEST_CASE(TensorView_ZeroCopySlicing) {
    std::vector<float> raw(24);
    for (size_t i = 0; i < 24; ++i) raw[i] = static_cast<float>(i);

    // 2D (4, 6)
    TensorView<float> mat(raw.data(), Shape{4, 6});

    // Slice rows [1, 3) -> 2 rows, 6 cols
    TensorView<float> row_slice = mat.slice(0, 1, 3);
    ASSERT_EQ(row_slice.size(0), 2);
    ASSERT_EQ(row_slice.size(1), 6);
    ASSERT_EQ(row_slice(0, 0), 6.0f); // row 1, col 0 in original
    ASSERT_EQ(row_slice(1, 5), 17.0f); // row 2, col 5 in original

    // Sub-slice columns [2, 5) -> 2 rows, 3 cols
    TensorView<float> col_slice = row_slice.slice(1, 2, 5);
    ASSERT_EQ(col_slice.size(0), 2);
    ASSERT_EQ(col_slice.size(1), 3);
    ASSERT_EQ(col_slice(0, 0), 8.0f);  // original (1, 2) = 1*6 + 2 = 8
    ASSERT_EQ(col_slice(1, 2), 16.0f); // original (2, 4) = 2*6 + 4 = 16
}

KINESIS_TEST_CASE(TensorView_ZeroCopyTranspose) {
    std::vector<float> raw = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };
    TensorView<float> mat(raw.data(), Shape{2, 3});
    TensorView<float> transposed = mat.t();

    ASSERT_EQ(transposed.size(0), 3);
    ASSERT_EQ(transposed.size(1), 2);
    ASSERT_EQ(transposed(0, 0), 1.0f);
    ASSERT_EQ(transposed(0, 1), 4.0f);
    ASSERT_EQ(transposed(1, 0), 2.0f);
    ASSERT_EQ(transposed(2, 1), 6.0f);
}

KINESIS_TEST_CASE(TensorView_ZeroCopyBroadcast) {
    std::vector<float> bias = {10.0f, 20.0f, 30.0f};
    // (1, 3)
    TensorView<float> view(bias.data(), Shape{1, 3});

    // Broadcast to (4, 3) without duplicating memory!
    TensorView<float> broadcasted = view.broadcast_to(Shape{4, 3});
    ASSERT_EQ(broadcasted.size(0), 4);
    ASSERT_EQ(broadcasted.size(1), 3);
    ASSERT_EQ(broadcasted.stride(0), 0); // Stride is 0 across broadcast dimension!

    for (int64_t r = 0; r < 4; ++r) {
        ASSERT_EQ(broadcasted(r, 0), 10.0f);
        ASSERT_EQ(broadcasted(r, 1), 20.0f);
        ASSERT_EQ(broadcasted(r, 2), 30.0f);
    }
}

KINESIS_TEST_CASE(Tensor_ArenaIntegrationAndElementwise) {
    ArenaAllocator arena(64 * 1024);

    Tensor<float> a(Shape{2, 3}, 5.0f, arena);
    Tensor<float> b(Shape{2, 3}, 2.0f, arena);
    Tensor<float> out(Shape{2, 3}, 0.0f, arena);

    // Elementwise add
    add(a.view(), b.view(), out.view());
    for (int64_t i = 0; i < 2; ++i) {
        for (int64_t j = 0; j < 3; ++j) {
            ASSERT_EQ(out(i, j), 7.0f);
        }
    }

    // Elementwise multiply
    mul(a.view(), b.view(), out.view());
    for (int64_t i = 0; i < 2; ++i) {
        for (int64_t j = 0; j < 3; ++j) {
            ASSERT_EQ(out(i, j), 10.0f);
        }
    }

    // ReLU
    Tensor<float> x(Shape{4}, arena);
    x(0) = -3.0f; x(1) = 0.0f; x(2) = 2.5f; x(3) = -0.1f;
    Tensor<float> y(Shape{4}, arena);
    relu(x.view(), y.view());
    ASSERT_EQ(y(0), 0.0f);
    ASSERT_EQ(y(1), 0.0f);
    ASSERT_EQ(y(2), 2.5f);
    ASSERT_EQ(y(3), 0.0f);
}

KINESIS_TEST_CASE(TensorView_DescriptorCompressionAndFastEquality) {
    // Verify 32-bit descriptor sizing
    ASSERT_EQ(sizeof(uint32_t), 4);
    // Shape has uint32_t rank + 6 uint32_t dims = 28 bytes
    ASSERT_EQ(sizeof(Shape), 28);

    Shape s1{2, 3, 4};
    Shape s2{2, 3, 4};
    Shape s3{2, 3, 5};
    Shape s4{2, 3};

    // Fast branchless equality
    ASSERT_TRUE(s1 == s2);
    ASSERT_FALSE(s1 == s3);
    ASSERT_FALSE(s1 == s4);
    ASSERT_TRUE(s1 != s3);
    ASSERT_TRUE(s1 != s4);

    // Strides fast equality
    Strides st1 = Strides::from_shape_row_major(s1);
    Strides st2 = Strides::from_shape_row_major(s2);
    Strides st3 = Strides::from_shape_row_major(s3);
    ASSERT_TRUE(st1 == st2);
    ASSERT_FALSE(st1 == st3);
}

KINESIS_TEST_CASE(TensorView_OperatorsAndIdentity) {
    // Test Tensor::eye factory
    auto eye4 = Tensor<float>::eye(4);
    ASSERT_EQ(eye4.rank(), 2);
    ASSERT_EQ(eye4.shape()[0], 4);
    ASSERT_EQ(eye4.shape()[1], 4);
    for (uint32_t i = 0; i < 4; ++i) {
        for (uint32_t j = 0; j < 4; ++j) {
            if (i == j) {
                ASSERT_EQ(eye4(i, j), 1.0f);
            } else {
                ASSERT_EQ(eye4(i, j), 0.0f);
            }
        }
    }

    // Test operator~ (transpose shortcut)
    std::vector<float> data = {1, 2, 3, 4, 5, 6};
    TensorView<float> mat(data.data(), Shape{2, 3});
    auto mat_t = ~mat;
    ASSERT_EQ(mat_t.shape()[0], 3);
    ASSERT_EQ(mat_t.shape()[1], 2);
    ASSERT_EQ(mat_t(0, 1), 4.0f); // original mat(1, 0)
    ASSERT_EQ(mat_t(2, 0), 3.0f); // original mat(0, 2)

    // Test free function transpose
    auto mat_t2 = transpose(mat);
    ASSERT_EQ(mat_t2.shape()[0], 3);
    ASSERT_EQ(mat_t2.shape()[1], 2);
    ASSERT_EQ(mat_t2(1, 1), 5.0f);

    // Test make_identity
    std::vector<float> buf(9, 0.0f);
    TensorView<float> square_view(buf.data(), Shape{3, 3});
    square_view.make_identity();
    ASSERT_EQ(square_view(0, 0), 1.0f);
    ASSERT_EQ(square_view(1, 1), 1.0f);
    ASSERT_EQ(square_view(2, 2), 1.0f);
    ASSERT_EQ(square_view(0, 1), 0.0f);
}

KINESIS_TEST_CASE(TensorView_LoopFissionBroadcastMultiDim) {
    // 1D tensor of shape {3}
    std::vector<float> data = {10.0f, 20.0f, 30.0f};
    TensorView<float> v1(data.data(), Shape{3});

    // Broadcast across multiple leading dimensions: {2, 4, 3}
    // rank_diff = 3 - 1 = 2 leading dims
    Shape target_shape{2, 4, 3};
    TensorView<float> bcast = v1.broadcast_to(target_shape);
    ASSERT_EQ(bcast.rank(), 3);
    ASSERT_EQ(bcast.size(0), 2);
    ASSERT_EQ(bcast.size(1), 4);
    ASSERT_EQ(bcast.size(2), 3);

    // Verify leading strides are 0
    ASSERT_EQ(bcast.stride(0), 0);
    ASSERT_EQ(bcast.stride(1), 0);
    ASSERT_EQ(bcast.stride(2), 1);

    // Verify all 24 elements read the broadcasted row values correctly
    for (int64_t i = 0; i < 2; ++i) {
        for (int64_t j = 0; j < 4; ++j) {
            ASSERT_EQ(bcast(i, j, 0), 10.0f);
            ASSERT_EQ(bcast(i, j, 1), 20.0f);
            ASSERT_EQ(bcast(i, j, 2), 30.0f);
        }
    }
}

