import paddle
from paddle.fluid.framework import _test_eager_guard

import numpy as np
import time
import torch

#data product by test_conv.py
indices = np.load("indices.npy") 
values = np.load("features.npy") 
kernels = np.load("kernels.npy")
out_indices = np.load("out_indices.npy")
out_values = np.load("out_feature.npy")
feature_grad = np.load("features_grad.npy")
kernel_grad = np.load("kernel_grad.npy")
#print(feature_grad)
#print(kernel_grad[0])

subm_indices = np.load("subm_indices.npy") 
subm_values = np.load("subm_features.npy") 
subm_kernels = np.load("subm_kernel.npy")
subm_out_indices = np.load("subm_out_indices.npy")
subm_out_values = np.load("subm_out_features.npy")
subm_feature_grad = np.load("subm_features_grad.npy")
subm_kernel_grad = np.load("subm_kernels_grad.npy")

#print(subm_indices)
#print(subm_values)
#print(subm_out_indices)
#print(subm_out_values)

with _test_eager_guard():
    # test sparse conv3d
    if False:
        sparse_x = paddle.sparse.sparse_coo_tensor(paddle.to_tensor(indices, dtype='int32'),
                paddle.to_tensor(values, dtype='float32'), shape = [2, 10, 400, 152, 32],
                stop_gradient=False)
        kernels_tensor = paddle.to_tensor(kernels, dtype='float32', stop_gradient=False)

        ###gpu warm up##
        out1 = paddle.sparse.functional.conv3d(sparse_x, kernels_tensor, None, [1, 1, 1],
                [0, 0, 0], [1, 1, 1], 1, "NDHWC")
        paddle.device.cuda.synchronize()
        ###gpu warm up##

        t = time.time()
        out = paddle.sparse.functional.conv3d(sparse_x, kernels_tensor, None, [1, 1, 1],
                [0, 0, 0], [1, 1, 1], 1, "NDHWC")
        out.backward(out)
        paddle.device.cuda.synchronize()
        print("sparse conv3d times:", time.time() - t)
        assert np.array_equal(out_indices, out.indices().numpy())
        assert np.allclose(out.values().numpy(), out_values, rtol=1e-3, atol=1e-3) 
        ##print(sparse_x.grad)
        ##print(kernels_tensor.grad.numpy()[0])
        assert np.allclose(sparse_x.grad.values().numpy(), feature_grad, rtol=1e-3, atol=1e-3)
        assert np.allclose(kernels_tensor.grad.numpy(), kernel_grad, rtol=1e-3, atol=1e-3)
        #print("compare success")
    
    # test sparse subm conv3d
    if True:
        torch_sparse_x = torch.sparse_coo_tensor(subm_indices, subm_values)
        torch_sparse_x = torch_sparse_x.coalesce()
        torch_sparse_out = torch.sparse_coo_tensor(subm_out_indices, subm_out_values)
        torch_sparse_out = torch_sparse_out.coalesce()

        sparse_x = paddle.sparse.sparse_coo_tensor(paddle.to_tensor(torch_sparse_x.indices().numpy(), dtype='int32'), paddle.to_tensor(torch_sparse_x.values().numpy(), dtype='float32'), shape = [2, 10, 400, 150, 32], stop_gradient=False)
        kernels_tensor = paddle.to_tensor(subm_kernels, dtype='float32', stop_gradient=False)

        ###gpu warm up##
        out1 = paddle.sparse.functional.subm_conv3d(sparse_x, kernels_tensor, None, [1, 1, 1],
                [0, 0, 0], [1, 1, 1], 1, "NDHWC")
        paddle.device.cuda.synchronize()
        ###gpu warm up##

        t = time.time()
        out = paddle.sparse.functional.subm_conv3d(sparse_x, kernels_tensor, None, [1, 1, 1],
                [0, 0, 0], [1, 1, 1], 1, "NDHWC")
        out.backward(out)
        paddle.device.cuda.synchronize()
        print("subm conv3d times:", time.time() - t)
        assert np.array_equal(torch_sparse_out.indices().numpy(), out.indices().numpy())
        assert np.allclose(out.values().numpy(), torch_sparse_out.values().numpy(), rtol=1e-3, atol=1e-3) 
        #assert np.allclose(sparse_x.grad.values().numpy(), subm_feature_grad, rtol=1e-3, atol=1e-3)
        #assert np.allclose(kernels_tensor.grad.numpy(), subm_kernel_grad, rtol=1e-3, atol=1e-3)
        
