# Copyright 2021 Yan Yan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from spconv.core import ConvAlgo

import spconv.pytorch as spconv
from spconv.test_utils import TestCase, generate_sparse_data, params_grid
from spconv.constants import FILTER_HWIO
# import sparseconvnet as scn

# we must disable tf32 to increase reference precision.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class SparseConv3dTestTorch(nn.Module):
    def __init__(self,
                 num_layers,
                 ndim,
                 shape,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 algo=spconv.ConvAlgo.MaskSplitImplicitGemm):
        super().__init__()
        self.algo = algo
        layers = [
            spconv.SparseConv3d(in_channels,
                                out_channels,
                                kernel_size,
                                stride,
                                padding=padding,
                                dilation=dilation,
                                bias=False,
                                algo=algo)
        ]
        for i in range(1, num_layers):
            layers.append(
                spconv.SparseConv3d(out_channels,
                                    out_channels,
                                    kernel_size,
                                    stride,
                                    padding=padding,
                                    dilation=dilation,
                                    bias=False,
                                    algo=algo))
        self.net = spconv.SparseSequential(*layers, )
        # self.grid = torch.full([3, *shape], -1, dtype=torch.int32).cuda()
        self.grid = None
        self.shape = shape

    def forward(self, features, coors, batch_size):
        coors = coors.int()
        x = spconv.SparseConvTensor(features, coors, self.shape, batch_size,
                                    self.grid)
        return self.net(x)  # .dense()


class SubMConv3dTestTorch(nn.Module):
    def __init__(self,
                 num_layers,
                 ndim,
                 shape,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 algo=spconv.ConvAlgo.Native):
        super().__init__()
        layers = [
            spconv.SubMConv3d(in_channels,
                              out_channels,
                              kernel_size,
                              stride,
                              padding=padding,
                              dilation=dilation,
                              bias=False,
                              algo=algo)
        ]
        for i in range(1, num_layers):
            layers.append(
                spconv.SubMConv3d(out_channels,
                                  out_channels,
                                  kernel_size,
                                  stride,
                                  padding=padding,
                                  dilation=dilation,
                                  bias=False,
                                  algo=algo))
        self.net = spconv.SparseSequential(*layers, )
        # self.grid = torch.full([3, *shape], -1, dtype=torch.int32).cuda()
        self.grid = None
        self.shape = shape

    def forward(self, features, coors, batch_size):
        coors = coors.int()  # .cpu()
        x = spconv.SparseConvTensor(features, coors, self.shape, batch_size,
                                    self.grid)
        return self.net(x)  # .dense()


class Conv3dTestTorch(nn.Module):
    def __init__(self, num_layers, ndim, shape, in_channels, out_channels,
                 kernel_size, stride, padding, dilation):
        super().__init__()
        layers = [
            nn.Conv3d(in_channels,
                      out_channels,
                      kernel_size,
                      stride,
                      padding=padding,
                      dilation=dilation,
                      bias=False)
        ]
        for i in range(1, num_layers):
            layers.append(
                nn.Conv3d(out_channels,
                          out_channels,
                          kernel_size,
                          stride,
                          padding=padding,
                          dilation=dilation,
                          bias=False))
        self.net = nn.Sequential(*layers, )
        self.shape = shape

    def forward(self, x):
        return self.net(x)  # .dense()


class SparseDeConv3dTestTorch(nn.Module):
    def __init__(self, num_layers, ndim, shape, in_channels, out_channels,
                 kernel_size, stride, padding, dilation):
        super().__init__()
        layers = [
            spconv.SparseConvTranspose3d(in_channels,
                                         out_channels,
                                         kernel_size,
                                         stride,
                                         padding=padding,
                                         dilation=dilation,
                                         bias=False)
        ]
        for i in range(1, num_layers):
            layers.append(
                spconv.SparseConvTranspose3d(out_channels,
                                             out_channels,
                                             kernel_size,
                                             stride,
                                             padding=padding,
                                             dilation=dilation,
                                             bias=False))
        self.net = spconv.SparseSequential(*layers, )
        self.shape = shape

    def forward(self, features, coors, batch_size):
        coors = coors.int()
        x = spconv.SparseConvTensor(features, coors, self.shape, batch_size)
        return self.net(x)  # .dense()


class DeConv3dTestTorch(nn.Module):
    def __init__(self, num_layers, ndim, shape, in_channels, out_channels,
                 kernel_size, stride, padding, dilation):
        super().__init__()
        layers = [
            nn.ConvTranspose3d(in_channels,
                               out_channels,
                               kernel_size,
                               stride,
                               padding=padding,
                               dilation=dilation,
                               bias=False)
        ]
        for i in range(1, num_layers):
            layers.append(
                nn.ConvTranspose3d(out_channels,
                                   out_channels,
                                   kernel_size,
                                   stride,
                                   padding=padding,
                                   dilation=dilation,
                                   bias=False))
        self.net = nn.Sequential(*layers, )
        self.shape = shape

    def forward(self, x):
        return self.net(x)  # .dense()


class SparseMaxPoolTestTorch(nn.Module):
    def __init__(self, num_layers, ndim, shape, kernel_size, stride, padding,
                 dilation):
        super().__init__()
        layers = [
            spconv.SparseMaxPool3d(kernel_size, stride, padding, dilation)
        ]
        for i in range(1, num_layers):
            layers.append(
                spconv.SparseMaxPool3d(kernel_size, stride, padding, dilation))
        self.net = spconv.SparseSequential(*layers, )
        self.shape = shape

        coors = coors.int()
        x = spconv.SparseConvTensor(features, coors, self.shape, batch_size)
        return self.net(x)  # .dense()


class MaxPool3dTestTorch(nn.Module):
    def __init__(self, num_layers, ndim, shape, kernel_size, stride, padding,
                 dilation):
        super().__init__()
        layers = [nn.MaxPool3d(kernel_size, stride, padding, dilation)]
        for i in range(1, num_layers):
            layers.append(nn.MaxPool3d(kernel_size, stride, padding, dilation))
        self.net = nn.Sequential(*layers, )
        self.shape = shape

    def forward(self, x):
        return self.net(x)  # .dense()


class SubmanifoldConvTestTorch(nn.Module):
    def __init__(self, num_layers, ndim, shape, in_channels, out_channels,
                 kernel_size, stride):
        super().__init__()
        layers = [
            spconv.SubMConv3d(in_channels,
                              out_channels,
                              kernel_size,
                              bias=False,
                              indice_key="subm0")
        ]
        for i in range(1, num_layers):
            layers.append(
                spconv.SubMConv3d(out_channels,
                                  out_channels,
                                  kernel_size,
                                  bias=False))
        self.net = nn.Sequential(*layers, )
        self.shape = shape

    def forward(self, features, coors, batch_size):
        coors = coors.int()
        x = spconv.SparseConvTensor(features, coors, self.shape, batch_size)
        return self.net(x)


class SCNCoupleDeConvTest(nn.Module):
    def __init__(self, num_layers, ndim, shape, in_channels, out_channels,
                 kernel_size, stride):
        super().__init__()
        self.scn_input = scn.InputLayer(ndim, shape, mode=0)
        self.net = nn.Sequential(
            scn.Convolution(ndim,
                            in_channels,
                            out_channels,
                            kernel_size,
                            stride,
                            bias=False),
            scn.Deconvolution(ndim,
                              out_channels,
                              in_channels,
                              kernel_size,
                              stride,
                              bias=False),
            scn.SparseToDense(ndim, in_channels),
        )

    def forward(self, features, coors, batch_size):
        coors = coors.long().cpu()
        x = self.scn_input((coors, features))
        return self.net(x)


class SparseCoupleDeConvTest(nn.Module):
    def __init__(self, num_layers, ndim, shape, in_channels, out_channels,
                 kernel_size, stride):
        super().__init__()
        self.net = spconv.SparseSequential(
            spconv.SparseConv3d(in_channels,
                                out_channels,
                                kernel_size,
                                stride,
                                indice_key="cp0",
                                bias=False),
            spconv.SparseInverseConv3d(out_channels,
                                       in_channels,
                                       kernel_size,
                                       indice_key="cp0",
                                       bias=False),
        )
        self.todense = spconv.ToDense()
        self.shape = shape

    def forward(self, features, coors, batch_size):
        coors = coors.int()
        x = spconv.SparseConvTensor(features, coors, self.shape, batch_size)
        return self.todense(self.net(x))  # .dense()


def gather_nd(params, indices):
    # this function has a limit that MAX_ADVINDEX_CALC_DIMS=5
    ndim = indices.shape[-1]
    output_shape = list(indices.shape[:-1]) + list(
        params.shape[indices.shape[-1]:])
    flatted_indices = indices.view(-1, ndim)
    slices = [flatted_indices[:, i] for i in range(ndim)]
    slices += [Ellipsis]
    return params[slices].view(*output_shape)


def scatter_nd(indices, updates, shape):
    """pytorch edition of tensorflow scatter_nd.
    this function don't contain except handle code. so use this carefully
    when indice repeats, don't support repeat add which is supported
    in tensorflow.
    """
    ret = torch.zeros(*shape, dtype=updates.dtype, device=updates.device)
    ndim = indices.shape[-1]
    output_shape = list(indices.shape[:-1]) + shape[indices.shape[-1]:]
    flatted_indices = indices.view(-1, ndim)
    slices = [flatted_indices[:, i] for i in range(ndim)]
    slices += [Ellipsis]
    ret[slices] = updates.view(*output_shape)
    return ret


class TestSpConv(TestCase):
    def testSpConv3d(self):
        np.random.seed(484)
        torch.manual_seed(48848)
        devices = ["cuda:0"]
        shapes = [[19, 18, 17]]
        batchsizes = [1, 2]

        in_channels = [32]
        out_channels = [32, 48, 64]
        ksizes = [2, 3]
        strides = [1, 2, 3]
        paddings = [0, 1, 2]
        dilations = [1, 2, 3]
        algos = [
            ConvAlgo.Native, ConvAlgo.MaskImplicitGemm,
            ConvAlgo.MaskSplitImplicitGemm
        ]
        algos = [ConvAlgo.MaskSplitImplicitGemm]

        for dev, shape, bs, IC, OC, k, s, p, d, al in params_grid(
                devices, shapes, batchsizes, in_channels, out_channels, ksizes,
                strides, paddings, dilations, algos):
            if all([s > 1, d > 1]):
                continue  # don't support this.
            print(k, s, p, d)
            device = torch.device(dev)
            num_points = [1000] * bs
            dtype = torch.float32
            net = SparseConv3dTestTorch(1,
                                        3,
                                        shape,
                                        IC,
                                        OC,
                                        k,
                                        s,
                                        p,
                                        d,
                                        algo=al).to(device).to(dtype)
            net_ref = Conv3dTestTorch(1, 3, shape, IC, OC, k, s, p,
                                      d).to(device).to(dtype)

            sparse_dict = generate_sparse_data(shape, num_points, IC)

            features = np.ascontiguousarray(sparse_dict["features"]).astype(
                np.float32)
            indices = np.ascontiguousarray(
                sparse_dict["indices"][:, [3, 0, 1, 2]]).astype(np.int32)
            features_dense = sparse_dict["features_dense"].astype(np.float32)
            indices_t = torch.from_numpy(indices).int().to(device)
            features_t = torch.from_numpy(features).to(device).to(dtype)
            features_t.requires_grad = True
            features_dense_t = torch.from_numpy(features_dense).to(device).to(
                dtype)
            features_dense_t.requires_grad = True
            if net.algo == ConvAlgo.Native:
                if FILTER_HWIO:
                    filters = np.random.uniform(-1, 1,
                                                size=[k, k, k, IC,
                                                      OC]).astype(np.float32)
                else:
                    filters = np.random.uniform(-1, 1,
                                                size=[k, k, k, OC,
                                                      IC]).astype(np.float32)
                filters_t = torch.from_numpy(filters).to(device).to(dtype)
                if FILTER_HWIO:
                    net_ref.net[0].weight.data[:] = filters_t.permute(
                        4, 3, 0, 1, 2).contiguous()
                else:
                    net_ref.net[0].weight.data[:] = filters_t.permute(
                        3, 4, 0, 1, 2).contiguous()
            else:
                filters = np.random.uniform(-1, 1,
                                            size=[OC, k, k, k,
                                                  IC]).astype(np.float32)
                filters_t = torch.from_numpy(filters).to(device).to(dtype)
                net_ref.net[0].weight.data[:] = filters_t.permute(
                    0, 4, 1, 2, 3).contiguous()

            net.net[0].weight.data[:] = filters_t
            out_ref = net_ref(features_dense_t)
            out = net(features_t, indices_t, bs).dense()
            out_np = out.detach().cpu().numpy()
            out_ref_np = out_ref.detach().cpu().numpy()
            self.assertAllClose(out_np, out_ref_np, atol=1e-4)

            dout = np.random.uniform(-0.2, 0.2,
                                     out_ref.shape).astype(features.dtype)
            dout_t = torch.from_numpy(dout).to(device)
            out.backward(dout_t)
            out_ref.backward(dout_t)
            din_dense = features_dense_t.grad.detach().permute(0, 2, 3, 4,
                                                               1).contiguous()
            din_sparse = gather_nd(din_dense, indices_t.long())
            din = features_t.grad.detach()

            din_np = din.cpu().numpy()
            din_sparse_np = din_sparse.cpu().numpy()
            for layer, layer_ref in zip(net.net, net_ref.net):
                dw = layer.weight.grad.detach().cpu().numpy()
                dw_ref = layer_ref.weight.grad.detach().cpu().numpy()
                if net.algo == ConvAlgo.Native:
                    if FILTER_HWIO:
                        dw = dw.transpose(4, 3, 0, 1, 2)
                    else:
                        dw = dw.transpose(3, 4, 0, 1, 2)
                else:
                    # OHWI -> OIHW
                    dw = dw.transpose(0, 4, 1, 2, 3)

                self.assertAllClose(dw, dw_ref, atol=1e-4)
            self.assertAllClose(din_np, din_sparse_np, atol=1e-4)

    def testSpDeConv3d(self):
        np.random.seed(484)
        devices = ["cuda:0"]
        shapes = [[19, 18, 17]]
        batchsizes = [1, 2]

        in_channels = [64]
        out_channels = [32, 48, 64]
        ksizes = [2, 3]
        strides = [2, 3]
        paddings = [0, 1, 2]
        dilations = [1, 2, 3]
        ksizes = [3]

        strides = [1]
        paddings = [0]
        dilations = [1]

        for dev, shape, bs, IC, OC, k, s, p, d in params_grid(
                devices, shapes, batchsizes, in_channels, out_channels, ksizes,
                strides, paddings, dilations):
            if all([s > 1, d > 1]):
                continue  # don't support this.
            device = torch.device(dev)
            num_points = [1000] * bs

            sparse_dict = generate_sparse_data(shape, num_points, IC)

            features = np.ascontiguousarray(sparse_dict["features"]).astype(
                np.float32)
            indices = np.ascontiguousarray(
                sparse_dict["indices"][:, [3, 0, 1, 2]]).astype(np.int32)
            features_dense = sparse_dict["features_dense"].astype(np.float32)
            if FILTER_HWIO:
                filters = np.random.uniform(0, 1, size=[k, k, k, IC,
                                                        OC]).astype(np.float32)
            else:
                filters = np.random.uniform(0, 1, size=[k, k, k, OC,
                                                        IC]).astype(np.float32)

            indices_t = torch.from_numpy(indices).int().to(device)
            features_t = torch.from_numpy(features).to(device)
            features_t.requires_grad = True
            features_dense_t = torch.from_numpy(features_dense).to(device)
            features_dense_t.requires_grad = True
            net = SparseDeConv3dTestTorch(1, 3, shape, IC, OC, k, s, p,
                                          d).to(device)
            net_ref = DeConv3dTestTorch(1, 3, shape, IC, OC, k, s, p,
                                        d).to(device)
            filters_t = torch.from_numpy(filters).to(device)
            print(net_ref.net[0].weight.shape)
            if FILTER_HWIO:
                net_ref.net[0].weight.data[:] = filters_t.permute(
                    3, 4, 0, 1, 2).contiguous()
            else:
                net_ref.net[0].weight.data[:] = filters_t.permute(
                    4, 3, 0, 1, 2).contiguous()
            net.net[0].weight.data[:] = filters_t
            out_ref = net_ref(features_dense_t)
            out = net(features_t, indices_t, bs).dense()
            out_np = out.detach().cpu().numpy()
            out_ref_np = out_ref.detach().cpu().numpy()
            self.assertAllClose(out_np, out_ref_np, atol=1e-4)

            dout = np.random.uniform(-0.2, 0.2,
                                     out_ref.shape).astype(features.dtype)
            dout_t = torch.from_numpy(dout).to(device)
            out.backward(dout_t)
            out_ref.backward(dout_t)
            din_dense = features_dense_t.grad.detach().permute(0, 2, 3, 4,
                                                               1).contiguous()
            din_sparse = gather_nd(din_dense, indices_t.long())
            din = features_t.grad.detach()
            din_np = din.cpu().numpy()
            din_sparse_np = din_sparse.cpu().numpy()
            self.assertAllClose(din_np, din_sparse_np, atol=1e-4)
            for layer, layer_ref in zip(net.net, net_ref.net):
                dw = layer.weight.grad.detach().cpu().numpy()
                dw_ref = layer_ref.weight.grad.detach().cpu().numpy()
                if FILTER_HWIO:
                    dw = dw.transpose(3, 4, 0, 1, 2)
                else:
                    dw = dw.transpose(4, 3, 0, 1, 2)
                self.assertAllClose(dw, dw_ref, atol=1e-4)

    def testSpCpConv3d(self):
        np.random.seed(484)
        devices = ["cuda:0", "cpu:0"]
        shapes = [[20, 20, 20]]
        batchsizes = [1, 2]

        in_channels = [64]
        out_channels = [32, 48, 64]
        ksizes = [2]
        strides = [2]
        paddings = [0, 1, 2]
        dilations = [1, 2, 3]

        for dev, shape, bs, IC, OC, k, s in params_grid(
                devices, shapes, batchsizes, in_channels, out_channels, ksizes,
                strides):
            device = torch.device(dev)
            num_points = [1000] * bs

            sparse_dict = generate_sparse_data(shape, num_points, IC)

            features = np.ascontiguousarray(sparse_dict["features"]).astype(
                np.float32)
            indices = np.ascontiguousarray(
                sparse_dict["indices"][:, [3, 0, 1, 2]]).astype(np.int32)
            features_dense = sparse_dict["features_dense"].astype(np.float32)
            filters = np.random.uniform(0, 1, size=[k, k, k, IC,
                                                    OC]).astype(np.float32)
            indices_t = torch.from_numpy(indices).int().to(device)
            indices_scn_t = torch.from_numpy(
                indices[:, [1, 2, 3, 0]]).int().to(device)
            features_t = torch.from_numpy(features).to(device)
            features_t.requires_grad = True
            features_ref_t = torch.from_numpy(features).to(device)
            features_ref_t.requires_grad = True

            net_ref = SCNCoupleDeConvTest(1, 3, shape, IC, OC, k, s).to(device)
            net = SparseCoupleDeConvTest(1, 3, shape, IC, OC, k, s).to(device)
            net_ref.net[0].weight.data[:] = net.net[0].weight.data[:].view(
                *net_ref.net[0].weight.shape)
            net_ref.net[1].weight.data[:] = net.net[1].weight.data[:].view(
                *net_ref.net[1].weight.shape)
            out_ref = net_ref(features_ref_t, indices_scn_t, bs)
            out = net(features_t, indices_t, bs)
            dout = np.random.uniform(-0.2, 0.2,
                                     out_ref.shape).astype(features.dtype)
            dout_t = torch.from_numpy(dout).to(device)
            out.backward(dout_t)
            out_ref.backward(dout_t)
            din = features_t.grad.detach()
            din_ref = features_ref_t.grad.detach()
            din_np = din.cpu().numpy()
            din_ref_np = din_ref.cpu().numpy()
            self.assertAllClose(din_ref_np, din_np, atol=1e-4)
            for layer, layer_ref in zip(net.net, net_ref.net):
                dw = layer.weight.grad.detach().cpu().numpy()
                dw_ref = layer_ref.weight.grad.detach().cpu().view(
                    *dw.shape).numpy()
                self.assertAllClose(dw, dw_ref, atol=1e-4)

            out_np = out.detach().cpu().numpy()
            out_ref_np = out_ref.detach().cpu().numpy()
            self.assertAllClose(out_np, out_ref_np, atol=1e-4)

    def testSpMaxPool3d(self):
        np.random.seed(485)
        devices = ["cuda:0"]
        #shapes = [[19, 18, 17]]
        shapes = [[1, 4, 4]]
        #batchsizes = [1, 2]
        batchsizes = [1, 2]

        in_channels = [64]
        out_channels = [64]
        ksizes = [2, 3]
        strides = [1, 2, 3]
        paddings = [0, 1]
        dilations = [1, 2, 3]
        # ksizes = [2]
        # strides = [2]
        # paddings = [0]
        # dilations = [1]

        for dev, shape, bs, IC, OC, k, s, p, d in params_grid(
                devices, shapes, batchsizes, in_channels, out_channels, ksizes,
                strides, paddings, dilations):
            if all([s > 1, d > 1]):
                continue  # don't support this.
            device = torch.device(dev)
            num_points = [1000] * bs

            # when data contains negative, sparse maxpool is not equal to dense maxpool.
            sparse_dict = generate_sparse_data(shape,
                                               num_points,
                                               IC,
                                               data_range=[0.1, 1])

            features = np.ascontiguousarray(sparse_dict["features"]).astype(
                np.float32)
            indices = np.ascontiguousarray(
                sparse_dict["indices"][:, [3, 0, 1, 2]]).astype(np.int32)
            features_dense = sparse_dict["features_dense"].astype(np.float32)
            filters = np.random.uniform(0, 1, size=[k, k, k, OC,
                                                    IC]).astype(np.float32)
            indices_t = torch.from_numpy(indices).int().to(device)
            features_t = torch.from_numpy(features).to(device)
            features_t.requires_grad = True
            features_dense_t = torch.from_numpy(features_dense).to(device)
            features_dense_t.requires_grad = True
            net = SparseMaxPoolTestTorch(1, 3, shape, k, s, p, d).to(device)
            net_ref = MaxPool3dTestTorch(1, 3, shape, k, s, p, d).to(device)

            out_ref = net_ref(features_dense_t)
            out = net(features_t, indices_t, bs)

            outids = out.indices
            outfeatures = out.features
            outids_dev = outids.float()
            out_dense = out.dense(channels_first=False)
            out = out_dense.permute(0, 4, 1, 2, 3).contiguous()
            out_np = out.detach().cpu().numpy()
            out_ref_np = out_ref.detach().cpu().numpy()
            self.assertAllClose(out_np, out_ref_np, atol=1e-4)

            dout_sparse = np.random.uniform(
                -0.2, 0.2, outfeatures.shape).astype(features.dtype)
            dout_sparse_t = torch.from_numpy(dout_sparse).to(device)
            dout_t = scatter_nd(outids.long(), dout_sparse_t,
                                list(out_dense.shape))
            dout_t = dout_t.permute(0, 4, 1, 2, 3).contiguous()
            out.backward(dout_t)
            out_ref.backward(dout_t)
            din_dense = features_dense_t.grad.detach().permute(0, 2, 3, 4,
                                                               1).contiguous()
            din_sparse = gather_nd(din_dense, indices_t.long())
            din = features_t.grad.detach()

            din_np = din.cpu().numpy()
            din_sparse_np = din_sparse.cpu().numpy()
            self.assertAllClose(din_np, din_sparse_np, atol=1e-4)


def main(algo=spconv.ConvAlgo.Native, dtype=torch.float32):
    # function for develop.
    np.random.seed(484)
    # devices = ["cuda:0"]
    devices = ["cuda:0"]
    shapes = [[10, 400, 152]]
    batchsizes = [2]

    in_channels = [32]
    out_channels = [32]
    ksizes = [(3, 3, 3)]
    strides = [1]
    paddings = [0]
    dilations = [1]

    for dev, shape, bs, IC, OC, k, s, p, d in params_grid(
            devices, shapes, batchsizes, in_channels, out_channels, ksizes,
            strides, paddings, dilations):
        if all([s > 1, d > 1]):
            continue
        device = torch.device(dev)
        num_points = [16000] * bs

        sparse_dict = generate_sparse_data(shape, num_points, IC)

        features = np.ascontiguousarray(sparse_dict["features"]).astype(
            np.float32)
        indices = np.ascontiguousarray(
            sparse_dict["indices"][:, [3, 0, 1, 2]]).astype(np.int32)
        features_dense = sparse_dict["features_dense"].astype(np.float32)
        indices_t = torch.from_numpy(indices)
        print(indices_t.size)
        filters = np.random.uniform(0, 1, size=[k[0], 3, 3, IC,
                                                OC]).astype(np.float32)
        print(filters.size)
        indices_t = torch.from_numpy(indices).int().to(device).to(dtype)
        features_t = torch.from_numpy(features).to(device).to(dtype)

        features_dense_t = torch.from_numpy(features_dense).to(device).to(
            dtype)
        net = SparseConv3dTestTorch(1, 3, shape, IC, OC, k, s, p, d,
                                    algo=algo).to(device).to(dtype)
        net_ref = Conv3dTestTorch(1, 3, shape, IC, OC, k, s, p,
                                  d).to(device).to(dtype)
        filters_t = torch.from_numpy(filters).to(device).to(dtype)
        tmp = filters_t.permute(4, 0, 1, 2, 3).contiguous()
        #net_ref.net[0].weight.requires_grad = False
        net.net[0].weight.requires_grad = False
        features_t.requires_grad_()
        #net.net[0].weight[:] = tmp
        net.net[0].weight[:] = filters_t
        net.net[0].weight.requires_grad = True
        #out_ref = net_ref(features_dense_t)
        #print("in_features:", features_t.detach().cpu())
        #print("in_indices:", indices_t.detach().cpu())
        #np.savetxt("features", features_t.detach().cpu().flatten(), fmt="%.8f");
        np.save("features", features_t.detach().cpu())
        #np.savetxt("indices", indices_t.detach().cpu().transpose(1,0).flatten(), fmt="%d");
        np.save("indices", indices_t.detach().cpu().transpose(1,0))
        #np.savetxt("kernels", net.net[0].weight.detach().data.cpu().flatten(), fmt="%.8f");
        np.save("kernels", net.net[0].weight.detach().data.cpu())
        #times = []
        #for i in range(10):
        #    t = time.time()
        #    out = net(features_t, indices_t, bs)
        #    torch.cuda.synchronize()
        #    times.append(time.time() - t)
        t = time.time()
        out = net(features_t, indices_t, bs)
        out._features.backward(out._features.detach())
        torch.cuda.synchronize(device)
        print("sparse conv3d times:", time.time() - t)
        #print("kernel size:", net.net[0].weight.data.size());
        #print("in_features size:", features_t.size(), "indices shape:", indices_t.size());
        #print("out feature shape:", out._features.size(), "indices shape:", out.indices.size())
        #np.savetxt("out_feature", out._features.detach().cpu().flatten(), fmt="%.5f")
        np.save("out_feature", out._features.detach().cpu())
        #np.savetxt("out_indices", out.indices.detach().cpu().transpose(1,0).flatten(), fmt="%d")
        np.save("out_indices", out.indices.detach().cpu().transpose(1,0))
        #print("kernels:", filters_t.cpu().numpy())
        # print((net.grid == -1).float().sum(), net.grid.numel())
        # print("spconv time", time.time() - t)
        #print("feature_t.grad:", features_t.grad)
        #np.savetxt("featuers_grad", features_t.grad.cpu().flatten(), fmt="%.5f")
        np.save("features_grad", features_t.grad.cpu())
        #np.savetxt("kernel_grad", net.net[0].weight.grad.cpu().flatten(), fmt="%.5f")
        np.save("kernel_grad", net.net[0].weight.grad.cpu())
        ## print(out.indices)
        #out = out.dense()
        #out_numpy = out.detach().cpu().numpy()

        #print(
        #    np.linalg.norm(out.detach().cpu().numpy() -
        #                   out_ref.detach().cpu().numpy()))
        #print(out_numpy.min(), out_numpy.max(), out_numpy.mean(),
        #      out_numpy.sum())


def main_subm(algo, dtype=torch.float32):
    # function for develop.
    np.random.seed(484)
    torch.manual_seed(50051)
    # devices = ["cuda:0"]
    devices = ["cuda:0"]
    shapes = [[10, 400, 150]]
    batchsizes = [2]

    in_channels = [32]
    out_channels = [32]
    ksizes = [(3, 3, 3)]
    strides = [1]
    paddings = [0]
    dilations = [1]
    for dev, shape, bs, IC, OC, k, s, p, d in params_grid(
            devices, shapes, batchsizes, in_channels, out_channels, ksizes,
            strides, paddings, dilations):
        if all([s > 1, d > 1]):
            continue
        device = torch.device(dev)
        num_points = [3] * bs

        sparse_dict = generate_sparse_data(shape, num_points, IC)

        features = np.ascontiguousarray(sparse_dict["features"]).astype(
            np.float32)
        indices = np.ascontiguousarray(
            sparse_dict["indices"][:, [3, 0, 1, 2]]).astype(np.int32)
        features_dense = sparse_dict["features_dense"].astype(np.float32)
        indices_t = torch.from_numpy(indices)
        filters = np.random.uniform(0, 1, size=[k[0], 3, 3, IC,
                                                OC]).astype(np.float32)
        indices_t = torch.from_numpy(indices).int().to(device).to(dtype)
        features_t = torch.from_numpy(features).to(device).to(dtype)

        features_dense_t = torch.from_numpy(features_dense).to(device).to(
            dtype)
        net = SubMConv3dTestTorch(1, 3, shape, IC, OC, k, s, p, d,
                                  algo=algo).to(device).to(dtype)
        net_ref = Conv3dTestTorch(1, 3, shape, IC, OC, k, s, p,
                                  d).to(device).to(dtype)
        filters_t = torch.from_numpy(filters).to(device).to(dtype)
        net_ref.net[0].weight.requires_grad = False
        #tmp = filters_t.permute(4,3, 0, 1, 2).contiguous()
        tmp = filters_t.permute(4,3, 0, 1, 2).contiguous()
        print(tmp.size())
        print(net_ref.net[0].weight[:].size())
        net_ref.net[0].weight[:] =  tmp 
        net.net[0].weight.requires_grad = False 
        features_t.requires_grad_()
        net.net[0].weight[:] = filters_t
        net.net[0].weight.requires_grad = True 
        #tmp = filters_t.permute(4, 0, 1, 2, 3).contiguous()
        #net.net[0].weight[:] = tmp 
        #out_ref = net_ref(features_dense_t)
        #times = []
        #for i in range(20):
        #    t = time.time()
        #    out = net(features_t, indices_t, bs)
        #    torch.cuda.synchronize()
        #    times.append(time.time() - t)
        # print((net.grid == -1).float().sum(), net.grid.numel())
        # print("spconv time", time.time() - t)
        #print("spconv time", np.mean(times[10:]))
        t = time.time()
        out = net(features_t, indices_t, bs)
        print("subm forward times:", time.time() - t)
        out._features.backward(out._features.detach())
        print("subm conv3d times:", time.time() - t)
        #print("indices:", indices_t.transpose(1, 0).flatten())
        #print("features:", features_t.flatten())
        #print("kernel:", net.net[0].weight[:].cpu().flatten())
        #print("out_indices:", out.indices.transpose(1,0).flatten())
        #print("out_features:", out._features.flatten())
        np.save("subm_indices", indices_t.cpu().transpose(1, 0))
        np.save("subm_features", features_t.detach().cpu())
        np.save("subm_kernel", net.net[0].weight[:].detach().cpu())
        np.save("subm_out_indices", out.indices.cpu().transpose(1,0))
        np.save("subm_out_features", out._features.detach().cpu())

        np.save("subm_features_grad", features_t.grad.cpu())
        np.save("subm_kernels_grad", net.net[0].weight.grad.cpu())

        #print("features_grad:", features_t.grad.flatten())
        #print("kernel_grad:", net.net[0].weight.grad.flatten())

        #out = out.dense()
        #out_numpy = out.detach().cpu().numpy()
        ## print(
        ##     np.linalg.norm(out.detach().cpu().numpy() -
        ##                    out_ref.detach().cpu().numpy()))
        #print(out_numpy.min(), out_numpy.max(), out_numpy.mean(),
        #      out_numpy.sum())
    #return out_numpy


if __name__ == '__main__':
    main_subm(algo=spconv.ConvAlgo.Native, dtype=torch.float32)
    #main(algo=spconv.ConvAlgo.Native, dtype=torch.float32)
    # TestCase().assertAllClose(out_my, out_ref)
    # unittest.main()
    #TestSpConv().testSpMaxPool3d()
