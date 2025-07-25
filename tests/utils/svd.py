"""Decomposition using PyTorch."""

import gc
from abc import ABC, abstractmethod
from functools import wraps
from typing import Optional, Union

import numpy as np

# NOTE disabled since torch isnt even a dependency

# import torch


# def np_wrapper(fn):
#     """
#     Make sure 1st arg is the one we want to wrap

#     @param fn:
#     @return:
#     """

#     @wraps(fn)
#     def _impl(self, *args, **kwargs):
#         x = args[0]

#         if type(x) == np.ndarray:
#             __return_np = True
#             x = torch.from_numpy(x)

#         else:
#             __return_np = False

#         args = tuple([x] + list(args[1:]))

#         ret = fn(self, *args, **kwargs)

#         if __return_np is True and type(ret) == torch.Tensor:
#             ret = ret.cpu().numpy()

#         return ret

#     return _impl


# class TorchSVDTransformABC(ABC):
#     def __init__(
#         self, n_components: int, device: Optional[torch.device] = torch.device("cuda:0")
#     ):
#         self.n_components = n_components
#         self.device = device

#         self.u = None
#         self.s = None
#         self.v = None

#     @property
#     def linear_layer(self):
#         assert self.v is not None, f"{self.__class__.__name__} not fitted"
#         return self.v

#     def to(self, device: torch.device):
#         if self.u is not None:
#             self.u = self.u.to(device)

#         if self.s is not None:
#             self.s = self.s.to(device)

#         if self.v is not None:
#             self.v = self.v.to(device)

#     @abstractmethod
#     def _fit(self, *args, **kwargs):
#         raise NotImplementedError

#     @np_wrapper
#     def fit(self, *args, **kwargs):
#         self._fit(*args, **kwargs)

#         gc.collect()
#         torch.cuda.empty_cache()

#     @np_wrapper
#     def transform(self, x: torch.Tensor):
#         assert self.v is not None, f"{self.__class__.__name__} not fitted"

#         x = x.to(self.device).type(torch.float)
#         x_transformed = torch.matmul(x, self.v)

#         return x_transformed

#     @np_wrapper
#     def inverse_transform(self, x: torch.Tensor):
#         assert self.v is not None, f"{self.__class__.__name__} not fitted"

#         x = x.to(self.device).type(torch.float)
#         x_untransformed = torch.matmul(x, torch.transpose(self.v, 0, 1))

#         return x_untransformed

#     @np_wrapper
#     def fit_transform(self, x: torch.Tensor):
#         x = x.to(self.device).type(torch.float)
#         self.fit(x)

#         return self.transform(x)


# class TorchLowRankSVDTransform(TorchSVDTransformABC):
#     def __init__(
#         self,
#         n_components: int,
#         n_oversamples: Optional[int] = 32,
#         n_iter: Optional[int] = 8,
#         device: Optional[torch.device] = torch.device("cuda:0"),
#         random_state: Optional[int] = None,
#     ):
#         super().__init__(n_components=n_components, device=device)
#         self.n_oversamples = n_oversamples
#         self.n_iter = n_iter
#         self.random_state = random_state

#     def _fit(self, x: Union[torch.Tensor, np.ndarray]):
#         if self.random_state is not None:
#             _ = torch.manual_seed(self.random_state)

#         nsamples = x.shape[0]
#         if self.n_components > nsamples:
#             self.n_components = nsamples

#         q = self.n_components + self.n_oversamples
#         if q > nsamples:
#             self.n_oversamples = nsamples - self.n_components
#             q = self.n_components + self.n_oversamples

#         _, _, v = torch.svd_lowrank(x, q=q)
#         self.v = v[:, : self.n_components]
