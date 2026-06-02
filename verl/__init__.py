# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

# Monkey-patch torch.library.wrap_triton for PyTorch < 2.6 compatibility.
# wrap_triton was introduced in PyTorch 2.6; in 2.5 it does not exist but
# some libraries (e.g. flash-attn) may reference it at runtime.
# This must live here (package __init__) so that every process – including
# Ray workers – gets the patch before any triton kernel is invoked.
import torch
import torch.library
if not hasattr(torch.library, "wrap_triton"):
    torch.library.wrap_triton = lambda fn: fn

from .protocol import DataProto


__all__ = ["DataProto"]
__version__ = "0.2.0.dev"
