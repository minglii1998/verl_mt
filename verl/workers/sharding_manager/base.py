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
"""
Sharding manager to implement HybridEngine
"""

from verl import DataProto


class BaseShardingManager:
    def __init__(self):
        # `timing` is used by fsdp_workers to aggregate latency information
        # Downstream code expects the sharding manager to always have this attribute.
        # For simple HF rollout that does not really shard, we keep an empty dict.
        self.timing: dict = {}

    def __enter__(self):
        # No special pre-processing for the default manager, but we still
        # return `self` so that advanced usages like `with manager as m:` work.
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def preprocess_data(self, data: DataProto) -> DataProto:
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        return data
