#!/bin/bash
# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


DATA_PATH="$1"
evalset="$2"
hmr_type="$3"
MogeSAM=/data3/zihanwa3/_Robotics/_vision/TAPIP3D/_raw_mega_priors


for seq in ${evalset[@]}; do
  python cvd_opt/post_process.py --scene_name $seq --input_dir $DATA_PATH --hmr_type $hmr_type --output_dir $MogeSAM
done

