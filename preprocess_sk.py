# Copyright 2024 - Valeo Comfort and Driving Assistance - valeo.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from tqdm import tqdm
from multiprocessing import Pool
from utils.utils import get_cpu_limit
from preprocess.SemanticKITTIPreprocess import KITTIPreprocessor

if __name__ == "__main__":
    SCAN_WINDOW = 40  # Number of frames to aggregate in the clusterization
    DOWNSAMPLING_RESOLUTION = [0.05,0.05,0.05,5]
    GROUND_METHOD = "patchworkpp"
    ds = KITTIPreprocessor(data_dir="datasets/semantickitti/",
                           scan_window=SCAN_WINDOW,
                           split='trainval',
                           downsampling_resolution=DOWNSAMPLING_RESOLUTION,
                           ground_method=GROUND_METHOD)
    num_cpus = get_cpu_limit()
    print(f"Using {num_cpus} threads")
    with Pool(num_cpus) as p:
        list(tqdm(p.imap(ds.__getitem__, range(len(ds))), total=len(ds)))
