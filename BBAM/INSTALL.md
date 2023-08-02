## Installation

### Requirements:
- torch <= 1.6
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9
- OpenCV
- CUDA >= 9.0


### Step-by-step installation

```bash

pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# maskrcnn_benchmark and coco api dependencies
pip install ninja yacs cython matplotlib tqdm opencv-python

# install pycocotools
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install
cd ..
cd ..

# install cityscapesScripts
git clone https://github.com/mcordts/cityscapesScripts.git
cd cityscapesScripts/
python setup.py build_ext install
cd ..

# install apex
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout apex_no_distributed
pip install -v --no-cache-dir ./
cd ..


# install BBAM
python setup.py build develop
```
