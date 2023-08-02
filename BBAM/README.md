# Installation

## Requirements:
- torch <= 1.6
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9
- OpenCV
- CUDA >= 9.0 


## Step-by-step installation

```bash
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install git+https://github.com/lucasb-eyer/pydensecrf.git

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


# Examples

#### Create pseudo ground-truth mask

- Obtain BBAMs for training images
```bash
python tools/BBAM/BBAM_training_images_multi.py --ckpt $path_model --config-file config.yml
```

- Create COCO style annotation files
```bash
python tools/BBAM/make_annotation/make_cocoann_topCRF_parallel.py
```

- Create semantic segmentation annotations
```bash
python tools/BBAM/make_annotation/create_semantic_labels.py
```