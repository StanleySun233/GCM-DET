# install

1. clone this repo
```shell
git clone https://github.com/StanleySun233/yolov8.git
cd yolov8
```

2. note the environment:
* python=3.9
* cuda>=12.6
* torch=2.3.0

```shell
# python -m pip install --upgrade pip
# pip install torch===2.3.0 torchvision torchaudio
pip install seaborn thop timm einops opencv-python scipy
cd selective_scan
pip install .
cd ..
```

```shell
!python -c "import torch; print(torch.cuda.is_available())"
```

3. run test demo for mamba
```shell
python mbyolo_train.py --task train --amp \
  --data ultralytics/cfg/datasets/coco8.yaml \
  --config ultralytics/cfg/models/v8/mamba-yolo/Mamba-YOLO-B.yaml \
  --project ./output_dir/test/mbyolo_coco8_test \
  --name mambayolo_n \
  --epoch 1 \
  --batch_size 16
```

```shell
python mbyolo_train.py --task train --amp \
  --data /home/sijin/datasets/cd5-det/data.yaml \
  --config ultralytics/cfg/models/v8/mamba-yolo/Mamba-YOLO-GCM.yaml \
  --project ./output_dir/test/cd5-det \
  --name mambayolo_n \
  --epoch 100 \
  --batch_size 8
```

4. TASKS

* yolov8+mambaB+carafe, container-damage-detection
```shell
python mbyolo_train.py --task train --amp \
  --data /home/sijin/datasets/cd5-det/data.yaml \
  --config ultralytics/cfg/models/v8/mamba-yolo/Mamba-YOLO-B-carafe.yaml \
  --project ./output_dir/test/mbyolo_coco8_test \
  --name mambayolo_cdt \
  --epoch 300 \
  --batch_size 8 # epoch=16时占用21.9G, but process is killed at 3rd train???
```

* yolov8+mambaB, container-damage-detection
```shell
python mbyolo_train.py --task train --amp \
  --data /workspace/container/data.yaml \
  --config ultralytics/cfg/models/v8/mamba-yolo/Mamba-YOLO-B.yaml \
  --project ./output_dir/test/mbyolo_coco8_test \
  --name mambayolo_cdt \
  --epoch 300 \
  --batch_size 8 
```

* yolov8+mambaB+carafe, neu-det
```shell
python mbyolo_train.py --task train --amp \
  --data /workspace/neu-det/data.yaml \
  --config ultralytics/cfg/models/v8/mamba-yolo/Mamba-YOLO-B-carafe.yaml \
  --project ./output_dir/test/mbyolo_coco8_test \
  --name mambayolo_cdt \
  --epoch 300 \
  --batch_size 8 
```

* yolov8+carafe, neu-det
```shell
python mbyolo_train.py --task train --amp \
  --data /workspace/neu-det/data.yaml \
  --config ultralytics/cfg/models/v8/yolov8-carafe.yaml \
  --project ./output_dir/test/mbyolo_coco8_test \
  --name mambayolo_cdt \
  --epoch 300 \
  --batch_size 16 
```

* yolov8+mamba, neu-det
```shell
python mbyolo_train.py --task train --amp \
  --data /workspace/neu-det/data.yaml \
  --config ultralytics/cfg/models/v8/mamba-yolo/Mamba-YOLO-B.yaml \
  --project ./output_dir/test/mbyolo_coco8_test \
  --name mambayolo_cdt \
  --epoch 300 \
  --batch_size 16 
```

* yolov8+mambaB+carafe, gc10-det
```shell
python mbyolo_train.py --task train --amp \
  --data /workspace/gc10-det/data.yaml \
  --config ultralytics/cfg/models/v8/mamba-yolo/Mamba-YOLO-B-carafe.yaml \
  --project ./output_dir/test/mbyolo_coco8_test \
  --name mambayolo_cdt \
  --epoch 300 \
  --batch_size 16
```

* yolov8+mambaB, gc10-det
```shell
python mbyolo_train.py --task train --amp \
  --data /workspace/gc10-det/data.yaml \
  --config ultralytics/cfg/models/v8/mamba-yolo/Mamba-YOLO-B.yaml \
  --project ./output_dir/test/mbyolo_coco8_test \
  --name mambayolo_cdt \
  --epoch 300 \
  --batch_size 16
```

* yolov8+carafe, gc10-det
```shell
python mbyolo_train.py --task train --amp \
  --data /workspace/gc10-det/data.yaml \
  --config ultralytics/cfg/models/v8/yolov8-carafe.yaml \
  --project ./output_dir/test/mbyolo_coco8_test \
  --name mambayolo_cdt \
  --epoch 300 \
  --batch_size 16
```

* yolov8+mambaB+carafe, cr7-det
```shell
python mbyolo_train.py --task train --amp \
  --data /workspace/cr7-det/data.yaml \
  --config ultralytics/cfg/models/v8/mamba-yolo/Mamba-YOLO-B-carafe.yaml \
  --project ./output_dir/test/mbyolo_coco8_test \
  --name mambayolo_cdt \
  --epoch 300 \
  --batch_size 16
```

* yolov8+mambaB, cr7-det
```shell
python mbyolo_train.py --task train --amp \
  --data /workspace/cr7-det/data.yaml \
  --config ultralytics/cfg/models/v8/mamba-yolo/Mamba-YOLO-B.yaml \
  --project ./output_dir/test/mbyolo_coco8_test \
  --name mambayolo_cdt \
  --epoch 300 \
  --batch_size 16
```

* yolov8+carafe, cr7-det
```shell
python mbyolo_train.py --task train --amp \
  --data /workspace/cr7-det/data.yaml \
  --config ultralytics/cfg/models/v8/yolov8-carafe.yaml \
  --project ./output_dir/test/mbyolo_coco8_test \
  --name mambayolo_cdt \
  --epoch 300 \
  --batch_size 8
```

* yolov8+MetaGamma, coco8
```shell
python mbyolo_train.py --task train --amp \
  --data ultralytics/cfg/datasets/coco8.yaml \
  --config ultralytics/cfg/models/v8/yolov8-mg.yaml \
  --project ./output_dir/test/mbyolo_coco8_test \
  --name mambayolo_cdt \
  --epoch 1 \
  --batch_size 8
```

* yolov8+MetaGamma, neu-det
```shell
python mbyolo_train.py --task train --amp \
  --data /workspace/neu-det/data.yaml \
  --config ultralytics/cfg/models/v8/yolov8-mg.yaml \
  --project ./output_dir/test/mbyolo_coco8_test \
  --name mambayolo_cdt \
  --epoch 300 \
  --batch_size 16 
```

5. DATASETS


# where it from
* yolov8 forked from [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
* mamba-yolo forked from [HZAI-ZJNU/Mamba-YOLO](https://github.com/HZAI-ZJNU/Mamba-YOLO)
* carafe forked from [XiaLiPKU/CARAFE](https://github.com/XiaLiPKU/CARAFE)