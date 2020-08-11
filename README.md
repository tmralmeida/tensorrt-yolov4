# tensorrt-yolov4
 Full pipeline from model training to deployment on Nvidia AGX Xavier. This is example is for BDD100k training dataset.


1 - Train model through: [bag-of-models repo](https://github.com/tmralmeida/bag-of-models/tree/master/CNNs). Then, Convert the final model to .weights file;

2 - .weights file in "yolo" directory and run from the root directory: 

```
python3 yolo_to_onnx.py --model yolov4-bdd-512
```

3 - From the root directory run:

```
python3 onnx_to_tensorrt.py --model yolov4-bdd-512
```

4 - At this point we have on "yolo" directory: .cfg, .weights, .onnx and .trt files. Finally to run the application, from the "scripts" directory: 

```
python3 run_yolo.py --model yolov4-bdd-512 --category_num 10 --video_dev 0
```


Acknowledgments

https://github.com/jkjung-avt/tensorrt_demos