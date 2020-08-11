# tensorrt-yolov4
 Full pipeline from model training to deployment on Nvidia AGX Xavier. This is example is for BDD100k training dataset.


1 - Train model through: [bag-of-models repo](https://github.com/tmralmeida/bag-of-models/tree/master/CNNs/2-Object_Detection). Then, Convert the final model to .weights file;

2 - .weights file in "yolo directory" and run: python3 yolo_to_onnx.py --model yolov4-bdd-512;

3 - python3 onnx_to_tensorrt.py --model yolov4-bdd-512;

4 - from "scripts" directory run: python3 run_yolo.py --model yolov4-bdd-512 --category_num 10 --video_dev 0;



Acknowledgments

https://github.com/jkjung-avt/tensorrt_demos