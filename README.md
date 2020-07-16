# Computer Pointer Controller

_TODO:_ Write a short introduction to your project

## Project Set Up and Installation

_TODO:_ Explain the setup procedures to run your project. For instance, this can include your project directory structure, the models you need to download and where to place them etc. Also include details about how to install the dependencies your project requires.

### Downloading models

```
python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name face-detection-adas-binary-0001 --output_dir '/media/psf/submissons/nd131-computer-pointer-controller/models'

python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name head-pose-estimation-adas-0001 --output_dir '/media/psf/submissons/nd131-computer-pointer-controller/models'

python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name landmarks-regression-retail-0009 --output_dir '/media/psf/submissons/nd131-computer-pointer-controller/models'

python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name gaze-estimation-adas-0002 --output_dir '/media/psf/submissons/nd131-computer-pointer-controller/models'
```

### Installing dependencies

```
python3 -m pip install -r requirements.txt
```

## Demo

```python

python3 src/main.py  --model_name models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001 \
                          --device CPU \
                          --video bin/demo.mp4

```

## Documentation

_TODO:_ Include any documentation that users might need to better understand your project code. For instance, this is a good place to explain the command line arguments that your project supports.

## Benchmarks

_TODO:_ Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

## Results

_TODO:_ Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Stand Out Suggestions

This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference

If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases

There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.
