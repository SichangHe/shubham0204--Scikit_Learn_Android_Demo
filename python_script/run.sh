#!/usr/bin/env bash

python3 sklearn_model.py
python3 -m onnxruntime.tools.convert_onnx_models_to_ort sklearn_model.onnx
