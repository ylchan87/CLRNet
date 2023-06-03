# Expected to be called from the CLRNet home folder i.e.
# cd CLRNet
# source weights/gen_trt_engine.bash

input=weights/llamas_dla34_dynamic_batch.onnx
output=weights/llamas_dla34_dynamic_batch.trt

echo "Generating TensorRT engine from ${input}"
python3 deploy/tensorrt/onnx2trt.py configs/clrnet/clr_dla34_llamas.py  --model ${input}  --trt_file ${output}
