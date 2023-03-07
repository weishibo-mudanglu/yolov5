'''
该脚本功能为通过onnx-tf将onnx模型转换为tensorflow模型
问题:tensorflow-1.12.0需要CUDA9
'''
import onnx
from onnx_tf.backend import prepare
onnx_model = onnx.load("test.onnx")  # load onnx model
tf_rep = prepare(onnx_model )
tf_rep.export_graph("test.pb")