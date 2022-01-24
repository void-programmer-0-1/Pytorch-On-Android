
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
from model import NeuralNetwork

def main():
	
	model_input = torch.rand(1,1)
	
	model = NeuralNetwork()
	model.load_state_dict(torch.load("linear_regression.pt"))

	traced_model = torch.jit.script(model)
	torch.jit.save(traced_model, "scripted_linear_regression.pt")
	
	optimized_model = optimize_for_mobile(traced_model)
	optimized_model._save_for_lite_interpreter("linear_regression.ptl")
	

def mobile():

	model_input = torch.rand(1,1)

	model = NeuralNetwork()
	model.load_state_dict(torch.load("linear_regression.pt"))
	
	quantized_model = torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
	scripted_model = torch.jit.trace(model,model_input)
	optimized_model = optimize_for_mobile(scripted_model)
	optimized_model._save_for_lite_interpreter("linear_regression.ptl")
	

def mobile2():

	model_input = torch.rand(1,1)

	model = NeuralNetwork()
	model.load_state_dict(torch.load("linear_regression.pt"))
	
	quantized_model = torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
	scripted_model = torch.jit.script(model)
	optimized_model = optimize_for_mobile(scripted_model)
	optimized_model._save_for_lite_interpreter("linear_regression.ptl")

if __name__ == '__main__':
	mobile2()










