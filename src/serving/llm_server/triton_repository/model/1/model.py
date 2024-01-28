import app
import json
import triton_python_backend_utils as pb_utils
import numpy as np
from transformers import pipeline


class TritonPythonModel:
    def initialize(self, args):
        self.generator = pipeline("text-generation", model="vilm/vinallama-2.7b-chat")

    def execute(self, requests):
        responses = []
        for request in requests:
            input = pb_utils.get_input_tensor_by_name(request, "prompt")
            input_string = input.as_numpy()[0].decode()
            
            pipeline_output = self.generator(input_string, do_sample=True, min_length=50)
            generated_txt = pipeline_output[0]["generated_text"]
            output = generated_txt
                            
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "generated_text",
                        np.array([output.encode()]),
                    )
                ]
            )
            responses.append(inference_response)
	    
        return responses

    def finalize(self, args):
         self.generator = None