"""LLM-Server converts LLM to TensorRT-LLM engines and hosts them with Triton Server."""
import argparse
import os

def main(args):
    #TODO: connect to Google cloud storage to load model checkpoint
    model = Model()
    inference_server = ModelServer(model, args.http)
    return inference_server.run()
