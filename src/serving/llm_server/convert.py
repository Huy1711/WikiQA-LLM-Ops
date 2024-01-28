from .conversion import ConversionOptions, convert

def main(args: argparse.Namespace) -> int:
    
    model = Model(model_type=args.type, world_size=args.world_size)
    
    conversion_opts = ConversionOptions(
        max_input_length=args.max_input_length,
        max_output_length=args.max_output_length,
        tensor_parallelism=args.tensor_parallelism,
        pipline_parallelism=args.pipeline_parallelism,
        quantization = args.quantization,
    )

    convert(model, conversion_opts)