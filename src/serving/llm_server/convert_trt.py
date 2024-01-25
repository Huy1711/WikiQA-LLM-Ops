"""This module contains the logic for exporting a PyTorch format LLM to TensorRT."""

import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_trt
from tensorrt_llm.builder import Builder
from tensorrt_llm.layers.attention import PositionEmbeddingType
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

def build_engine(
    builder: Builder,
    builder_config: tensorrt_llm.builder.BuilderConfig,
    engine_name,
    rank,
    args,
):
    """Build the engine.

    Parameters
    ----------
    args: 
        The cmd line arguments.
    
    Returns
    -------
    engine
        The built engine.
    """

    dtype = str_dtype_to_trt(args.dtype)
    mapping = Mapping(
        world_size=args.world_size,
        rank=rank,
        tp_size=args.tp_size,
        pp_size=args.pp_size,
    )

    assert (
        args.n_layer % args.pp_size == 0
    ), f"num_layers {args.n_layer} must be a multiple of pipeline parallelism size {args.pp_size}"

    # Initialize Module
    tensorrt_llm_llama = tensorrt_llm.models.LLaMAForCausalLM(
        num_layers=args.n_layer,
        num_heads=args.n_head,
        num_kv_heads=args.n_kv_head,
        hidden_size=args.n_embd,
        vocab_size=args.vocab_size,
        hidden_act=args.hidden_act,
        max_position_embeddings=args.n_positions,
        dtype=dtype,
        mlp_hidden_size=args.inter_size,
        position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
        mapping=mapping,
        rotary_base=args.rotary_base,
        rotary_scaling=args.rotary_scaling,
        use_parallel_embedding=args.use_parallel_embedding,
        embedding_sharding_dim=args.embedding_sharding_dim,
        quant_mode=args.quant_mode,
        rms_norm_eps=args.rms_norm_eps,
    )
    if args.use_smooth_quant:
        tensorrt_llm_llama = smooth_quantize(tensorrt_llm_llama, args.quant_mode)
    elif args.use_weight_only:
        if args.weight_only_precision == "int8":
            tensorrt_llm_llama = weight_only_quantize(
                tensorrt_llm_llama, args.quant_mode
            )
        elif args.weight_only_precision == "int4":
            tensorrt_llm_llama = weight_only_quantize(
                tensorrt_llm_llama, args.quant_mode
            )
        elif args.weight_only_precision == "int4_awq":
            tensorrt_llm_llama = weight_only_groupwise_quantize(
                model=tensorrt_llm_llama,
                quant_mode=args.quant_mode,
                group_size=args.group_size,
                zero=False,
                pre_quant_scale=True,
                exclude_modules=[],
            )
        elif args.weight_only_precision == "int4_gptq":
            tensorrt_llm_llama = weight_only_groupwise_quantize(
                model=tensorrt_llm_llama,
                quant_mode=args.quant_mode,
                group_size=args.group_size,
                zero=True,
                pre_quant_scale=False,
            )
    elif args.enable_fp8 or args.fp8_kv_cache:
        logger.info(f"Loading scaling factors from " f"{args.quantized_fp8_model_path}")
        quant_scales = get_scaling_factors(
            args.quantized_fp8_model_path,
            num_layers=args.n_layer,
            quant_mode=args.quant_mode,
        )
        tensorrt_llm_llama = fp8_quantize(
            tensorrt_llm_llama, quant_mode=args.quant_mode, quant_scales=quant_scales
        )
    if args.per_group:
        load_func = (
            load_from_awq_llama
            if args.weight_only_precision == "int4_awq"
            else load_from_gptq_llama
        )
        load_func(
            tensorrt_llm_llama=tensorrt_llm_llama,
            quant_ckpt_path=args.quant_ckpt_path,
            mapping=mapping,
            dtype=args.dtype,
        )
    elif args.meta_ckpt_dir is not None:
        load_from_meta_llama(
            tensorrt_llm_llama, args.meta_ckpt_dir, mapping, args.dtype
        )
    elif args.model_dir is not None:
        logger.info(f"Loading HF LLaMA ... from {args.model_dir}")
        tik = time.time()
        hf_llama = LlamaForCausalLM.from_pretrained(
            args.model_dir,
            device_map={"model": "cpu", "lm_head": "cpu"},  # Load to CPU memory
            torch_dtype="auto",
        )
        tok = time.time()
        t = time.strftime("%H:%M:%S", time.gmtime(tok - tik))
        logger.info(f"HF LLaMA loaded. Total time: {t}")
        load_from_hf_llama(
            tensorrt_llm_llama, hf_llama, mapping=mapping, dtype=args.dtype
        )
        del hf_llama
    elif args.ft_model_dir is not None:
        load_from_binary(
            tensorrt_llm_llama,
            args.ft_model_dir,
            mapping,
            fp16=(args.dtype == "float16"),
            multi_query_mode=(args.n_kv_head != args.n_head),
        )

    # Module -> Network
    network = builder.create_network()
    network.trt_network.name = engine_name
    if args.use_gpt_attention_plugin:
        network.plugin_config.set_gpt_attention_plugin(
            dtype=args.use_gpt_attention_plugin
        )
    if args.use_gemm_plugin:
        network.plugin_config.set_gemm_plugin(dtype=args.use_gemm_plugin)
    if args.use_rmsnorm_plugin:
        network.plugin_config.set_rmsnorm_plugin(dtype=args.use_rmsnorm_plugin)

    # Quantization plugins.
    if args.use_smooth_quant:
        network.plugin_config.set_smooth_quant_gemm_plugin(dtype=args.dtype)
        network.plugin_config.set_rmsnorm_quantization_plugin(dtype=args.dtype)
        network.plugin_config.set_quantize_tensor_plugin()
        network.plugin_config.set_quantize_per_token_plugin()
    assert not (args.enable_context_fmha and args.enable_context_fmha_fp32_acc)
    if args.enable_context_fmha:
        network.plugin_config.set_context_fmha(ContextFMHAType.enabled)
    if args.enable_context_fmha_fp32_acc:
        network.plugin_config.set_context_fmha(ContextFMHAType.enabled_with_fp32_acc)
    if args.use_weight_only:
        if args.per_group:
            network.plugin_config.set_weight_only_groupwise_quant_matmul_plugin(
                dtype="float16"
            )
        else:
            network.plugin_config.set_weight_only_quant_matmul_plugin(dtype="float16")
    if args.world_size > 1:
        network.plugin_config.set_nccl_plugin(args.dtype, args.use_custom_all_reduce)
    if args.remove_input_padding:
        network.plugin_config.enable_remove_input_padding()
    if args.paged_kv_cache:
        network.plugin_config.enable_paged_kv_cache(args.tokens_per_block)

    with net_guard(network):
        # Prepare
        network.set_named_parameters(tensorrt_llm_llama.named_parameters())

        # Forward
        inputs = tensorrt_llm_llama.prepare_inputs(
            args.max_batch_size,
            args.max_input_len,
            args.max_output_len,
            True,
            args.max_beam_width,
            args.max_num_tokens,
        )
        tensorrt_llm_llama(*inputs)
        if args.enable_debug_output:
            # mark intermediate nodes' outputs
            for k, v in tensorrt_llm_llama.named_network_outputs():
                v = v.trt_tensor
                v.name = k
                network.trt_network.mark_output(v)
                v.dtype = dtype
 

    tensorrt_llm.graph_rewriting.optimize(network)

    engine = None

    # Network -> Engine
    engine = builder.build_engine(network, builder_config)
    if rank == 0:
        config_path = os.path.join(args.output_dir, "config.json")
        builder.save_config(builder_config, config_path)
    return engine


if __name__ == "__main__":
    builder = Builder()

    builder_config = builder.create_builder_config(
                name=MODEL_NAME,
                precision=args.dtype,
                timing_cache=args.timing_cache if cache is None else cache,
                tensor_parallel=args.tp_size,
                pipeline_parallel=args.pp_size,
                parallel_build=args.parallel_build,
                num_layers=args.n_layer,
                num_heads=args.n_head,
                num_kv_heads=args.n_kv_head,
                hidden_size=args.n_embd,
                vocab_size=args.vocab_size,
                hidden_act=args.hidden_act,
                max_position_embeddings=args.n_positions,
                max_batch_size=args.max_batch_size,
                max_input_len=args.max_input_len,
                max_output_len=args.max_output_len,
                max_num_tokens=args.max_num_tokens,
                int8=int8_trt_flag,
                fp8=args.quant_mode.has_fp8_qdq(),
                quant_mode=args.quant_mode,
                strongly_typed=args.strongly_typed,
                opt_level=args.builder_opt,
            )

    engine_name = get_engine_name(
                MODEL_NAME, args.dtype, args.tp_size, args.pp_size, cur_rank
            )

    engine = build_engine(builder, builder_config, engine_name, cur_rank, args)