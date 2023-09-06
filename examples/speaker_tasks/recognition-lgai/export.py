import argparse
import os
import sys

sys.path.append(os.path.dirname(__file__))

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.core.module import _jit_is_scripting
from speaker_reco import LgaiSpeakerModel

from nemo.core import ModelPT
from nemo.core.classes import Exportable
from nemo.core.config.pytorch_lightning import TrainerConfig
from nemo.utils import logging

try:
    from contextlib import nullcontext
except ImportError:
    # handle python < 3.7
    from contextlib import suppress as nullcontext

try:
    torch.set_float32_matmul_precision("highest")
except AttributeError:
    pass


def get_args(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=f"Export NeMo models to ONNX/Torchscript",
    )
    parser.add_argument("source", help="Source .nemo file")
    parser.add_argument("out", help="Location to write result to")
    parser.add_argument("--checkpoint")
    parser.add_argument("--autocast", action="store_true", help="Use autocast when exporting")
    parser.add_argument("--runtime-check", action="store_true", help="Runtime check of exported net result")
    parser.add_argument("--verbose", default=None, help="Verbose level for logging, numeric")
    parser.add_argument("--max-batch", type=int, default=256, help="Max batch size for model export")
    parser.add_argument("--max-dim", type=int, default=48000, help="Max dimension(s) for model export")
    parser.add_argument("--min-length", type=int, default=8000, help="Max dimension(s) for model export")
    parser.add_argument("--onnx-opset", type=int, default=None, help="ONNX opset for model export")
    parser.add_argument(
        "--cache_support", action="store_true", help="enables caching inputs for the models support it."
    )
    parser.add_argument("--device", default="cuda", help="Device to export for")
    parser.add_argument("--check-tolerance", type=float, default=0.01, help="tolerance for verification")
    parser.add_argument(
        "--export-config",
        metavar="KEY=VALUE",
        nargs='+',
        help="Set a number of key-value pairs to model.export_config dictionary "
        "(do not put spaces before or after the = sign). "
        "Note that values are always treated as strings.",
    )
    parser.add_argument("--tensorrt", action="store_true", help="Use tensorrt runtime when exporting")

    args = parser.parse_args(argv)
    return args


def nemo_export(argv):
    args = get_args(argv)
    loglevel = logging.INFO
    # assuming loglevel is bound to the string value obtained from the
    # command line argument. Convert to upper case to allow the user to
    # specify --log=DEBUG or --log=debug
    if args.verbose is not None:
        numeric_level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError('Invalid log level: %s' % numeric_level)
        loglevel = numeric_level
    logging.setLevel(loglevel)
    logging.info("Logging level set to {}".format(loglevel))

    """Convert a .nemo saved model into .riva Riva input format."""
    nemo_in = args.source
    out = args.out

    # Create a PL trainer object which is required for restoring Megatron models
    cfg_trainer = TrainerConfig(
        accelerator="cuda",
        strategy="ddp",
        num_nodes=1,
        devices=1,
        # Need to set the following two to False as ExpManager will take care of them differently.
        logger=False,
        enable_checkpointing=False,
    )
    trainer = Trainer(**cfg_trainer.__dict__)

    logging.info("Restoring NeMo model from '{}'".format(nemo_in))
    try:
        with torch.inference_mode():
            # Restore instance from .nemo file using generic model restore_from
            model = ModelPT.restore_from(restore_path=nemo_in, trainer=trainer)
            if args.checkpoint is not None:
                model = model.load_from_checkpoint(args.checkpoint)
    except Exception as e:
        logging.error(
            "Failed to restore model from NeMo file : {}. Please make sure you have the latest NeMo package installed with [all] dependencies.".format(
                nemo_in
            )
        )
        raise e

    logging.info("Model {} restored from '{}'".format(model.__class__.__name__, nemo_in))

    if not isinstance(model, Exportable):
        logging.error("Your NeMo model class ({}) is not Exportable.".format(model.__class__.__name__))
        sys.exit(1)

    #
    #  Add custom export parameters here
    #
    check_trace = args.runtime_check

    in_args = {}
    if args.max_batch is not None:
        in_args["max_batch"] = args.max_batch
    if args.max_dim is not None:
        in_args["max_dim"] = args.max_dim
    if args.min_length is not None:
        in_args["min_length"] = args.min_length

    if args.cache_support:
        model.set_export_config({"cache_support": "True"})

    if args.export_config:
        kv = {}
        for key_value in args.export_config:
            lst = key_value.split("=")
            if len(lst) != 2:
                raise Exception("Use correct format for --export_config: k=v")
            k, v = lst
            kv[k] = v
        model.set_export_config(kv)

    autocast = nullcontext
    if args.autocast:
        autocast = torch.cuda.amp.autocast
    try:
        with autocast(), torch.inference_mode():
            model.to(device=args.device).freeze()
            model.eval()
            input_example = None

            # input_example = model.input_module.input_example(**in_args)
            # input_example = [x.to(model.device) for x in input_example]
            # # check_trace = [input_example]
            # for key, arg in in_args.items():
            #     in_args[key] = (arg + 1) // 2
            # input_example2 = model.input_module.input_example(**in_args)
            # # check_trace.append([x.to(model.device) for x in input_example2])
            # logging.info(f"Using additional check args: {in_args}")

            _, descriptions = model.export(
                out,
                input_example=input_example,
                check_trace=check_trace,
                check_tolerance=args.check_tolerance,
                onnx_opset_version=args.onnx_opset,
                verbose=bool(args.verbose),
            )

            traced_titanet = torch.jit.load(out)

            if args.tensorrt:
                import torch_tensorrt

                trt_model = torch_tensorrt.compile(
                    traced_titanet,
                    inputs=[
                        torch_tensorrt.Input(
                            min_shape=(args.max_batch, 80, 50),
                            opt_shape=(args.max_batch, 80, 200),
                            max_shape=(args.max_batch, 80, 300),
                            dtype=torch.float32,
                        ),
                        torch_tensorrt.Input(
                            min_shape=(args.max_batch,),
                            opt_shape=(args.max_batch,),
                            max_shape=(args.max_batch,),
                            dtype=torch.int32,
                        ),
                    ],
                    enabled_precisions={torch.half},
                )

                benchmark(traced_titanet, input_shape=[(args.max_batch, 80, 300), (args.max_batch,)], nruns=100)
                benchmark(
                    trt_model, input_shape=[(args.max_batch, 80, 300), (args.max_batch,)], nruns=100, dtype="fp16"
                )
                traced_titanet = trt_model

            with torch.jit.optimized_execution(True), _jit_is_scripting():
                input_example = model.preprocessor.input_example(**in_args)
                input_example = [x.cuda() for x in input_example]
                input_example = {"input_signal": input_example[0], "length": input_example[1]}
                preprocessor = model.preprocessor.featurizer.eval()
                preprocessor(**input_example)
                scripted_preprocessor = torch.jit.script(preprocessor)
                scripted_preprocessor.eval()
                x = scripted_preprocessor(**input_example)

                titanet_ = TitanetScriptable(scripted_preprocessor, traced_titanet)
                titanet_script = torch.jit.script(titanet_)
                titanet_script.eval()
                titanet_script = torch.jit.freeze(titanet_script)
                titanet_script.save(out)

    except Exception as e:
        logging.error(
            "Export failed. Please make sure your NeMo model class ({}) has working export() and that you have the latest NeMo package installed with [all] dependencies.".format(
                model.__class__
            )
        )
        raise e


class TitanetScriptable(torch.nn.Module):
    def __init__(self, preprocessor, traced_titanet):
        super().__init__()
        self.preprocessor = preprocessor
        self.traced_titanet = traced_titanet

    def forward(self, input_signal, length):
        processed_signal, processed_signal_len = self.preprocessor(input_signal=input_signal, length=length,)
        logits, embs = self.traced_titanet(processed_signal, processed_signal_len)
        return logits, embs


import time
import numpy as np
import torch.backends.cudnn as cudnn

cudnn.benchmark = True


def benchmark(model, input_shape=(1024, 3, 512, 512), dtype='fp32', nwarmup=50, nruns=1000):
    input_data = torch.randn(input_shape)
    input_data = input_data.to("cuda")
    if dtype == 'fp16':
        input_data = input_data.half()

    print("Warm up ...")
    with torch.no_grad():
        for _ in range(nwarmup):
            features = model(input_data)
    torch.cuda.synchronize()
    print("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in range(1, nruns + 1):
            start_time = time.time()
            pred_loc = model(input_data)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if i % 10 == 0:
                print('Iteration %d/%d, avg batch time %.2f ms' % (i, nruns, np.mean(timings) * 1000))

    print("Input shape:", input_data.size())


if __name__ == '__main__':
    nemo_export(sys.argv[1:])
