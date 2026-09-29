from __future__ import annotations

import contextlib
import io
import os
import tempfile
import warnings
from pathlib import Path
from typing import Any

import onnx
import torch
from torchvision.models import resnet18


class ONNXExporter:
    def export(self, ctx: Any, checkpoint: Path) -> list[Path]:
        variant = ctx.cfg["variant"]
        dataset = variant["dataset"]
        infer = variant["infer"]
        checkpoint_document = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        model = resnet18(
            weights=None,
            num_classes=len(dataset["classes"]),
        )
        model.load_state_dict(checkpoint_document["model_state"])
        model.eval()
        example = torch.zeros(tuple(infer["input_shape"]), dtype=torch.float32)
        destination = ctx.paths.export / "model.onnx"
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=".model.",
            suffix=".onnx",
            dir=ctx.paths.export,
        )
        os.close(file_descriptor)
        temporary = Path(temporary_name)
        try:
            diagnostics = io.StringIO()
            with (
                contextlib.redirect_stdout(diagnostics),
                contextlib.redirect_stderr(diagnostics),
                warnings.catch_warnings(),
            ):
                warnings.simplefilter("ignore", FutureWarning)
                torch.onnx.export(
                    model,
                    (example,),
                    temporary,
                    input_names=[infer["input_name"]],
                    output_names=[infer["output_name"]],
                    opset_version=infer["opset_version"],
                    dynamo=True,
                    external_data=False,
                )
            document = onnx.load(temporary, load_external_data=False)
            onnx.checker.check_model(document)
            _validate_signature(document, infer)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            return [destination]
        finally:
            temporary.unlink(missing_ok=True)


def _validate_signature(document: Any, infer: Any) -> None:
    if [entry.version for entry in document.opset_import if entry.domain == ""] != [
        infer["opset_version"]
    ]:
        raise ValueError("ONNX opset is invalid")
    if len(document.graph.input) != 1 or len(document.graph.output) != 1:
        raise ValueError("ONNX graph signature is invalid")
    input_value = document.graph.input[0]
    output_value = document.graph.output[0]
    if (
        input_value.name != infer["input_name"]
        or output_value.name != infer["output_name"]
    ):
        raise ValueError("ONNX graph names are invalid")
    dimensions = input_value.type.tensor_type.shape.dim
    shape = tuple(dimension.dim_value for dimension in dimensions)
    if shape != tuple(infer["input_shape"]):
        raise ValueError("ONNX input shape is invalid")
