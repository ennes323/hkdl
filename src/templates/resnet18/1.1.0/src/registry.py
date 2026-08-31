from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18

from export import ONNXExporter

TRAIN_COMPONENTS = {
    "model",
    "loss",
    "optimizer",
    "dataloader",
    "trainer",
}
EVAL_COMPONENTS = {
    "model",
    "dataloader",
    "evaluator",
}
EXPORT_COMPONENTS = {"exporter"}
SOURCE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = SOURCE_ROOT / "data"
CLASSES = ("daisy", "sunflowers")
SPLIT_COUNTS = {"train": 16, "eval": 8}
MAX_DATASET_BYTES = 1 << 20
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)
MANIFEST_SOURCE = {
    "title": "TF-Flowers",
    "author": "Ye Xu",
    "doi": "10.6084/m9.figshare.19166516.v1",
    "url": "https://figshare.com/articles/dataset/TF-Flowers/19166516",
    "license": "CC BY 4.0",
    "license_url": "https://creativecommons.org/licenses/by/4.0/",
    "archive": "TF-Flowers.zip",
    "archive_md5": "4dc888d4b39b63404feed35ce61e75c0",
    "selection": (
        "first 24 JPEG names per class in UTF-8 byte order; "
        "first 16 train and next 8 eval"
    ),
}
MANIFEST_PROCESSING = {
    "mode": "RGB",
    "crop": "center-square",
    "size": [64, 64],
    "resize": "bilinear",
    "format": "JPEG",
    "quality": 90,
    "metadata": "removed",
}


class Trainer:
    def fit(self, ctx: Any) -> dict[str, Path]:
        train = ctx.cfg["variant"]["train"]
        dataset = ctx.cfg["variant"]["dataset"]
        torch.manual_seed(ctx.exec.seed)
        model = ctx.components["model"]().to(ctx.exec.device)
        loss_fn = ctx.components["loss"]()
        optimizer = ctx.components["optimizer"](model.parameters())
        batches = ctx.components["dataloader"](ctx.exec.seed)
        completed_steps = 0
        if ctx.resume_from is not None:
            checkpoint_document = torch.load(
                ctx.resume_from,
                map_location="cpu",
                weights_only=True,
            )
            completed_steps = _validate_resume_checkpoint(
                checkpoint_document,
                seed=ctx.exec.seed,
                classes=len(dataset["classes"]),
                maximum_steps=train["steps"],
            )
            model.load_state_dict(checkpoint_document["model_state"])
            optimizer.load_state_dict(checkpoint_document["optimizer_state"])
            _move_optimizer_state(optimizer, ctx.exec.device)
        model.train()
        for step, (inputs, targets) in enumerate(batches):
            if step >= train["steps"]:
                break
            if step < completed_steps:
                continue
            batch_started_at = time.perf_counter()
            inputs = inputs.to(ctx.exec.device)
            targets = targets.to(ctx.exec.device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(inputs), targets)
            loss.backward()
            optimizer.step()
            ctx.tracker.log_scalar("train.loss", float(loss.detach().cpu()), step)
            last = ctx.paths.checkpoints / "last.pt"
            checkpoint_document = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": step + 1,
                "seed": ctx.exec.seed,
                "classes": len(dataset["classes"]),
            }
            _atomic_torch_save(checkpoint_document, last)
            ctx.report_checkpoint(last)
            ctx.tracker.log_scalar(
                "train.batch_seconds",
                max(0.0, time.perf_counter() - batch_started_at),
                step,
            )
        if ctx.exec.device == "mps":
            torch.mps.synchronize()
        last = ctx.paths.checkpoints / "last.pt"
        if not last.is_file():
            raise ValueError("training produced no checkpoint")
        checkpoint_document = torch.load(last, map_location="cpu", weights_only=True)
        best = ctx.paths.checkpoints / "best.pt"
        _atomic_torch_save(checkpoint_document, best)
        return {"best_checkpoint": best, "last_checkpoint": last}


class Evaluator:
    def evaluate(self, ctx: Any, checkpoint: Path) -> dict[str, Any]:
        model = ctx.components["model"]().to(ctx.exec.device)
        checkpoint_document = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(checkpoint_document["model_state"])
        batches = ctx.components["dataloader"](ctx.exec.seed)
        model.eval()
        correct = 0
        total = 0
        predictions_document = []
        with torch.no_grad():
            for inputs, targets in batches:
                inputs = inputs.to(ctx.exec.device)
                targets = targets.to(ctx.exec.device)
                predictions = model(inputs).argmax(dim=1)
                correct += int((predictions == targets).sum().cpu())
                total += int(targets.numel())
                predictions_document.extend(
                    {
                        "target": int(target),
                        "prediction": int(prediction),
                    }
                    for target, prediction in zip(
                        targets.detach().cpu().tolist(),
                        predictions.detach().cpu().tolist(),
                        strict=True,
                    )
                )
        if ctx.exec.device == "mps":
            torch.mps.synchronize()
        result = ctx.paths.results / "predictions.json"
        _atomic_json_save(
            {
                "schema_version": 1,
                "evaluation_case": ctx.cfg["runtime"]["target"]["evaluation_case"],
                "predictions": predictions_document,
            },
            result,
        )
        return {
            "values": {"accuracy": correct / total},
            "files": [result],
        }


class Entrypoint:
    def registry(self):
        yield "model", "resnet18", resnet18
        yield "loss", "cross_entropy", nn.CrossEntropyLoss
        yield "optimizer", "sgd", torch.optim.SGD
        yield "dataloader", "imagefolder", _image_batches
        yield "trainer", "default", Trainer
        yield "evaluator", "default", Evaluator
        yield "exporter", "onnx", ONNXExporter

    def validate(self, action, cfg, selected, exec_info):
        if action == "train":
            if set(selected) != TRAIN_COMPONENTS:
                raise ValueError("invalid training selection")
        elif action == "eval":
            if set(selected) != EVAL_COMPONENTS:
                raise ValueError("invalid evaluation selection")
        elif action == "export":
            if set(selected) != EXPORT_COMPONENTS:
                raise ValueError("invalid export selection")
        else:
            raise ValueError("invalid action")
        variant = cfg["variant"]
        tracker = dict(variant["tracker"])
        if set(tracker) != {"backend"}:
            raise ValueError("tracker must contain only backend")
        backend = tracker["backend"]
        if isinstance(backend, str):
            valid_tracker = backend in {"none", "local", "mlflow"}
        else:
            valid_tracker = (
                isinstance(backend, tuple)
                and bool(backend)
                and all(
                    isinstance(item, str) and item in {"local", "mlflow"}
                    for item in backend
                )
                and len(backend) == len(set(backend))
            )
        if not valid_tracker:
            raise ValueError("invalid tracker.backend")
        dataset = variant["dataset"]
        _validate_dataset_bundle(dataset)
        if action == "train":
            _validate_train(variant["train"])
            _validate_split_coverage("train", variant["train"])
        elif action == "eval":
            evaluation = _evaluation_case(cfg)
            _validate_eval(evaluation)
            _validate_split_coverage(
                "eval",
                evaluation,
                selected_class=evaluation["selector"].get("class"),
            )
        else:
            _validate_infer(variant["infer"])
        seed = exec_info["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        if action == "train":
            identity = {
                "dataset": _dataset_identity(dataset),
                "train": dict(variant["train"]),
            }
        elif action == "eval":
            identity = {
                "dataset": _dataset_identity(dataset),
                "case": _plain_json(_evaluation_case(cfg)),
            }
        else:
            identity = {"infer": _plain_json(variant["infer"])}
        return {
            "exec": {
                "seed": seed,
                "device": _resolve_device(exec_info["device"]),
            },
            "identity": identity,
        }

    def assemble(self, action, cfg, selected):
        if action == "train" and set(selected) == TRAIN_COMPONENTS:
            return self._assemble_train(cfg)
        if action == "eval" and set(selected) == EVAL_COMPONENTS:
            return self._assemble_eval(cfg)
        if action == "export" and set(selected) == EXPORT_COMPONENTS:
            return {"exporter": ONNXExporter()}
        raise ValueError("invalid action selection")

    @staticmethod
    def _assemble_train(cfg):
        dataset = cfg["variant"]["dataset"]
        train = cfg["variant"]["train"]
        return {
            "model": lambda: resnet18(
                weights=None,
                num_classes=len(dataset["classes"]),
            ),
            "loss": nn.CrossEntropyLoss,
            "optimizer": lambda parameters: torch.optim.SGD(
                parameters,
                lr=float(train["learning_rate"]),
            ),
            "dataloader": lambda seed: _image_batches(
                dataset["train_dir"],
                seed,
                batch_size=train["batch_size"],
                image_size=tuple(dataset["image_size"]),
                classes=tuple(dataset["classes"]),
                shuffle=True,
            ),
            "trainer": Trainer(),
        }

    @staticmethod
    def _assemble_eval(cfg):
        dataset = cfg["variant"]["dataset"]
        evaluation = _evaluation_case(cfg)
        selected_class = evaluation["selector"].get("class")
        return {
            "model": lambda: resnet18(
                weights=None,
                num_classes=len(dataset["classes"]),
            ),
            "dataloader": lambda seed: _image_batches(
                dataset["eval_dir"],
                seed,
                batch_size=evaluation["batch_size"],
                image_size=tuple(dataset["image_size"]),
                classes=tuple(dataset["classes"]),
                shuffle=False,
                selected_class=selected_class,
            ),
            "evaluator": Evaluator(),
        }


def entrypoint() -> Entrypoint:
    return Entrypoint()


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        if sys.platform == "darwin" and torch.backends.mps.is_available():
            return "mps"
        if sys.platform.startswith("linux") and torch.cuda.is_available():
            return "cuda:0"
        return "cpu"
    if requested == "cpu":
        return "cpu"
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS is unavailable")
        return "mps"
    if requested == "cuda":
        requested = "cuda:0"
    if requested.startswith("cuda:"):
        try:
            index = int(requested.removeprefix("cuda:"))
        except ValueError as error:
            raise ValueError("CUDA device is invalid") from error
        if (
            index < 0
            or not torch.cuda.is_available()
            or index >= torch.cuda.device_count()
        ):
            raise ValueError("CUDA device is unavailable")
        return f"cuda:{index}"
    raise ValueError("device is invalid")


def _validate_train(train: Any) -> None:
    if set(train) != {"steps", "batch_size", "learning_rate"}:
        raise ValueError("train fields are invalid")
    for field in ("steps", "batch_size"):
        value = train[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"train.{field} is invalid")
    learning_rate = train["learning_rate"]
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("train.learning_rate is invalid")


def _validate_eval(evaluation: Any) -> None:
    if set(evaluation) != {
        "dataset",
        "selector",
        "metrics",
        "steps",
        "batch_size",
    }:
        raise ValueError("eval fields are invalid")
    for field in ("steps", "batch_size"):
        value = evaluation[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"eval.{field} is invalid")
    if dict(evaluation["dataset"]) != {"split": "eval"}:
        raise ValueError("eval dataset identity is invalid")
    selector = dict(evaluation["selector"])
    if selector not in ({}, {"class": "daisy"}):
        raise ValueError("eval selector is invalid")
    if dict(evaluation["metrics"]) != {
        "primary": "accuracy",
        "report": ("accuracy",),
    }:
        raise ValueError("metrics fields are invalid")


def _validate_infer(infer: Any) -> None:
    if set(infer) != {
        "format",
        "opset_version",
        "input_shape",
        "input_name",
        "output_name",
    }:
        raise ValueError("infer fields are invalid")
    if (
        infer["format"] != "onnx"
        or infer["opset_version"] != 18
        or infer["input_shape"] != (1, 3, 64, 64)
        or infer["input_name"] != "input"
        or infer["output_name"] != "logits"
    ):
        raise ValueError("infer configuration is invalid")


def _validate_dataset_bundle(dataset: Any) -> None:
    if set(dataset) != {
        "kind",
        "train_dir",
        "eval_dir",
        "image_size",
        "classes",
    }:
        raise ValueError("dataset fields are invalid")
    if dataset["kind"] != "bundled_imagefolder":
        raise ValueError("dataset.kind must be bundled_imagefolder")
    if dataset["train_dir"] != "data/train" or dataset["eval_dir"] != "data/eval":
        raise ValueError("dataset split paths are invalid")
    if dataset["image_size"] != (3, 64, 64):
        raise ValueError("dataset.image_size must be 3x64x64")
    if dataset["classes"] != CLASSES:
        raise ValueError("dataset.classes are invalid")
    _validate_data_tree()


def _validate_data_tree() -> None:
    attribution = DATA_ROOT / "ATTRIBUTION.md"
    manifest_path = DATA_ROOT / "manifest.json"
    if (
        not attribution.is_file()
        or attribution.is_symlink()
        or not manifest_path.is_file()
        or manifest_path.is_symlink()
    ):
        raise ValueError("dataset metadata is invalid")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("dataset manifest is invalid") from error
    if (
        not isinstance(manifest, dict)
        or set(manifest)
        != {"schema_version", "source", "processing", "classes", "entries"}
        or manifest["schema_version"] != 1
        or manifest["source"] != MANIFEST_SOURCE
        or manifest["processing"] != MANIFEST_PROCESSING
        or manifest["classes"] != list(CLASSES)
        or not isinstance(manifest["entries"], list)
    ):
        raise ValueError("dataset manifest is invalid")

    expected_paths = {"ATTRIBUTION.md", "manifest.json"}
    source_paths: dict[str, set[str]] = {"train": set(), "eval": set()}
    counts = {
        (split, class_name): 0 for split in SPLIT_COUNTS for class_name in CLASSES
    }
    for entry in manifest["entries"]:
        if not isinstance(entry, dict) or set(entry) != {
            "source",
            "split",
            "class",
            "path",
            "sha256",
        }:
            raise ValueError("dataset manifest entry is invalid")
        split = entry["split"]
        class_name = entry["class"]
        path_value = entry["path"]
        source = entry["source"]
        digest = entry["sha256"]
        if split not in SPLIT_COUNTS or class_name not in CLASSES:
            raise ValueError("dataset manifest entry is invalid")
        if not isinstance(path_value, str) or not isinstance(source, str):
            raise ValueError("dataset manifest entry is invalid")
        expected_prefix = f"{split}/{class_name}/"
        if (
            not path_value.startswith(expected_prefix)
            or "/" in path_value.removeprefix(expected_prefix)
            or not path_value.endswith(".jpg")
            or source != f"TF-Flowers/{class_name}/{Path(path_value).name}"
        ):
            raise ValueError("dataset manifest path is invalid")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("dataset manifest digest is invalid")
        if path_value in expected_paths or source in source_paths[split]:
            raise ValueError("dataset manifest entry is duplicated")

        image_path = DATA_ROOT / path_value
        if not image_path.is_file() or image_path.is_symlink():
            raise ValueError("dataset image is missing")
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != digest:
            raise ValueError("dataset image digest does not match manifest")
        try:
            with Image.open(image_path) as image:
                image.load()
                if image.format != "JPEG" or image.mode != "RGB":
                    raise ValueError("dataset image encoding is invalid")
                if image.size != (64, 64):
                    raise ValueError("dataset image size is invalid")
        except (OSError, SyntaxError) as error:
            raise ValueError("dataset image cannot be decoded") from error

        expected_paths.add(path_value)
        source_paths[split].add(source)
        counts[(split, class_name)] += 1

    if source_paths["train"] & source_paths["eval"]:
        raise ValueError("dataset splits overlap")
    for split, expected_count in SPLIT_COUNTS.items():
        for class_name in CLASSES:
            if counts[(split, class_name)] != expected_count:
                raise ValueError("dataset class count is invalid")

    actual_paths: set[str] = set()
    actual_directories: set[str] = set()
    total_bytes = 0
    for path in DATA_ROOT.rglob("*"):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise ValueError("dataset tree contains an invalid entry")
        if path.is_file():
            actual_paths.add(path.relative_to(DATA_ROOT).as_posix())
            total_bytes += path.stat().st_size
        elif path.is_dir():
            actual_directories.add(path.relative_to(DATA_ROOT).as_posix())
    if actual_paths != expected_paths:
        raise ValueError("dataset tree entries are invalid")
    expected_directories = {split for split in SPLIT_COUNTS} | {
        f"{split}/{class_name}" for split in SPLIT_COUNTS for class_name in CLASSES
    }
    if actual_directories != expected_directories:
        raise ValueError("dataset directories are invalid")
    if total_bytes > MAX_DATASET_BYTES:
        raise ValueError("dataset fixture exceeds size limit")


def _validate_split_coverage(
    split: str,
    settings: Any,
    *,
    selected_class: str | None = None,
) -> None:
    expected = SPLIT_COUNTS[split] * (1 if selected_class is not None else len(CLASSES))
    if settings["steps"] * settings["batch_size"] != expected:
        raise ValueError(f"{split} steps and batch_size must cover the split")


def _image_batches(
    split_dir: str,
    seed: int,
    *,
    batch_size: int,
    image_size: tuple[int, int, int],
    classes: tuple[str, ...],
    shuffle: bool,
    selected_class: str | None = None,
):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size[1:], antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]
    )
    images = ImageFolder(DATA_ROOT.parent / split_dir, transform=transform)
    if tuple(images.classes) != classes:
        raise ValueError("dataset class mapping is invalid")
    if selected_class is not None:
        if selected_class not in images.class_to_idx:
            raise ValueError("selected dataset class is invalid")
        selected_index = images.class_to_idx[selected_class]
        images = Subset(
            images,
            [
                index
                for index, (_, target) in enumerate(images.samples)
                if target == selected_index
            ],
        )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return DataLoader(
        images,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )


def _atomic_torch_save(document: dict[str, Any], checkpoint: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{checkpoint.name}.candidate-",
        dir=checkpoint.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(document, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, checkpoint)
        directory = os.open(checkpoint.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_save(document: dict[str, Any], path: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.candidate-",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _evaluation_case(cfg: Any) -> Any:
    runtime = cfg.get("runtime")
    if runtime is None:
        case_name = "default"
    else:
        case_name = runtime["target"]["evaluation_case"]
    cases = cfg["variant"]["eval"]["cases"]
    if case_name not in cases:
        raise ValueError("evaluation case is unavailable")
    return cases[case_name]


def _dataset_identity(dataset: Any) -> dict[str, Any]:
    return {
        "kind": dataset["kind"],
        "manifest_sha256": hashlib.sha256(
            (DATA_ROOT / "manifest.json").read_bytes()
        ).hexdigest(),
        "classes": list(dataset["classes"]),
    }


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_json(item) for item in value]
    return value


def _validate_resume_checkpoint(
    document: Any,
    *,
    seed: int,
    classes: int,
    maximum_steps: int,
) -> int:
    if not isinstance(document, dict) or not {
        "model_state",
        "optimizer_state",
        "epoch",
        "seed",
        "classes",
    }.issubset(document):
        raise ValueError("resume checkpoint fields are invalid")
    completed = document["epoch"]
    if (
        isinstance(completed, bool)
        or not isinstance(completed, int)
        or completed < 0
        or completed > maximum_steps
        or document["seed"] != seed
        or document["classes"] != classes
    ):
        raise ValueError("resume checkpoint identity is invalid")
    return completed


def _move_optimizer_state(optimizer: Any, device: str) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)
