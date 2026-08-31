# Copyright (C) 2026 HKDL contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from PIL import Image

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
CLASSES = ("circle", "rectangle")
SPLIT_COUNTS = {"train": 8, "val": 4}
IMAGE_SIZE = 128
MAX_DATASET_BYTES = 1 << 20
METRICS = {
    "primary": "map50_95",
    "report": ("map50_95", "map50", "precision", "recall"),
}
DISABLED_INTEGRATIONS = {
    "clearml",
    "comet",
    "dvc",
    "hub",
    "mlflow",
    "neptune",
    "raytune",
    "tensorboard",
    "wandb",
}


class Trainer:
    def fit(self, ctx: Any) -> dict[str, Path]:
        train = ctx.cfg["variant"]["train"]
        with tempfile.TemporaryDirectory(
            prefix=".yolo-train-",
            dir=ctx.paths.run_dir,
        ) as temporary:
            workspace = Path(temporary)
            YOLO = _configure_ultralytics(workspace)
            dataset_yaml = _prepare_dataset(workspace)
            model = YOLO("yolo26n.yaml", task="detect", verbose=False)
            epoch_started_at: float | None = None

            def epoch_step(vendor_trainer: Any) -> int:
                step = vendor_trainer.epoch
                if isinstance(step, bool) or not isinstance(step, int) or step < 0:
                    raise ValueError("Ultralytics epoch is invalid")
                return step

            def on_train_epoch_start(vendor_trainer: Any) -> None:
                del vendor_trainer
                nonlocal epoch_started_at
                epoch_started_at = time.perf_counter()

            def on_model_save(vendor_trainer: Any) -> None:
                nonlocal epoch_started_at
                if epoch_started_at is None:
                    raise ValueError("Ultralytics epoch timer was not started")
                loss = _finite_loss(vendor_trainer.tloss)
                step = epoch_step(vendor_trainer)
                ctx.tracker.log_scalar("train_loss", loss, step)
                ctx.tracker.log_scalar(
                    "train.epoch_seconds",
                    max(0.0, time.perf_counter() - epoch_started_at),
                    step,
                )
                epoch_started_at = None
                last = ctx.paths.checkpoints / "last.pt"
                _atomic_copy(Path(vendor_trainer.last), last)
                ctx.report_checkpoint(last)

            model.add_callback("on_train_epoch_start", on_train_epoch_start)
            model.add_callback("on_model_save", on_model_save)
            resume = str(ctx.resume_from) if ctx.resume_from is not None else False
            model.train(
                data=str(dataset_yaml),
                epochs=train["epochs"],
                batch=train["batch_size"],
                imgsz=IMAGE_SIZE,
                device=ctx.exec.device,
                workers=0,
                optimizer="SGD",
                lr0=float(train["learning_rate"]),
                seed=ctx.exec.seed,
                deterministic=True,
                pretrained=False,
                resume=resume,
                save=True,
                save_period=-1,
                save_dir=str(workspace / "train"),
                val=False,
                plots=False,
                verbose=False,
                amp=False,
                cache=False,
                mosaic=0.0,
                mixup=0.0,
                copy_paste=0.0,
                hsv_h=0.0,
                hsv_s=0.0,
                hsv_v=0.0,
                degrees=0.0,
                translate=0.0,
                scale=0.0,
                shear=0.0,
                perspective=0.0,
                flipud=0.0,
                fliplr=0.0,
                bgr=0.0,
                close_mosaic=0,
                patience=train["epochs"],
            )
            vendor_trainer = model.trainer
            vendor_last = Path(vendor_trainer.last)
            vendor_best = (
                Path(vendor_trainer.best)
                if Path(vendor_trainer.best).is_file()
                else vendor_last
            )
            last = ctx.paths.checkpoints / "last.pt"
            best = ctx.paths.checkpoints / "best.pt"
            _atomic_copy(vendor_last, last)
            _atomic_copy(vendor_best, best)
            ctx.report_checkpoint(last, best)
            return {"best_checkpoint": best, "last_checkpoint": last}


class Evaluator:
    def evaluate(self, ctx: Any, checkpoint: Path) -> dict[str, Any]:
        evaluation = _evaluation_case(ctx.cfg)
        with tempfile.TemporaryDirectory(
            prefix=".yolo-eval-",
            dir=ctx.paths.run_dir,
        ) as temporary:
            workspace = Path(temporary)
            YOLO = _configure_ultralytics(workspace)
            dataset_yaml = _prepare_dataset(workspace)
            model = YOLO(str(checkpoint), task="detect", verbose=False)
            metrics = model.val(
                data=str(dataset_yaml),
                split="val",
                imgsz=IMAGE_SIZE,
                batch=evaluation["batch_size"],
                conf=evaluation["confidence"],
                iou=evaluation["iou"],
                device=ctx.exec.device,
                workers=0,
                save_json=False,
                plots=False,
                save=False,
                save_dir=str(workspace / "val"),
                verbose=False,
            )
            values = {
                "map50_95": _finite_number(metrics.box.map, "map50_95"),
                "map50": _finite_number(metrics.box.map50, "map50"),
                "precision": _finite_number(metrics.box.mp, "precision"),
                "recall": _finite_number(metrics.box.mr, "recall"),
            }
            image_root = workspace / "data" / "images" / "val"
            image_paths = sorted(
                image_root.glob("*.png"),
                key=lambda path: path.name.encode("utf-8"),
            )
            results = model.predict(
                source=[str(path) for path in image_paths],
                imgsz=IMAGE_SIZE,
                conf=evaluation["confidence"],
                iou=evaluation["iou"],
                device=ctx.exec.device,
                save=False,
                stream=False,
                verbose=False,
            )
            predictions = []
            for image_path, result in zip(image_paths, results, strict=True):
                detections = []
                boxes = result.boxes
                if boxes is not None:
                    for xyxy, confidence, class_value in zip(
                        boxes.xyxy.detach().cpu().tolist(),
                        boxes.conf.detach().cpu().tolist(),
                        boxes.cls.detach().cpu().tolist(),
                        strict=True,
                    ):
                        class_id = int(class_value)
                        if class_value != class_id or class_id not in range(
                            len(CLASSES)
                        ):
                            raise ValueError("prediction class is invalid")
                        numeric_box = [
                            _finite_number(value, "prediction box") for value in xyxy
                        ]
                        if len(numeric_box) != 4 or any(
                            value < -1e-5 or value > IMAGE_SIZE + 1e-5
                            for value in numeric_box
                        ):
                            raise ValueError("prediction box is outside the image")
                        numeric_confidence = _finite_number(
                            confidence,
                            "prediction confidence",
                        )
                        if not 0.0 <= numeric_confidence <= 1.0:
                            raise ValueError("prediction confidence is invalid")
                        detections.append(
                            {
                                "class_id": class_id,
                                "class_name": CLASSES[class_id],
                                "confidence": round(numeric_confidence, 8),
                                "xyxy": [
                                    round(min(max(value, 0.0), IMAGE_SIZE), 6)
                                    for value in numeric_box
                                ],
                            }
                        )
                detections.sort(
                    key=lambda item: (
                        -item["confidence"],
                        item["class_id"],
                        item["xyxy"],
                    )
                )
                predictions.append(
                    {
                        "image": f"images/val/{image_path.name}",
                        "detections": detections,
                    }
                )
            result_path = ctx.paths.results / "predictions.json"
            _atomic_json_save(
                {
                    "schema_version": 1,
                    "evaluation_case": ctx.cfg["runtime"]["target"]["evaluation_case"],
                    "classes": list(CLASSES),
                    "predictions": predictions,
                },
                result_path,
            )
            return {"values": values, "files": [result_path]}


class ONNXExporter:
    def export(self, ctx: Any, checkpoint: Path) -> list[Path]:
        infer = ctx.cfg["variant"]["infer"]
        with tempfile.TemporaryDirectory(
            prefix=".yolo-export-",
            dir=ctx.paths.run_dir,
        ) as temporary:
            workspace = Path(temporary)
            YOLO = _configure_ultralytics(workspace)
            local_checkpoint = workspace / "model.pt"
            _atomic_copy(checkpoint, local_checkpoint)
            model = YOLO(str(local_checkpoint), task="detect", verbose=False)
            exported = Path(
                model.export(
                    format="onnx",
                    imgsz=IMAGE_SIZE,
                    batch=1,
                    opset=infer["opset_version"],
                    dynamic=False,
                    simplify=False,
                    nms=False,
                    device=ctx.exec.device,
                    verbose=False,
                )
            )
            output = ctx.paths.export / "model.onnx"
            license_path = ctx.paths.export / "LICENSE-AGPL-3.0-or-later.txt"
            _atomic_copy(exported, output)
            _atomic_copy(SOURCE_ROOT / "LICENSE", license_path)
            _validate_onnx(output, infer)
            return [license_path, output]


class Entrypoint:
    def registry(self):
        yield "model", "yolo26n", _registered_component
        yield "loss", "ultralytics", _registered_component
        yield "optimizer", "sgd", _registered_component
        yield "dataloader", "yolo", _registered_component
        yield "trainer", "default", Trainer
        yield "evaluator", "default", Evaluator
        yield "exporter", "onnx", ONNXExporter

    def validate(self, action, cfg, selected, exec_info):
        required = {
            "train": TRAIN_COMPONENTS,
            "eval": EVAL_COMPONENTS,
            "export": EXPORT_COMPONENTS,
        }
        if action not in required or set(selected) != required[action]:
            raise ValueError("invalid action selection")
        variant = cfg["variant"]
        _validate_tracker(variant["tracker"])
        _validate_dataset_bundle(variant["dataset"])
        if action == "train":
            _validate_train(variant["train"])
            identity = {
                "dataset": _dataset_identity(variant["dataset"]),
                "train": _plain_json(variant["train"]),
            }
        elif action == "eval":
            evaluation = _evaluation_case(cfg)
            _validate_evaluation(evaluation)
            identity = {
                "dataset": _dataset_identity(variant["dataset"]),
                "case": _plain_json(evaluation),
            }
        else:
            _validate_infer(variant["infer"])
            identity = {"infer": _plain_json(variant["infer"])}
        seed = exec_info["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        return {
            "exec": {
                "seed": seed,
                "device": _resolve_device(exec_info["device"]),
            },
            "identity": identity,
        }

    def assemble(self, action, cfg, selected):
        del cfg
        if action == "train" and set(selected) == TRAIN_COMPONENTS:
            return {
                "model": "yolo26n",
                "loss": "ultralytics",
                "optimizer": "sgd",
                "dataloader": "yolo",
                "trainer": Trainer(),
            }
        if action == "eval" and set(selected) == EVAL_COMPONENTS:
            return {
                "model": "yolo26n",
                "dataloader": "yolo",
                "evaluator": Evaluator(),
            }
        if action == "export" and set(selected) == EXPORT_COMPONENTS:
            return {"exporter": ONNXExporter()}
        raise ValueError("invalid action selection")


def entrypoint() -> Entrypoint:
    return Entrypoint()


def _registered_component() -> None:
    return None


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


def _validate_tracker(tracker: Any) -> None:
    if set(tracker) != {"backend"}:
        raise ValueError("tracker must contain only backend")
    backend = tracker["backend"]
    if isinstance(backend, str):
        valid = backend in {"none", "local", "mlflow"}
    else:
        valid = (
            isinstance(backend, tuple)
            and bool(backend)
            and all(item in {"local", "mlflow"} for item in backend)
            and len(backend) == len(set(backend))
        )
    if not valid:
        raise ValueError("invalid tracker.backend")


def _validate_train(train: Any) -> None:
    if set(train) != {"epochs", "batch_size", "learning_rate"}:
        raise ValueError("train fields are invalid")
    epochs = train["epochs"]
    batch_size = train["batch_size"]
    if (
        isinstance(epochs, bool)
        or not isinstance(epochs, int)
        or not 1 <= epochs <= 1000
    ):
        raise ValueError("train.epochs is invalid")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or not 1 <= batch_size <= 128
    ):
        raise ValueError("train.batch_size is invalid")
    learning_rate = train["learning_rate"]
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(float(learning_rate))
        or not 0.0 < float(learning_rate) <= 1.0
    ):
        raise ValueError("train.learning_rate is invalid")


def _validate_evaluation(evaluation: Any) -> None:
    if set(evaluation) != {
        "dataset",
        "selector",
        "metrics",
        "batch_size",
        "confidence",
        "iou",
    }:
        raise ValueError("evaluation fields are invalid")
    if dict(evaluation["dataset"]) != {"split": "val"}:
        raise ValueError("evaluation dataset is invalid")
    if dict(evaluation["selector"]) != {}:
        raise ValueError("evaluation selector is invalid")
    if dict(evaluation["metrics"]) != METRICS:
        raise ValueError("evaluation metrics are invalid")
    batch_size = evaluation["batch_size"]
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or not 1 <= batch_size <= 64
    ):
        raise ValueError("evaluation batch_size is invalid")
    for field in ("confidence", "iou"):
        value = evaluation[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(f"evaluation {field} is invalid")


def _validate_infer(infer: Any) -> None:
    if set(infer) != {
        "format",
        "opset_version",
        "input_shape",
        "output_shape",
    }:
        raise ValueError("infer fields are invalid")
    if (
        infer["format"] != "onnx"
        or infer["opset_version"] != 18
        or infer["input_shape"] != (1, 3, IMAGE_SIZE, IMAGE_SIZE)
        or infer["output_shape"] != (1, 300, 6)
    ):
        raise ValueError("infer configuration is invalid")


def _validate_dataset_bundle(dataset: Any) -> None:
    if set(dataset) != {"kind", "root", "image_size", "classes"}:
        raise ValueError("dataset fields are invalid")
    if (
        dataset["kind"] != "bundled_yolo_detection"
        or dataset["root"] != "data"
        or dataset["image_size"] != (3, IMAGE_SIZE, IMAGE_SIZE)
        or dataset["classes"] != CLASSES
    ):
        raise ValueError("dataset configuration is invalid")
    _validate_data_tree()


def _validate_data_tree() -> None:
    manifest_path = DATA_ROOT / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("dataset manifest is invalid")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("dataset manifest is invalid") from error
    if (
        not isinstance(manifest, dict)
        or set(manifest)
        != {"schema_version", "generator", "image_size", "classes", "entries"}
        or manifest["schema_version"] != 1
        or manifest["generator"] != "HKDL synthetic shapes v1"
        or manifest["image_size"] != [IMAGE_SIZE, IMAGE_SIZE]
        or manifest["classes"] != list(CLASSES)
        or not isinstance(manifest["entries"], list)
    ):
        raise ValueError("dataset manifest is invalid")

    expected_files = {"manifest.json"}
    seen_stems = set()
    counts = {(split, class_id): 0 for split in SPLIT_COUNTS for class_id in range(2)}
    for entry in manifest["entries"]:
        if not isinstance(entry, dict) or set(entry) != {
            "split",
            "stem",
            "class_id",
            "image",
            "label",
            "image_sha256",
            "label_sha256",
        }:
            raise ValueError("dataset manifest entry is invalid")
        split = entry["split"]
        stem = entry["stem"]
        class_id = entry["class_id"]
        if (
            split not in SPLIT_COUNTS
            or not isinstance(stem, str)
            or stem in seen_stems
            or isinstance(class_id, bool)
            or class_id not in range(len(CLASSES))
        ):
            raise ValueError("dataset manifest entry is invalid")
        image_relative = f"images/{split}/{stem}.png"
        label_relative = f"labels/{split}/{stem}.txt"
        if entry["image"] != image_relative or entry["label"] != label_relative:
            raise ValueError("dataset manifest path is invalid")
        image_path = DATA_ROOT / image_relative
        label_path = DATA_ROOT / label_relative
        _validate_digest_file(image_path, entry["image_sha256"])
        _validate_digest_file(label_path, entry["label_sha256"])
        try:
            with Image.open(image_path) as image:
                image.load()
                if (
                    image.format != "PNG"
                    or image.mode != "RGB"
                    or image.size != (IMAGE_SIZE, IMAGE_SIZE)
                ):
                    raise ValueError("dataset image encoding is invalid")
        except (OSError, SyntaxError) as error:
            raise ValueError("dataset image cannot be decoded") from error
        _validate_label(label_path, class_id)
        expected_files.update({image_relative, label_relative})
        seen_stems.add(stem)
        counts[(split, class_id)] += 1

    if len(seen_stems) != sum(SPLIT_COUNTS.values()):
        raise ValueError("dataset sample count is invalid")
    for split, count in SPLIT_COUNTS.items():
        expected_per_class = count // len(CLASSES)
        for class_id in range(len(CLASSES)):
            if counts[(split, class_id)] != expected_per_class:
                raise ValueError("dataset class count is invalid")

    actual_files = set()
    actual_directories = set()
    total_bytes = 0
    for path in DATA_ROOT.rglob("*"):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise ValueError("dataset tree contains an invalid entry")
        relative = path.relative_to(DATA_ROOT).as_posix()
        if path.is_file():
            actual_files.add(relative)
            total_bytes += path.stat().st_size
        else:
            actual_directories.add(relative)
    expected_directories = {
        "images",
        "labels",
        "images/train",
        "images/val",
        "labels/train",
        "labels/val",
    }
    if (
        actual_files != expected_files
        or actual_directories != expected_directories
        or total_bytes > MAX_DATASET_BYTES
    ):
        raise ValueError("dataset tree entries are invalid")


def _validate_digest_file(path: Path, digest: Any) -> None:
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or not path.is_file()
        or path.is_symlink()
        or hashlib.sha256(path.read_bytes()).hexdigest() != digest
    ):
        raise ValueError("dataset file digest is invalid")


def _validate_label(path: Path, expected_class: int) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        fields = lines[0].split()
    except (OSError, UnicodeError, IndexError) as error:
        raise ValueError("dataset label is invalid") from error
    if len(lines) != 1 or len(fields) != 5:
        raise ValueError("dataset label is invalid")
    try:
        class_id = int(fields[0])
        x, y, width, height = (float(value) for value in fields[1:])
    except ValueError as error:
        raise ValueError("dataset label is invalid") from error
    if (
        str(class_id) != fields[0]
        or class_id != expected_class
        or not all(math.isfinite(value) for value in (x, y, width, height))
        or not 0.0 < width <= 1.0
        or not 0.0 < height <= 1.0
        or not 0.0 <= x - width / 2
        or x + width / 2 > 1.0
        or not 0.0 <= y - height / 2
        or y + height / 2 > 1.0
    ):
        raise ValueError("dataset label is invalid")


def _evaluation_case(cfg: Any) -> Any:
    runtime = cfg.get("runtime")
    name = "default" if runtime is None else runtime["target"]["evaluation_case"]
    cases = cfg["variant"]["eval"]["cases"]
    if name not in cases:
        raise ValueError("evaluation case is unavailable")
    return cases[name]


def _dataset_identity(dataset: Any) -> dict[str, Any]:
    return {
        "kind": dataset["kind"],
        "manifest_sha256": hashlib.sha256(
            (DATA_ROOT / "manifest.json").read_bytes()
        ).hexdigest(),
        "classes": list(dataset["classes"]),
    }


def _configure_ultralytics(workspace: Path):
    config_root = workspace / "config"
    config_root.mkdir()
    os.environ["YOLO_CONFIG_DIR"] = str(config_root)
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_VERBOSE"] = "false"
    os.environ["ULTRALYTICS_SAFE_LOAD"] = "true"
    os.environ["ULTRALYTICS_API_KEY"] = ""
    os.environ["MPLCONFIGDIR"] = str(config_root / "matplotlib")
    os.environ["XDG_CACHE_HOME"] = str(config_root / "cache")
    from ultralytics import YOLO, settings

    if not DISABLED_INTEGRATIONS.issubset(settings):
        raise ValueError("Ultralytics integration settings are unavailable")
    settings.update(
        {
            "sync": False,
            "api_key": "",
            "datasets_dir": str(workspace / "datasets"),
            "weights_dir": str(workspace / "weights"),
            "runs_dir": str(workspace / "runs"),
            **{name: False for name in DISABLED_INTEGRATIONS},
        }
    )
    return YOLO


def _prepare_dataset(workspace: Path) -> Path:
    destination = workspace / "data"
    shutil.copytree(DATA_ROOT, destination)
    document = {
        "path": str(destination),
        "train": "images/train",
        "val": "images/val",
        "names": {index: name for index, name in enumerate(CLASSES)},
    }
    path = workspace / "dataset.yaml"
    path.write_text(
        json.dumps(document, ensure_ascii=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return path


def _finite_loss(values: Any) -> float:
    if not isinstance(values, Mapping) or not values:
        raise ValueError("Ultralytics loss is invalid")
    total = sum(_finite_number(value, "train loss") for value in values.values())
    return _finite_number(total, "train loss")


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{field} is invalid")
        value = value.detach().cpu().item()
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field} is invalid")
    return float(value)


def _validate_onnx(path: Path, infer: Any) -> None:
    import onnx

    model = onnx.load(path)
    onnx.checker.check_model(model)
    opsets = {
        item.domain: item.version
        for item in model.opset_import
        if item.domain in {"", "ai.onnx"}
    }
    if opsets.get("", opsets.get("ai.onnx")) != infer["opset_version"]:
        raise ValueError("ONNX opset is invalid")
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError("ONNX signature is invalid")
    input_shape = _onnx_shape(model.graph.input[0])
    output_shape = _onnx_shape(model.graph.output[0])
    if input_shape != list(infer["input_shape"]) or output_shape != list(
        infer["output_shape"]
    ):
        raise ValueError("ONNX signature is invalid")


def _onnx_shape(value: Any) -> list[int]:
    result = []
    for dimension in value.type.tensor_type.shape.dim:
        if not dimension.HasField("dim_value"):
            raise ValueError("ONNX shape must be fixed")
        result.append(dimension.dim_value)
    return result


def _atomic_copy(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"required file is invalid: {source}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.candidate-",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
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
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_json(item) for item in value]
    return value
