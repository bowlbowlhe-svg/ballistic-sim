"""由 pydantic 模型字段自动生成 tkinter 表单。"""

from __future__ import annotations

import ast
import json
import tkinter as tk
from tkinter import ttk
from typing import Any, Dict, List, Type, Union, cast, get_args, get_origin

from pydantic import BaseModel

_Literal = getattr(__import__("typing"), "Literal")
_Union = getattr(__import__("typing"), "Union")


def _is_optional(annotation: Any) -> bool:
    """判断 ``annotation`` 是否为 ``Optional[T]`` / ``Union[T, None]``。"""
    origin = get_origin(annotation)
    if origin is Union:
        args = get_args(annotation)
        return type(None) in args and len(args) == 2
    return False


def _inner_optional(annotation: Any) -> Any:
    """从 ``Optional[T]`` 中提取 ``T``。"""
    args = [a for a in get_args(annotation) if a is not type(None)]
    return args[0] if args else str


def _is_literal(annotation: Any) -> bool:
    return get_origin(annotation) is _Literal


def _literal_values(annotation: Any) -> List[str]:
    return [str(v) for v in get_args(annotation)]


def _is_list(annotation: Any) -> bool:
    origin = get_origin(annotation)
    return origin is list or (isinstance(origin, type) and issubclass(origin, list))


def _inner_type(annotation: Any) -> Any:
    """返回 List[T] 或 Optional[List[T]] 中的 T。"""
    if _is_optional(annotation):
        annotation = _inner_optional(annotation)
    if not _is_list(annotation):
        return None
    args = get_args(annotation)
    return args[0] if args else None


def _is_model_type(annotation: Any) -> bool:
    """判断注解是否为 pydantic BaseModel（支持 Optional/Union 包装）。"""
    if _is_optional(annotation):
        annotation = _inner_optional(annotation)
    try:
        return isinstance(annotation, type) and issubclass(annotation, BaseModel)
    except TypeError:
        return False


def _non_optional_type(annotation: Any) -> str:
    """对非 Optional 注解归类。"""
    if _is_literal(annotation):
        return "literal"
    origin = get_origin(annotation)
    if origin is not None and _is_list(annotation):
        inner = _inner_type(annotation)
        if _is_model_type(inner):
            return "list_model"
        return "list"
    if annotation is bool:
        return "bool"
    if annotation is int:
        return "int"
    if annotation is float:
        return "float"
    return "str"


def _field_type(annotation: Any) -> str:
    """将字段注解归类，Optional 字段带 ``optional_`` 前缀。"""
    if _is_optional(annotation):
        inner = _inner_optional(annotation)
        return f"optional_{_non_optional_type(inner)}"
    return _non_optional_type(annotation)


def _default_value(field_info: Any) -> Any:
    """获取字段默认值。"""
    default = field_info.default
    if default is not None and default is not ...:
        return default
    default_factory = getattr(field_info, "default_factory", None)
    if default_factory is not None:
        return default_factory()
    return None


def _var_for_type(field_type: str, value: Any) -> tk.Variable:
    """根据字段类型构造对应的 tkinter 变量。"""
    if field_type == "bool":
        return tk.BooleanVar(value=bool(value) if value is not None else False)
    if field_type == "int":
        return tk.IntVar(value=int(value) if value is not None else 0)
    if field_type == "float":
        return tk.DoubleVar(value=float(value) if value is not None else 0.0)
    if field_type.startswith("optional_"):
        text = "" if value is None else str(value)
        return tk.StringVar(value=text)
    if field_type == "literal":
        return tk.StringVar(value=str(value) if value is not None else "")
    if field_type in ("list", "optional_list"):
        if isinstance(value, (list, tuple)):
            text = ", ".join(str(v) for v in value)
        elif value is None:
            text = ""
        else:
            text = str(value)
        return tk.StringVar(value=text)
    if field_type in ("list_model", "optional_list_model"):
        if isinstance(value, (list, tuple)):
            text = json.dumps(value, ensure_ascii=False, indent=2)
        elif value is None:
            text = ""
        else:
            text = str(value)
        return tk.StringVar(value=text)
    text = "" if value is None else str(value)
    return tk.StringVar(value=text)


def _widget_for_field(
    parent: tk.Widget,
    field_type: str,
    var: tk.Variable,
    annotation: Any,
) -> tk.Widget:
    """为字段创建合适的输入控件。"""
    if field_type == "bool":
        return ttk.Checkbutton(parent, variable=var)

    inner_annotation = annotation
    if _is_optional(annotation):
        inner_annotation = _inner_optional(annotation)

    if _is_literal(inner_annotation):
        values = _literal_values(inner_annotation)
        combo = ttk.Combobox(parent, textvariable=var, values=values, state="readonly")
        if values and not var.get():
            var.set(values[0])
        return combo
    return ttk.Entry(parent, textvariable=var)


def _parse_list_value(text: str) -> List[Any]:
    """将逗号分隔字符串解析为列表。"""
    if not text.strip():
        return []
    parts = [p.strip() for p in text.split(",")]
    out: List[Any] = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError:
            try:
                out.append(float(p))
            except ValueError:
                out.append(p)
    return out


def _model_cls(model: Union[BaseModel, Type[BaseModel]]) -> Type[BaseModel]:
    """统一处理 ``model`` 为实例或类的情况。"""
    return cast(Type[BaseModel], model if isinstance(model, type) else type(model))


def build_form(
    frame: tk.Widget, model: Union[BaseModel, Type[BaseModel]]
) -> Dict[str, tk.Variable]:
    """在 ``frame`` 中根据 ``model`` 的字段生成表单，返回字段名到变量的映射。

    支持 ``int`` / ``float`` / ``str`` / ``bool`` / ``Literal`` / ``Optional[T]`` /
    ``List`` 等常见类型。列表类型用逗号分隔的文本输入表示。
    """
    variables: Dict[str, tk.Variable] = {}
    fields = _model_cls(model).model_fields
    for row, (name, field_info) in enumerate(fields.items()):
        annotation = field_info.annotation
        field_type = _field_type(annotation)
        default = _default_value(field_info)

        label_text = name
        if field_info.description:
            label_text = f"{name} ({field_info.description})"
        ttk.Label(frame, text=label_text).grid(row=row, column=0, sticky="w", padx=4, pady=2)

        var = _var_for_type(field_type, default)
        widget = _widget_for_field(frame, field_type, var, annotation)
        widget.grid(row=row, column=1, sticky="ew", padx=4, pady=2)
        variables[name] = var

    frame.columnconfigure(1, weight=1)
    return variables


def _parse_optional(field_type: str, raw: Any) -> Any:
    """解析 Optional 字段的值，空字符串映射为 ``None``。"""
    text = str(raw).strip()
    if text == "":
        return None
    inner = field_type.replace("optional_", "")
    if inner == "int":
        try:
            return int(text)
        except ValueError:
            return None
    if inner == "float":
        try:
            return float(text)
        except ValueError:
            return None
    if inner == "bool":
        return text.lower() in ("true", "1", "yes", "on")
    return text


def read_model_variables(
    model_cls: Type[BaseModel], variables: Dict[str, tk.Variable]
) -> Dict[str, Any]:
    """从 tkinter 变量中读取值，构造 ``model_cls`` 可接受的字典。"""
    data: Dict[str, Any] = {}
    for name, var in variables.items():
        field_info = model_cls.model_fields[name]
        annotation = field_info.annotation
        field_type = _field_type(annotation)
        raw = var.get()

        if field_type == "bool":
            data[name] = bool(raw)
        elif field_type == "int":
            try:
                data[name] = int(raw) if raw != "" else 0
            except (ValueError, TypeError):
                data[name] = 0
        elif field_type == "float":
            try:
                data[name] = float(raw) if raw != "" else 0.0
            except (ValueError, TypeError):
                data[name] = 0.0
        elif field_type == "list":
            data[name] = _parse_list_value(str(raw))
        elif field_type == "list_model":
            data[name] = _parse_list_model_value(str(raw), annotation)
        elif field_type == "optional_list_model":
            data[name] = _parse_list_model_value(str(raw), annotation) if str(raw).strip() else None
        elif field_type.startswith("optional_"):
            data[name] = _parse_optional(field_type, raw)
        else:
            data[name] = str(raw) if raw != "" else None
    return data


def _parse_list_model_value(text: str, annotation: Any) -> List[Any]:
    """将 JSON/YAML/Py repr 字符串解析为 BaseModel 列表。"""
    text = text.strip()
    if not text:
        return []
    item_cls = _inner_type(annotation)
    if item_cls is None:
        raise ValueError(f"无法识别列表元素类型: {annotation}")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"列表格式错误: {exc}") from exc
    if not isinstance(parsed, list):
        raise ValueError("列表字段应为一个数组")
    return [item_cls(**item) if isinstance(item, dict) else item_cls(item) for item in parsed]


def update_model_variables(variables: Dict[str, tk.Variable], model: BaseModel) -> None:
    """用 ``model`` 的字段值更新已有的 tkinter 变量。"""
    data = model.model_dump(mode="json")
    for name, var in variables.items():
        value = data.get(name)
        if isinstance(var, tk.BooleanVar):
            var.set(bool(value))
        elif isinstance(var, tk.IntVar):
            var.set(int(value) if value is not None else 0)
        elif isinstance(var, tk.DoubleVar):
            var.set(float(value) if value is not None else 0.0)
        elif isinstance(var, tk.StringVar):
            if isinstance(value, (list, tuple)):
                if value and isinstance(value[0], dict):
                    var.set(json.dumps(list(value), ensure_ascii=False, indent=2))
                else:
                    var.set(", ".join(str(v) for v in value))
            elif value is None:
                var.set("")
            else:
                var.set(str(value))
