# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
#
# This file is based on https://github.com/deepset-ai/haystack
# See: https://github.com/deepset-ai/haystack/blob/main/rest_api/rest_api/controller/utils.py

import inspect
import json
from typing import Type

from fastapi import Form
from pydantic import BaseModel


def as_form(cls: Type[BaseModel]):
    """
    Adds an as_form class method to decorated models. The as_form class method
    can be used with FastAPI endpoints.
    """
    new_params = [
        inspect.Parameter(
            field_name,
            inspect.Parameter.POSITIONAL_ONLY,
            default=(Form(field_info.default) if field_info.default is not Ellipsis else Form(...)),
        )
        for field_name, field_info in cls.model_fields.items()
    ]

    async def _as_form(**data):
        return cls(**data)

    sig = inspect.signature(_as_form)
    sig = sig.replace(parameters=new_params)
    _as_form.__signature__ = sig  # type: ignore
    setattr(cls, "as_form", _as_form)
    return cls


def serialize_complex_objects(obj):
    """Convert complex objects to serializable strings."""
    try:
        # Try standard serialization
        json.dumps(obj)
        return obj
    except (TypeError, OverflowError):
        # If it fails, convert to string
        return str(obj)


def make_serializable(data):
    if isinstance(data, dict):
        return {k: make_serializable(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [make_serializable(item) for item in data]
    else:
        return serialize_complex_objects(data)
