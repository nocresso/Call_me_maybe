from pydantic import BaseModel
from enum import Enum
from typing import Dict, Any, List, Union
from pathlib import Path
import os
import json


class ParameterType(str, Enum):
    """
    Supported parameter types for function definitions.
    """
    NUMBER = "number"
    STRING = "string"
    BOOLEAN = "boolean"
    INTEGER = "integer"
    FLOAT = "float"


class TypeDefinition(BaseModel):
    """
    Type definition for a function parameter or return value.
    """
    type: ParameterType


class FunctionDefinition(BaseModel):
    """
    Represents a callable function with its parameters
    """
    name: str
    description: str
    parameters: Dict[str, TypeDefinition]
    returns: TypeDefinition


def load_json_file(file_path: Union[str, Path]) -> Any:
    """
    Load and parse JSON file.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"{file_path} does not exist")

    if not path.is_file():
        raise FileNotFoundError(f"{file_path} is not a valid file")

    if path.suffix != ".json":
        raise ValueError(f"expected a .json file, got: '{path.suffix}'")
    if not os.access(path, os.R_OK):
        raise PermissionError(f"do not have permission to READ from '{path}'")
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON in {file_path}: {e}")


def load_prompts(file_path: Union[str, Path]) -> List[str]:
    """
    Load a list of prompts.
    """
    data = load_json_file(file_path)

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("prompts file must contain a non-empty array")

    if isinstance(data[0], str):
        if not all(isinstance(p, str) for p in data):
            raise ValueError("All prompts must be strings")
        return data
    elif isinstance(data[0], dict):
        prompts = []
        for i, entry in enumerate(data):
            if "prompt" not in entry:
                raise ValueError(f"entry {i} is missing the 'prompt' key")
            if not entry["prompt"].strip():
                raise ValueError(f"entry {i}: 'prompt' cannot be empty")
            if not isinstance(entry["prompt"], str):
                raise ValueError(f"entry {i}: 'prompt' must be a string")
            prompts.append(entry["prompt"])
        return prompts
    else:
        raise ValueError("prompts must be strings or dict with 'prompt' key")


def load_functions(file_path: Union[str, Path]) -> List[FunctionDefinition]:
    """
    Load and validate function definitions.
    """
    data = load_json_file(file_path)

    if not isinstance(data, list):
        raise ValueError("Function definitions file must contain a JSON array")
    functions = []
    for f in data:
        try:
            functions.append(FunctionDefinition(**f))
        except Exception as e:
            raise ValueError(f"Invalid function definition: {e}")
    return functions
