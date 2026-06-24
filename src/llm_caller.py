from llm_sdk import Small_LLM_Model  # type: ignore
from typing import Dict, Any, List, Tuple, cast
from src.input_validation import (FunctionDefinition,
                                  TypeDefinition, ParameterType)
import numpy as np
import json


def load_vocabulary(model: Small_LLM_Model) -> Dict[str, str]:
    """
    Download and load the Qwen3 tokenizer vocabulary.
    """
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return {str(v): k for k, v in vocab.items()}


def precompute_token_sets(
        vocabulary: Dict[str, str]) -> Dict[str, List[Tuple[int, str]]]:
    """
    Pre-compute lists of tuples (token_id, token_str) for each parameter
    type.
    """
    number_tokens: List[Tuple[int, str]] = []
    boolean_tokens: List[Tuple[int, str]] = []
    fn_name_tokens: List[Tuple[int, str]] = []

    for token_id_str, token_str in vocabulary.items():
        tid = int(token_id_str)

        if all(c.isdigit() or c in "-.,}" for c in token_str.strip()):
            number_tokens.append((tid, token_str))

        if "True".startswith(token_str) or "False".startswith(token_str):
            boolean_tokens.append((tid, token_str))

        if all(c.isalnum() or c in '_"' for c in token_str):
            fn_name_tokens.append((tid, token_str))

    return {
        "number": number_tokens,
        "boolean": boolean_tokens,
        "fn_name": fn_name_tokens,
    }


def build_system_context(functions: List[FunctionDefinition]) -> str:
    """
    Build the system context prompt with available functions.
    """
    context = ("Given a user request, select the correct"
               " function and extract its arguments.\n"
               "IMPORTANT: Match the verb/action in the user request to the"
               " function description or function name carefully,"
               " find coincidences.\n Available functions:\n")
    for fn in functions:
        context += (f"- {fn.name}: {fn.description},"
                    f" parameters: {fn.parameters}\n")
    context += "\nRespond only with valid JSON, no extra text.\n"
    return context


def generate_function_name(
        model: Small_LLM_Model,
        system_ids: List[int],
        generated_json: str,
        functions: List[FunctionDefinition],
        token_sets: Dict[str, List[Tuple[int, str]]],
) -> str:
    """
    Use constrained decoding to select a valid function name.
    """
    fn_name = ""
    function_names = [f.name for f in functions]
    json_ids: List[int] = model.encode(generated_json)[0].tolist()

    while True:
        input_ids = system_ids + json_ids
        logits = model.get_logits_from_input_ids(input_ids)
        mask = [float("-inf")] * len(logits)

        for tid, token_str in token_sets["fn_name"]:
            candidate = fn_name + token_str
            for name in function_names:
                if (name.startswith(candidate) or
                        candidate.startswith(name + '"')):
                    mask[tid] = logits[tid]
                    break

        next_token = int(np.argmax(mask))
        next_text: str = cast(str, model.decode([next_token]))
        json_ids += model.encode(next_text)[0].tolist()
        fn_name += next_text

        if '"' in fn_name:
            break

    return fn_name[:fn_name.index('"')]


def generate_argument(
        model: Small_LLM_Model,
        system_ids: List[int],
        generated_json: str,
        param_type: TypeDefinition,
        token_sets: Dict[str, List[Tuple[int, str]]],
        max_string_tokens: int) -> str:
    """
    Use constrained decoding to get argument values.
    """
    len_before = len(generated_json)
    json_ids: List[int] = model.encode(generated_json)[0].tolist()
    tokens_generated = 0
    while True:
        tokens_generated += 1
        if (param_type.type == ParameterType.STRING and
                tokens_generated > max_string_tokens):
            raw_content = generated_json[len_before:]
            try:
                value = json.loads('"' + raw_content + '"')
            except (json.JSONDecodeError, ValueError):
                value = raw_content
            generated_json = (generated_json[:len_before]
                              + json.dumps(value)[1:-1] + '"')
            break
        input_ids = system_ids + json_ids
        logits = model.get_logits_from_input_ids(input_ids)
        mask = [float("-inf")] * len(logits)
        result = generated_json[len_before:]

        if (param_type.type == ParameterType.NUMBER or
                param_type.type == ParameterType.INTEGER or
                param_type.type == ParameterType.FLOAT):
            for tid, token_str in token_sets["number"]:
                candidate = result + token_str
                if all(c.isdigit() or c in ".-,}" for c in candidate.strip()):
                    mask[tid] = logits[tid]

        elif param_type.type == ParameterType.STRING:
            mask = list(logits)

        elif param_type.type == ParameterType.BOOLEAN:
            for tid, token_str in token_sets["boolean"]:
                candidate = result + token_str
                if ("True".startswith(candidate) or
                        "False".startswith(candidate)):
                    mask[tid] = logits[tid]
        else:
            raise ValueError(f"unknown parameter type: {param_type.type}")

        next_token = int(np.argmax(mask))
        next_text = model.decode([next_token])
        if (param_type.type == ParameterType.STRING and
                not result and
                len(next_text) > 1 and
                next_text.startswith('"')):
            next_text = next_text[1:]
        json_ids += model.encode(next_text)[0].tolist()
        result += next_text
        generated_json += next_text

        if (param_type.type == ParameterType.NUMBER or
                param_type.type == ParameterType.INTEGER or
                param_type.type == ParameterType.FLOAT):
            if "," in next_text or "}" in next_text or "\n" in next_text:
                generated_json = generated_json.rstrip(",}")
                break
        elif param_type.type == ParameterType.STRING:
            if '"' in next_text:
                partial = generated_json[len_before:]
                i = 0
                found = False
                while i < len(partial):
                    if partial[i] == '\\' and i + 1 < len(partial):
                        i += 2
                    elif partial[i] == '"':
                        raw_content = partial[:i]
                        try:
                            value = json.loads('"' + raw_content + '"')
                        except (json.JSONDecodeError, ValueError):
                            value = raw_content
                        generated_json = (generated_json[:len_before]
                                          + json.dumps(value)[1:-1] + '"')
                        found = True
                        break
                    else:
                        i += 1
                if found:
                    break
        elif param_type.type == ParameterType.BOOLEAN:
            if generated_json[len_before:].strip() in ["True", "False"]:
                break

    return generated_json


def cast_args(args: Dict[str, Any],
              fn_def: FunctionDefinition) -> Dict[str, Any]:
    """
    Cast each argument to the type declared in the function definition.
    """
    casted: Dict[str, Any] = {}
    for param_name, param_type in fn_def.parameters.items():
        if param_name not in args:
            continue
        value = args[param_name]
        try:
            if param_type.type == ParameterType.INTEGER:
                casted[param_name] = int(float(value))
            elif (param_type.type == ParameterType.NUMBER or
                  param_type.type == ParameterType.FLOAT):
                casted[param_name] = float(value)
            else:
                casted[param_name] = value
        except (ValueError, TypeError):
            casted[param_name] = value
    return casted


def process_prompt(
        model: Small_LLM_Model,
        prompt: str,
        functions: List[FunctionDefinition],
        token_sets: Dict[str, List[Tuple[int, str]]]) -> Dict[str, Any]:
    """
    Process a single prompt and return the function call result.
    """
    system_context = build_system_context(functions)
    max_string_tokens = len(prompt) * 3
    safe_prompt = prompt.replace('"', '\\"')
    json_safe_prompt = json.dumps(prompt)[1:-1]
    system_context += (f'Given this request: {safe_prompt}\n'
                       ' The most appropiate function is: "')
    generated_json = f'{{"prompt": "{json_safe_prompt}", "fn_name": "'

    system_ids = model.encode(system_context)[0].tolist()
    fn_name = generate_function_name(
        model, system_ids, generated_json, functions, token_sets)
    generated_json += f'{fn_name}", "args": {{'

    fn_def = next(fn for fn in functions if fn.name == fn_name)
    params_items = list(fn_def.parameters.items())
    for i, (param_name, param_type) in enumerate(params_items):
        generated_json += f'"{param_name}": '
        if param_type.type == ParameterType.STRING:
            generated_json += '"'
        generated_json = generate_argument(
            model, system_ids, generated_json, param_type,
            token_sets, max_string_tokens)
        if i < len(params_items) - 1:
            generated_json += ", "
    generated_json = generated_json.rstrip(", \n") + "}}"
    try:
        parsed = json.loads(generated_json)
        return {
            "prompt": prompt,
            "name": parsed["fn_name"],
            "parameters": cast_args(parsed["args"], fn_def)
        }
    except Exception:
        return {
            "prompt": prompt,
            "name": fn_name,
            "parameters": {}
        }
