import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.input_validation import load_prompts, load_functions  # noqa: E402
from src.llm_caller import (process_prompt, load_vocabulary,  # noqa: E402
                            precompute_token_sets)
from llm_sdk import Small_LLM_Model  # type: ignore  # noqa: E402
from src.output_writer import write_output  # noqa: E402
import argparse  # noqa: E402


def main() -> None:
    """
    Entry point of the program.
    Parses --input and --output arguments, loads prompts and function
    definitions, runs constrained decoding for each prompt and writes
    the results to a JSON file.
    """
    try:
        BASE_DIR = Path(__file__).parent.parent
        default_input = (BASE_DIR / "data" / "input"
                         / "function_calling_tests.json")
        default_functions = (BASE_DIR / "data" /
                             "input" / "functions_definition.json")
        default_output = BASE_DIR / "data" / "output"

        parser = argparse.ArgumentParser()
        parser.add_argument("--input", default=default_input)
        parser.add_argument("--functions_definition",
                            default=default_functions)
        parser.add_argument("--output", default=default_output)
        args = parser.parse_args()

        input_path = Path(args.input)
        functions_path = Path(args.functions_definition)
        output_path = Path(args.output)
        if output_path.is_dir() or not output_path.suffix:
            output_path = output_path / "function_calling_results.json"
        if output_path.suffix and output_path.suffix != ".json":
            raise ValueError("Output file must be a .json file.")

        prompts = load_prompts(input_path)
        functions = load_functions(functions_path)

        model = Small_LLM_Model()
        vocab = load_vocabulary(model)
        token_sets = precompute_token_sets(vocab)

        results = []
        for prompt in prompts:
            result = process_prompt(model, prompt, functions, token_sets)
            results.append(result)
        write_output(results, output_path)

    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.stderr.write(f"Unexpected error: {e}")
