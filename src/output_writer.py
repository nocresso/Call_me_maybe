import json
from pathlib import Path
from typing import Dict, Any, List, Union


def write_output(result: List[Dict[str, Any]],
                 output_path: Union[str, Path]) -> None:
    """
    Write the function call results to a JSON file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
