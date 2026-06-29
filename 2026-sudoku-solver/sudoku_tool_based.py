#!/usr/bin/env python3
"""Sudoku solver using LLM-generated code with Mellea tools and requirements.

This version:
1. Extracts sudoku numbers from colored/highlighted grid images
2. Generates Python code via LLM to solve the sudoku
3. Uses Mellea python_tools.py requirements for validation
4. Uses python_tool from mellea/stdlib/tools/interpreter.py for execution
5. Enables tool_use=True for the instruct call

Usage:
    python sudoku_tool_based.py <image_path>

Prerequisites:
    - Tesseract OCR installed: brew install tesseract
    - Python packages: pip install pytesseract opencv-python numpy
    - Ollama running with: ollama pull granite4.1:3b
    - Mellea installed
"""

import copy
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

try:
    import cv2
except ImportError:
    raise ImportError(
        "OpenCV is required for image processing. "
        "Install it with: pip install opencv-python"
    )

try:
    import pytesseract
except ImportError:
    raise ImportError(
        "pytesseract is required for OCR. "
        "Install it with: pip install pytesseract "
        "and ensure Tesseract is installed (brew install tesseract on macOS)"
    )

import numpy as np
from pydantic import BaseModel, Field

from mellea import start_session
from mellea.backends import ModelOption
from mellea.stdlib.context import ChatContext
from mellea.stdlib.requirements import uses_tool
from mellea.stdlib.requirements.python_tools import python_code_generation_requirements
from mellea.stdlib.sampling import RepairTemplateStrategy
from mellea.stdlib.tools.interpreter import python_tool

# Set Mellea logging to INFO level
logging.getLogger("mellea").setLevel(logging.INFO)

# Constants for sudoku grid and image processing
SUDOKU_SIZE = 9
SUDOKU_BOX_SIZE = 3
SUDOKU_MIN_NUMBERS = 17  # Minimum numbers for valid sudoku puzzle

# Image processing constants
OCR_DARK_PIXEL_THRESHOLD = 0.02
OCR_DARK_PIXEL_VALUE = 180
OCR_THRESHOLD_VALUES = [100, 120, 150, 180, 200]
CELL_MARGIN_DIVISOR = 6
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
BINARY_THRESHOLD = 200

# LLM code generation constants
CODE_OUTPUT_LIMIT = 10_000  # Characters
CODE_TIMEOUT_SECONDS = 10
REPAIR_LOOP_BUDGET = 2  # Initial + 1 repair attempt

# Difficulty estimation thresholds (based on number of given cells)
DIFFICULTY_EASY_THRESHOLD = 50
DIFFICULTY_MEDIUM_THRESHOLD = 35
DIFFICULTY_HARD_THRESHOLD = 25

# LLM prompt for sudoku code generation
SUDOKU_SOLVER_PROMPT_TEMPLATE = """Generate code for python_tool to solve the following sudoku puzzle.

IMPORTANT: Generate code to be executed by the python_tool. Use the python_tool to run this code.

INPUT SUDOKU (0 = empty cell):
{grid_str}

REQUIREMENTS:
1. Write a function named 'def solve_sudoku(grid):' that takes a 9x9 sudoku grid as a LIST OF LISTS
2. The input grid is a list of 9 rows, where each row is a list of 9 integers
3. Access cells using grid[row][col] notation (2D indexing), NOT flattened 1D indexing
4. The function should solve using backtracking or constraint propagation
5. Do NOT modify any non-zero (given) numbers from the input - preserve all given numbers
6. Fill empty cells (0s) with numbers 1-9 following sudoku rules
7. Return the completed 9x9 grid as a list of lists (same 2D format as input)
8. Include necessary helper functions (e.g., is_valid, find_empty, solve, etc.)
9. After defining the function, call it with the provided grid and print the solution as JSON

CRITICAL IMPLEMENTATION NOTES:
- Use grid[row][col] to access and modify cells (row and col are 0-8)
- When searching for empty cells, iterate: for row in range(9): for col in range(9):
- Validate row constraint: check all cells in grid[row][0:9]
- Validate column constraint: check all cells in grid[0:9][col]
- Validate 3x3 box constraint: for i in range(3): for j in range(3): check grid[box_row+i][box_col+j]

SUDOKU RULES:
- Each row must have unique numbers 1-9 (no duplicates except 0)
- Each column must have unique numbers 1-9 (no duplicates except 0)
- Each 3x3 box must have unique numbers 1-9 (no duplicates except 0)

EXECUTION REQUIREMENTS:
1. Define the solve_sudoku function with all helper functions
2. Create the input grid from the provided numbers
3. Call: solution = solve_sudoku(input_grid)
4. Print the result as JSON: import json; print(json.dumps({{"solved_grid": solution}}))

EXAMPLE IMPLEMENTATION STRUCTURE:
```python
def solve_sudoku(grid):
    solved = [row[:] for row in grid]

    def is_valid(row, col, num):
        # Check if num can be placed at solved[row][col]
        for i in range(9):
            if solved[row][i] == num or solved[i][col] == num:
                return False
        box_row, box_col = (row // 3) * 3, (col // 3) * 3
        for i in range(3):
            for j in range(3):
                if solved[box_row + i][box_col + j] == num:
                    return False
        return True

    def find_empty():
        for row in range(9):
            for col in range(9):
                if solved[row][col] == 0:
                    return (row, col)
        return None

    def solve():
        cell = find_empty()
        if cell is None:
            return True
        row, col = cell
        for num in range(1, 10):
            if is_valid(row, col, num):
                solved[row][col] = num
                if solve():
                    return True
                solved[row][col] = 0
        return False

    solve()
    return solved

# Execute with the provided sudoku grid
input_grid = {grid}
solution = solve_sudoku(input_grid)
import json
print(json.dumps({{"solved_grid": solution}}))
```

Generate ONLY the Python code. Make sure to:
- Define all helper functions
- Include the input grid from the provided numbers
- Call the solver
- Print the JSON result"""


class SudokuGrid(BaseModel):
    """Represents a 9x9 sudoku grid extracted from an image."""

    grid: list[list[int]] = Field(
        ...,
        description="9x9 grid where 0 represents empty cells and 1-9 represent given numbers",
    )


class SudokuSolution(BaseModel):
    """Represents a solved sudoku grid with metadata.

    Contains the completed grid, estimated solving steps, and difficulty level
    based on the number of given numbers in the original puzzle.
    """

    solved_grid: list[list[int]] = Field(
        ..., description="The completed 9x9 sudoku grid with all cells filled (1-9)"
    )
    steps_to_solve: int = Field(
        ...,
        description="Estimated number of logical steps required to solve the puzzle",
    )
    difficulty: str = Field(
        ..., description="Estimated difficulty level: easy, medium, hard, or expert"
    )


def extract_numbers_from_colored_grid(image_path: str) -> list[list[int]]:
    """Extract sudoku numbers from a colored grid image.

    Optimized for sudoku images with colored backgrounds (blue, yellow, etc).

    Args:
        image_path: Path to sudoku image

    Returns:
        9x9 grid of integers (0 for empty cells, 1-9 for filled)

    Raises:
        ValueError: If image file cannot be read
        ValueError: If sudoku grid cannot be detected in the image
    """
    # Read image in color
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Convert to RGB for processing
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to HSV to handle colors better
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Extract value channel (brightness) - works well for both colored and white cells
    _, _, v = cv2.split(img_hsv)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE
    )
    enhanced = clahe.apply(v)

    # Threshold to binary - dark text on light backgrounds
    _, binary = cv2.threshold(enhanced, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    # Find grid lines using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("Could not detect sudoku grid")

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Extract grid region from original image
    grid_region = img_rgb[y : y + h, x : x + w]

    # Cell dimensions
    cell_w = w // SUDOKU_SIZE
    cell_h = h // SUDOKU_SIZE

    # Extract numbers from each cell
    grid = []
    for row_idx in range(SUDOKU_SIZE):
        grid_row = []
        for col_idx in range(SUDOKU_SIZE):
            # Cell boundaries
            x1 = col_idx * cell_w
            y1 = row_idx * cell_h
            x2 = x1 + cell_w
            y2 = y1 + cell_h

            # Extract cell
            cell_img = grid_region[y1:y2, x1:x2]

            # Get center region of cell (skip borders)
            margin = max(cell_w // CELL_MARGIN_DIVISOR, cell_h // CELL_MARGIN_DIVISOR)
            center = cell_img[margin:-margin, margin:-margin]

            if center.size == 0:
                grid_row.append(0)
                continue

            # Convert to grayscale
            center_gray = cv2.cvtColor(center, cv2.COLOR_RGB2GRAY)

            # Check if cell has significant dark content (text/number)
            dark_pixels = np.sum(center_gray < OCR_DARK_PIXEL_VALUE)
            total_pixels = center_gray.size

            if dark_pixels < total_pixels * OCR_DARK_PIXEL_THRESHOLD:
                # Cell is mostly empty/light
                grid_row.append(0)
            else:
                # Try multiple thresholding strategies
                extracted_num = 0

                try:
                    # Try different threshold values to find the best one
                    best_result = None

                    for thresh_val in OCR_THRESHOLD_VALUES:
                        _, cell_binary = cv2.threshold(
                            center_gray, thresh_val, 255, cv2.THRESH_BINARY_INV
                        )

                        # Dilate to connect broken characters
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                        cell_binary = cv2.dilate(cell_binary, kernel, iterations=1)

                        # Try Tesseract
                        text = pytesseract.image_to_string(
                            cell_binary,
                            config="--psm 10 -c tessedit_char_whitelist=0123456789",
                        )

                        # Extract digit
                        digits = [int(c) for c in text if c.isdigit()]

                        if digits:
                            num = digits[0]
                            if 1 <= num <= 9:
                                best_result = num
                                break  # Found a good result

                    if best_result:
                        extracted_num = best_result

                except Exception as e:
                    logging.debug(
                        f"OCR extraction failed for cell ({row_idx}, {col_idx}): {e}"
                    )

                grid_row.append(extracted_num)

        grid.append(grid_row)

    return grid


def generate_and_execute_sudoku_solver_with_tools(
    session: Any, grid: SudokuGrid
) -> SudokuSolution:
    """Generate Python code to solve sudoku using Mellea tools and requirements.

    Uses LLM to generate a sudoku solver function, validates the code using
    mellea/stdlib/requirements/python_tools.py requirements, and executes it
    using mellea/stdlib/tools/interpreter.py python_tool.

    Args:
        session: Mellea session for LLM inference
        grid: SudokuGrid with the puzzle to solve

    Returns:
        SudokuSolution with the solved grid and metadata

    Raises:
        ValueError: If code execution fails or returns no solution
        ValueError: If generated solution fails verification checks
        ValueError: If user declines to execute fallback code
    """
    # Format grid as string for LLM
    grid_str = "\n".join(
        " ".join(str(cell) if cell != 0 else "0" for cell in row) for row in grid.grid
    )

    prompt = SUDOKU_SOLVER_PROMPT_TEMPLATE.format(grid_str=grid_str, grid=grid.grid)

    # Create python_tool for execution
    print("\n📋 Creating python_tool for code execution...")
    tool = python_tool(
        tier="local_unsafe",  # Local execution for speed
        name="python",  # Standard tool name
    )
    print("   ✓ python_tool created with tier='local_unsafe'")

    print("\n🔄 Calling session.instruct() with tool_use...")
    print("   Tool name: 'python'")
    print("   Tool tier: local_unsafe")
    print("   Tool calls enabled: True\n")

    # Create Python code generation requirements
    code_requirements = python_code_generation_requirements(
        output_limit_chars=CODE_OUTPUT_LIMIT,
        timeout_seconds=CODE_TIMEOUT_SECONDS,
        use_sandbox=False,  # Local execution
    )
    # Call session.instruct() with tool_use to let the LLM generate and execute code
    # Telemetry (tokens, latency, errors) is automatically recorded via Mellea's metrics plugins
    result = session.instruct(
        prompt,
        requirements=[*code_requirements, uses_tool("python")],  # Require tool use first
        model_options={ModelOption.TOOLS: [tool]},
        tool_calls=True,  # Enable tool calling
        return_sampling_results=True,  # Get full sampling results for debugging
        strategy=RepairTemplateStrategy(loop_budget=REPAIR_LOOP_BUDGET),
    )

    # Extract the main result (the successful or best attempt)
    generated_code = None
    solution_grid = None

    # Get the chosen generation from the sampling result
    if hasattr(result, "sample_generations") and result.result_index >= 0:
        chosen_generation = result.sample_generations[result.result_index]
    else:
        chosen_generation = result

    # First, try to extract code from tool calls if LLM used them
    if (
        chosen_generation.tool_calls is not None
        and len(chosen_generation.tool_calls) > 0
    ):
        for _, tool_call in chosen_generation.tool_calls.items():
            # Execute the python_tool which runs the code
            exec_result = tool_call.call_func()

            if not exec_result.success:
                error_msg = (
                    exec_result.skip_message or exec_result.stderr or "Unknown error"
                )
                print(f"⚠️  Tool execution warning: {error_msg}")
            else:
                # Parse the tool output to extract the solution
                if hasattr(exec_result, "stdout") and exec_result.stdout:
                    try:
                        # Try parsing stdout as JSON
                        output_data = json.loads(exec_result.stdout)
                        if (
                            isinstance(output_data, dict)
                            and "solved_grid" in output_data
                        ):
                            solution_grid = output_data["solved_grid"]
                    except (json.JSONDecodeError, ValueError) as e:
                        logging.debug(
                            f"Failed to parse tool output as JSON: {e}\nOutput: {exec_result.stdout}"
                        )

                if solution_grid:
                    print("✓ Solution extracted from tool execution")
            break

    # If tool didn't work, fallback to extracting code from text response
    if not solution_grid:
        text_output = (
            str(chosen_generation.value).strip() if chosen_generation.value else ""
        )

        # Try to extract code block with python markers from the LLM response
        match = re.search(r"```(?:python)?\n(.*?)```", text_output, re.DOTALL)
        if not match:
            # Try code block without language tag
            match = re.search(r"```\n?(.*?)```", text_output, re.DOTALL)

        if match:
            generated_code = match.group(1)
        else:
            # Last resort: look for function definition
            if "def solve_sudoku" in text_output:
                lines = text_output.split("\n")
                start_idx = None
                for i, line in enumerate(lines):
                    if "def solve_sudoku" in line:
                        start_idx = i
                        break
                if start_idx is not None:
                    generated_code = "\n".join(lines[start_idx:])

        if generated_code:
            # Strip markdown code fences if present
            if generated_code.startswith("```"):
                lines = generated_code.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip().startswith("```"):
                    lines = lines[:-1]
                generated_code = "\n".join(lines).strip()

            # Ask user confirmation before executing untrusted code
            print(
                "\n⚠️  WARNING: Tool-based execution failed. Fallback to direct code execution."
            )
            print("The following code will be executed locally:\n")
            print("--- CODE START ---")
            print(generated_code)
            print("--- CODE END ---\n")
            confirm = input("Execute this code? (yes/no): ").strip().lower()
            if confirm != "yes":
                raise ValueError("User declined to execute generated code")

            # Execute the extracted code
            namespace: dict[str, Any] = {}
            exec(generated_code, namespace)

            if "solve_sudoku" not in namespace:
                raise ValueError("Generated code doesn't contain solve_sudoku function")

            solve_sudoku = namespace["solve_sudoku"]
            grid_to_solve = copy.deepcopy(grid.grid)

            try:
                solution_grid = solve_sudoku(grid_to_solve)
                print("✓ Solution obtained from fallback code execution")
            except Exception as e:
                raise ValueError(f"Error executing solve_sudoku: {e}")

    if not solution_grid:
        raise ValueError("Could not execute code or extract solution")

    # Verify the solution (checks format, values, and constraints)
    if not verify_sudoku_solution(solution_grid, grid.grid):
        raise ValueError("Generated solution failed verification")

    print("✓ Solution verified")

    # Estimate difficulty and estimated solving steps based on given numbers
    given_count = sum(1 for row in grid.grid for cell in row if cell != 0)
    if given_count >= DIFFICULTY_EASY_THRESHOLD:
        difficulty = "easy"
        steps = 5  # Estimated logical steps for easy puzzles
    elif given_count >= DIFFICULTY_MEDIUM_THRESHOLD:
        difficulty = "medium"
        steps = 12  # Estimated logical steps for medium puzzles
    elif given_count >= DIFFICULTY_HARD_THRESHOLD:
        difficulty = "hard"
        steps = 25  # Estimated logical steps for hard puzzles
    else:
        difficulty = "expert"
        steps = 40  # Estimated logical steps for expert puzzles

    return SudokuSolution(
        solved_grid=solution_grid, steps_to_solve=steps, difficulty=difficulty
    )


def verify_sudoku_solution(
    solution: list[list[int]], original: list[list[int]]
) -> bool:
    """Verify that the solution meets all sudoku requirements.

    Args:
        solution: The 9x9 solution grid
        original: The original puzzle grid

    Returns:
        True if solution meets all requirements, False otherwise
    """
    # 1. Format: exactly 9 lines
    if len(solution) != SUDOKU_SIZE:
        return False

    # 2. Format: 9 numbers per line
    for row in solution:
        if len(row) != SUDOKU_SIZE:
            return False

    # 3. Valid range: 1-9 only
    for row in solution:
        for cell in row:
            if not (1 <= cell <= SUDOKU_SIZE):
                return False

    # 4. Row uniqueness
    for row in solution:
        if len(set(row)) != SUDOKU_SIZE:
            return False

    # 5. Column uniqueness
    for col in range(SUDOKU_SIZE):
        column = [solution[row][col] for row in range(SUDOKU_SIZE)]
        if len(set(column)) != SUDOKU_SIZE:
            return False

    # 6. 3x3 box uniqueness
    for box_row in range(SUDOKU_BOX_SIZE):
        for box_col in range(SUDOKU_BOX_SIZE):
            box = []
            for i in range(SUDOKU_BOX_SIZE):
                for j in range(SUDOKU_BOX_SIZE):
                    row_idx = box_row * SUDOKU_SIZE // SUDOKU_BOX_SIZE + i
                    col_idx = box_col * SUDOKU_SIZE // SUDOKU_BOX_SIZE + j
                    box.append(solution[row_idx][col_idx])
            if len(set(box)) != SUDOKU_SIZE:
                return False

    # 7. All cells filled (no zeros)
    for row in solution:
        if 0 in row:
            return False

    # 8. Respects givens (doesn't change original numbers)
    for row_idx, row in enumerate(original):
        for j, cell in enumerate(row):
            if cell != 0 and solution[row_idx][j] != cell:
                return False

    return True


def format_sudoku_grid(grid: list[list[int]]) -> str:
    """Format a sudoku grid for pretty printing.

    Args:
        grid: 9x9 sudoku grid with values 1-9

    Returns:
        Formatted string representation with 3x3 box separators
    """
    lines = []
    for i, row in enumerate(grid):
        if i % SUDOKU_BOX_SIZE == 0 and i != 0:
            lines.append("------+-------+------")
        row_str = ""
        for j, cell in enumerate(row):
            if j % SUDOKU_BOX_SIZE == 0 and j != 0:
                row_str += "| "
            row_str += str(cell) + " "
        lines.append(row_str)
    return "\n".join(lines)


def main_sync() -> None:
    """Main entry point for the sudoku tool-based solver.

    Raises:
        SystemExit: With exit code 1 if no image path is provided
        SystemExit: With exit code 1 if image file not found or image processing fails
        SystemExit: With exit code 1 if sudoku solving fails or verification fails
    """
    if len(sys.argv) < 2:
        print("Usage: python sudoku_tool_based.py <image_path>")
        print("\nExample: python sudoku_tool_based.py sudoku_puzzle.jpg")
        print("\nRequirements:")
        print("  - Tesseract OCR: brew install tesseract")
        print("  - Python packages: pip install pytesseract opencv-python numpy")
        print("  - Ollama: ollama pull granite4.1:3b")
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        # Validate image exists
        img_file = Path(image_path)
        if not img_file.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        print("=" * 70)
        print("SUDOKU SOLVER - Tool-Based Code Generation + Execution (via Mellea)")
        print("=" * 70)
        print(f"\n📷 Sudoku image: {img_file.name}\n")

        # Extract numbers
        grid_data = extract_numbers_from_colored_grid(image_path)
        extracted_grid = SudokuGrid(grid=grid_data)

        print("\n📋 Extracted puzzle:")
        print(format_sudoku_grid(extracted_grid.grid))

        # Validate
        empty_count = sum(1 for row in extracted_grid.grid for cell in row if cell == 0)
        given_count = SUDOKU_SIZE * SUDOKU_SIZE - empty_count
        print(f"\n✓ Grid extracted ({given_count} numbers, {empty_count} empty cells)")

        if given_count < SUDOKU_MIN_NUMBERS:
            print(
                f"⚠️  Warning: Typical sudoku has ≥{SUDOKU_MIN_NUMBERS} numbers. Check extraction accuracy."
            )

        # Solve with code generation and tool-based execution
        print("\n⚙️  Generating and executing sudoku solver code...")
        ctx = ChatContext()
        m = start_session(model_id="granite4.1:3b", ctx=ctx)
        solution = generate_and_execute_sudoku_solver_with_tools(m, grid=extracted_grid)

        print(f"\n✨ Solved! ({solution.difficulty}, {solution.steps_to_solve} steps)")

        print("\n🎯 Solution:")
        print(format_sudoku_grid(solution.solved_grid))

        print("\n" + "=" * 70)
        print("Success!")
        print("=" * 70)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main_sync()
