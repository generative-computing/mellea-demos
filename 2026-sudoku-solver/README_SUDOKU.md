---
title: Sudoku Solver - LLM-Powered Code Generation & Execution
description: LLM-based sudoku solver combining image recognition, code generation, and Mellea's tool-use framework
sidebar_label: Sudoku Solver
---

A sophisticated sudoku solver that combines image recognition, LLM-based code generation, and
Mellea's tool-use framework to automatically solve sudoku puzzles from images.

## Overview

This program demonstrates a cutting-edge approach to sudoku solving:

1. **Image Extraction**: Reads sudoku puzzles from colored/highlighted grid images using OpenCV
   and Tesseract OCR
2. **LLM Code Generation**: Uses an LLM (via Mellea) to generate Python solver code tailored to
   the extracted puzzle
3. **Tool-Based Execution**: Executes the generated code safely using Mellea's `python_tool` with
   validation
4. **Fallback Safety**: If tool execution fails, falls back to direct code execution with explicit
   user confirmation
5. **Solution Verification**: Validates the solution against all sudoku constraints (rows, columns,
   3×3 boxes)

## Features

- **OCR-Powered Grid Extraction**: Automatically detects and extracts sudoku numbers from images
- **Smart Image Processing**: Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) for
  robust extraction
- **LLM-Guided Solving**: Leverages LLM capabilities to generate optimized solving algorithms
- **Safe Code Execution**: Tool-based execution with fallback confirmation for security
- **Comprehensive Validation**: 8-point verification (format, values, uniqueness, constraints)
- **Difficulty Estimation**: Automatically estimates puzzle difficulty based on given numbers
- **Pretty Printing**: Formatted output with 3×3 box separators

## Requirements

### System Dependencies

- **Tesseract OCR**: For optical character recognition

  ```bash
  # macOS
  brew install tesseract

  # Ubuntu/Debian
  sudo apt-get install tesseract-ocr
  ```

- **Ollama**: For running the LLM locally

  ```bash
  # Install from https://ollama.ai
  # Pull the required model
  ollama pull granite4.1:3b
  ```

### Python Dependencies

```bash
# Using pip
pip install pytesseract opencv-python numpy pydantic mellea

# Or with uv (recommended for Mellea)
uv sync --extra backends
```

## Installation

1. Clone the repository and navigate to the project directory:

   ```bash
   cd mellea
   ```

2. Install system dependencies (Tesseract, Ollama)

3. Install Python dependencies:

   ```bash
   uv sync --extra backends --all-groups
   ```

4. Start Ollama:

   ```bash
   ollama serve
   ```

## Usage

Run the sudoku solver with an image of a sudoku puzzle:

```bash
python sudoku_tool_based.py <image_path>
```

### Example

```bash
python sudoku_tool_based.py sudoku_puzzle.jpg
```

### Output

The program will:

1. Extract the sudoku grid from the image
2. Display the extracted puzzle
3. Generate and execute solving code
4. Verify the solution
5. Display the solved puzzle with difficulty estimation

Example output:

```text
======================================================================
SUDOKU SOLVER - Tool-Based Code Generation + Execution (via Mellea)
======================================================================

📷 Sudoku image: sudoku_puzzle.jpg

📋 Extracted puzzle:
1 2 3 | 4 5 6 | 7 8 0
4 5 6 | 7 8 9 | 1 2 3
7 8 9 | 1 2 3 | 4 5 6
------+-------+------
...

✓ Grid extracted (45 numbers, 36 empty cells)

⚙️  Generating and executing sudoku solver code...

📋 Creating python_tool for code execution...
   ✓ python_tool created with tier='local_unsafe'

🔄 Calling session.instruct() with tool_use...
   Tool name: 'python'
   Tool tier: local_unsafe
   Tool calls enabled: True

✓ Solution extracted from tool execution
✓ Solution verified

✨ Solved! (medium, 12 steps)

🎯 Solution:
1 2 3 | 4 5 6 | 7 8 9
4 5 6 | 7 8 9 | 1 2 3
7 8 9 | 1 2 3 | 4 5 6
------+-------+------
...

======================================================================
Success!
======================================================================
```

## How It Works

### 1. Image Extraction (`extract_numbers_from_colored_grid`)

- Reads image with OpenCV
- Converts to HSV color space for robust detection
- Applies CLAHE for enhanced contrast
- Thresholds to binary image
- Detects grid using contour analysis
- Extracts each cell and applies OCR (Tesseract)
- Tries multiple threshold values for robustness

**Key Constants:**

- `CLAHE_CLIP_LIMIT = 2.0`: Contrast enhancement strength
- `CLAHE_TILE_GRID_SIZE = (8, 8)`: Processing tile size
- `BINARY_THRESHOLD = 200`: Binary conversion threshold
- `OCR_THRESHOLD_VALUES = [100, 120, 150, 180, 200]`: Multi-attempt thresholds

### 2. LLM Code Generation (`generate_and_execute_sudoku_solver_with_tools`)

- Formats extracted grid as text
- Sends detailed prompt to LLM with:
  - Input sudoku puzzle
  - Implementation requirements
  - Sudoku rules
  - Example code structure
- LLM generates complete solver function
- Mellea validates code against requirements

**Key Features:**

- Tool-use enabled for safe execution
- Repair strategy: 1 initial + 1 repair attempt
- Output limit: 10,000 characters
- Timeout: 10 seconds

### 3. Fallback Execution

If tool-based execution fails:

1. Extracts code from text response using flexible regex patterns
2. Shows code to user for review
3. **Requires explicit "yes" confirmation** before execution
4. Executes in isolated namespace

### 4. Solution Verification (`verify_sudoku_solution`)

Performs 8-point validation:

1. ✓ Exactly 9 rows
2. ✓ Exactly 9 cells per row
3. ✓ All cells contain 1-9 (no zeros)
4. ✓ Row uniqueness (no duplicates)
5. ✓ Column uniqueness (no duplicates)
6. ✓ 3×3 box uniqueness (no duplicates)
7. ✓ All cells filled (no empty cells)
8. ✓ Respects original givens (unchanged)

### 5. Difficulty Estimation

Based on number of given cells:

- **Easy** (≥50 numbers): 5 estimated steps
- **Medium** (≥35 numbers): 12 estimated steps
- **Hard** (≥25 numbers): 25 estimated steps
- **Expert** (<25 numbers): 40 estimated steps

## Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                    sudoku_tool_based.py                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ extract_numbers_from_colored_grid()                 │   │
│  │ • OpenCV image processing                           │   │
│  │ • Tesseract OCR                                     │   │
│  │ • Multi-threshold retry logic                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ SudokuGrid (Pydantic)                               │   │
│  │ • Validates 9x9 structure                           │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ generate_and_execute_sudoku_solver_with_tools()     │   │
│  │ • Format grid for LLM                               │   │
│  │ • LLM code generation via Mellea                    │   │
│  │ • Tool-based execution (local_unsafe)               │   │
│  │ • Fallback with user confirmation                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ verify_sudoku_solution()                            │   │
│  │ • 8-point validation                                │   │
│  │ • Constraint checking                               │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ SudokuSolution (Pydantic)                           │   │
│  │ • Solved grid                                       │   │
│  │ • Estimated steps & difficulty                      │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Constants

### Sudoku Grid

- `SUDOKU_SIZE = 9`: Grid dimensions
- `SUDOKU_BOX_SIZE = 3`: 3×3 box dimensions
- `SUDOKU_MIN_NUMBERS = 17`: Minimum valid puzzle numbers

### Image Processing

- `OCR_DARK_PIXEL_THRESHOLD = 0.02`: Dark pixel detection ratio
- `OCR_DARK_PIXEL_VALUE = 180`: Gray value threshold for dark pixels
- `OCR_THRESHOLD_VALUES = [100, 120, 150, 180, 200]`: Multi-attempt thresholds
- `CELL_MARGIN_DIVISOR = 6`: Cell margin from borders
- `CLAHE_CLIP_LIMIT = 2.0`: CLAHE enhancement strength
- `CLAHE_TILE_GRID_SIZE = (8, 8)`: CLAHE tile size
- `BINARY_THRESHOLD = 200`: Binary conversion threshold

### LLM Code Generation

- `CODE_OUTPUT_LIMIT = 10_000`: Max output characters
- `CODE_TIMEOUT_SECONDS = 10`: Execution timeout
- `REPAIR_LOOP_BUDGET = 2`: Initial + repair attempts

### Difficulty Thresholds

- `DIFFICULTY_EASY_THRESHOLD = 50`
- `DIFFICULTY_MEDIUM_THRESHOLD = 35`
- `DIFFICULTY_HARD_THRESHOLD = 25`

## Error Handling

The program handles various failure modes:

1. **Image Not Found**: FileNotFoundError with clear message
2. **Image Cannot Be Read**: ValueError from OpenCV
3. **Grid Not Detected**: ValueError from contour detection
4. **OCR Failures**: Debug logging with cell coordinates
5. **Code Generation Failure**: ValueError with details
6. **JSON Parsing Failure**: Debug logging with raw output
7. **Solution Verification Failure**: ValueError with specific constraint violated
8. **User Declines Execution**: ValueError when user responds "no" to confirmation

All exceptions are caught in main() with full traceback printed to stderr.

## Security Considerations

### Safe Execution

1. **Tool-Based Primary**: Executes code via Mellea's safe `python_tool` tier="local_unsafe"
2. **Explicit Fallback Confirmation**: Requires user to type "yes" before executing fallback code
3. **Code Review**: User reviews entire generated code before confirmation
4. **Input Validation**: Pydantic models validate extracted grids
5. **Output Validation**: 8-point solution verification before returning

### Limitations

- Uses `tier="local_unsafe"` for local-only execution (not suitable for untrusted inputs)
- Fallback path uses `exec()` which is inherently risky (mitigated by user confirmation)
- OCR accuracy depends on image quality

## Troubleshooting

### Common Issues

#### "Could not detect sudoku grid"

- Ensure image has clear grid lines and high contrast
- Try a different sudoku puzzle image
- Check that Tesseract is properly installed

#### "Ollama refused" or connection errors

- Start Ollama: `ollama serve`
- Verify model is installed: `ollama list`
- Check model: `ollama pull granite4.1:3b`

#### OCR extraction not working

- Enable debug logging: `logging.basicConfig(level=logging.DEBUG)`
- Check image quality and contrast
- Adjust OCR constants if needed

#### Tool execution failed, prompting for confirmation

- Review the displayed code carefully
- Only type "yes" if the code looks safe
- The code should define `solve_sudoku()` and print JSON output

## Development

### Code Organization

- `extract_numbers_from_colored_grid()`: Image processing and OCR
- `generate_and_execute_sudoku_solver_with_tools()`: LLM interaction and execution
- `verify_sudoku_solution()`: Constraint validation
- `format_sudoku_grid()`: Output formatting
- `main_sync()`: CLI entry point

### Testing

```bash
# Run with a test image
python sudoku_tool_based.py test_sudoku.jpg

# With debug logging
DEBUG=1 python sudoku_tool_based.py test_sudoku.jpg
```

### Extending

To customize:

1. Modify image processing constants for different image types
2. Adjust LLM prompt (`SUDOKU_SOLVER_PROMPT_TEMPLATE`) for different solvers
3. Change difficulty thresholds for different estimation logic
4. Add new validation checks in `verify_sudoku_solution()`

## Project Context

This program is part of the **Mellea** project, demonstrating:

- LLM tool-use frameworks
- Safe code execution patterns
- Multi-step AI workflows
- Integration of computer vision with LLM capabilities

For more information, see the [Mellea documentation](../README.md).

## License

Same as Mellea project

## Author

Generated for Mellea demonstration

## Contributing

Contributions welcome! Areas for improvement:

- Additional OCR strategies for lower-quality images
- Alternative solving algorithms from LLM
- Performance optimizations
- Batch processing for multiple puzzles
