# Seam Carving With Visualization

A Python implementation of the content-aware image resizing technique known as Seam Carving, with visualization capabilities and parallel processing options.

## Introduction

Seam Carving is an algorithm for content-aware image resizing that preserves important features while removing less noticeable pixels. Unlike traditional resizing methods that uniformly scale the entire image, Seam Carving intelligently identifies and removes "seams" (paths of least important pixels) to achieve size reduction while maintaining the image's key visual elements.

This project provides:
- A standard implementation of the Seam Carving algorithm
- A parallel implementation for improved performance
- Visualization of the seam removal process

## Installation

```bash
# Clone the repository
git clone https://github.com/AbdelrahmanGalhom/Seam-Carving-With-Visualization.git
cd Seam-Carving-With-Visualization

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Standard Implementation

```bash
python main.py --input <input_image> --height <target_height> --width <target_width> [--visualize]
```

### Parallel Implementation

```bash
python parallel/main_parallel.py --input <input_image> --height <target_height> --width <target_width> [--visualize] [--workers <num_workers>] [--compare]
```

### Arguments

**Standard Implementation**
- `--input`: Path to input image (default: "input.jpg")
- `--height`: Target output height (required)
- `--width`: Target output width (required)
- `--visualize`: Display visualization after processing

**Parallel Implementation**
- `--input`: Path to input image (default: "input.jpg")
- `--height`: Target output height (required)
- `--width`: Target output width (required)
- `--visualize`: Display visualization after processing
- `--workers`: Number of worker processes/threads (0 = auto)
- `--compare`: Compare with non-parallelized version

## Examples

### Basic Usage

Resize an image to 400x300 pixels:

```bash
python main.py --input img.png --height 400 --width 300
```

### With Visualization

Resize with visualization:

```bash
python main.py --input img.png --height 400 --width 300 --visualize
```
#### Example of output
![Image1](https://github.com/user-attachments/assets/aa71a001-5cf4-4425-85d4-0b09bc5eb665)
![Image2](https://github.com/user-attachments/assets/d4952ca0-a985-4c37-8401-9036ac44fb2c)

### Parallel Processing

Resize using parallel implementation with 4 worker threads:

```bash
python parallel/main_parallel.py --input parallel/landscape.jpg --height 500 --width 400 --workers 4
```

### Performance Comparison

Compare performance between standard and parallel implementations:

```bash
python parallel/main_parallel.py --input parallel/landscape.jpg --height 500 --width 400 --compare
```

## Output Files

For each input image processed, the following output files are generated:

1. `<image_name>_output.jpg` - The resized image
2. `<image_name>_visualization.jpg` - Visualization of the seam carving process

For the parallel implementation, the output files are named `<image_name>_output_parallel.jpg` and `<image_name>_visualization_parallel.jpg`.

## Implementation Details

### Standard Implementation

The standard implementation is contained in the SeamCarving.py module and uses dynamic programming to identify optimal seams for removal. The DynamicMapping class is used to track pixel positions between the original and resized images.

### Parallel Implementation

The parallel implementation in parallel_seam_carving.py uses numba's JIT compilation and parallel processing to optimize the seam carving process, significantly improving performance for larger images.

## Algorithm Overview

1. **Energy Calculation**: Calculate the energy of each pixel using gradient filters
2. **Seam Identification**: Use dynamic programming to find the path of least energy
3. **Seam Removal**: Remove the identified seam and update the image
4. **Visualization**: Track and visualize the removed seams


## Performance

The parallel implementation typically offers significant performance improvements, especially for larger images. The exact speedup depends on:
- Image size
- Number of seams to remove
- Available CPU cores
- Hardware specifications


## Credits

Implementation by Abdelrahman Galhom & Menna Noseer.

Based on the Seam Carving algorithm by Shai Avidan and Ariel Shamir.