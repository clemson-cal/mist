"""
plot_products.py - Plot products from mist simulation output files

Usage:
    python -m mist.plot_products prods.dat
    python -m mist.plot_products prods.bin --fields concentration cell_x
    python -m mist.plot_products prods.dat -o plot.png
    python -m mist.plot_products prods.dat --list-fields
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def load_products(filename):
    """Load products file and return dict of field_name -> assembled array."""
    from . import mist_archive as ma

    data = ma.load(filename)
    result = {}

    for field_name, partitions in data.items():
        # Determine total size from partitions
        total_size = 0
        for p in partitions:
            start = p["start"]
            shape = p["shape"]
            # Handle 1D case
            if len(start) == 1:
                end = start[0] + shape[0]
                total_size = max(total_size, end)

        # Assemble the full array
        assembled = np.zeros(total_size)
        for p in partitions:
            start = int(p["start"][0])
            data_arr = p["data"]
            assembled[start : start + len(data_arr)] = data_arr

        result[field_name] = assembled

    return result


def list_fields(filename):
    """List available fields in products file."""
    from . import mist_archive as ma

    data = ma.load(filename)
    print(f"Fields in {filename}:")
    for field_name, partitions in data.items():
        total_size = 0
        for p in partitions:
            start = p["start"]
            shape = p["shape"]
            if len(start) == 1:
                end = start[0] + shape[0]
                total_size = max(total_size, end)
        print(f"  {field_name}: {total_size} elements, {len(partitions)} partitions")


def plot_products(filename, fields=None, x_field=None, output=None, title=None):
    """Plot products from file.

    Args:
        filename: Path to products file (.dat or .bin)
        fields: List of field names to plot (None = all except x_field)
        x_field: Field to use as x-axis (None = auto-detect or use index)
        output: Output filename for plot (None = show interactively)
        title: Plot title (None = use filename)
    """
    import matplotlib.pyplot as plt

    products = load_products(filename)
    available_fields = list(products.keys())

    # Auto-detect x_field if not specified
    if x_field is None:
        x_candidates = ["cell_x", "x", "position", "radius", "r"]
        for candidate in x_candidates:
            if candidate in available_fields:
                x_field = candidate
                break

    # Get x data
    if x_field and x_field in products:
        x_data = products[x_field]
        x_label = x_field
    else:
        # Use index as x
        first_field = available_fields[0]
        x_data = np.arange(len(products[first_field]))
        x_label = "index"

    # Determine fields to plot
    if fields is None:
        fields = [f for f in available_fields if f != x_field]
    else:
        # Validate requested fields
        for f in fields:
            if f not in available_fields:
                print(f"Warning: field '{f}' not found in {filename}")
        fields = [f for f in fields if f in available_fields]

    if not fields:
        print("No fields to plot")
        return

    # Create plot
    fig, axes = plt.subplots(len(fields), 1, figsize=(10, 3 * len(fields)), squeeze=False)

    for ax, field_name in zip(axes.flat, fields):
        y_data = products[field_name]
        ax.plot(x_data, y_data, "-", linewidth=1)
        ax.set_xlabel(x_label)
        ax.set_ylabel(field_name)
        ax.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title)
    else:
        fig.suptitle(Path(filename).name)

    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150)
        print(f"Saved plot to {output}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot products from mist simulation output files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s prods.dat                     # Plot all fields
  %(prog)s prods.bin -f concentration    # Plot specific field
  %(prog)s prods.dat -o plot.png         # Save to file
  %(prog)s prods.dat --list-fields       # List available fields
  %(prog)s prods.dat -x cell_x           # Use cell_x as x-axis
""",
    )

    parser.add_argument("filename", help="Products file (.dat or .bin)")
    parser.add_argument(
        "-f",
        "--fields",
        nargs="+",
        help="Fields to plot (default: all)",
    )
    parser.add_argument(
        "-x",
        "--x-field",
        help="Field to use as x-axis (default: auto-detect cell_x, x, etc.)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output filename for plot (default: show interactively)",
    )
    parser.add_argument(
        "-t",
        "--title",
        help="Plot title (default: filename)",
    )
    parser.add_argument(
        "--list-fields",
        action="store_true",
        help="List available fields and exit",
    )

    args = parser.parse_args()

    if not Path(args.filename).exists():
        print(f"Error: file not found: {args.filename}")
        sys.exit(1)

    if args.list_fields:
        list_fields(args.filename)
        sys.exit(0)

    plot_products(
        args.filename,
        fields=args.fields,
        x_field=args.x_field,
        output=args.output,
        title=args.title,
    )


if __name__ == "__main__":
    main()
