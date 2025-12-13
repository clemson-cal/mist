"""
plot_products.py - Plot products from mist simulation output files

Usage:
    python -m mist.plot_products prods.dat
    python -m mist.plot_products prods.bin --fields concentration cell_x
    python -m mist.plot_products prods.dat -o plot.png
    python -m mist.plot_products prods.dat --list-fields
    python -m mist.plot_products lo:prods_256.dat hi:prods_8192.dat -o compare.pdf
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


def parse_file_spec(spec):
    """Parse a file specification like 'label:filename' or just 'filename'."""
    if ":" in spec:
        label, filename = spec.split(":", 1)
        return label, filename
    return None, spec


def plot_products(
    filenames, fields=None, x_field=None, output=None, title=None, figwidth=4.0
):
    """Plot products from one or more files.

    Args:
        filenames: List of file specs ('label:path' or just 'path')
        fields: List of field names to plot (None = all except x_field)
        x_field: Field to use as x-axis (None = auto-detect or use index)
        output: Output filename for plot (None = show interactively)
        title: Plot title (None = use filename)
        figwidth: Figure width in inches (default 4.0)
    """
    import matplotlib.pyplot as plt

    # Parse file specifications
    datasets = []
    for spec in filenames:
        label, filename = parse_file_spec(spec)
        products = load_products(filename)
        if label is None:
            label = Path(filename).stem
        datasets.append((label, products))

    # Use first dataset to determine fields and x_field
    first_products = datasets[0][1]
    available_fields = list(first_products.keys())

    # Auto-detect x_field if not specified
    if x_field is None:
        x_candidates = ["cell_x", "x", "position", "radius", "r"]
        for candidate in x_candidates:
            if candidate in available_fields:
                x_field = candidate
                break

    x_label = x_field if x_field else "index"

    # Determine fields to plot
    if fields is None:
        fields = [f for f in available_fields if f != x_field]
    else:
        # Validate requested fields
        for f in fields:
            if f not in available_fields:
                print(f"Warning: field '{f}' not found")
        fields = [f for f in fields if f in available_fields]

    if not fields:
        print("No fields to plot")
        return

    # Create plot with shared x-axis and 4:3 aspect ratio per subplot
    subplot_height = figwidth * 0.5
    fig, axes = plt.subplots(
        len(fields),
        1,
        figsize=(figwidth, subplot_height * len(fields)),
        squeeze=False,
        sharex=True,
    )

    for i, (ax, field_name) in enumerate(zip(axes.flat, fields)):
        for label, products in datasets:
            # Get x data for this dataset
            if x_field and x_field in products:
                x_data = products[x_field]
            else:
                x_data = np.arange(len(products[field_name]))

            y_data = products[field_name]
            ax.plot(x_data, y_data, "-", linewidth=0.8, label=label)

        # Only show x-label on bottom subplot
        if i == len(fields) - 1:
            ax.set_xlabel(x_label)
        ax.set_ylabel(field_name)
        ax.grid(True, alpha=0.3)
        if len(datasets) > 1:
            ax.legend(loc="best", fontsize="small")

    if title:
        fig.suptitle(title)
    elif len(filenames) == 1:
        fig.suptitle(Path(filenames[0]).name)

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
  %(prog)s lo:p256.dat hi:p8192.dat      # Compare two runs with labels
""",
    )

    parser.add_argument(
        "filenames",
        nargs="+",
        help="Products file(s) (.dat or .bin), optionally with label:path format",
    )
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
        "-w",
        "--width",
        type=float,
        default=4.0,
        help="Figure width in inches (default: 4.0)",
    )
    parser.add_argument(
        "--list-fields",
        action="store_true",
        help="List available fields and exit",
    )

    args = parser.parse_args()

    # Check files exist
    for spec in args.filenames:
        _, filename = parse_file_spec(spec)
        if not Path(filename).exists():
            print(f"Error: file not found: {filename}")
            sys.exit(1)

    if args.list_fields:
        for spec in args.filenames:
            _, filename = parse_file_spec(spec)
            list_fields(filename)
        sys.exit(0)

    plot_products(
        args.filenames,
        fields=args.fields,
        x_field=args.x_field,
        output=args.output,
        title=args.title,
        figwidth=args.width,
    )


if __name__ == "__main__":
    main()
