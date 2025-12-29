"""Terminal UI for mist simulations.

Usage: python -m mist path/to/executable
"""

import argparse
import sys
from typing import Optional


try:
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
    from textual.widgets import (
        Button,
        Checkbox,
        Input,
        Label,
        RichLog,
        Rule,
        Static,
        TabbedContent,
        TabPane,
    )
    from textual.reactive import reactive
except ImportError:
    print("Error: textual is required for the mist TUI")
    print("Install with: pip install textual")
    sys.exit(1)

try:
    from textual_plotext import PlotextPlot

    HAVE_PLOTEXT = True
except ImportError:
    HAVE_PLOTEXT = False

from .mist_exe import Mist


class ConfigInput(Horizontal):
    """A labeled input for config values."""

    DEFAULT_CSS = """
    ConfigInput {
        height: auto;
        width: 100%;
    }
    ConfigInput Label {
        width: 16;
        text-align: right;
        padding: 0 1 0 0;
    }
    ConfigInput Input {
        width: 1fr;
    }
    """

    def __init__(self, key: str, value: str, config_type: str, **kwargs):
        super().__init__(**kwargs)
        self.config_key = key
        self.config_type = config_type  # "physics" or "initial"
        self._value = value

    def compose(self) -> ComposeResult:
        yield Label(self.config_key)
        yield Input(value=self._value, id=f"cfg-{self.config_type}-{self.config_key}")


class SimulationInfo(Static):
    """Widget showing current simulation state."""

    time = reactive(0.0)
    iteration = reactive(0)
    dt = reactive(0.0)
    initialized = reactive(False)

    def render(self) -> str:
        status = "Initialized" if self.initialized else "Not initialized"
        return (
            f"[bold]Status:[/] {status}  "
            f"[bold]Time:[/] {self.time:.6g}  "
            f"[bold]Iteration:[/] {self.iteration}  "
            f"[bold]dt:[/] {self.dt:.6g}"
        )


class MistTUI(App):
    """Terminal UI for mist simulations."""

    CSS = """
    Screen {
        layout: grid;
        grid-size: 1 4;
        grid-rows: auto 2fr 1fr auto;
    }

    #info-bar {
        height: auto;
        padding: 0 1;
        background: $surface;
    }

    #main-area {
        height: 1fr;
    }

    #sidebar {
        width: 40;
        height: 1fr;
        padding: 1;
        border-right: solid $primary;
    }

    #content-tabs {
        width: 1fr;
        height: 1fr;
    }

    #plot-content {
        width: 1fr;
        height: 1fr;
    }

    #physics-text, #initial-text, #profiler-text {
        width: 1fr;
        height: 1fr;
        padding: 1;
    }

    #console {
        height: 1fr;
        border-top: solid $primary;
        background: $surface-darken-1;
    }

    #controls {
        height: auto;
        padding: 0 1;
        background: $surface;
    }

    #control-buttons {
        height: auto;
        align: center middle;
    }

    #control-buttons Button {
        min-width: 8;
        margin: 0 1;
    }

    #advance-input {
        width: 12;
        margin: 0 1;
    }

    .section-title {
        text-style: bold;
        margin: 1 0 0 0;
    }

    #physics-config, #initial-config, #products-section {
        height: auto;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("i", "init", "Init"),
        Binding("s", "step", "Step"),
        Binding("r", "reset", "Reset"),
        Binding("g", "advance", "Go"),
    ]

    def __init__(self, executable: str):
        super().__init__()
        self.executable = executable
        self.sim: Optional[Mist] = None
        self.advance_target = 0.1

    def compose(self) -> ComposeResult:
        yield SimulationInfo(id="info-bar")
        with Horizontal(id="main-area"):
            with ScrollableContainer(id="sidebar"):
                yield Static("Physics", classes="section-title")
                with Vertical(id="physics-config"):
                    pass
                yield Static("Initial", classes="section-title")
                with Vertical(id="initial-config"):
                    pass
                yield Static("Products", classes="section-title")
                with Vertical(id="products-section"):
                    pass
            with TabbedContent(id="content-tabs"):
                with TabPane("Plot", id="tab-plot"):
                    if HAVE_PLOTEXT:
                        yield PlotextPlot(id="plot-content")
                    else:
                        yield Static("Install textual-plotext", id="plot-content")
                with TabPane("Physics", id="tab-physics"):
                    yield RichLog(id="physics-text", highlight=False, markup=False)
                with TabPane("Initial", id="tab-initial"):
                    yield RichLog(id="initial-text", highlight=False, markup=False)
                with TabPane("Profiler", id="tab-profiler"):
                    yield RichLog(id="profiler-text", highlight=False, markup=False)
        yield RichLog(id="console", highlight=True, markup=True)
        with Horizontal(id="controls"):
            with Horizontal(id="control-buttons"):
                yield Button("Init", id="btn-init", variant="success")
                yield Button("Step", id="btn-step")
                yield Button("Go", id="btn-advance", variant="primary")
                yield Input(value="0.1", id="advance-input", placeholder="t")
                yield Button("Reset", id="btn-reset", variant="warning")

    def on_mount(self) -> None:
        """Connect to the simulation on startup."""
        self.title = f"mist - {self.executable}"
        self.log_message(f"Connecting to {self.executable}...")
        try:
            self.sim = Mist(self.executable, init=False)
            self.log_message("Connected successfully")
            self.log_message("Press 'i' or click Init to initialize")
            self.refresh_configs()
            self.refresh_tab_content()
            self.refresh_all()
        except Exception as e:
            self.log_message(f"[red]Error connecting: {e}[/]")

    def refresh_tab_content(self) -> None:
        """Update the content in the config/profiler tabs."""
        if not self.sim:
            return
        try:
            for log_id, content in [
                ("#physics-text", self.sim.physics_text),
                ("#initial-text", self.sim.initial_text),
                ("#profiler-text", self.sim.profiler_text),
            ]:
                log = self.query_one(log_id, RichLog)
                log.clear()
                log.write(content)
        except Exception as e:
            self.log_message(f"[red]Error updating tabs: {e}[/]")

    def on_tabbed_content_tab_activated(
        self, event: TabbedContent.TabActivated
    ) -> None:
        """Refresh tab content when switching tabs."""
        self.refresh_tab_content()

    def refresh_configs(self) -> None:
        """Populate config inputs from simulation."""
        if not self.sim:
            return

        try:
            # Physics config
            physics_container = self.query_one("#physics-config", Vertical)
            physics_data = self.sim.physics.to_dict()
            for key, value in physics_data.items():
                input_id = f"cfg-physics-{key}"
                if not self.query(f"#{input_id}"):
                    widget = ConfigInput(key, str(value), "physics")
                    physics_container.mount(widget)

            # Initial config
            initial_container = self.query_one("#initial-config", Vertical)
            initial_data = self.sim.initial.to_dict()
            for key, value in initial_data.items():
                input_id = f"cfg-initial-{key}"
                if not self.query(f"#{input_id}"):
                    widget = ConfigInput(key, str(value), "initial")
                    initial_container.mount(widget)

        except Exception as e:
            self.log_message(f"[red]Error loading configs: {e}[/]")

    def log_message(self, message: str) -> None:
        """Add a message to the console log."""
        log = self.query_one("#console", RichLog)
        log.write(message)

    def update_info_bar(self) -> None:
        """Update the info bar with current state."""
        info = self.query_one("#info-bar", SimulationInfo)
        if self.sim:
            info.initialized = self.sim._initialized
            if self.sim._initialized:
                info.time = self.sim.time
                info.iteration = self.sim.iteration
                info.dt = self.sim.dt

    def refresh_products(self) -> None:
        """Refresh the products checkboxes."""
        if not self.sim or not self.sim._initialized:
            return
        try:
            container = self.query_one("#products-section", Vertical)
            existing_ids = {cb.id for cb in container.query(Checkbox)}

            names = self.sim.product_names
            # Exclude coordinate products since they're used as x-axis
            plot_names = [n for n in names if n not in ("cell_x", "cell_r")]

            # Only add checkboxes that don't exist yet
            for name in plot_names:
                cb_id = f"product-{name}"
                if cb_id not in existing_ids:
                    checkbox = Checkbox(name, value=True, id=cb_id)
                    container.mount(checkbox)

            # Auto-update plot after refreshing products
            self.update_plot()
        except Exception as e:
            self.log_message(f"[red]Error refreshing products: {e}[/]")

    def refresh_all(self) -> None:
        """Refresh all displays."""
        self.update_info_bar()
        self.refresh_products()  # Also updates plot
        self.refresh_tab_content()

    def action_init(self) -> None:
        """Initialize the simulation."""
        if not self.sim:
            return
        if self.sim._initialized:
            self.log_message("[yellow]Already initialized[/]")
            return
        try:
            self.log_message("Initializing...")
            self.sim.init()
            self.log_message("[green]Initialized successfully[/]")
            self.show_iteration_info()
            self.update_initial_inputs_state()
            self.refresh_all()
        except Exception as e:
            self.log_message(f"[red]Error: {e}[/]")

    def action_step(self) -> None:
        """Advance by one timestep."""
        if not self.sim:
            return
        if not self.sim._initialized:
            self.log_message("[yellow]Not initialized - press Init first[/]")
            return
        try:
            self.sim.advance_by(self.sim.dt or 0.001)
            self.show_iteration_info()
            self.update_info_bar()
            self.update_plot()
            self.refresh_tab_content()
        except Exception as e:
            self.log_message(f"[red]Error: {e}[/]")

    def action_advance(self) -> None:
        """Advance to target time."""
        if not self.sim:
            return
        if not self.sim._initialized:
            self.log_message("[yellow]Not initialized - press Init first[/]")
            return
        try:
            inp = self.query_one("#advance-input", Input)
            target = float(inp.value)
            self.sim.advance_to(target)
            self.log_message(f"[green]Advanced to t={self.sim.time:.6g}[/]")
            self.show_iteration_info()
            self.update_info_bar()
            self.update_plot()
            self.refresh_tab_content()
        except ValueError:
            self.log_message("[red]Invalid target value[/]")
        except Exception as e:
            self.log_message(f"[red]Error: {e}[/]")

    def action_reset(self) -> None:
        """Reset the simulation."""
        if not self.sim:
            return
        try:
            self.log_message("Resetting...")
            self.sim.reset()
            self.log_message("[green]Reset complete[/]")
            self.update_initial_inputs_state()
            self.refresh_all()
        except Exception as e:
            self.log_message(f"[red]Error: {e}[/]")

    def update_plot(self) -> None:
        """Update the plot with selected products vs cell_x."""
        if not HAVE_PLOTEXT:
            return
        if not self.sim or not self.sim._initialized:
            return

        try:
            container = self.query_one("#products-section", Vertical)
            selected = [
                cb.id.replace("product-", "", 1)
                for cb in container.query(Checkbox)
                if cb.value
            ]
        except Exception as e:
            self.log_message(f"[red]Error getting selected: {e}[/]")
            selected = []

        try:
            plot_widget = self.query_one("#plot-content", PlotextPlot)
            plt = plot_widget.plt
            plt.clear_figure()

            # Determine x-axis coordinate product (cell_x or cell_r)
            x_coord = None
            x_label = "x"
            if "cell_x" in self.sim.product_names:
                x_coord = "cell_x"
                x_label = "x"
            elif "cell_r" in self.sim.product_names:
                x_coord = "cell_r"
                x_label = "r"

            if selected and x_coord:
                # Select products including coordinate for x-axis
                self.sim.select_products([x_coord] + selected)

                x_values = list(self.sim.products[x_coord])
                for name in selected:
                    values = list(self.sim.products[name])
                    plt.plot(x_values, values, label=name)

                plt.title(f"t = {self.sim.time:.6g}")
                plt.xlabel(x_label)
                plt.ylabel("Value")

            plot_widget.refresh()
        except Exception as e:
            self.log_message(f"[red]Error plotting: {e}[/]")

    def show_iteration_info(self) -> None:
        """Log the current iteration info."""
        if not self.sim or not self.sim._initialized:
            return
        t = self.sim.time
        n = self.sim.iteration
        dt = self.sim.dt
        zps = self.sim.zps
        self.log_message(f"[{n:06d}] t={t:+.6e} dt={dt:.6e} zps={zps:.2e}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        if button_id == "btn-init":
            self.action_init()
        elif button_id == "btn-step":
            self.action_step()
        elif button_id == "btn-advance":
            self.action_advance()
        elif button_id == "btn-reset":
            self.action_reset()

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Handle checkbox changes for product selection."""
        if not event.checkbox.id or not event.checkbox.id.startswith("product-"):
            return
        if not self.sim or not self.sim._initialized:
            return
        self.update_plot()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle config input changes on Enter."""
        if not event.input.id or not self.sim:
            return

        input_id = event.input.id
        if not input_id.startswith("cfg-"):
            return

        # Parse input id: cfg-{type}-{key}
        parts = input_id.split("-", 2)
        if len(parts) != 3:
            return

        config_type = parts[1]  # "physics" or "initial"
        key = parts[2]
        value = event.value

        try:
            if config_type == "physics":
                self.sim.physics[key] = value
                self.log_message(f"[green]Set physics.{key} = {value}[/]")
            elif config_type == "initial":
                if self.sim._initialized:
                    self.log_message(
                        f"[yellow]Cannot modify initial.{key} - state exists, reset first[/]"
                    )
                else:
                    self.sim.initial[key] = value
                    self.log_message(f"[green]Set initial.{key} = {value}[/]")
        except Exception as e:
            self.log_message(f"[red]Error setting {config_type}.{key}: {e}[/]")

    def update_initial_inputs_state(self) -> None:
        """Enable/disable initial config inputs based on initialization state."""
        if not self.sim:
            return
        try:
            container = self.query_one("#initial-config", Vertical)
            for inp in container.query(Input):
                inp.disabled = self.sim._initialized
        except Exception:
            pass

    def on_unmount(self) -> None:
        """Clean up on exit."""
        if self.sim:
            self.sim.close()


def main():
    parser = argparse.ArgumentParser(
        description="Terminal UI for mist simulations",
        prog="python -m mist",
    )
    parser.add_argument(
        "executable",
        help="Path to the mist simulation executable",
    )
    args = parser.parse_args()

    app = MistTUI(args.executable)
    app.run()


if __name__ == "__main__":
    main()
