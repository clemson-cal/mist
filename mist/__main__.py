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
        Button, Checkbox, Footer, Header, Input, Label, Log, Static,
        TabbedContent, TabPane, DataTable, Rule
    )
    from textual.reactive import reactive
except ImportError:
    print("Error: textual is required for the mist TUI")
    print("Install with: pip install textual")
    sys.exit(1)

from .mist_exe import Mist


class ConfigTable(DataTable):
    """A table showing key-value configuration."""

    def update_config(self, config: dict):
        """Update the table with new config data."""
        self.clear()
        if not self.columns:
            self.add_columns("Key", "Value")
        for key, value in sorted(config.items()):
            self.add_row(str(key), str(value))


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
        grid-rows: auto 1fr 12 auto;
    }

    #info-bar {
        height: 3;
        padding: 1;
        background: $surface;
        border-bottom: solid $primary;
    }

    #main-content {
        height: 1fr;
    }

    #controls {
        height: auto;
        padding: 1;
        background: $surface;
        border-top: solid $primary;
    }

    #control-buttons {
        height: auto;
        align: center middle;
    }

    #control-buttons Button {
        margin: 0 1;
    }

    #advance-input {
        width: 20;
        margin: 0 1;
    }

    #console {
        height: 1fr;
        border: solid $primary;
        background: $surface-darken-1;
    }

    .tab-content {
        padding: 1;
    }

    ConfigTable {
        height: 1fr;
    }

    #products-checkboxes {
        height: auto;
    }

    .section-title {
        text-style: bold;
        padding: 0 0 1 0;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("i", "init", "Init"),
        Binding("s", "step", "Step"),
        Binding("r", "reset", "Reset"),
        Binding("a", "advance", "Advance"),
        Binding("f5", "refresh", "Refresh"),
    ]

    def __init__(self, executable: str):
        super().__init__()
        self.executable = executable
        self.sim: Optional[Mist] = None
        self.advance_target = 0.1

    def compose(self) -> ComposeResult:
        yield Header()
        yield SimulationInfo(id="info-bar")
        with Container(id="main-content"):
            with TabbedContent():
                with TabPane("Console", id="tab-console"):
                    yield Log(id="console", highlight=True)
                with TabPane("Physics", id="tab-physics"):
                    with ScrollableContainer(classes="tab-content"):
                        yield ConfigTable(id="physics-table")
                with TabPane("Initial", id="tab-initial"):
                    with ScrollableContainer(classes="tab-content"):
                        yield ConfigTable(id="initial-table")
                with TabPane("Products", id="tab-products"):
                    with ScrollableContainer(classes="tab-content"):
                        yield Vertical(id="products-checkboxes")
                with TabPane("Timeseries", id="tab-timeseries"):
                    with ScrollableContainer(classes="tab-content"):
                        yield Static(id="timeseries-content")
                with TabPane("State", id="tab-state"):
                    with ScrollableContainer(classes="tab-content"):
                        yield Static(id="state-content")
                with TabPane("Profiler", id="tab-profiler"):
                    with ScrollableContainer(classes="tab-content"):
                        yield Static(id="profiler-content")
        with Container(id="controls"):
            with Horizontal(id="control-buttons"):
                yield Button("Init", id="btn-init", variant="success")
                yield Button("Step", id="btn-step")
                yield Button("Advance to:", id="btn-advance", variant="primary")
                yield Input(value="0.1", id="advance-input", placeholder="target")
                yield Button("Reset", id="btn-reset", variant="warning")
                yield Button("Refresh", id="btn-refresh")
        yield Footer()

    def on_mount(self) -> None:
        """Connect to the simulation on startup."""
        self.title = f"mist - {self.executable}"
        self.log_message(f"Connecting to {self.executable}...")
        try:
            self.sim = Mist(self.executable, init=False)
            self.log_message("Connected successfully")
            self.log_message("Press 'i' or click Init to initialize")
            self.refresh_all()
        except Exception as e:
            self.log_message(f"[red]Error connecting: {e}[/]")

    def log_message(self, message: str) -> None:
        """Add a message to the console log."""
        log = self.query_one("#console", Log)
        log.write_line(message)

    def update_info_bar(self) -> None:
        """Update the info bar with current state."""
        info = self.query_one("#info-bar", SimulationInfo)
        if self.sim:
            info.initialized = self.sim._initialized
            if self.sim._initialized:
                info.time = self.sim.time
                info.iteration = self.sim.iteration
                info.dt = self.sim.dt

    def refresh_physics(self) -> None:
        """Refresh the physics config tab."""
        if not self.sim:
            return
        try:
            table = self.query_one("#physics-table", ConfigTable)
            table.update_config(self.sim.physics.to_dict())
        except Exception as e:
            self.log_message(f"[red]Error refreshing physics: {e}[/]")

    def refresh_initial(self) -> None:
        """Refresh the initial config tab."""
        if not self.sim:
            return
        try:
            table = self.query_one("#initial-table", ConfigTable)
            table.update_config(self.sim.initial.to_dict())
        except Exception as e:
            self.log_message(f"[red]Error refreshing initial: {e}[/]")

    def refresh_products(self) -> None:
        """Refresh the products tab with checkboxes."""
        if not self.sim or not self.sim._initialized:
            return
        try:
            container = self.query_one("#products-checkboxes", Vertical)
            names = self.sim.product_names

            # Remove all existing checkboxes and recreate
            container.remove_children()
            for name in names:
                cb_id = f"product-{name}"
                checkbox = Checkbox(name, value=True, id=cb_id)
                container.mount(checkbox)
        except Exception as e:
            self.log_message(f"[red]Error refreshing products: {e}[/]")

    def refresh_timeseries(self) -> None:
        """Refresh the timeseries tab."""
        if not self.sim or not self.sim._initialized:
            return
        try:
            content = self.query_one("#timeseries-content", Static)
            names = self.sim.timeseries_names
            lines = ["[bold]Available Timeseries:[/]", ""]
            for name in names:
                lines.append(f"  - {name}")
            if not names:
                lines.append("  (none)")
            content.update("\n".join(lines))
        except Exception as e:
            self.log_message(f"[red]Error refreshing timeseries: {e}[/]")

    def refresh_state(self) -> None:
        """Refresh the state tab."""
        if not self.sim:
            return
        try:
            content = self.query_one("#state-content", Static)
            state = self.sim.state
            lines = ["[bold]Simulation State:[/]", ""]
            lines.append(f"  Initialized: {bool(state.get('initialized', 0))}")
            lines.append(f"  Zone count: {state.get('zone_count', 0)}")
            lines.append("")
            lines.append("[bold]Time Variables:[/]")
            for entry in state.get("times", []):
                lines.append(f"  {entry.get('key', '?')}: {entry.get('value', 0):.6g}")
            content.update("\n".join(lines))
        except Exception as e:
            self.log_message(f"[red]Error refreshing state: {e}[/]")

    def refresh_profiler(self) -> None:
        """Refresh the profiler tab."""
        if not self.sim or not self.sim._initialized:
            return
        try:
            content = self.query_one("#profiler-content", Static)
            data = self.sim.profiler
            lines = ["[bold]Profiler Data:[/]", ""]
            for key, value in sorted(data.items()):
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.6g}")
                else:
                    lines.append(f"  {key}: {value}")
            if len(data) == 0:
                lines.append("  (no profiler data)")
            content.update("\n".join(lines))
        except Exception as e:
            self.log_message(f"[red]Error refreshing profiler: {e}[/]")

    def refresh_all(self) -> None:
        """Refresh all tabs and info."""
        self.update_info_bar()
        self.refresh_physics()
        self.refresh_initial()
        self.refresh_products()
        self.refresh_timeseries()
        self.refresh_state()
        self.refresh_profiler()

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
            self.log_message(f"Advancing to t={target}...")
            self.sim.advance_to(target)
            self.log_message(f"[green]Advanced to t={self.sim.time:.6g}[/]")
            self.show_iteration_info()
            self.update_info_bar()
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
            self.refresh_all()
        except Exception as e:
            self.log_message(f"[red]Error: {e}[/]")

    def action_refresh(self) -> None:
        """Refresh all displays."""
        self.log_message("Refreshing...")
        self.refresh_all()
        self.log_message("[green]Refreshed[/]")

    def show_iteration_info(self) -> None:
        """Log the current iteration info."""
        if not self.sim or not self.sim._initialized:
            return
        t = self.sim.time
        n = self.sim.iteration
        dt = self.sim.dt
        self.log_message(f"[{n:06d}] t={t:+.6e} dt={dt:.6e}")

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
        elif button_id == "btn-refresh":
            self.action_refresh()

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Handle checkbox changes for product selection."""
        if not event.checkbox.id or not event.checkbox.id.startswith("product-"):
            return
        if not self.sim or not self.sim._initialized:
            return
        try:
            container = self.query_one("#products-checkboxes", Vertical)
            selected = [
                cb.id.replace("product-", "", 1)
                for cb in container.query(Checkbox)
                if cb.value
            ]
            self.sim.select_products(selected)
            self.log_message(f"Selected products: {selected}")
        except Exception as e:
            self.log_message(f"[red]Error selecting products: {e}[/]")

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
