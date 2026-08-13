"""
ipywidgets control panels for the drift / coherency workflow.

Kept here rather than in ``quantem`` on purpose: the analysis in
``quantem.imaging.interface_coherency`` is pure numpy/scipy/matplotlib, and adding an
``ipywidgets`` import to it would put a notebook-only dependency into a library module.  The
interactive *canvas* tools (``RegionSelector``, ``SiteEditor``) do live in quantem, because
they need nothing beyond matplotlib event handling.

Everything here needs ``%matplotlib widget`` (ipympl) to be active, otherwise the figures
render as static PNGs and the sliders redraw into nothing.
"""

from __future__ import annotations

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from quantem.imaging.interface_coherency import InterfaceCoherency

__all__ = [
    "ParameterPanel",
    "CoherencyExplorer",
    "coherency_explorer",
    "site_editor_panel",
    "use_backend",
    "interactive_available",
]


def interactive_available() -> bool:
    """
    Whether live widgets make sense right now.

    False when ``ZBDRIFT_STATIC=1`` (which ``build_notebooks.py`` sets while executing a
    notebook unattended) or when there is no IPython kernel at all.  The point is that the
    interactive cells in the notebooks can be *real* cells -- run for a human, skipped for a
    build -- rather than commented-out recipes that nobody ever checks still work.
    """
    if os.environ.get("ZBDRIFT_STATIC") == "1":
        return False
    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except Exception:
        return False


def use_backend(interactive: bool | None = None) -> str:
    """
    Select the matplotlib backend and return its name.

    ``interactive=True`` asks for ``ipympl`` (``%matplotlib widget``), which is what makes
    click-to-edit and the slider panels work.  ``False`` uses the inline backend so figures
    render as static images that survive being saved into the ``.ipynb`` and viewed without
    the widget extension installed.  ``None`` decides with :func:`interactive_available`.

    Outside IPython this falls back to ``Agg``: the notebooks double as plain scripts, and a
    script that pops GUI windows cannot run in a build.
    """
    interactive = interactive_available() if interactive is None else interactive
    try:
        from IPython import get_ipython

        ip = get_ipython()
    except Exception:
        ip = None

    if ip is None:
        matplotlib.use("Agg", force=True)
        return "agg"
    if interactive:
        # Colab serves third-party widgets only after this is enabled, and without it ipympl
        # renders as an empty output area rather than raising -- a silent failure, which is the
        # worst kind for an interactive cell.  Enabling it is harmless if it is already on.
        try:
            from google.colab import output as _colab_output

            _colab_output.enable_custom_widget_manager()
        except Exception:
            pass
        try:
            ip.run_line_magic("matplotlib", "widget")
            return "widget"
        except Exception:
            pass  # ipympl missing -- inline is still better than crashing
    ip.run_line_magic("matplotlib", "inline")
    return "inline"


class CoherencyExplorer:
    """
    Slider panel over the coherency measurement.

    The parameters worth exploring interactively are the ones where a defensible choice
    depends on the data in front of you: how many rows to include, how many to skip either
    side of the interface (the rows immediately at it are chemically and structurally mixed),
    whether to remove each row's own offset, and -- the comparison this whole project is
    about -- whether to use drift-corrected coordinates.

    >>> ex = CoherencyExplorer(image, sites, interface_y=657, drift=sid, pixel_size=0.12)
    >>> ex.panel        # display this
    """

    def __init__(self, image, sites, interface_y, drift=None, pixel_size=None,
                 figsize=(9.5, 3.6), max_rows=10, interactive=None, n_rows=4, skip_rows=1):
        self.image = np.asarray(image, float)
        self.sites_raw = np.asarray(sites, float)
        self.interface_y = float(interface_y)
        self.drift = drift
        self.pixel_size = pixel_size
        self._figsize = figsize
        self._defaults = dict(n_rows=n_rows, skip_rows=skip_rows, corrected=drift is not None,
                              offset=True, stack=0.0)
        self.interactive = interactive_available() if interactive is None else interactive

        if not self.interactive:
            # No frontend, so build no widgets: an ipywidgets Output with
            # clear_output(wait=True) inside it blocks forever waiting for a comm that will
            # never answer, and with nbclient's default 1800 s timeout that is a half-hour
            # stall in the middle of an automated notebook build.  Measure once, expose the
            # result, and skip the machinery.
            self.panel = None
            self._widgets = {}
            self.coherency = self._measure(**self._defaults)
            return

        import ipywidgets as W

        self.w_corrected = W.Checkbox(value=drift is not None, description="drift corrected",
                                      disabled=drift is None, indent=False,
                                      tooltip="apply M^-1 to the site coordinates before measuring")
        self.w_nrows = W.IntSlider(value=4, min=1, max=max_rows,
                                   description="rows of substrate and film to plot",
                                   continuous_update=False,
                                   style={"description_width": "initial"},
                                   layout=W.Layout(width="420px"))
        # Fixed at 0 and not shown.  Skipping rows is a judgement about where the interface stops
        # being the interface; leaving it on a slider invited fiddling with it to move the answer.
        # Still reachable as ``explorer.w_skip.value`` when a genuinely intermixed interface needs it.
        self.w_skip = W.IntSlider(value=0, min=0, max=6, description="skip rows")
        # A dropdown rather than a checkbox: "substrate" is a distinct and arguably better view,
        # not a half-way state.  It flattens the ruler and leaves the film's registry visible.
        self.w_offset = W.Dropdown(value="both", options=["both", "substrate", "film", "none"],
                                   description="start rows at the origin:",
                                   style={"description_width": "initial"},
                                   layout=W.Layout(width="320px"))
        self.w_stack = W.FloatSlider(value=0.0, min=0.0, max=3.0, step=0.1,
                                     description="fan rows apart (display only)",
                                     continuous_update=False,
                                     style={"description_width": "initial"},
                                     layout=W.Layout(width="420px"))
        self.w_out = W.Output()
        self.w_text = W.HTML()

        controls = W.VBox([
            W.HBox([self.w_corrected, self.w_offset]),
            self.w_nrows,
            self.w_stack,
        ])
        self.panel = W.VBox([controls, self.w_text, self.w_out])
        self._widgets = dict(corrected=self.w_corrected, offset=self.w_offset,
                             nrows=self.w_nrows, skip=self.w_skip, stack=self.w_stack)

        for w in self._widgets.values():
            w.observe(lambda *_: self.update(), names="value")
        self.update()

    # ------------------------------------------------------------------
    def _measure(self, n_rows, skip_rows, corrected, **_):
        """Run the measurement once; shared by the widget and the headless paths."""
        import warnings

        if corrected and self.drift is not None:
            xy = self.drift.apply_to_coordinates(self.sites_raw)
            yi = self.interface_y / self.drift.solution["k"]
            label = "drift-corrected"
        else:
            xy, yi, label = self.sites_raw, self.interface_y, "uncorrected"
        coh = InterfaceCoherency.from_sites(xy, yi, self.pixel_size, label=label)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coh.measure(n_rows_each_side=n_rows, skip_rows=skip_rows)
        return coh

    def update(self):
        try:
            coh = self._measure(n_rows=self.w_nrows.value, skip_rows=self.w_skip.value,
                                corrected=self.w_corrected.value)
        except Exception as exc:  # keep the panel alive on a bad parameter combination
            self.w_text.value = f"<pre style='color:#b00'>{type(exc).__name__}: {exc}</pre>"
            return
        self.coherency = coh
        s = coh.summary()
        px = self.pixel_size
        unit, key = ("A", "spacing_A") if px else ("px", "spacing_px")
        pkey = "row_pitch_A" if px else "row_pitch_px"
        # Measured numbers only.  The categorical COHERENT / RELAXED verdict is deliberately not
        # here: it collapses the mismatch, the residual ratio and the row-to-row agreement into one
        # word, and on a rumpled film the same word covers rows that plainly disagree.  The numbers
        # below and the row-by-row plot are what the classification was standing in for.
        self.w_text.value = (
            f"<div style='font-family:monospace;font-size:12px'>"
            f"substrate {s['substrate'].get(key, float('nan')):.4f} {unit} in-plane, "
            f"{s['substrate'].get(pkey, float('nan')):.4f} {unit} pitch &nbsp;|&nbsp; "
            f"film {s['film'].get(key, float('nan')):.4f} {unit} in-plane, "
            f"{s['film'].get(pkey, float('nan')):.4f} {unit} pitch<br>"
            f"mismatch {s.get('mismatch', float('nan')):+.2%} &nbsp;|&nbsp; "
            f"residual ratio {s.get('residual_ratio', float('nan')):.2f}"
            f"</div>")

        with self.w_out:
            self.w_out.clear_output(wait=True)
            fig, ax = plt.subplots(1, 2, figsize=self._figsize)
            offset = {"both": True, "none": False}.get(self.w_offset.value, self.w_offset.value)
            coh.plot(remove_row_offset=offset,
                     stack_offset=self.w_stack.value, axes=ax)
            fig.suptitle(coh.label, fontsize=9)
            plt.show()


def coherency_explorer(*args, **kwargs):
    """Build a :class:`CoherencyExplorer` and return its panel, ready to display."""
    return CoherencyExplorer(*args, **kwargs).panel


def site_editor_panel(editor, on_done=None):
    """
    Wrap a :class:`~quantem.imaging.interface_coherency.SiteEditor` with refine and done
    buttons, a live count, and the refine report.

    The done button matters more than it looks: without an explicit "done", the natural thing
    is to read ``editor.sites`` in the next cell, and it is easy to do that before finishing the
    edits and then wonder why the plot did not change.

    The refine button is the same thing as pressing ``f`` on the canvas, but with the window
    radius exposed -- because that number is the one judgement call in the operation.  It is both
    the half-width of the window each site is fitted in and the furthest a site is allowed to
    move.  Too small and the window clips the column; larger than half the site spacing and it
    starts to contain the neighbours.  The report line says what happened, and it is one ``u`` to
    undo.
    """
    import ipywidgets as W

    count = W.HTML()
    report = W.HTML()
    radius = W.BoundedFloatText(value=editor.snap_radius, min=1.0, max=60.0, step=1.0,
                                description="window ± px", style={"description_width": "80px"},
                                layout=W.Layout(width="180px"))
    refine = W.Button(description="refine", button_style="info", icon="crosshairs",
                      tooltip="fit a gaussian in a window around each site and move it there")
    only_added = W.Checkbox(value=editor.refine_only_added, description="clicked only",
                            indent=False, layout=W.Layout(width="130px"))
    button = W.Button(description="use these sites", button_style="success")
    result = {}

    def refresh(*_):
        count.value = (f"<span style='font-family:monospace'>{len(editor.sites)} sites"
                       f"{' — accepted' if result else ''}</span>")

    def do_refine(_):
        # An accepted set is now stale: the sites it was read from have moved.
        result.clear()
        info = editor.refine(only_added=only_added.value, max_move=radius.value)
        report.value = (f"<span style='font-family:monospace;font-size:90%'>"
                        f"{info.get('text', '')}</span>")
        refresh()

    def click(_):
        result["sites"] = editor.sites.copy()
        refresh()
        if on_done:
            on_done(result["sites"])

    refine.on_click(do_refine)
    button.on_click(click)
    editor.fig.canvas.mpl_connect("button_release_event", refresh)
    editor.fig.canvas.mpl_connect("key_release_event", refresh)
    refresh()
    controls = W.HBox([refine, radius, only_added, button, count])
    return W.VBox([controls, report]), result

class ParameterPanel:
    """
    A small form for the handful of numbers a run actually depends on.

    Not decoration: these are the values where a wrong choice produces a plausible-looking
    wrong answer rather than an error -- the pixel size (which sets every reported length), the
    reference angle and axial ratio (the crystal you are claiming to see), the flyback crop,
    and the interface row.  Having them in one visible place beats having them scattered
    through cells as literals.

    ``.values`` is a plain dict, so the notebook reads it the same way whether or not
    ipywidgets is available.

    >>> pp = ParameterPanel(pixel_size=0.12, interface_row=650)
    >>> pp.panel                      # display
    >>> pp.values["pixel_size"]       # read, later
    """

    SPECS = (
        ("pixel_size", "pixel size (Å/px)", 0.12, 0.005, 1.0, 0.005, "%.4f"),
        ("axial_angle_deg", "reference angle (°)", 90.0, 60.0, 120.0, 0.5, "%.1f"),
        ("axial_ratio", "axial ratio |c|/|a|", 1.41421356, 0.5, 3.0, 0.001, "%.5f"),
        ("flyback_rows", "flyback rows to drop", 12, 0, 60, 1, "%d"),
        ("interface_row", "interface row", 650, 0, 4096, 1, "%d"),
    )

    def __init__(self, **overrides):
        self.values = {k: overrides.get(k, d) for k, _, d, *_ in self.SPECS}
        try:
            import ipywidgets as W
        except Exception:
            self.panel = None
            self._widgets = {}
            return

        self._widgets = {}
        rows = []
        for key, label, default, lo, hi, step, fmt in self.SPECS:
            val = self.values[key]
            cls = W.BoundedIntText if fmt == "%d" else W.BoundedFloatText
            w = cls(value=val, min=lo, max=hi, step=step, description=label,
                    style={"description_width": "160px"},
                    layout=W.Layout(width="330px"))
            w.observe(self._sync, names="value")
            self._widgets[key] = w
            rows.append(w)
        self.panel = W.VBox(rows)

    def _sync(self, *_):
        for key, w in self._widgets.items():
            self.values[key] = w.value

    def __getitem__(self, key):
        self._sync()
        return self.values[key]

    def geometry(self, site_basis=((0.0, 0.0), (0.5, 0.5)), **kw):
        """A :class:`ReferenceGeometry` from the angle and ratio currently in the form."""
        from quantem.imaging.drift_single_image import ReferenceGeometry

        self._sync()
        return ReferenceGeometry.from_angle_and_ratio(
            self.values["axial_angle_deg"], self.values["axial_ratio"],
            site_basis=site_basis, **kw)
