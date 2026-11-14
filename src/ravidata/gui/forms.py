"""Auto-generated parameter forms (processing ParamSpecs and RadarConfig fields)."""

from __future__ import annotations

import typing
from dataclasses import fields
from typing import Callable

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QLineEdit, QSpinBox, QWidget)

from ..config import RadarConfig
from ..processing.base import ParamSpec

CHOICES = {
    "iq_order": ["imag_low", "real_low"],
    "cube_order": ["tx_rx_dop_range", "dop_tx_rx_range"],
    "window_range": ["", "hann", "rect"],
    "window_doppler": ["", "hann", "rect"],
    "cmd_order": ["p2", "v3", "baseband_last"],
}
TOOLTIPS = {
    "fft_mode": "1 = range only (1D). 2 = several intervals per frame (range-Doppler if dop_fft).",
    "range_max_mm": "Sets the chirp slope (only range_mm does!). Keep >= 3747 mm for the 400 MHz/us limit.",
    "range_res_mm": "Bin size. range/res = ADC samples (32..1024).",
    "chirp_num": "On-chip complex averaging of N chirps (power of 2). May lose signal on moving targets.",
    "freq_max_mhz": "Second argument of `mmwc freq`; semantics unverified. Leave empty.",
    "hpf_code": "Raw `mmwc hpf 255 <code>`; unit unknown (legacy used 16). Empty = chip default (0.5 MHz).",
    "baudrate": "Data UART baud (<= 2 Mbaud). The cube must fit into one frame period.",
    "extra_commands": "Raw mmwc commands separated by ';', sent after the baseband config.",
    "iq_order": "Decoding convention. HIF manual: imag in low 16 bits.",
    "cmd_order": "Order of the mmwc commands. p2 = legacy perception2 order (known to stream on a dev board).",
    "power_save": "`mmwc ps`; empty = not sent.",
}


def param_form(specs: list[ParamSpec], values: dict, on_change: Callable[[str, object], None]) -> tuple[QWidget, dict]:
    w = QWidget()
    lay = QFormLayout(w)
    lay.setContentsMargins(2, 2, 2, 2)
    widgets = {}
    for s in specs:
        v = values.get(s.name, s.default)
        if s.kind == "bool":
            e = QCheckBox()
            e.setChecked(bool(v))
            e.toggled.connect(lambda val, n=s.name: on_change(n, bool(val)))
        elif s.kind == "choice":
            e = QComboBox()
            e.addItems(s.choices)
            e.setCurrentText(str(v))
            e.currentTextChanged.connect(lambda val, n=s.name: on_change(n, val))
        elif s.kind == "int":
            e = QSpinBox()
            e.setRange(int(s.lo if s.lo is not None else -10**6), int(s.hi if s.hi is not None else 10**6))
            e.setSingleStep(int(s.step or 1))
            e.setValue(int(v))
            e.valueChanged.connect(lambda val, n=s.name: on_change(n, int(val)))
        else:
            e = QDoubleSpinBox()
            e.setDecimals(3)
            e.setRange(float(s.lo if s.lo is not None else -1e6), float(s.hi if s.hi is not None else 1e6))
            e.setSingleStep(float(s.step or 0.1))
            e.setValue(float(v))
            e.valueChanged.connect(lambda val, n=s.name: on_change(n, float(val)))
        e.setToolTip(s.help)
        lay.addRow(s.name, e)
        widgets[s.name] = e
    return w, widgets


def set_param_widget(e, value):
    e.blockSignals(True)
    try:
        if isinstance(e, QCheckBox):
            e.setChecked(bool(value))
        elif isinstance(e, QComboBox):
            e.setCurrentText(str(value))
        elif isinstance(e, QSpinBox):
            e.setValue(int(value))
        elif isinstance(e, QDoubleSpinBox):
            e.setValue(float(value))
    finally:
        e.blockSignals(False)


class ConfigForm(QWidget):
    """Editable RadarConfig. Empty text = None (keep the chip default)."""

    SKIP = {"n_rx", "n_tx"}

    def __init__(self, on_change: Callable[[], None]):
        super().__init__()
        self.on_change = on_change
        self.lay = QFormLayout(self)
        self.lay.setContentsMargins(2, 2, 2, 2)
        self.hints = typing.get_type_hints(RadarConfig)
        self.widgets: dict[str, QWidget] = {}
        for f in fields(RadarConfig):
            if f.name in self.SKIP:
                continue
            hint = self.hints[f.name]
            if f.name in CHOICES:
                e = QComboBox()
                e.addItems(CHOICES[f.name])
                e.currentTextChanged.connect(lambda *_: self.on_change())
            elif _is_bool(hint):
                e = QComboBox()
                e.addItems(["", "on", "off"] if _optional(hint) else ["on", "off"])
                e.currentTextChanged.connect(lambda *_: self.on_change())
            else:
                e = QLineEdit()
                e.editingFinished.connect(self.on_change)
            e.setToolTip(TOOLTIPS.get(f.name, ""))
            self.lay.addRow(f.name, e)
            self.widgets[f.name] = e

    def set_config(self, cfg: RadarConfig):
        for name, e in self.widgets.items():
            v = getattr(cfg, name)
            e.blockSignals(True)
            if isinstance(e, QComboBox):
                if v is None:
                    e.setCurrentText("")
                elif isinstance(v, bool):
                    e.setCurrentText("on" if v else "off")
                else:
                    e.setCurrentText(str(v))
            else:
                if isinstance(v, list):
                    e.setText("; ".join(v))
                else:
                    e.setText("" if v is None else str(v))
            e.blockSignals(False)

    def get_config(self) -> tuple[RadarConfig, list[str]]:
        cfg = RadarConfig()
        errors = []
        for name, e in self.widgets.items():
            hint = self.hints[name]
            txt = e.currentText() if isinstance(e, QComboBox) else e.text().strip()
            try:
                if name == "extra_commands":
                    val = [c.strip() for c in txt.split(";") if c.strip()]
                elif txt == "":
                    if not _optional(hint) and name not in ("notes", "window_range", "window_doppler"):
                        raise ValueError("required")
                    val = "" if name == "notes" else None
                elif _is_bool(hint):
                    val = txt == "on"
                elif _base(hint) is int:
                    val = int(float(txt))
                elif _base(hint) is float:
                    val = float(txt)
                else:
                    val = txt
                setattr(cfg, name, val)
            except Exception as ex:
                errors.append(f"{name}: {ex}")
        return cfg, errors


def _optional(h) -> bool:
    return type(None) in typing.get_args(h)


def _base(h):
    args = [a for a in typing.get_args(h) if a is not type(None)]
    return args[0] if args else h


def _is_bool(h) -> bool:
    return _base(h) is bool
