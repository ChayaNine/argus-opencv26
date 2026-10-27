"""Asset registry: what the plant knows about each instrument.

In a real deployment this comes from the CMMS / asset database (tag, range,
alarm limits, criticality).  The registry is deliberately the *only* source of
calibration; the vision stack verifies it against the dial's tick marks.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

STATUS_ORDER = ["normal", "warning", "alarm", "instrument_fault", "uncertain"]


@dataclass
class GaugeSpec:
    id: str
    name: str
    units: str
    min_value: float
    max_value: float
    theta_min: float = -135.0
    theta_max: float = 135.0
    major_ticks: int = 9
    minor_per_major: int = 4
    low_alarm: Optional[float] = None
    low_warn: Optional[float] = None
    high_warn: Optional[float] = None
    high_alarm: Optional[float] = None
    criticality: str = "medium"  # high | medium | low
    needle_color: str = "black"  # black | red
    marker_id: int = 0
    dial_radius_mm: float = 50.0
    location: str = ""

    @property
    def span(self) -> float:
        return self.max_value - self.min_value

    def limits(self) -> dict:
        return {k: getattr(self, k) for k in ("low_alarm", "low_warn", "high_warn", "high_alarm")}

    def limits_text(self) -> str:
        parts = [f"{k.replace('_', ' ')} {v:g}" for k, v in self.limits().items() if v is not None]
        return ", ".join(parts) + f" {self.units}" if parts else "none"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["span"] = self.span
        return d

    def status_of(self, value: float) -> str:
        """Ground-truth style status for a point value."""
        if self.low_alarm is not None and value < self.low_alarm:
            return "alarm"
        if self.high_alarm is not None and value > self.high_alarm:
            return "alarm"
        if self.low_warn is not None and value < self.low_warn:
            return "warning"
        if self.high_warn is not None and value > self.high_warn:
            return "warning"
        return "normal"

    def thresholds(self) -> list[tuple[float, str]]:
        out = []
        for k in ("low_alarm", "low_warn", "high_warn", "high_alarm"):
            v = getattr(self, k)
            if v is not None:
                out.append((v, k))
        return sorted(out)


@dataclass
class Registry:
    gauges: dict[str, GaugeSpec] = field(default_factory=dict)

    def add(self, spec: GaugeSpec) -> None:
        self.gauges[spec.id] = spec

    def by_marker(self, marker_id: int) -> Optional[GaugeSpec]:
        for g in self.gauges.values():
            if g.marker_id == marker_id:
                return g
        return None

    def __iter__(self):
        return iter(self.gauges.values())

    def __len__(self):
        return len(self.gauges)


# A small, realistic pump-skid / utility-room catalogue.  Scenes sample from it.
CATALOGUE: list[dict] = [
    dict(name="Pump P-101 discharge pressure", units="bar", min_value=0, max_value=16,
         low_alarm=2, low_warn=4, high_warn=10, high_alarm=12, criticality="high"),
    dict(name="Pump P-101 suction pressure", units="bar", min_value=-1, max_value=3, major_ticks=9,
         low_alarm=-0.5, low_warn=0.0, high_warn=2.0, high_alarm=2.5, criticality="high"),
    dict(name="Filter F-201 differential pressure", units="kPa", min_value=0, max_value=100, major_ticks=11,
         high_warn=60, high_alarm=80, criticality="medium"),
    dict(name="Boiler feed line temperature", units="°C", min_value=0, max_value=160, major_ticks=9,
         low_warn=60, high_warn=120, high_alarm=140, criticality="high", needle_color="red"),
    dict(name="Instrument air header", units="psi", min_value=0, max_value=160, major_ticks=9,
         low_alarm=70, low_warn=85, high_warn=125, criticality="medium"),
    dict(name="Cooling water return", units="bar", min_value=0, max_value=10, major_ticks=11,
         low_warn=1.5, high_warn=6, high_alarm=8, criticality="low"),
    dict(name="Nitrogen blanket pressure", units="mbar", min_value=0, max_value=250, major_ticks=6,
         low_alarm=20, low_warn=40, high_warn=180, high_alarm=220, criticality="high", needle_color="red"),
    dict(name="Hydraulic unit HPU-3 pressure", units="bar", min_value=0, max_value=250, major_ticks=6,
         low_warn=120, high_warn=210, high_alarm=230, criticality="medium"),
    dict(name="Chiller CH-2 condenser pressure", units="psi", min_value=0, max_value=300, major_ticks=7,
         high_warn=220, high_alarm=260, criticality="medium"),
    dict(name="Sample cooler outlet temperature", units="°C", min_value=0, max_value=100, major_ticks=11,
         high_warn=45, high_alarm=60, criticality="low", needle_color="red"),
]

TAG_PREFIX = {"bar": "PG", "kPa": "PDG", "psi": "PG", "mbar": "PG", "°C": "TG"}


def make_spec(idx: int, marker_id: int, entry: dict | None = None) -> GaugeSpec:
    entry = dict(entry or CATALOGUE[idx % len(CATALOGUE)])
    prefix = TAG_PREFIX.get(entry["units"], "G")
    gid = f"{prefix}-{101 + idx}"
    return GaugeSpec(id=gid, marker_id=marker_id, **entry)
