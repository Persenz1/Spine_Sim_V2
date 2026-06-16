"""Plot style configuration loaded from plot_styles/*.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_STYLE = "debug"
VALID_SAVE_FORMATS = {"png", "svg", "pdf"}


@dataclass(frozen=True)
class PlotStyle:
    """Runtime plotting style options."""

    name: str
    figure_size: tuple[float, float] = (6.4, 4.4)
    dpi: int = 180
    font_family: str = "DejaVu Sans"
    font_size: float = 10.0
    line_width: float = 1.6
    marker_size: float = 5.0
    colormap: str = "viridis"
    language: str = "en"
    save_format: str = "png"

    def path(self, outdir: str | Path, stem: str) -> Path:
        """Return a figure path using the style's save format."""
        return Path(outdir) / f"{stem}.{self.save_format}"

    def scaled_size(self, *, width_scale: float = 1.0, height_scale: float = 1.0) -> tuple[float, float]:
        """Return the style figure size scaled for panel layouts."""
        return (self.figure_size[0] * width_scale, self.figure_size[1] * height_scale)


def load_plot_style(style: str | Path = DEFAULT_STYLE) -> PlotStyle:
    """Load a style by name or YAML path."""
    style_path = _resolve_style_path(style)
    data = _load_style_document(style_path)
    name = style_path.stem if style_path is not None else str(style)
    save_format = str(data.get("save_format", "png")).lower()
    if save_format not in VALID_SAVE_FORMATS:
        raise ValueError(f"Unsupported save_format {save_format!r}; expected one of {sorted(VALID_SAVE_FORMATS)}.")
    figure_size = data.get("figure_size", [6.4, 4.4])
    if isinstance(figure_size, str):
        figure_size = _parse_list(figure_size)
    if not isinstance(figure_size, (list, tuple)) or len(figure_size) != 2:
        raise ValueError("style figure_size must contain exactly two numeric values.")
    return PlotStyle(
        name=name,
        figure_size=(float(figure_size[0]), float(figure_size[1])),
        dpi=int(data.get("dpi", 180)),
        font_family=str(data.get("font_family", "DejaVu Sans")),
        font_size=float(data.get("font_size", 10.0)),
        line_width=float(data.get("line_width", 1.6)),
        marker_size=float(data.get("marker_size", 5.0)),
        colormap=str(data.get("colormap", "viridis")),
        language=str(data.get("language", "en")),
        save_format=save_format,
    )


def apply_plot_style(plt: Any, style: PlotStyle) -> None:
    """Apply style values to matplotlib rcParams."""
    plt.rcParams.update(
        {
            "font.family": style.font_family,
            "font.size": style.font_size,
            "axes.titlesize": style.font_size + 1,
            "axes.labelsize": style.font_size,
            "legend.fontsize": max(style.font_size - 2, 6),
            "xtick.labelsize": max(style.font_size - 1, 6),
            "ytick.labelsize": max(style.font_size - 1, 6),
            "lines.linewidth": style.line_width,
            "lines.markersize": style.marker_size,
            "savefig.dpi": style.dpi,
        }
    )


def require_columns(df: Any, columns: list[str] | tuple[str, ...], *, dataset_name: str) -> None:
    """Raise a clear error when a dataframe is missing required columns."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {missing}")


def _resolve_style_path(style: str | Path) -> Path | None:
    path = Path(style)
    if path.suffix:
        if not path.exists():
            raise FileNotFoundError(f"Plot style file not found: {path}")
        return path
    root = Path(__file__).resolve().parents[2]
    candidate = root / "plot_styles" / f"{style}.yaml"
    if not candidate.exists():
        raise FileNotFoundError(
            f"Plot style {style!r} not found at {candidate}. "
            "Expected one of plot_styles/debug.yaml, report.yaml, paper.yaml."
        )
    return candidate


def _load_style_document(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        return _parse_simple_yaml(text)
    loaded = yaml.safe_load(text) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Plot style must be a YAML mapping: {path}")
    return dict(loaded)


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        data[key.strip()] = _parse_scalar(value.strip())
    return data


def _parse_scalar(value: str) -> Any:
    if value.startswith("[") and value.endswith("]"):
        return _parse_list(value)
    cleaned = value.strip("'\"")
    lowered = cleaned.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in cleaned:
            return float(cleaned)
        return int(cleaned)
    except ValueError:
        return cleaned


def _parse_list(value: str) -> list[Any]:
    body = value.strip()[1:-1]
    if not body.strip():
        return []
    return [_parse_scalar(item.strip()) for item in body.split(",")]
