# config.py
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Literal, Tuple, Any
import csv


Label = Literal["LB", "MEI", "UNK"]
LabelMode = Literal["declared", "blind", "external"]

Group = Literal["Ctrl", "Exp", "UNK"]
GroupMode = Literal["declared_ctrl_exp", "by_date", "external_ctrl_exp"]


@dataclass(frozen=True)
class ExperimentConfig:
    """
    Declarative experiment configuration.

    Key design:
      - Always supports a double-blind run (labels/groups are UNK).
      - Optionally supports declared or externally provided labels/groups (unblinding).
      - Keeps experiment semantics (label/group) separate from data processing.
    """

    # Roots
    source_root: Path
    target_root: Path

    # Calendar / identifiers
    dates: List[str] = field(default_factory=list)
    labs: List[str] = field(default_factory=list)
    default_shift: Optional[str] = None

    # Optional canonical naming / renaming (raw_name -> canonical_name)
    rename_map: Dict[str, str] = field(default_factory=dict)

    # Labeling (LB/MEI)
    label_mode: LabelMode = "blind"
    default_label: Label = "UNK"
    declared_label_map: Dict[str, Label] = field(default_factory=dict)  # keyed by canonical_name
    labels_csv: Optional[Path] = None  # for label_mode="external"

    # Grouping (Ctrl/Exp)
    group_mode: GroupMode = "by_date"
    default_group: Group = "UNK"
    date_to_group: Dict[str, Group] = field(default_factory=dict)  # for group_mode="declared_ctrl_exp"
    groups_csv: Optional[Path] = None  # for group_mode="external_ctrl_exp"

    # ----------------------------
    # Validation
    # ----------------------------
    def validate(self) -> None:
        # Roots
        if not isinstance(self.source_root, Path) or not isinstance(self.target_root, Path):
            raise TypeError("source_root and target_root must be pathlib.Path")

        # Modes
        if self.label_mode not in ("declared", "blind", "external"):
            raise ValueError(f"Invalid label_mode: {self.label_mode}")
        if self.group_mode not in ("declared_ctrl_exp", "by_date", "external_ctrl_exp"):
            raise ValueError(f"Invalid group_mode: {self.group_mode}")

        # Dates 
        if not isinstance(self.dates, list):
            raise TypeError("dates must be a list[str]")
        for d in self.dates:
            if not isinstance(d, str) or not d.strip():
                raise ValueError("dates must contain non-empty strings")

        # Labs
        if not isinstance(self.labs, list):
            raise TypeError("labs must be a list[str]")
        for lab in self.labs:
            if not isinstance(lab, str) or not lab.strip():
                raise ValueError("labs must contain non-empty strings")
        


        # Shift (optional, legacy field)
        if self.default_shift is not None:
            if not isinstance(self.default_shift, str) or not self.default_shift.strip():
                raise ValueError("default_shift must be None or a non-empty string")


        # Defaults
        if self.default_label not in ("LB", "MEI", "UNK"):
            raise ValueError(f"Invalid default_label: {self.default_label}")
        if self.default_group not in ("Ctrl", "Exp", "UNK"):
            raise ValueError(f"Invalid default_group: {self.default_group}")

        # Declared groups require a complete mapping over dates (if dates are provided)
        if self.group_mode == "declared_ctrl_exp":
            if self.dates:
                missing = [d for d in self.dates if d not in self.date_to_group]
                if missing:
                    raise ValueError(f"Missing date_to_group entries for dates: {missing}")
            for d, g in self.date_to_group.items():
                if g not in ("Ctrl", "Exp", "UNK"):
                    raise ValueError(f"Invalid group value for date '{d}': {g}")

        # External groups require a file path (and file must exist)
        if self.group_mode == "external_ctrl_exp":
            if self.groups_csv is None:
                raise ValueError("group_mode='external_ctrl_exp' requires groups_csv")
            if not isinstance(self.groups_csv, Path):
                raise TypeError("groups_csv must be pathlib.Path")
            if not self.groups_csv.exists():
                raise FileNotFoundError(f"groups_csv does not exist: {self.groups_csv}")

        # Declared labels validation
        if self.label_mode == "declared":
            for name, lab in self.declared_label_map.items():
                if lab not in ("LB", "MEI", "UNK"):
                    raise ValueError(f"Invalid label value for '{name}': {lab}")

        # External labels require a file path (and file must exist)
        if self.label_mode == "external":
            if self.labels_csv is None:
                raise ValueError("label_mode='external' requires labels_csv")
            if not isinstance(self.labels_csv, Path):
                raise TypeError("labels_csv must be pathlib.Path")
            if not self.labels_csv.exists():
                raise FileNotFoundError(f"labels_csv does not exist: {self.labels_csv}")

        # Rename map sanity
        for raw, canon in self.rename_map.items():
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError("rename_map keys must be non-empty strings")
            if not isinstance(canon, str) or not canon.strip():
                raise ValueError("rename_map values must be non-empty strings")

    # ----------------------------
    # Serialization helpers (used by print_config)
    # ----------------------------
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Convert Paths to strings for JSON serialization
        d["source_root"] = str(self.source_root)
        d["target_root"] = str(self.target_root)
        d["labels_csv"] = str(self.labels_csv) if self.labels_csv is not None else None
        d["groups_csv"] = str(self.groups_csv) if self.groups_csv is not None else None
        return d

    def summary(self) -> Dict[str, Any]:
        #    
        return {
            "source_root": str(self.source_root),
            "target_root": str(self.target_root),
            "dates_count": len(self.dates),
            "dates": list(self.dates),
            "labs_count": len(self.labs),
            "labs": list(self.labs),
            "default_shift": self.default_shift,
            "label_mode": self.label_mode,
            "group_mode": self.group_mode,
            "declared_labels_count": len(self.declared_label_map),
            "declared_groups_count": len(self.date_to_group),
            "rename_map_count": len(self.rename_map),
            "labels_csv": str(self.labels_csv) if self.labels_csv else None,
            "groups_csv": str(self.groups_csv) if self.groups_csv else None,
        }
        
    # ----------------------------
    # Canonical name helper
    # ----------------------------
    def canonical_name_for(self, raw_name: str) -> str:
        """Return canonical name for a raw file name, if rename_map provides it."""
        return self.rename_map.get(raw_name, raw_name)

    # ----------------------------
    # Label helpers
    # ----------------------------
    def label_for(
        self,
        *,
        lab: Optional[str] = None,
        mid: Optional[str] = None,
        canonical_name: Optional[str] = None,
    ) -> Label:
        """
        Return LB/MEI/UNK depending on label_mode:
          - blind: always default_label (UNK)
          - declared: declared_label_map[canonical_name] (fallback default_label)
          - external: labels_csv mapping by mid or canonical_name (fallback default_label)
        """
        #==============================
        if self.label_mode == "blind":
            return self.default_label

        if self.label_mode == "declared":
            if canonical_name is None:
                return self.default_label
            return self.declared_label_map.get(canonical_name, self.default_label)

        # external
        mapping_lab_mid, mapping_mid, mapping_lab_name, mapping_name = self._load_external_labels()

        if lab is not None and mid is not None and (lab, mid) in mapping_lab_mid:
            return mapping_lab_mid[(lab, mid)]

        if mid is not None and mid in mapping_mid:
            return mapping_mid[mid]

        if lab is not None and canonical_name is not None and (lab, canonical_name) in mapping_lab_name:
            return mapping_lab_name[(lab, canonical_name)]

        if canonical_name is not None and canonical_name in mapping_name:
            return mapping_name[canonical_name]

        return self.default_label
    #==================
    #==============================
    def _load_external_labels(
        self,
    ) -> Tuple[
        Dict[Tuple[str, str], Label],
        Dict[str, Label],
        Dict[Tuple[str, str], Label],
        Dict[str, Label],
    ]:
        """
        Read labels CSV.

        Accepted key patterns:
        - ('lab', 'mid')
        - ('mid',)
        - ('lab', 'canonical_name')
        - ('canonical_name',)

        Required:
        - column 'label'
        - plus at least one key pattern above
        """
        if self.labels_csv is None:
            return {}, {}, {}, {}

        path = Path(self.labels_csv)
        if not path.exists():
            raise FileNotFoundError(f"labels_csv does not exist: {path}")

        mapping_lab_mid: Dict[Tuple[str, str], Label] = {}
        mapping_mid: Dict[str, Label] = {}
        mapping_lab_name: Dict[Tuple[str, str], Label] = {}
        mapping_name: Dict[str, Label] = {}

        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            cols = set(reader.fieldnames or [])

            if "label" not in cols:
                raise ValueError("labels_csv must include a 'label' column")

            has_lab = "lab" in cols
            has_mid = "mid" in cols
            has_name = "canonical_name" in cols

            if not has_mid and not has_name:
                raise ValueError(
                    "labels_csv must include 'mid' or 'canonical_name' column"
                )

            for row in reader:
                label_value = (row.get("label") or "").strip()
                if label_value not in ("LB", "MEI", "UNK"):
                    raise ValueError(f"Invalid label '{label_value}' in labels_csv")

                lab_value = (row.get("lab") or "").strip() if has_lab else ""

                if has_mid:
                    mid_value = (row.get("mid") or "").strip()
                    if mid_value:
                        if lab_value:
                            mapping_lab_mid[(lab_value, mid_value)] = label_value
                        else:
                            mapping_mid[mid_value] = label_value

                if has_name:
                    name_value = (row.get("canonical_name") or "").strip()
                    if name_value:
                        if lab_value:
                            mapping_lab_name[(lab_value, name_value)] = label_value
                        else:
                            mapping_name[name_value] = label_value

        return mapping_lab_mid, mapping_mid, mapping_lab_name, mapping_name
#==================
    # ----------------------------
    # Group helpers
    # ----------------------------
    def group_for(
        self,
        *,
        date: Optional[str] = None,
        lab: Optional[str] = None,
        mid: Optional[str] = None,
        canonical_name: Optional[str] = None,
    ) -> Group:
        """
        Return Ctrl/Exp/UNK depending on group_mode:
          - by_date: always default_group (UNK)
          - declared_ctrl_exp: date_to_group[date] (fallback default_group)
          - external_ctrl_exp: groups_csv mapping by date/mid/canonical_name (fallback default_group)
        """
        if self.group_mode == "by_date":
            return self.default_group

        if self.group_mode == "declared_ctrl_exp":
            if date is None:
                return self.default_group
            return self.date_to_group.get(date, self.default_group)

        # external_ctrl_exp
        mapping_date, mapping_mid, mapping_name = self._load_external_groups()
        if date is not None and date in mapping_date:
            return mapping_date[date]
        if mid is not None and mid in mapping_mid:
            return mapping_mid[mid]
        if canonical_name is not None and canonical_name in mapping_name:
            return mapping_name[canonical_name]
        return self.default_group

    def _load_external_groups(self) -> Tuple[Dict[str, Group], Dict[str, Group], Dict[str, Group]]:
        """
        Read groups CSV.
        Required:
          - column 'group'
          - plus at least one key column: 'date' or 'mid' or 'canonical_name'
        """
        if self.groups_csv is None:
            return {}, {}, {}

        path = Path(self.groups_csv)
        if not path.exists():
            raise FileNotFoundError(f"groups_csv does not exist: {path}")

        mapping_date: Dict[str, Group] = {}
        mapping_mid: Dict[str, Group] = {}
        mapping_name: Dict[str, Group] = {}

        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            cols = set(reader.fieldnames or [])
            if "group" not in cols:
                raise ValueError("groups_csv must include a 'group' column")

            has_date = "date" in cols
            has_mid = "mid" in cols
            has_name = "canonical_name" in cols
            if not (has_date or has_mid or has_name):
                raise ValueError("groups_csv must include at least one key column: 'date', 'mid', or 'canonical_name'")

            for row in reader:
                grp = (row.get("group") or "").strip()
                if grp not in ("Ctrl", "Exp", "UNK"):
                    raise ValueError(f"Invalid group '{grp}' in groups_csv")

                if has_date:
                    key = (row.get("date") or "").strip()
                    if key:
                        mapping_date[key] = grp

                if has_mid:
                    key = (row.get("mid") or "").strip()
                    if key:
                        mapping_mid[key] = grp

                if has_name:
                    key = (row.get("canonical_name") or "").strip()
                    if key:
                        mapping_name[key] = grp

        return mapping_date, mapping_mid, mapping_name
