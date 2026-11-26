"""Minimal readers: NRRD (attached data, raw/gzip) and Anima multi-compartment models (.mcm/.mcmx).

An Anima MCM is an XML file (.mcm) listing a weights image and one image per compartment
(<Compartment><Type>Tensor</Type><FileName>mcm_2.nrrd</FileName></Compartment>), stored
in a folder named after the .mcm file; a .mcmx is the zip of both.
"""
from __future__ import annotations
import gzip
import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union

import numpy as np

_NRRD_TYPES = {
    "signed char": "i1", "int8": "i1", "int8_t": "i1",
    "uchar": "u1", "unsigned char": "u1", "uint8": "u1", "uint8_t": "u1",
    "short": "i2", "short int": "i2", "signed short": "i2", "signed short int": "i2", "int16": "i2", "int16_t": "i2",
    "ushort": "u2", "unsigned short": "u2", "unsigned short int": "u2", "uint16": "u2", "uint16_t": "u2",
    "int": "i4", "signed int": "i4", "int32": "i4", "int32_t": "i4",
    "uint": "u4", "unsigned int": "u4", "uint32": "u4", "uint32_t": "u4",
    "longlong": "i8", "long long": "i8", "long long int": "i8", "int64": "i8", "int64_t": "i8",
    "ulonglong": "u8", "unsigned long long": "u8", "unsigned long long int": "u8", "uint64": "u8", "uint64_t": "u8",
    "float": "f4", "double": "f8",
}
LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])


@dataclass
class Nrrd:
    data: np.ndarray            # shape = NRRD sizes (fastest axis first), e.g. (6, X, Y, Z)
    header: Dict[str, str]
    kinds: List[str] = field(default_factory=list)

    @property
    def domain_axes(self) -> List[int]:
        return [i for i, k in enumerate(self.kinds) if k == "domain"] or list(range(self.data.ndim))

    def affine_ras(self) -> np.ndarray:
        """4x4 voxel -> world RAS affine of the domain axes."""
        dirs = [d for d in _vectors(self.header.get("space directions", "")) if d is not None]
        origin = (_vectors(self.header.get("space origin", "")) or [None])[0]
        affine = np.eye(4)
        if dirs:
            affine[:3, :3] = np.array(dirs[:3]).T
        if origin is not None:
            affine[:3, 3] = origin
        space = self.header.get("space", "left-posterior-superior").lower()
        if space in ("left-posterior-superior", "lps"):
            affine = LPS_TO_RAS @ affine
        return affine

    def voxels_last(self) -> np.ndarray:
        """Data with the domain axes first and the non-domain (vector) axes last."""
        dom = self.domain_axes
        other = [i for i in range(self.data.ndim) if i not in dom]
        return np.transpose(self.data, dom + other)


def _vectors(text: str) -> List:
    """'(1,0,0) none (0,1,0)' -> [array, None, array]."""
    out = []
    for token in re.findall(r"\([^)]*\)|none", text):
        out.append(None if token == "none" else np.array([float(v) for v in token[1:-1].split(",")]))
    return out


def read_nrrd(source: Union[str, Path, bytes]) -> Nrrd:
    raw = source if isinstance(source, (bytes, bytearray)) else Path(source).read_bytes()
    sep = raw.find(b"\n\n")
    if not raw.startswith(b"NRRD") or sep < 0:
        raise ValueError("Not an NRRD file with attached data.")
    header: Dict[str, str] = {}
    for line in raw[:sep].decode("latin-1").splitlines()[1:]:
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition(":=") if ":=" in line and ": " not in line else line.partition(":")
        header[key.strip().lower()] = value.strip()
    if "data file" in header or "datafile" in header:
        raise ValueError("Detached NRRD data is not supported.")
    sizes = [int(s) for s in header["sizes"].split()]
    dtype = np.dtype(_NRRD_TYPES[header["type"].lower()])
    if dtype.itemsize > 1:
        dtype = dtype.newbyteorder("<" if header.get("endian", "little") == "little" else ">")
    body = raw[sep + 2:]
    encoding = header.get("encoding", "raw").lower()
    if encoding in ("gzip", "gz"):
        body = gzip.decompress(body)
    elif encoding != "raw":
        raise ValueError(f"NRRD encoding '{encoding}' is not supported.")
    count = int(np.prod(sizes))
    data = np.frombuffer(body, dtype=dtype, count=count).reshape(sizes, order="F")
    kinds = header.get("kinds", "").split() or ["domain"] * len(sizes)
    return Nrrd(data=data, header=header, kinds=kinds)


@dataclass
class Compartment:
    type: str
    data: np.ndarray     # (X, Y, Z) or (X, Y, Z, C)
    weight: np.ndarray   # (X, Y, Z)


@dataclass
class MCM:
    compartments: List[Compartment]
    affine: np.ndarray   # voxel -> RAS
    shape: Tuple[int, int, int]


def read_mcm(path: Union[str, Path]) -> MCM:
    """Read an Anima .mcmx (zip) or .mcm (XML next to its folder of images)."""
    path = Path(path)
    if path.suffix.lower() == ".mcmx":
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            xml_name = next((n for n in names if n.lower().endswith(".mcm") and "/" not in n.strip("/")), None) \
                or next((n for n in names if n.lower().endswith(".mcm")), None)
            if xml_name is None:
                raise ValueError(f"{path.name}: no .mcm file in the archive.")
            folder = xml_name[: -len(".mcm")]
            return _parse_mcm(archive.read(xml_name), lambda f: archive.read(f"{folder}/{f}"))
    folder = path.with_suffix("")
    return _parse_mcm(path.read_bytes(), lambda f: (folder / f).read_bytes())


def _parse_mcm(xml: bytes, read: Callable[[str], bytes]) -> MCM:
    root = ET.fromstring(xml)
    weights = read_nrrd(read(root.findtext("Weights").strip()))
    w = weights.voxels_last()  # (X, Y, Z, n)
    comps = []
    for i, node in enumerate(root.findall("Compartment")):
        nrrd = read_nrrd(read(node.findtext("FileName").strip()))
        comps.append(Compartment(type=node.findtext("Type").strip(), data=nrrd.voxels_last(), weight=w[..., i]))
    return MCM(compartments=comps, affine=weights.affine_ras(), shape=tuple(w.shape[:3]))
