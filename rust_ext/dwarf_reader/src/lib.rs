//! kernel_talk_dwarf_rs — Fast DWARF parser exposed to Python via PyO3.
//!
//! Parses DWARF debug information from a Linux vmlinux binary to extract:
//!   - Function symbols (DW_TAG_subprogram) with address ranges and source locations
//!   - Struct/union names (DW_TAG_structure_type, DW_TAG_union_type)
//!   - Line number tables (file:line → PC address mapping)
//!
//! Python API:
//!   parse_dwarf(path: str, verbose: bool = False) -> dict
//!   get_function_ranges(path: str) -> dict  # functions only, faster

use std::collections::HashMap;
use std::fs;

use gimli::{AttributeValue, Dwarf, EndianSlice, LittleEndian, SectionId};
use object::{Object, ObjectSection};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

// ─── Types ────────────────────────────────────────────────────────────────────

type R<'a> = EndianSlice<'a, LittleEndian>;
type MyDwarf<'a> = Dwarf<R<'a>>;
type MyUnit<'a> = gimli::Unit<R<'a>>;

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Convert a gimli reader into an owned String, returning empty on error.
fn reader_str(r: R<'_>) -> String {
    r.to_string().map(|c| c.to_owned()).unwrap_or_default()
}

fn load_sections(path: &str) -> Result<HashMap<String, Vec<u8>>, String> {
    let data = fs::read(path).map_err(|e| format!("Cannot read {path}: {e}"))?;
    let obj = object::File::parse(data.as_slice())
        .map_err(|e| format!("Cannot parse ELF {path}: {e}"))?;
    let mut sections: HashMap<String, Vec<u8>> = HashMap::new();
    for section in obj.sections() {
        if let Ok(name) = section.name() {
            if name.starts_with(".debug_") {
                if let Ok(d) = section.data() {
                    sections.insert(name.to_string(), d.to_vec());
                }
            }
        }
    }
    Ok(sections)
}

fn get_section<'a>(sections: &'a HashMap<String, Vec<u8>>, name: &str) -> R<'a> {
    EndianSlice::new(sections.get(name).map(|v| v.as_slice()).unwrap_or(&[]), LittleEndian)
}

// ─── Data types ───────────────────────────────────────────────────────────────

struct FunctionSym {
    name: String,
    addr_start: u64,
    addr_end: u64,
    file_path: String,
    line: u64,
}

struct StructSym {
    name: String,
}

struct LineEntry {
    address: u64,
    file_path: String,
    line: u64,
}

// ─── File index ───────────────────────────────────────────────────────────────

fn build_file_index(unit: &MyUnit<'_>, dwarf: &MyDwarf<'_>) -> Vec<String> {
    let mut index = vec![String::new()]; // slot 0 unused per DWARF spec

    let program = match unit.line_program.clone() {
        Some(p) => p,
        None => return index,
    };
    let header = program.header();

    // comp_dir lives on the Unit in gimli 0.31
    let comp_dir: String = unit.comp_dir
        .as_ref()
        .map(|r| reader_str(*r))
        .unwrap_or_default();

    for file in header.file_names() {
        let dir: String = match file.directory(header) {
            Some(dir_attr) => match dwarf.attr_string(unit, dir_attr) {
                Ok(s) => reader_str(s),
                Err(_) => comp_dir.clone(),
            },
            None => comp_dir.clone(),
        };

        let fname: String = match dwarf.attr_string(unit, file.path_name()) {
            Ok(s) => reader_str(s),
            Err(_) => String::new(),
        };

        let full = if dir.is_empty() || fname.starts_with('/') {
            fname
        } else {
            format!("{dir}/{fname}")
        };
        index.push(full);
    }
    index
}

// ─── Function extraction ──────────────────────────────────────────────────────

fn extract_functions(unit: &MyUnit<'_>, dwarf: &MyDwarf<'_>, file_index: &[String], out: &mut Vec<FunctionSym>) {
    let mut entries = unit.entries();
    while let Ok(Some((_delta, entry))) = entries.next_dfs() {
        if entry.tag() != gimli::DW_TAG_subprogram { continue; }

        // Skip declarations (no code)
        if let Ok(Some(a)) = entry.attr(gimli::DW_AT_declaration) {
            if let AttributeValue::Flag(true) = a.value() { continue; }
        }

        let name: String = match entry.attr(gimli::DW_AT_name) {
            Ok(Some(a)) => match dwarf.attr_string(unit, a.value()) {
                Ok(s) => reader_str(s),
                Err(_) => continue,
            },
            _ => continue,
        };
        if name.is_empty() { continue; }

        let lo_pc: u64 = match entry.attr(gimli::DW_AT_low_pc) {
            Ok(Some(a)) => match a.value() {
                AttributeValue::Addr(v) => v,
                _ => continue,
            },
            _ => continue,
        };
        if lo_pc == 0 { continue; }

        let hi_pc: u64 = entry.attr(gimli::DW_AT_high_pc).ok().flatten()
            .and_then(|a| match a.value() {
                AttributeValue::Addr(v) => Some(v),
                AttributeValue::Udata(offset) => Some(lo_pc + offset),
                _ => None,
            })
            .unwrap_or(lo_pc);

        let file_idx: usize = entry.attr(gimli::DW_AT_decl_file).ok().flatten()
            .and_then(|a| match a.value() {
                AttributeValue::FileIndex(v) => Some(v as usize),
                _ => None,
            })
            .unwrap_or(0);
        let file_path = file_index.get(file_idx).cloned().unwrap_or_default();

        let line: u64 = entry.attr(gimli::DW_AT_decl_line).ok().flatten()
            .and_then(|a| match a.value() {
                AttributeValue::Udata(v) => Some(v),
                _ => None,
            })
            .unwrap_or(0);

        out.push(FunctionSym { name, addr_start: lo_pc, addr_end: hi_pc, file_path, line });
    }
}

// ─── Struct extraction ────────────────────────────────────────────────────────

fn extract_structs(unit: &MyUnit<'_>, dwarf: &MyDwarf<'_>, out: &mut Vec<StructSym>) {
    let mut entries = unit.entries();
    while let Ok(Some((_delta, entry))) = entries.next_dfs() {
        let tag = entry.tag();
        if tag != gimli::DW_TAG_structure_type && tag != gimli::DW_TAG_union_type { continue; }

        if let Ok(Some(a)) = entry.attr(gimli::DW_AT_declaration) {
            if let AttributeValue::Flag(true) = a.value() { continue; }
        }

        let name: String = match entry.attr(gimli::DW_AT_name) {
            Ok(Some(a)) => match dwarf.attr_string(unit, a.value()) {
                Ok(s) => reader_str(s),
                Err(_) => continue,
            },
            _ => continue,
        };
        if !name.is_empty() {
            out.push(StructSym { name });
        }
    }
}

// ─── Line table extraction ────────────────────────────────────────────────────

fn extract_lines(unit: &MyUnit<'_>, file_index: &[String], out: &mut Vec<LineEntry>) {
    let program = match unit.line_program.clone() {
        Some(p) => p,
        None => return,
    };
    let mut rows = program.rows();
    while let Ok(Some((_header, row))) = rows.next_row() {
        if row.end_sequence() { continue; }
        let file_idx = row.file_index() as usize;
        let file_path = file_index.get(file_idx).cloned().unwrap_or_default();
        let line: u64 = row.line().map(|l| l.get()).unwrap_or(0);
        out.push(LineEntry { address: row.address(), file_path, line });
    }
}

// ─── Python entry points ──────────────────────────────────────────────────────

#[pyfunction]
#[pyo3(signature = (path, verbose=false, functions_only=false))]
fn parse_dwarf(py: Python<'_>, path: &str, verbose: bool, functions_only: bool) -> PyResult<PyObject> {
    let sections = load_sections(path).map_err(PyRuntimeError::new_err)?;

    let dwarf = gimli::Dwarf::load(|id: SectionId| -> Result<R<'_>, gimli::Error> {
        Ok(get_section(&sections, id.name()))
    }).map_err(|e| PyRuntimeError::new_err(format!("DWARF load: {e}")))?;

    let mut functions: Vec<FunctionSym> = Vec::new();
    let mut structs: Vec<StructSym> = Vec::new();
    let mut lines: Vec<LineEntry> = Vec::new();

    let mut cu_iter = dwarf.units();
    let mut cu_count = 0u32;

    while let Ok(Some(header)) = cu_iter.next() {
        let unit = match dwarf.unit(header) {
            Ok(u) => u,
            Err(_) => continue,
        };
        cu_count += 1;
        if verbose && cu_count % 1000 == 0 {
            eprintln!("[dwarf-rs] {cu_count} CUs, {} funcs, {} structs",
                functions.len(), structs.len());
        }

        let file_index = build_file_index(&unit, &dwarf);
        extract_functions(&unit, &dwarf, &file_index, &mut functions);

        if !functions_only {
            extract_structs(&unit, &dwarf, &mut structs);
            extract_lines(&unit, &file_index, &mut lines);
        }
    }

    if verbose {
        eprintln!("[dwarf-rs] Done: {} CUs, {} functions, {} structs, {} line entries",
            cu_count, functions.len(), structs.len(), lines.len());
    }

    let result = PyDict::new_bound(py);

    let py_functions = PyList::empty_bound(py);
    for f in &functions {
        let d = PyDict::new_bound(py);
        d.set_item("name", &f.name)?;
        d.set_item("addr_start", f.addr_start)?;
        d.set_item("addr_end", f.addr_end)?;
        d.set_item("file_path", &f.file_path)?;
        d.set_item("line", f.line)?;
        py_functions.append(&d)?;
    }
    result.set_item("functions", &py_functions)?;

    let py_structs = PyList::empty_bound(py);
    for s in &structs {
        let d = PyDict::new_bound(py);
        d.set_item("name", &s.name)?;
        d.set_item("fields", PyList::empty_bound(py))?;
        py_structs.append(&d)?;
    }
    result.set_item("structs", &py_structs)?;

    let py_lines = PyList::empty_bound(py);
    for le in &lines {
        let d = PyDict::new_bound(py);
        d.set_item("address", le.address)?;
        d.set_item("file_path", &le.file_path)?;
        d.set_item("line", le.line)?;
        py_lines.append(&d)?;
    }
    result.set_item("line_entries", &py_lines)?;

    Ok(result.into())
}

#[pyfunction]
#[pyo3(signature = (path, verbose=false))]
fn get_function_ranges(py: Python<'_>, path: &str, verbose: bool) -> PyResult<PyObject> {
    parse_dwarf(py, path, verbose, true)
}

#[pymodule]
fn kernel_talk_dwarf_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_dwarf, m)?)?;
    m.add_function(wrap_pyfunction!(get_function_ranges, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
