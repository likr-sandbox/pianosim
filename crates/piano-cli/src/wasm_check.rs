//! Wasm binary section parser verifying genuine SIMD128 (`0xfd`) opcodes.

const WASM_MAGIC: [u8; 4] = [0x00, 0x61, 0x73, 0x6d];
const WASM_VERSION: [u8; 4] = [0x01, 0x00, 0x00, 0x00];

/// Summary of parsed Wasm binary sections and SIMD128 instruction counts.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct WasmSimdReport {
    pub import_count: u32,
    pub code_bodies: u32,
    pub v128_locals: u32,
    pub simd_opcodes: usize,
}

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.pos)
    }

    fn read_u8(&mut self) -> Result<u8, String> {
        let b = *self
            .bytes
            .get(self.pos)
            .ok_or_else(|| String::from("unexpected EOF in Wasm binary"))?;
        self.pos += 1;
        Ok(b)
    }

    fn read_slice(&mut self, len: usize) -> Result<&'a [u8], String> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| String::from("Wasm length overflow"))?;
        let slice = self
            .bytes
            .get(self.pos..end)
            .ok_or_else(|| String::from("truncated Wasm section or payload"))?;
        self.pos = end;
        Ok(slice)
    }

    fn skip(&mut self, len: usize) -> Result<(), String> {
        let _ = self.read_slice(len)?;
        Ok(())
    }

    fn read_uleb128(&mut self) -> Result<u32, String> {
        let mut result = 0_u32;
        let mut shift = 0_u32;
        for _ in 0..5 {
            let byte = self.read_u8()?;
            result |= u32::from(byte & 0x7f) << shift;
            if (byte & 0x80) == 0 {
                return Ok(result);
            }
            shift += 7;
        }
        Err(String::from("malformed u32 LEB128 in Wasm binary"))
    }

    fn skip_sleb128(&mut self) -> Result<(), String> {
        for _ in 0..10 {
            let byte = self.read_u8()?;
            if (byte & 0x80) == 0 {
                return Ok(());
            }
        }
        Err(String::from("malformed signed LEB128 in Wasm binary"))
    }
}

/// Parses Wasm sections (LEB128) and scans Code section (ID 10) for `0xfd` opcodes.
pub fn inspect_wasm_simd(bytes: &[u8]) -> Result<WasmSimdReport, String> {
    if bytes.len() < 8 || bytes[0..4] != WASM_MAGIC || bytes[4..8] != WASM_VERSION {
        return Err(String::from("invalid Wasm magic or version header"));
    }

    let mut cur = Cursor::new(&bytes[8..]);
    let mut report = WasmSimdReport::default();

    while cur.remaining() > 0 {
        let section_id = cur.read_u8()?;
        let section_size = cur.read_uleb128()? as usize;
        let payload = cur.read_slice(section_size)?;
        if section_id == 2 {
            let mut imp_cur = Cursor::new(payload);
            report.import_count = imp_cur.read_uleb128()?;
        } else if section_id == 10 {
            scan_code_section(payload, &mut report)?;
        }
    }

    Ok(report)
}

fn scan_code_section(payload: &[u8], report: &mut WasmSimdReport) -> Result<(), String> {
    let mut sec = Cursor::new(payload);
    let func_count = sec.read_uleb128()?;
    report.code_bodies = func_count;

    for _ in 0..func_count {
        let body_size = sec.read_uleb128()? as usize;
        let body_bytes = sec.read_slice(body_size)?;
        let mut body = Cursor::new(body_bytes);

        let local_decls = body.read_uleb128()?;
        for _ in 0..local_decls {
            let count = body.read_uleb128()?;
            let valtype = body.read_u8()?;
            if valtype == 0x7b {
                report.v128_locals = report.v128_locals.saturating_add(count);
            }
        }

        while body.remaining() > 0 {
            let op = body.read_u8()?;
            skip_opcode_immediates(op, &mut body, report)?;
        }
    }

    Ok(())
}

fn skip_opcode_immediates(
    op: u8,
    body: &mut Cursor<'_>,
    report: &mut WasmSimdReport,
) -> Result<(), String> {
    match op {
        0x00 | 0x01 | 0x05 | 0x0b | 0x0f | 0x1a | 0x1b | 0xd1 | 0x45..=0xc4 => {}
        0x02..=0x04 | 0x41 | 0x42 => body.skip_sleb128()?,
        0x0c | 0x0d | 0x10 | 0x20..=0x26 | 0x3f | 0x40 | 0xd2 => {
            let _ = body.read_uleb128()?;
        }
        0x0e => {
            let count = body.read_uleb128()?;
            for _ in 0..=count {
                let _ = body.read_uleb128()?;
            }
        }
        0x11 | 0x28..=0x3e => {
            let _ = body.read_uleb128()?;
            let _ = body.read_uleb128()?;
        }
        0x1c => {
            let count = body.read_uleb128()? as usize;
            body.skip(count)?;
        }
        0x43 => body.skip(4)?,
        0x44 => body.skip(8)?,
        0xd0 => {
            let _ = body.read_u8()?;
        }
        0xfc => skip_misc_opcode(body)?,
        0xfd => {
            report.simd_opcodes += 1;
            skip_simd_opcode(body)?;
        }
        _ => {
            return Err(format!(
                "unrecognized Wasm opcode 0x{op:02x} in code section"
            ));
        }
    }
    Ok(())
}

fn skip_misc_opcode(body: &mut Cursor<'_>) -> Result<(), String> {
    let sub = body.read_uleb128()?;
    match sub {
        0..=7 => {}
        8 => {
            let _ = body.read_uleb128()?;
            let _ = body.read_u8()?;
        }
        9 | 13 | 15..=17 => {
            let _ = body.read_uleb128()?;
        }
        10 => {
            let _ = body.read_u8()?;
            let _ = body.read_u8()?;
        }
        11 => {
            let _ = body.read_u8()?;
        }
        12 | 14 => {
            let _ = body.read_uleb128()?;
            let _ = body.read_uleb128()?;
        }
        _ => return Err(format!("unrecognized 0xfc sub-opcode {sub}")),
    }
    Ok(())
}

fn skip_simd_opcode(body: &mut Cursor<'_>) -> Result<(), String> {
    let sub = body.read_uleb128()?;
    match sub {
        0..=11 | 92 | 93 => {
            let _ = body.read_uleb128()?;
            let _ = body.read_uleb128()?;
        }
        12 | 13 => body.skip(16)?,
        21..=34 => {
            let _ = body.read_u8()?;
        }
        84..=91 => {
            let _ = body.read_uleb128()?;
            let _ = body.read_uleb128()?;
            let _ = body.read_u8()?;
        }
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_wasm_with_body(body: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&WASM_MAGIC);
        out.extend_from_slice(&WASM_VERSION);
        let mut sec = vec![1_u8];
        sec.push(body.len() as u8);
        sec.extend_from_slice(body);
        out.push(10);
        out.push(sec.len() as u8);
        out.extend_from_slice(&sec);
        out
    }

    #[test]
    fn wasm_simd_scanner_distinguishes_f32_const_bytes_from_real_0xfd_opcodes() {
        // f32.const containing 0xfd bytes + drop + end -> MUST report 0 SIMD opcodes.
        let fake = minimal_wasm_with_body(&[0x00, 0x43, 0xfd, 0xfd, 0xfd, 0x3f, 0x1a, 0x0b]);
        let fake_report = inspect_wasm_simd(&fake).expect("valid fake wasm");
        assert_eq!(fake_report.simd_opcodes, 0);

        // Real v128.const + v128.const + f32x4.add (0xfd 0xe4 0x01) + drop + end -> 3 SIMD opcodes.
        let mut real_body = vec![0x00];
        real_body.extend_from_slice(&[0xfd, 12]);
        real_body.extend_from_slice(&[0_u8; 16]);
        real_body.extend_from_slice(&[0xfd, 12]);
        real_body.extend_from_slice(&[0_u8; 16]);
        real_body.extend_from_slice(&[0xfd, 0xe4, 0x01, 0x1a, 0x0b]);
        let real = minimal_wasm_with_body(&real_body);
        let real_report = inspect_wasm_simd(&real).expect("valid real wasm");
        assert_eq!(real_report.simd_opcodes, 3);
    }
}
