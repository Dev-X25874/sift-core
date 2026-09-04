//! Append-only write-ahead log.
//!
//! Record wire format (little-endian, all fixed-width fields up front so a
//! reader can validate a record before trusting its length-prefixed body):
//!
//!   [checksum: u64][seq: u64][op: u8][key_len: u32][key][val_len: u32][val]
//!
//! On replay we stop at the first record that fails its checksum or runs
//! past EOF — that's the normal shape of a torn write after a crash.

use crate::Seq;
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Put = 0,
    Delete = 1,
}

pub struct WalRecord {
    pub seq: Seq,
    pub op: Op,
    pub key: Vec<u8>,
    pub val: Vec<u8>,
}

/// FNV-1a 64-bit — fast corruption detection, same tradeoff as LevelDB's CRC32C.
fn fnv1a(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;
    let mut hash = OFFSET;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

pub struct Wal {
    writer: BufWriter<File>,
}

impl Wal {
    /// Opens (creating if needed) a WAL file in append mode.
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
        })
    }

    pub fn append(&mut self, seq: Seq, op: Op, key: &[u8], val: &[u8]) -> io::Result<()> {
        let mut body = Vec::with_capacity(8 + 1 + 4 + key.len() + 4 + val.len());
        body.extend_from_slice(&seq.to_le_bytes());
        body.push(op as u8);
        body.extend_from_slice(&(key.len() as u32).to_le_bytes());
        body.extend_from_slice(key);
        body.extend_from_slice(&(val.len() as u32).to_le_bytes());
        body.extend_from_slice(val);

        let checksum = fnv1a(&body);
        self.writer.write_all(&checksum.to_le_bytes())?;
        self.writer.write_all(&body)?;
        Ok(())
    }

    /// Callers decide fsync frequency — that's a latency/durability knob we expose.
    pub fn sync(&mut self) -> io::Result<()> {
        self.writer.flush()?;
        self.writer.get_ref().sync_data()
    }

    /// Truncates the WAL to zero. Called after memtable flush makes the log redundant.
    pub fn reset(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
        })
    }
}

/// Replays every well-formed record, stopping silently at the first
/// short read / bad checksum / unknown op byte.
pub fn replay(path: impl AsRef<Path>) -> io::Result<Vec<WalRecord>> {
    let path = path.as_ref();
    if !path.exists() {
        return Ok(Vec::new());
    }
    let mut reader = BufReader::new(File::open(path)?);
    let mut out = Vec::new();

    loop {
        let mut checksum_buf = [0u8; 8];
        if reader.read_exact(&mut checksum_buf).is_err() {
            break; // clean EOF or torn checksum field
        }
        let expected_checksum = u64::from_le_bytes(checksum_buf);

        let mut seq_buf = [0u8; 8];
        if reader.read_exact(&mut seq_buf).is_err() {
            break;
        }
        let mut op_buf = [0u8; 1];
        if reader.read_exact(&mut op_buf).is_err() {
            break;
        }
        let mut key_len_buf = [0u8; 4];
        if reader.read_exact(&mut key_len_buf).is_err() {
            break;
        }
        let key_len = u32::from_le_bytes(key_len_buf) as usize;
        let mut key = vec![0u8; key_len];
        if reader.read_exact(&mut key).is_err() {
            break;
        }
        let mut val_len_buf = [0u8; 4];
        if reader.read_exact(&mut val_len_buf).is_err() {
            break;
        }
        let val_len = u32::from_le_bytes(val_len_buf) as usize;
        let mut val = vec![0u8; val_len];
        if reader.read_exact(&mut val).is_err() {
            break;
        }

        let mut body = Vec::with_capacity(8 + 1 + 4 + key_len + 4 + val_len);
        body.extend_from_slice(&seq_buf);
        body.extend_from_slice(&op_buf);
        body.extend_from_slice(&key_len_buf);
        body.extend_from_slice(&key);
        body.extend_from_slice(&val_len_buf);
        body.extend_from_slice(&val);

        if fnv1a(&body) != expected_checksum {
            break; // torn/corrupt tail record
        }

        // Unknown op bytes are treated as corruption — stop, don't silently
        // interpret them as Delete.
        let op = match op_buf[0] {
            0 => Op::Put,
            1 => Op::Delete,
            _ => break,
        };
        let seq = u64::from_le_bytes(seq_buf);
        out.push(WalRecord { seq, op, key, val });
    }

    Ok(out)
}
