//! Compact binary serialization for TØR-G graphs.
//!
//! Binary format specification:
//! ```text
//! Header:     [0x54, 0x47] "TG" magic + version:u8 + flags:u8
//! Inputs:     count:u16 (LE) + [id:u16; count]
//! Nodes:      count:u16 (LE) + [Node; count]
//!   Node:     id:u16 + op:u8 + left:Source + right:Source
//!   Source:   tag:u8 (0=Id, 1=True, 2=False) + payload:u16 if Id
//! Outputs:    count:u16 (LE) + [id:u16; count]
//! ```

use std::io::{Cursor, Read};
use torg_core::{BoolOp, Graph, Node, Source};

use crate::error::SerdeError;

/// Magic bytes identifying TØR-G binary format.
const MAGIC: [u8; 2] = [0x54, 0x47]; // "TG"

/// Current binary format version.
const VERSION: u8 = 1;

/// Source tag for Id variant.
const SOURCE_ID: u8 = 0;
/// Source tag for True variant.
const SOURCE_TRUE: u8 = 1;
/// Source tag for False variant.
const SOURCE_FALSE: u8 = 2;

/// Op code for Or.
const OP_OR: u8 = 0;
/// Op code for Nor.
const OP_NOR: u8 = 1;
/// Op code for Xor.
const OP_XOR: u8 = 2;

/// Serialize a graph to compact binary format.
///
/// The format is designed for compactness (~7 bytes per node) and
/// fast parsing. All multi-byte values use little-endian encoding.
pub fn to_bytes(graph: &Graph) -> Vec<u8> {
    let mut buf = Vec::with_capacity(estimate_size(graph));

    // Header: magic + version + flags
    buf.extend_from_slice(&MAGIC);
    buf.push(VERSION);
    buf.push(0); // Reserved flags

    // Inputs section
    write_u16(&mut buf, graph.inputs.len() as u16);
    for &id in &graph.inputs {
        write_u16(&mut buf, id);
    }

    // Nodes section
    write_u16(&mut buf, graph.nodes.len() as u16);
    for node in &graph.nodes {
        write_u16(&mut buf, node.id);
        buf.push(encode_op(node.op));
        write_source(&mut buf, &node.left);
        write_source(&mut buf, &node.right);
    }

    // Outputs section
    write_u16(&mut buf, graph.outputs.len() as u16);
    for &id in &graph.outputs {
        write_u16(&mut buf, id);
    }

    buf
}

/// Deserialize a graph from binary format.
///
/// Returns an error if the format is invalid or the data is corrupted.
pub fn from_bytes(bytes: &[u8]) -> Result<Graph, SerdeError> {
    let mut cursor = Cursor::new(bytes);
    let mut pos = 0usize;

    // Read and validate header
    let magic = read_bytes(&mut cursor, 2, &mut pos)?;
    if magic[0] != MAGIC[0] || magic[1] != MAGIC[1] {
        return Err(SerdeError::InvalidMagic(magic[0], magic[1]));
    }

    let version = read_u8(&mut cursor, &mut pos)?;
    if version != VERSION {
        return Err(SerdeError::UnsupportedVersion(version));
    }

    let _flags = read_u8(&mut cursor, &mut pos)?; // Reserved

    // Read inputs
    let input_count = read_u16(&mut cursor, &mut pos)?;
    let mut inputs = Vec::with_capacity(input_count as usize);
    for _ in 0..input_count {
        inputs.push(read_u16(&mut cursor, &mut pos)?);
    }

    // Read nodes
    let node_count = read_u16(&mut cursor, &mut pos)?;
    let mut nodes = Vec::with_capacity(node_count as usize);
    for _ in 0..node_count {
        let id = read_u16(&mut cursor, &mut pos)?;
        let op = decode_op(read_u8(&mut cursor, &mut pos)?, pos)?;
        let left = read_source(&mut cursor, &mut pos)?;
        let right = read_source(&mut cursor, &mut pos)?;
        nodes.push(Node::new(id, op, left, right));
    }

    // Read outputs
    let output_count = read_u16(&mut cursor, &mut pos)?;
    let mut outputs = Vec::with_capacity(output_count as usize);
    for _ in 0..output_count {
        outputs.push(read_u16(&mut cursor, &mut pos)?);
    }

    Ok(Graph {
        inputs,
        nodes,
        outputs,
    })
}

/// Estimate serialized size for pre-allocation.
fn estimate_size(graph: &Graph) -> usize {
    4 // Header
    + 2 + graph.inputs.len() * 2 // Inputs
    + 2 + graph.nodes.len() * 7 // Nodes (worst case: 2 + 1 + 3 + 3 = 9, but typical is 7)
    + 2 + graph.outputs.len() * 2 // Outputs
}

/// Write a u16 in little-endian format.
#[inline]
fn write_u16(buf: &mut Vec<u8>, val: u16) {
    buf.extend_from_slice(&val.to_le_bytes());
}

/// Encode a BoolOp as a single byte.
#[inline]
fn encode_op(op: BoolOp) -> u8 {
    match op {
        BoolOp::Or => OP_OR,
        BoolOp::Nor => OP_NOR,
        BoolOp::Xor => OP_XOR,
    }
}

/// Write a Source operand.
fn write_source(buf: &mut Vec<u8>, source: &Source) {
    match source {
        Source::Id(id) => {
            buf.push(SOURCE_ID);
            write_u16(buf, *id);
        }
        Source::True => buf.push(SOURCE_TRUE),
        Source::False => buf.push(SOURCE_FALSE),
    }
}

/// Read exactly `n` bytes.
fn read_bytes(
    cursor: &mut Cursor<&[u8]>,
    n: usize,
    pos: &mut usize,
) -> Result<Vec<u8>, SerdeError> {
    let mut buf = vec![0u8; n];
    cursor
        .read_exact(&mut buf)
        .map_err(|_| SerdeError::UnexpectedEof(*pos))?;
    *pos += n;
    Ok(buf)
}

/// Read a single byte.
fn read_u8(cursor: &mut Cursor<&[u8]>, pos: &mut usize) -> Result<u8, SerdeError> {
    let mut buf = [0u8; 1];
    cursor
        .read_exact(&mut buf)
        .map_err(|_| SerdeError::UnexpectedEof(*pos))?;
    *pos += 1;
    Ok(buf[0])
}

/// Read a u16 in little-endian format.
fn read_u16(cursor: &mut Cursor<&[u8]>, pos: &mut usize) -> Result<u16, SerdeError> {
    let mut buf = [0u8; 2];
    cursor
        .read_exact(&mut buf)
        .map_err(|_| SerdeError::UnexpectedEof(*pos))?;
    *pos += 2;
    Ok(u16::from_le_bytes(buf))
}

/// Decode a BoolOp from a single byte.
fn decode_op(byte: u8, _pos: usize) -> Result<BoolOp, SerdeError> {
    match byte {
        OP_OR => Ok(BoolOp::Or),
        OP_NOR => Ok(BoolOp::Nor),
        OP_XOR => Ok(BoolOp::Xor),
        _ => Err(SerdeError::InvalidOpCode(byte)),
    }
}

/// Read a Source operand.
fn read_source(cursor: &mut Cursor<&[u8]>, pos: &mut usize) -> Result<Source, SerdeError> {
    let tag = read_u8(cursor, pos)?;
    match tag {
        SOURCE_ID => {
            let id = read_u16(cursor, pos)?;
            Ok(Source::Id(id))
        }
        SOURCE_TRUE => Ok(Source::True),
        SOURCE_FALSE => Ok(Source::False),
        _ => Err(SerdeError::InvalidSourceTag(tag)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_graph() -> Graph {
        Graph {
            inputs: vec![0, 1],
            nodes: vec![
                Node::new(2, BoolOp::Or, Source::Id(0), Source::Id(1)),
                Node::new(3, BoolOp::Nor, Source::Id(2), Source::True),
            ],
            outputs: vec![3],
        }
    }

    #[test]
    fn test_roundtrip() {
        let graph = sample_graph();
        let bytes = to_bytes(&graph);
        let restored = from_bytes(&bytes).unwrap();
        assert_eq!(graph, restored);
    }

    #[test]
    fn test_empty_graph() {
        let graph = Graph::default();
        let bytes = to_bytes(&graph);
        let restored = from_bytes(&bytes).unwrap();
        assert_eq!(graph, restored);
    }

    #[test]
    fn test_all_source_types() {
        let graph = Graph {
            inputs: vec![0],
            nodes: vec![
                Node::new(1, BoolOp::Or, Source::Id(0), Source::True),
                Node::new(2, BoolOp::Xor, Source::False, Source::Id(1)),
            ],
            outputs: vec![2],
        };
        let bytes = to_bytes(&graph);
        let restored = from_bytes(&bytes).unwrap();
        assert_eq!(graph, restored);
    }

    #[test]
    fn test_compact_size() {
        let graph = sample_graph();
        let bytes = to_bytes(&graph);
        // Header(4) + inputs(2+4) + nodes(2 + 2*(2+1+3+3)) + outputs(2+2) = 4+6+20+4 = 34
        // With Source::Id taking 3 bytes: 2+1+3+3 = 9 per node with two Id sources
        // Actually: node 1: 2+1+3+3=9, node 2: 2+1+3+1=7 (True is 1 byte)
        assert!(
            bytes.len() < 50,
            "Expected compact encoding, got {} bytes",
            bytes.len()
        );
    }

    #[test]
    fn test_invalid_magic() {
        let bytes = [0x00, 0x00, VERSION, 0, 0, 0, 0, 0, 0, 0];
        let result = from_bytes(&bytes);
        assert!(matches!(result, Err(SerdeError::InvalidMagic(0x00, 0x00))));
    }

    #[test]
    fn test_unsupported_version() {
        let bytes = [MAGIC[0], MAGIC[1], 99, 0, 0, 0, 0, 0, 0, 0];
        let result = from_bytes(&bytes);
        assert!(matches!(result, Err(SerdeError::UnsupportedVersion(99))));
    }

    #[test]
    fn test_unexpected_eof() {
        let bytes = [MAGIC[0], MAGIC[1], VERSION, 0];
        let result = from_bytes(&bytes);
        assert!(matches!(result, Err(SerdeError::UnexpectedEof(_))));
    }
}
