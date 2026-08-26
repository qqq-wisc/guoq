//! Classical constraints on a symbolic sub-circuit.
//!
//! A symbolic rewrite rule is valid only when the arbitrary sub-circuit sitting in its
//! hole acts, on the rule's *boundary qubits*, like one of a listed set of permutations of
//! the computational basis. This module represents those permutations and checks a
//! concrete sub-circuit against them.
//!
//! # Width
//!
//! The reference fixed the boundary at exactly two qubits. `Optimizer.parseConstraints`
//! allocated `new boolean[2]` and read exactly four booleans per entry; `findSymb` then
//! compared pattern qubit names against the string literals `"q0"` and `"q1"` and indexed
//! `e.getKey()[0]` and `[1]` directly. A three-qubit boundary was not expressible, and
//! every shipped constraint is width 2 as a result.
//!
//! [`BasisPermutation`] carries its width, so a rule may have any number of boundary
//! qubits. The legacy reader produces width-2 permutations; nothing downstream assumes it.

use std::fmt;

use crate::error::{Result, RuleError};

/// A permutation of the `2^width` computational basis states of the boundary qubits.
///
/// `image[i]` is the basis state that `i` is mapped to. Bit `j` of a basis index belongs
/// to boundary qubit `j`, at bit position `width - 1 - j` — the same big-endian convention
/// the unitary builder uses, so a constraint index and a matrix index mean the same thing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BasisPermutation {
    width: usize,
    image: Vec<u32>,
}

impl BasisPermutation {
    /// Build from an explicit image table.
    pub fn new(width: usize, image: Vec<u32>) -> Result<Self> {
        let n = 1usize << width;
        if image.len() != n {
            return Err(RuleError::Malformed(format!(
                "constraint of width {width} needs {n} entries, got {}",
                image.len()
            )));
        }
        let mut seen = vec![false; n];
        for &v in &image {
            let v = v as usize;
            if v >= n {
                return Err(RuleError::Malformed(format!(
                    "constraint image {v} out of range for width {width}"
                )));
            }
            if std::mem::replace(&mut seen[v], true) {
                return Err(RuleError::Malformed(format!(
                    "constraint is not a permutation: {v} appears twice"
                )));
            }
        }
        Ok(Self { width, image })
    }

    /// The identity permutation.
    pub fn identity(width: usize) -> Self {
        Self {
            width,
            image: (0..(1u32 << width)).collect(),
        }
    }

    /// Build from `(input bits, output bits)` pairs, as the legacy format writes them.
    ///
    /// `bits[j]` is boundary qubit `j`.
    pub fn from_bit_pairs(pairs: &[(Vec<bool>, Vec<bool>)]) -> Result<Self> {
        let Some((first, _)) = pairs.first() else {
            return Err(RuleError::Malformed("empty constraint".into()));
        };
        let width = first.len();
        let n = 1usize << width;
        if pairs.len() != n {
            return Err(RuleError::Malformed(format!(
                "constraint of width {width} needs {n} entries, got {}",
                pairs.len()
            )));
        }
        let mut image = vec![u32::MAX; n];
        for (input, output) in pairs {
            if input.len() != width || output.len() != width {
                return Err(RuleError::Malformed(
                    "constraint entries have inconsistent width".into(),
                ));
            }
            let i = bits_to_index(input);
            let o = bits_to_index(output);
            if image[i] != u32::MAX {
                return Err(RuleError::Malformed(format!(
                    "constraint lists input {i} twice"
                )));
            }
            image[i] = o as u32;
        }
        if image.contains(&u32::MAX) {
            return Err(RuleError::Malformed("constraint is not total".into()));
        }
        Self::new(width, image)
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn len(&self) -> usize {
        self.image.len()
    }

    pub fn is_empty(&self) -> bool {
        self.image.is_empty()
    }

    /// The basis state `index` is mapped to.
    pub fn apply(&self, index: usize) -> usize {
        self.image[index] as usize
    }

    pub fn image(&self) -> &[u32] {
        &self.image
    }

    /// `(input, output)` bit vectors, in index order.
    pub fn bit_pairs(&self) -> Vec<(Vec<bool>, Vec<bool>)> {
        (0..self.len())
            .map(|i| {
                (
                    index_to_bits(i, self.width),
                    index_to_bits(self.apply(i), self.width),
                )
            })
            .collect()
    }

    pub fn is_identity(&self) -> bool {
        self.image.iter().enumerate().all(|(i, &v)| i as u32 == v)
    }
}

/// Basis index from per-qubit bits, big-endian: `bits[0]` is the most significant.
pub fn bits_to_index(bits: &[bool]) -> usize {
    bits.iter()
        .fold(0usize, |acc, &b| (acc << 1) | usize::from(b))
}

/// Per-qubit bits from a basis index, inverse of [`bits_to_index`].
pub fn index_to_bits(index: usize, width: usize) -> Vec<bool> {
    (0..width)
        .map(|j| (index >> (width - 1 - j)) & 1 == 1)
        .collect()
}

impl fmt::Display for BasisPermutation {
    /// The legacy spelling, so a constraint round-trips through a rule file.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{")?;
        for (i, (input, output)) in self.bit_pairs().iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}={}", fmt_bits(input), fmt_bits(output))?;
        }
        write!(f, "}}")
    }
}

fn fmt_bits(bits: &[bool]) -> String {
    format!(
        "[{}]",
        bits.iter()
            .map(|b| b.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// Parse the legacy constraint list,
/// `[{[false, false]=[true, false], ...}, {...}]`.
///
/// The reference parsed this by splitting on `"},"` then `"],"` then running four
/// `String.replace` calls over each fragment (`Optimizer.parseConstraints`), which only
/// worked for exactly two booleans per side. This reads the structure, so any width
/// parses.
pub fn parse_constraints(text: &str) -> Result<Vec<BasisPermutation>> {
    let mut out = Vec::new();
    let bytes: Vec<char> = text.chars().collect();
    let mut i = 0;

    while i < bytes.len() {
        if bytes[i] != '{' {
            i += 1;
            continue;
        }
        let start = i;
        let end = bytes[i..]
            .iter()
            .position(|&c| c == '}')
            .map(|p| i + p)
            .ok_or_else(|| RuleError::Malformed(format!("unclosed constraint at {start}")))?;
        let body: String = bytes[i + 1..end].iter().collect();
        if !body.trim().is_empty() {
            out.push(parse_one(&body)?);
        }
        i = end + 1;
    }

    if out.is_empty() {
        return Err(RuleError::Malformed(format!(
            "no constraints found in `{text}`"
        )));
    }
    Ok(out)
}

fn parse_one(body: &str) -> Result<BasisPermutation> {
    let mut pairs: Vec<(Vec<bool>, Vec<bool>)> = Vec::new();
    // Entries look like `[a, b]=[c, d]`, separated by `, ` at bracket depth zero.
    let mut depth = 0usize;
    let mut current = String::new();
    let mut entries: Vec<String> = Vec::new();
    for c in body.chars() {
        match c {
            '[' => {
                depth += 1;
                current.push(c);
            }
            ']' => {
                depth = depth.saturating_sub(1);
                current.push(c);
            }
            ',' if depth == 0 => {
                entries.push(std::mem::take(&mut current));
            }
            _ => current.push(c),
        }
    }
    if !current.trim().is_empty() {
        entries.push(current);
    }

    for entry in entries {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        let (lhs, rhs) = entry
            .split_once("]=[")
            .ok_or_else(|| RuleError::Malformed(format!("bad constraint entry `{entry}`")))?;
        pairs.push((parse_bits(lhs)?, parse_bits(rhs)?));
    }
    BasisPermutation::from_bit_pairs(&pairs)
}

fn parse_bits(text: &str) -> Result<Vec<bool>> {
    text.trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .split(',')
        .map(|t| match t.trim() {
            "true" => Ok(true),
            "false" => Ok(false),
            other => Err(RuleError::Malformed(format!("bad boolean `{other}`"))),
        })
        .collect()
}

/// Render a constraint list in the legacy spelling.
pub fn format_constraints(constraints: &[BasisPermutation]) -> String {
    format!(
        "[{}]",
        constraints
            .iter()
            .map(BasisPermutation::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_index_round_trip() {
        for width in 1..=4 {
            for i in 0..(1usize << width) {
                let bits = index_to_bits(i, width);
                assert_eq!(bits.len(), width);
                assert_eq!(bits_to_index(&bits), i);
            }
        }
        // Big-endian: bits[0] is the most significant.
        assert_eq!(bits_to_index(&[true, false]), 0b10);
        assert_eq!(index_to_bits(0b10, 2), vec![true, false]);
    }

    #[test]
    fn identity_permutation() {
        let p = BasisPermutation::identity(2);
        assert_eq!(p.width(), 2);
        assert_eq!(p.len(), 4);
        assert!(p.is_identity());
        for i in 0..4 {
            assert_eq!(p.apply(i), i);
        }
    }

    #[test]
    fn rejects_non_permutations() {
        assert!(BasisPermutation::new(2, vec![0, 0, 2, 3]).is_err());
        assert!(BasisPermutation::new(2, vec![0, 1, 2]).is_err());
        assert!(BasisPermutation::new(2, vec![0, 1, 2, 9]).is_err());
    }

    #[test]
    fn parses_a_real_legacy_constraint() {
        let text = "[{[false, false]=[false, false], [true, false]=[true, true], \
                    [false, true]=[false, true], [true, true]=[true, false]}]";
        let cs = parse_constraints(text).unwrap();
        assert_eq!(cs.len(), 1);
        let c = &cs[0];
        assert_eq!(c.width(), 2);
        assert_eq!(c.apply(0b00), 0b00);
        assert_eq!(c.apply(0b10), 0b11);
        assert_eq!(c.apply(0b01), 0b01);
        assert_eq!(c.apply(0b11), 0b10);
    }

    #[test]
    fn parses_a_constraint_list() {
        let text = "[{[false, false]=[false, false], [true, false]=[true, false], \
                    [false, true]=[false, true], [true, true]=[true, true]}, \
                    {[false, false]=[false, true], [true, false]=[true, true], \
                    [false, true]=[false, false], [true, true]=[true, false]}]";
        let cs = parse_constraints(text).unwrap();
        assert_eq!(cs.len(), 2);
        assert!(cs[0].is_identity());
        assert!(!cs[1].is_identity());
    }

    /// The representation is not limited to two boundary qubits, which is the point.
    #[test]
    fn parses_and_builds_wider_constraints() {
        // A three-qubit constraint: swap the low two bits.
        let pairs: Vec<(Vec<bool>, Vec<bool>)> = (0..8)
            .map(|i| {
                let b = index_to_bits(i, 3);
                (b.clone(), vec![b[0], b[2], b[1]])
            })
            .collect();
        let p = BasisPermutation::from_bit_pairs(&pairs).unwrap();
        assert_eq!(p.width(), 3);
        assert_eq!(p.apply(0b001), 0b010);
        assert_eq!(p.apply(0b101), 0b110);

        // ...and it round-trips through the textual form.
        let text = format_constraints(std::slice::from_ref(&p));
        let back = parse_constraints(&text).unwrap();
        assert_eq!(back.len(), 1);
        assert_eq!(back[0], p);
    }

    #[test]
    fn round_trips_through_text() {
        let text = "[{[false, false]=[true, false], [true, false]=[false, false], \
                    [false, true]=[true, true], [true, true]=[false, true]}]";
        let cs = parse_constraints(text).unwrap();
        let back = parse_constraints(&format_constraints(&cs)).unwrap();
        assert_eq!(cs, back);
    }

    #[test]
    fn rejects_malformed_text() {
        assert!(parse_constraints("").is_err());
        assert!(parse_constraints("[]").is_err());
        assert!(parse_constraints("[{[false]=[maybe]}]").is_err());
        assert!(parse_constraints("[{[false, false]=[false, false]}]").is_err()); // not total
        assert!(parse_constraints("[{oops}]").is_err());
    }

    #[test]
    fn bit_pairs_round_trip() {
        let text = "[{[false, false]=[true, true], [true, false]=[false, true], \
                    [false, true]=[true, false], [true, true]=[false, false]}]";
        let c = &parse_constraints(text).unwrap()[0];
        let rebuilt = BasisPermutation::from_bit_pairs(&c.bit_pairs()).unwrap();
        assert_eq!(*c, rebuilt);
    }

    /// Every constraint in every shipped rule file must parse.
    #[test]
    fn all_shipped_constraints_parse() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("rules");
        let mut files: Vec<_> = std::fs::read_dir(&root)
            .unwrap()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.to_string_lossy().contains("_symb"))
            .collect();
        files.sort();
        assert!(!files.is_empty(), "no symbolic rule files found");

        let mut total = 0usize;
        let mut widths = std::collections::BTreeSet::new();
        for path in files {
            let text = std::fs::read_to_string(&path).unwrap();
            for (lineno, line) in text.lines().enumerate() {
                if line.trim().is_empty() {
                    continue;
                }
                let fields: Vec<&str> = line.split(" | ").collect();
                assert!(
                    fields.len() >= 3,
                    "{}:{}: symbolic rule needs three fields",
                    path.display(),
                    lineno + 1
                );
                let cs = parse_constraints(fields[2])
                    .unwrap_or_else(|e| panic!("{}:{}: {e}", path.display(), lineno + 1));
                for c in &cs {
                    widths.insert(c.width());
                    total += 1;
                }
            }
        }
        println!("parsed {total} constraints, widths {widths:?}");
        // 11,060 constraints across the six symbolic rule files (four entries each, two
        // bracket groups per entry, hence the 88,480 bracket groups a naive grep counts).
        assert!(total > 10_000, "expected the full corpus, saw {total}");
        // Every shipped constraint is width 2 -- the hardcoding this module removes.
        assert_eq!(widths, [2].into_iter().collect());
    }
}
