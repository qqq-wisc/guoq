//! Gate sets, loaded from data rather than hardcoded.

use rustc_hash::FxHashMap;
use std::path::Path;

use crate::angle::AngleExpr;
use crate::error::{CircuitError, Result};
use crate::gate::GateRegistry;

const BUILTIN_GATESETS: &str = include_str!("gatesets.toml");

/// A gate with parameters pinned to specific values, for hardware that exposes a rotation
/// only at fixed angles (Rigetti's `rx`).
#[derive(Debug, Clone, PartialEq)]
pub struct FixedParamGate {
    pub gate: String,
    pub params: Vec<AngleExpr>,
}

/// A named target gate set.
#[derive(Debug, Clone)]
pub struct GateSet {
    pub name: String,
    pub aliases: Vec<String>,
    pub description: String,
    /// Gates the circuit may contain.
    pub gates: Vec<String>,
    /// Name the resynthesis backend uses for this gate set.
    pub resynth_target: String,
    pub default_rules: String,
    pub default_symb_rules: String,
    /// The angle basis rule synthesis enumerates over.
    pub synth_angles: Vec<AngleExpr>,
    /// Fixed-angle instantiations to enumerate in addition to `synth_angles`.
    pub fixed_params: Vec<FixedParamGate>,
    /// Gates whose angle parameters must not repeat an expression already used anywhere
    /// in the circuit being enumerated.
    ///
    /// The reference's enumerator applies this to `u1`/`u2`/`u3`, `rxx`, and `ms`
    /// (`containsAngle` in `Synthesizer.java`): each basis expression is single-use per
    /// circuit for these gates. For multi-angle gates it is what keeps enumeration
    /// tractable at all — without it, `u3` alone offers 60 variants per placement with
    /// unlimited reuse, and the ibmo space exhausts memory before size 3.
    pub single_use_angles: Vec<String>,
}

impl GateSet {
    /// `true` if `name` is this gate set's name or one of its aliases, case-insensitively.
    pub fn matches(&self, name: &str) -> bool {
        self.name.eq_ignore_ascii_case(name)
            || self.aliases.iter().any(|a| a.eq_ignore_ascii_case(name))
    }

    /// `true` if every gate in this set is known to `registry`.
    pub fn validate(&self, registry: &GateRegistry) -> Result<()> {
        for g in &self.gates {
            if !registry.contains(g) {
                return Err(CircuitError::UnknownGate(format!(
                    "{g} (referenced by gate set `{}`)",
                    self.name
                )));
            }
        }
        for fp in &self.fixed_params {
            let def = registry.get(&fp.gate)?;
            if def.num_params != fp.params.len() {
                return Err(CircuitError::ParamCount {
                    gate: fp.gate.clone(),
                    expected: def.num_params,
                    got: fp.params.len(),
                });
            }
        }
        Ok(())
    }

    /// `true` if `gate` belongs to this set.
    pub fn contains_gate(&self, gate: &str) -> bool {
        self.gates.iter().any(|g| g == gate)
    }
}

/// A collection of gate sets, looked up by name or alias.
#[derive(Debug, Clone)]
pub struct GateSetLibrary {
    sets: Vec<GateSet>,
}

impl GateSetLibrary {
    /// The gate sets shipped with the binary.
    pub fn builtin() -> Self {
        Self::from_toml(BUILTIN_GATESETS).expect("built-in gate sets must parse")
    }

    /// Parse gate sets from TOML text.
    pub fn from_toml(text: &str) -> Result<Self> {
        let raw: RawLibrary =
            toml::from_str(text).map_err(|e| CircuitError::Other(format!("gate set TOML: {e}")))?;
        let sets = raw
            .gateset
            .into_iter()
            .map(GateSet::try_from)
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { sets })
    }

    /// Load additional gate sets from a file, overriding any of the same name.
    pub fn extend_from_file(&mut self, path: &Path) -> Result<()> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| CircuitError::Other(format!("reading {}: {e}", path.display())))?;
        let extra = Self::from_toml(&text)?;
        for set in extra.sets {
            match self.sets.iter_mut().find(|s| s.name == set.name) {
                Some(existing) => *existing = set,
                None => self.sets.push(set),
            }
        }
        Ok(())
    }

    pub fn get(&self, name: &str) -> Result<&GateSet> {
        self.sets
            .iter()
            .find(|s| s.matches(name))
            .ok_or_else(|| CircuitError::Other(format!("unknown gate set `{name}`")))
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.sets.iter().map(|s| s.name.as_str())
    }

    pub fn iter(&self) -> impl Iterator<Item = &GateSet> {
        self.sets.iter()
    }

    pub fn len(&self) -> usize {
        self.sets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sets.is_empty()
    }

    /// Map from gate-set name to its resynthesis target name.
    pub fn resynth_targets(&self) -> FxHashMap<String, String> {
        self.sets
            .iter()
            .map(|s| (s.name.clone(), s.resynth_target.clone()))
            .collect()
    }
}

impl Default for GateSetLibrary {
    fn default() -> Self {
        Self::builtin()
    }
}

// --- TOML shapes -----------------------------------------------------------------

#[derive(serde::Deserialize)]
struct RawLibrary {
    #[serde(default)]
    gateset: Vec<RawGateSet>,
}

#[derive(serde::Deserialize)]
struct RawGateSet {
    name: String,
    #[serde(default)]
    aliases: Vec<String>,
    #[serde(default)]
    description: String,
    gates: Vec<String>,
    #[serde(default)]
    resynth_target: String,
    #[serde(default)]
    default_rules: String,
    #[serde(default)]
    default_symb_rules: String,
    #[serde(default)]
    synth_angles: Vec<String>,
    #[serde(default)]
    fixed_params: Vec<RawFixedParams>,
    #[serde(default)]
    single_use_angles: Vec<String>,
}

#[derive(serde::Deserialize)]
struct RawFixedParams {
    gate: String,
    #[serde(default)]
    params: Vec<String>,
}

impl TryFrom<RawGateSet> for GateSet {
    type Error = CircuitError;

    fn try_from(r: RawGateSet) -> Result<Self> {
        let parse_angle = |s: &str| -> Result<AngleExpr> {
            // Angle expressions reuse the QASM expression grammar by parsing a throwaway
            // gate call, so there is exactly one expression syntax in the system.
            let dag = crate::qasm::parse(&format!("rz({s}) __angle;"))?;
            let idx = *dag
                .topological_gates()
                .first()
                .ok_or_else(|| CircuitError::Other(format!("empty angle expression `{s}`")))?;
            Ok(dag.gate(idx).params[0].clone())
        };

        Ok(GateSet {
            name: r.name,
            aliases: r.aliases,
            description: r.description,
            gates: r.gates,
            resynth_target: r.resynth_target,
            default_rules: r.default_rules,
            default_symb_rules: r.default_symb_rules,
            synth_angles: r
                .synth_angles
                .iter()
                .map(|s| parse_angle(s))
                .collect::<Result<_>>()?,
            fixed_params: r
                .fixed_params
                .into_iter()
                .map(|f| {
                    Ok(FixedParamGate {
                        gate: f.gate,
                        params: f
                            .params
                            .iter()
                            .map(|s| parse_angle(s))
                            .collect::<Result<_>>()?,
                    })
                })
                .collect::<Result<_>>()?,
            single_use_angles: r.single_use_angles,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn builtin_library_loads() {
        let lib = GateSetLibrary::builtin();
        assert_eq!(lib.len(), 6);
        let mut names: Vec<&str> = lib.names().collect();
        names.sort_unstable();
        assert_eq!(
            names,
            ["cliffordt", "ibmn", "ibmo", "ion", "nam", "rigetti"]
        );
    }

    #[test]
    fn every_builtin_gateset_validates_against_the_registry() {
        let lib = GateSetLibrary::builtin();
        let reg = GateRegistry::with_builtins();
        for set in lib.iter() {
            set.validate(&reg)
                .unwrap_or_else(|e| panic!("gate set {} invalid: {e}", set.name));
        }
    }

    #[test]
    fn lookup_by_name_and_alias_is_case_insensitive() {
        let lib = GateSetLibrary::builtin();
        for probe in ["nam", "NAM", "Nam"] {
            assert_eq!(lib.get(probe).unwrap().name, "nam");
        }
        for probe in ["ibmn", "IBMN", "ibmnew", "ibm-eagle"] {
            assert_eq!(lib.get(probe).unwrap().name, "ibmn");
        }
        assert_eq!(lib.get("ibm").unwrap().name, "ibmo");
        assert_eq!(lib.get("clifford+t").unwrap().name, "cliffordt");
    }

    #[test]
    fn unknown_gateset_is_an_error() {
        assert!(GateSetLibrary::builtin().get("nope").is_err());
    }

    /// These are the mappings the reference kept in `Params.GATE_SET_RULES_MAP`, except
    /// where a deliberate departure is recorded in `gatesets.toml`: every gate set whose
    /// symbolic corpus could be regenerated as a strict superset of the reference's
    /// valid rules — all but ibmo and ion — defaults to the regenerated file.
    #[test]
    fn default_rule_files_match_the_reference() {
        let lib = GateSetLibrary::builtin();
        let expect = [
            (
                "nam",
                "rules_q3_s6_nam.txt",
                "generated/rules_q3_s3_nam_symb.txt",
            ),
            ("ibmo", "rules_q3_s4_ibm.txt", "rules_q3_s3_ibm_symb.txt"),
            (
                "ibmn",
                "rules_q3_s6_ibmnew.txt",
                "generated/rules_q3_s3_ibmnew_symb.txt",
            ),
            (
                "rigetti",
                "rules_q3_s5_rigetti.txt",
                "generated/rules_q3_s3_rigetti_symb.txt",
            ),
            ("ion", "rules_q3_s3_ion.txt", "rules_q3_s3_ion_symb.txt"),
            (
                "cliffordt",
                "rules_q3_s6_cliffordt.txt",
                "generated/rules_q3_s3_cliffordt_symb.txt",
            ),
        ];
        for (name, rules, symb) in expect {
            let s = lib.get(name).unwrap();
            assert_eq!(s.default_rules, rules, "{name}");
            assert_eq!(s.default_symb_rules, symb, "{name}");
        }
    }

    /// These are the mappings the reference kept in `Params.GATE_SET_RESYNTH_MAP`.
    #[test]
    fn resynth_targets_match_the_reference() {
        let lib = GateSetLibrary::builtin();
        let t = lib.resynth_targets();
        assert_eq!(t["nam"], "nam");
        assert_eq!(t["ibmo"], "none");
        assert_eq!(t["ibmn"], "ibm_new");
        assert_eq!(t["rigetti"], "");
        assert_eq!(t["ion"], "ion");
        assert_eq!(t["cliffordt"], "none");
    }

    #[test]
    fn gate_membership() {
        let lib = GateSetLibrary::builtin();
        let nam = lib.get("nam").unwrap();
        assert!(nam.contains_gate("cx"));
        assert!(nam.contains_gate("rz"));
        assert!(!nam.contains_gate("sx"));
    }

    #[test]
    fn synth_angles_parse_as_expressions() {
        let lib = GateSetLibrary::builtin();
        let nam = lib.get("nam").unwrap();
        assert_eq!(nam.synth_angles.len(), 3);
        assert_eq!(nam.synth_angles[0].free_vars(), vec!["theta1"]);
        assert_eq!(nam.synth_angles[2].free_vars(), vec!["theta1", "theta2"]);

        let ion = lib.get("ion").unwrap();
        // `pi` and `pi/2` are concrete.
        let concrete: Vec<f64> = ion.synth_angles.iter().filter_map(|a| a.eval()).collect();
        assert!(concrete.iter().any(|v| (v - PI).abs() < 1e-12));
        assert!(concrete.iter().any(|v| (v - PI / 2.0).abs() < 1e-12));

        assert!(lib.get("cliffordt").unwrap().synth_angles.is_empty());
    }

    #[test]
    fn rigetti_fixed_angles_are_parsed() {
        let lib = GateSetLibrary::builtin();
        let r = lib.get("rigetti").unwrap();
        assert_eq!(r.fixed_params.len(), 3);
        let vals: Vec<f64> = r
            .fixed_params
            .iter()
            .map(|f| f.params[0].eval().unwrap())
            .collect();
        for want in [PI / 2.0, 3.0 * PI / 2.0, PI] {
            assert!(vals.iter().any(|v| (v - want).abs() < 1e-12), "{want}");
        }
    }

    /// A new gate set can be added without recompiling, which the reference could not do.
    #[test]
    fn library_extends_from_toml() {
        let mut lib = GateSetLibrary::builtin();
        let before = lib.len();
        let extra = r#"
[[gateset]]
name = "mine"
gates = ["h", "cx", "ccz"]
resynth_target = "none"
synth_angles = ["theta1"]
"#;
        let added = GateSetLibrary::from_toml(extra).unwrap();
        for s in added.iter() {
            assert!(s.validate(&GateRegistry::with_builtins()).is_ok());
        }
        lib.sets.extend(added.sets);
        assert_eq!(lib.len(), before + 1);
        assert!(lib.get("mine").unwrap().contains_gate("ccz"));
    }

    #[test]
    fn overriding_a_builtin_replaces_it() {
        let mut lib = GateSetLibrary::builtin();
        let dir = std::env::temp_dir().join(format!("gs-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("override.toml");
        std::fs::write(
            &path,
            "[[gateset]]\nname = \"nam\"\ngates = [\"h\"]\nresynth_target = \"x\"\n",
        )
        .unwrap();
        lib.extend_from_file(&path).unwrap();
        assert_eq!(lib.len(), 6, "override must not add a duplicate");
        assert_eq!(lib.get("nam").unwrap().gates, vec!["h".to_string()]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn gateset_referencing_unknown_gate_fails_validation() {
        let lib =
            GateSetLibrary::from_toml("[[gateset]]\nname = \"bad\"\ngates = [\"nosuchgate\"]\n")
                .unwrap();
        let reg = GateRegistry::with_builtins();
        assert!(lib.get("bad").unwrap().validate(&reg).is_err());
    }

    #[test]
    fn malformed_toml_is_an_error() {
        assert!(GateSetLibrary::from_toml("[[gateset]]\nname = 3\n").is_err());
    }
}
