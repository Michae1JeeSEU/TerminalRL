# Landing reserve basis (2026-09-24)

This note records the evidence boundary for the frozen `research_evtol`
`min_landing_soc = 0.25` setting. It does not change the configuration, energy
model, emergency gate, or landing-success semantics.

## Frozen implementation meaning

The current normalized model uses `energy_capacity = 1500` s-equivalent and
`power_reference = 1.0` unit/s. Therefore `0.25 * 1500 = 375` normalized energy
units under the reference model. This arithmetic is not a claim that the vehicle
has 375 seconds of real-flight reserve: phase powers, usable aircraft capacity,
and the energy required for go-around or diversion are not externally calibrated
in this project. The 25% floor is a frozen engineering research assumption, not a
verified EH216-S AFM or type-certification limit.

## Primary evidence

### Yang et al., *Joule* (2021)

The author PDF is [available from Penn State ECEC](https://ecec.me.psu.edu/Pubs/2021_Yang_Joule.pdf),
and the publication record gives DOI [10.1016/j.joule.2021.05.001](https://doi.org/10.1016/j.joule.2021.05.001).
The local extraction is `results/viewer_repair_20260924/references/yang_joule_2021.txt`.

- PDF p. 4 uses a 10-minute cruise-reserve calculation that produces 24% for
  design A and 13.5% for design C under the paper's stated assumptions.
- PDF p. 11 uses EoD SOC above 25% as an example criterion for 10 minutes of
  additional cruise and discusses retaining landing power.
- PDF p. 12 labels 25% as the 10-minute reserve-energy limit in the described
  cell-cycle example.

These are scenario calculations and an example EoD criterion; they do not define
a universal 25% aircraft operating or certification limit.

### CAAC EH216-S special conditions

The public [SC-21-002 PDF](https://www.caac.gov.cn/XXGK/XXGK/BZGF/ZYTJHHM/202202/P020220222611863596239.pdf)
contains PEU.E025(a)(3) on PDF p. 19 (printed p. 12): the powerplant installation
must provide an appropriate electrical-energy margin across permitted and possible
operating conditions while considering possible component failures. The public text
checked here does not specify 25%; it uses an appropriate-margin requirement.
The local extraction is `results/viewer_repair_20260924/references/caac_2022.txt`.

### EASA operational method reference

The [EASA Easy Access Rules for Air Operations](https://www.easa.europa.eu/en/document-library/easy-access-rules/online-publications/easy-access-rules-air-operations?erules-id=ERULES-1963177438-23868),
Revision 24 (March 2026), provides a methodological reference in UAM.OP.VCA.191(c)(4)
for basing reserve on the representative go-around time from the landing decision
point (LDP) back to LDP, using certified minimum performance (CMP), the relevant
configuration and conservative conditions. AMC4(b) specifies at least five minutes
in the relevant configuration. This is a manned-VCA
operational context and is not a direct Chinese unmanned-aircraft rule for EH216-S.

## Calibration boundary

The project keeps the 25% floor unchanged for reproducible fixed-rule experiments.
Aircraft-specific calibration would require measured power by phase, usable battery
capacity and degradation, go-around/diversion energy, and uncertainty bounds. Until
those quantities are available, the normalized 375-unit value remains a model
reference rather than a real-time flight-time guarantee.
