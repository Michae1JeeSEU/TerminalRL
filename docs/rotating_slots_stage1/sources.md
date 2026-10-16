# Source classifications and implementation limits

This table records the frozen contract appendix B. The archive records below
identify the exact public copies retrieved for provenance; they do not claim
complete OEM validation. The original supplied three files remain unchanged.
Any unavailable source archive remains an explicit gap; secondary news values
must not replace the frozen historical values.

| Contract source | Classification and permitted use |
|---|---|
| S1 CAAC TC0078A Rev 0, 2023-10-12 | REF: 620 kg maximum mass, 25 m/s horizontal-flight reference, height/rotor/battery/motor data, 25% touchdown reference. Fixed 620 kg execution mass is DESIGN. This version is not asserted to be the newest approval. |
| S2 EHang EH216-S public product page | REF: public vehicle positioning/basic performance cross-check only; design speed does not relax the 25 m/s execution limit. |
| S3 Waltz, Okhrin, Schultz (2024), TRC 167:104806, DOI 10.1016/j.trc.2024.104806 | Background only: its 800 m boundary/200 m zone and Mavic setting are not transplanted as this contract's vehicle or geometry. |
| S4 Shao, Shao, Lu (2021), TRC 132:103385, DOI 10.1016/j.trc.2021.103385 | Background only: its nodes, waits, ring scales and simplified energy are not current implementation parameters. |
| S5 Pradeep and Wei (2019), JAIS 16(7):263–277, DOI 10.2514/1.I010710 | LIT-PROXY: momentum/coaxial/shallow-descent structure and EH184 effective drag areas 2.11/1.47 m². These are not EH216-S empirical calibration. |
| S6 Pradeep et al. (2022), Frontiers in Aerospace Engineering, DOI 10.3389/fpace.2022.1064142 | LIT-PROXY: profile-power form, solidity 0.055, mean profile drag 0.0089, Fp 0.97, tip speed 120.48 m/s. The original concept aircraft is not EH216-S. |
| U1 historical dynamics upgrade plan | Historical implementation context for point-mass/checked-equals-executed/PyVista. Its former qualification does not qualify this contract. |
| U2 frozen user research choices | DESIGN: geometry, 80 m, dynamics/jerk limits, synthetic demand/SOC/service distributions, emergency trigger/costs, scheduling/event rules, 85 V equivalent constant voltage, efficiency 0.90 and cumulative 120 s peak budget. |

## Archived public copies and provenance

The following files are the actual bytes retrieved for this registry. Hashes are
SHA-256 over the saved files. The archive records do not replace the source
classifications above and do not assert that either source is a current approval
or an approved flight manual.

### S1 CAAC TC0078A Revision 0

- Archive: `source_archive/S1_CAAC_TC0078A_Rev0_2023-10-12_xuefeiji.pdf`
- Selected URL: `https://www.xuefeiji.org/public/uploads/files/20231028/1_20231028170409faa88.pdf`
- Retrieved UTC: `2026-10-16T08:43:47.9553929Z`
- Filesize: `1,555,086` bytes
- SHA-256: `FCEFCEE38FE49F5CAC41432133F8BA60FD06FDE05AC1806400F64853D08FF3CA`
- Validation: `%PDF-1.6`; 5 pages; not encrypted.
- Visible title/revision/date: page 1, `TYPE CERTIFICATE DATA SHEET`,
  TC0078A, Revision 0, model EH216-S, date 2023-10-12. Relevant contract
  values are on pages 2–3: page 2 contains the technical-data heading and
  85 V motor rating; page 3 contains the 25% battery floor, 90 km/h maximum
  level-flight speed, 620 kg maximum takeoff weight and 120 m AGL maximum
  operating altitude. Page 5 records TC0078A Revision 0 as initially issued
  on 2023-10-12.

### S5 Pradeep and Wei (2019)

- Archive: `source_archive/S5_Pradeep_Wei_2019_JAIS_UAMArrTraj.pdf`
- Selected URL: `https://bpb-us-w2.wpmucdn.com/web.seas.gwu.edu/dist/9/15/files/2019/07/JAIS19_UAMArrTraj.pdf`
- Retrieved UTC: `2026-10-16T08:44:50.9353647Z`
- Filesize: `1,629,913` bytes
- SHA-256: `D5BA2A391586FC60CC53538A18E67A90EEB703A002BF514CEA2124C88C1607A7`
- Validation: `%PDF-1.5`; 27 pages; not encrypted. PDF metadata creation date:
  2019-03-26; no revision identifier is stated in the PDF.
- Visible title: page 1, `Energy Efficient Arrival with RTA Constraint for
  Multirotor eVTOL in Urban Air Mobility`, by Priyank Pradeep and Peng Wei.
  Relevant cited data are on page 8, Table 1 (`Performance Data`), including
  EHang 184 mass 240 kg, rotor diameter 1.6 m, and equivalent front/top plate
  areas 2.11/1.47 m²; pages 7–10 contain the surrounding drag, momentum and
  power equations.

The appendix refers to `energy_reference_v1_0.py`, but the supplied input folder
contains only the contract, config and catalog. A repository reference program
reconstructed from appendix A is implementation material, not the missing
original attachment or an independent empirical oracle. Formula self-checks
must be labelled accordingly in reports.

The vehicle is a performance-constrained point-mass research surrogate, not
full 6DOF, contact dynamics, a validated digital twin or certified flight control.
Energy has no measured EH216-S calibration, auxiliary load, voltage sag,
temperature, SOH, battery imbalance, regeneration or ground-service energy.
The flat local ENU/AGL world has zero wind and position noise. Synthetic Poisson
demand, truncated-normal initial SOC and service classes are DESIGN workloads,
not measured capacity classes. Emergency eligibility never waives physical
motion, energy-domain, pad or continuous 80 m safety constraints. NOT_FOUND is
search failure rather than an infeasibility proof. Computational timing is an
implementation diagnostic, not a selected stage 2 baseline or realtime guarantee.
