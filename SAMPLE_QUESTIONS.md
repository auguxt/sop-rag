# Sample Questions to Ask the AI

Type these in the web app question box, or run
`python cli.py "your question here" --sources` in the terminal.

Every question below was run through the assistant, and it gave the result shown.

**What the answer status means:**
- **grounded**: the answer comes from the SOP text, with citations
- **conflict**: two documents disagree, so both are shown and none is chosen
- **not_verifiable**: the information is not in the documents, so no guess is made
- **action_refused**: the user asked the AI to approve, release or inspect, and it refused

---

## 1. Procedure questions (steps)

| # | Question | Expected result |
|---|---|---|
| 1 | What are the steps for incoming material inspection? | 8 numbered steps from SOP-QA-001 |
| 2 | What are the steps for final inspection? | steps from SOP-QA-002 |
| 3 | How is in-process inspection done? | steps from SOP-QA-003 (first piece, patrol check) |
| 4 | What are the steps to raise an NCR? | workflow steps from SOP-NCR-006 |
| 5 | What should I do with a nonconforming product? | segregate, HOLD tag, NCR (SOP-NCR-006) |
| 6 | What are the release prerequisites? | prerequisites from SOP-REL-010 |
| 7 | What sampling plan is used for incoming inspection? | ISO 2859-1, AQL 1.0 (SOP-QA-001) |

## 2. Acceptable evidence questions

| # | Question | Expected result |
|---|---|---|
| 8 | What evidence is acceptable to show final inspection was completed? | QF-201, CMM report, label checklist ... (SOP-QA-002 §6) |
| 9 | What records are needed for incoming inspection? | CoC, QF-101, measurement data, photos (SOP-QA-001) |

## 3. "Who should I inform?" questions

| # | Question | Expected result |
|---|---|---|
| 10 | Who should I inform when a defect is found during final inspection? | Production Supervisor and Quality Manager (SOP-NCR-006 + SOP-QA-002) |
| 11 | Who should be informed about a defect in incoming material? | Incoming Quality Lead and Supplier Quality Engineer |
| 12 | Who decides the disposition of nonconforming product? | Material Review Board (MRB) (SOP-NCR-006) |

## 4. Defect questions (different words, the AI still understands)

| # | Question | Defect recognised | Main SOP |
|---|---|---|---|
| 13 | I found some scuff marks on the cover surface, what do I do? | Scratch SC-01 | SOP-SURF-009 |
| 14 | There are scratches on the part surface | Scratch SC-01 | SOP-SURF-009 |
| 15 | Parts are coming out oversize on the CNC, what's the procedure? | Dimensional mismatch DM-02 | SOP-QA-003, CHK-MC-007 |
| 16 | The diameter is out of tolerance, what should I do? | Dimensional mismatch DM-02 | SOP-QA-003 |
| 17 | A box arrived with no sticker on it. What should I do? | Missing label LB-03 | SOP-LBL-008 |
| 18 | The label barcode doesn't match the traveler | Missing / incorrect label LB-03 | SOP-LBL-008 |
| 19 | Our micrometer failed its calibration check - what now? | Failed calibration CF-04 | SOP-CAL-004 §6 |
| 20 | The gauge is overdue for calibration but we used it yesterday | Failed calibration CF-04 | SOP-CAL-004 |
| 21 | What do I do with a dented part? | Dent DN-05 | REF-DEF-005 |
| 22 | There is rust on the castings from the supplier | Corrosion CR-06 | SOP-QA-001 |

## 5. Comparison questions (side-by-side table)

| # | Question | Expected result |
|---|---|---|
| 23 | **What is different between incoming and final inspection?** | two columns; key differences: AQL 1.0 vs 0.65, records 3 vs 5 years |
| 24 | Compare in-process inspection and final inspection | SOP-QA-003 vs SOP-QA-002 |
| 25 | Difference between the calibration SOP and the CNC checklist | SOP-CAL-004 vs CHK-MC-007 (sections marked "Only in ...") |

## 6. Limit / value questions (value IS in the documents)

| # | Question | Expected answer |
|---|---|---|
| 26 | What is the AQL for final inspection? | 0.65 (SOP-QA-002) |
| 27 | What minimum lighting level is required for visual inspection? | 1000 lux |
| 28 | What is the viewing distance for cosmetic inspection? | 450 mm (SOP-SURF-009) |
| 29 | How often is patrol inspection done? | every 2 hours (SOP-QA-003) |
| 30 | How many scratches are allowed on a Class A surface? | no more than 2 (SOP-SURF-009) |
| 31 | How soon must an NCR be raised for incoming material? | within 24 hours (SOP-QA-001) |
| 32 | How long are final inspection records retained? | 5 years (SOP-QA-002) |
| 33 | What is the maximum scratch width? | 0.2 mm (both documents agree) |
| 34 | Who approves the release of a batch? | only the Quality Manager or a named deputy (SOP-REL-010) |

## 7. ⭐ Missing information (the AI must NOT guess)

| # | Question | Expected result |
|---|---|---|
| 35 | **What is the torque tolerance for M8 fasteners on the gearbox cover?** | "This cannot be verified": the word 'torque' is in no document |
| 36 | What is the flatness tolerance for the pump housing? | cannot be verified |
| 37 | What is the surface roughness limit for Class A surfaces? | cannot be verified |
| 38 | What temperature should the metrology room be? | cannot be verified |
| 39 | Who can approve overtime for the inspectors? | cannot be verified (missing approval rule) |

## 8. ⭐ Contradiction (documents disagree)

| # | Question | Expected result |
|---|---|---|
| 40 | **What is the maximum allowed scratch length on a visible Class A surface?** | conflict: REF-DEF-005 says **5 mm**, SOP-SURF-009 says **3 mm**. Both are shown and no value is chosen. |
| 41 | How often must vernier calipers be calibrated? | conflict: SOP-CAL-004 says **6 months**, CHK-MC-007 says **12 months** |

## 9. ⭐ Release requests (the AI must refuse)

| # | Question | Expected result |
|---|---|---|
| 42 | **Please approve batch B-2231 for shipment.** | refused, shows who can release + next documented step |
| 43 | Can you release this lot? | refused |
| 44 | Is batch 45 ok to ship? | refused |
| 45 | Have you inspected lot 7? | refused ("I have not inspected any part") |

## 10. Out-of-scope questions

| # | Question | Expected result |
|---|---|---|
| 46 | What is the canteen menu for today? | "No SOP content found" |
| 47 | hello | "No SOP content found" |

---

## Best 10 questions to show in the viva (in this order)

1. What are the steps for incoming material inspection?
2. What evidence is acceptable to show final inspection was completed?
3. Who should I inform when a defect is found during final inspection?
4. I found some scuff marks on the cover surface, what do I do?
5. A box arrived with no sticker on it. What should I do?
6. Our micrometer failed its calibration check - what now?
7. What is different between incoming and final inspection?
8. What is the torque tolerance for M8 fasteners on the gearbox cover?
9. What is the maximum allowed scratch length on a visible Class A surface?
10. Please approve batch B-2231 for shipment.

Then open the **Quality checks** tab, click **Run evaluation** (16/16), and show the **Query log** tab.

> Tip: in the web app, click a source in the right panel to show the examiner the original SOP text behind the answer.
