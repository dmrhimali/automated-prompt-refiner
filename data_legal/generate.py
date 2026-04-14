"""
Generate a synthetic legal document review dataset for the
Luminant/Vistra vs. Cipher matter.

Produces input.jsonl + gold.jsonl for dev, canary, and test splits.

The documents are synthetic — no real case data is used.  They are designed
to exercise the six relevance categories and common false-positive traps
(tangential energy docs, generic ERCOT filings, unrelated internal comms).

Run once:
    uv run python data_legal/generate.py
"""

from __future__ import annotations

import json
import pathlib
import random

ROOT = pathlib.Path(__file__).parent

# ── Seed for reproducibility ─────────────────────────────────────────────────
random.seed(42)

# ── Query used for all rows (single-issue review) ───────────────────────────
QUERY = (
    "Documents relevant to the Luminant/Vistra vs. Cipher matter, "
    "including agreements, communications, invoices, ERCOT approval, "
    "Odessa Plant operations, and project references (Bitfury/Cipher/Cypher)."
)

# ── Document templates ───────────────────────────────────────────────────────
# Each entry: (document_text, label, category)
# label: 1 = relevant, 0 = not relevant

RELEVANT_DOCS: list[tuple[str, str]] = [
    # ── Hard relevant: subtle signals, no obvious keywords ────────────
    # These test whether the classifier can identify relevance from
    # context rather than keyword matching.
    (
        "Email from J. Roberts to S. Patel: 'The tenant at the West Texas "
        "facility called again about the power fluctuations. They want a "
        "credit for last month. I told them we need to review the contract "
        "first. Can you pull the file?' No party names in body — only "
        "identifiable from the Odessa/West Texas location and power context.",
        "hard_relevant",
    ),
    (
        "Handwritten note scanned to PDF: 'Call T.W. re: mining facility "
        "electrical specs. 300MW req. Check if substation can handle. "
        "Also need La Frontera lease copy.' Abbreviations and informal "
        "language with no full party names.",
        "hard_relevant",
    ),
    (
        "Spreadsheet tab labeled 'Project B' showing monthly MWh delivery "
        "figures from March 2019 to December 2022 for an unnamed customer "
        "at the Odessa location. Column headers: Month, Contracted MWh, "
        "Delivered MWh, Shortfall, Credit Amount. Internal codename only.",
        "hard_relevant",
    ),
    (
        "Legal hold notice from Vistra in-house counsel: 'Effective "
        "immediately, preserve all documents relating to the West Texas "
        "data center customer dispute. This includes emails, contracts, "
        "invoices, and internal memos.' Does not name Cipher explicitly.",
        "hard_relevant",
    ),
    (
        "Travel expense report for J. Roberts showing flights DFW-MAF "
        "(Midland-Odessa) on four dates in 2022, with hotel receipts in "
        "Odessa. Purpose listed as 'customer site visit — data center.' "
        "Relevant because it documents visits to the Cipher facility.",
        "hard_relevant",
    ),
    (
        "Email from Luminant accounting: 'The receivable for the mining "
        "customer is now 90 days past due. Total outstanding: $8.7M. "
        "Legal has been notified.' No mention of Cipher by name.",
        "hard_relevant",
    ),
    (
        "Draft press release (never published): 'Luminant announces "
        "innovative partnership to provide dedicated power to cutting-edge "
        "blockchain technology operations in the Permian Basin.' Refers to "
        "the Cipher deal without naming the company.",
        "hard_relevant",
    ),
    (
        "Insurance claim filed by Luminant for transformer damage at the "
        "'Odessa Substation — data center interconnection point.' The claim "
        "describes equipment serving the dedicated mining load but refers "
        "to the customer only as 'the interconnected load customer.'",
        "hard_relevant",
    ),

    # ── Standard relevant: clear keywords ─────────────────────────────
    # Category 1: Agreements & Negotiations
    (
        "LEASE AGREEMENT dated March 15, 2019 between Cipher Mining Inc. "
        "(Tenant) and La Frontera Holdings LLC (Landlord) for the premises "
        "located at 4200 Industrial Blvd, Odessa, TX 79762. The Tenant "
        "shall use the premises exclusively for cryptocurrency mining "
        "operations and related data center activities.",
        "agreements",
    ),
    (
        "PURCHASE AND SALE AGREEMENT between Vistra Corp. and Cipher "
        "Mining Inc. dated June 1, 2020. Vistra agrees to sell and Cipher "
        "agrees to purchase electrical capacity from the Odessa generating "
        "facility at the rates specified in Schedule A.",
        "agreements",
    ),
    (
        "Email from J. Roberts (Luminant) to M. Chen (Cipher): 'Attached "
        "is the redline of the amended Power Purchase Agreement. Please "
        "review Section 4.2 regarding the revised delivery point at the "
        "Substation and confirm Cipher's acceptance by Friday.'",
        "agreements",
    ),
    (
        "Internal memo from Vistra Legal: 'The term sheet with Cipher has "
        "been reviewed. Key open items: (1) liability cap for Substation "
        "equipment damage, (2) force majeure definition, (3) minimum "
        "consumption commitment. Scheduling a call with Cipher counsel "
        "for next Tuesday.'",
        "agreements",
    ),
    (
        "Amendment No. 3 to the Master Services Agreement between Luminant "
        "Generation Company LLC and Cipher Mining Inc. This amendment "
        "modifies the pricing structure in Exhibit B and extends the term "
        "through December 31, 2024.",
        "agreements",
    ),
    # Category 2: Communications
    (
        "Email from T. Williams (Cipher) to S. Patel (Luminant): 'We are "
        "experiencing intermittent power drops at the data mining center. "
        "The facility logged three outages last week, each lasting 15-30 "
        "minutes. Can we schedule a call to discuss reliability of the "
        "feed from the Odessa Plant?'",
        "communications",
    ),
    (
        "Slack message from D. Kim (Vistra Operations) to internal team: "
        "'Cipher is pushing back on the maintenance window schedule. They "
        "claim each hour of downtime costs them $45K in lost mining "
        "revenue. Need to coordinate with the Odessa Plant team to find "
        "an overnight window.'",
        "communications",
    ),
    (
        "Letter from Cipher Mining Inc. to Luminant Generation Company: "
        "'RE: Notice of Breach. Cipher hereby provides notice that "
        "Luminant has failed to deliver the contracted minimum of 300 MW "
        "to the Substation on 14 occasions during Q3 2022, constituting "
        "a material breach under Section 7.1 of the Agreement.'",
        "communications",
    ),
    (
        "Email from external counsel (Baker & Associates) to Vistra Legal: "
        "'Per our discussion, Cipher's position on the indemnification "
        "clause is unreasonable. Recommend we propose a mutual cap at "
        "2x annual fees. Attaching a comparison of the Cipher and La "
        "Frontera lease indemnification provisions.'",
        "communications",
    ),
    (
        "Meeting minutes — Luminant/Cipher quarterly review, October 12, "
        "2022. Attendees: J. Roberts, S. Patel (Luminant), T. Williams, "
        "R. Novak (Cipher). Topics: Substation upgrade timeline, Q3 "
        "invoice dispute, upcoming ERCOT compliance filing.",
        "communications",
    ),
    (
        "Email chain between Luminant engineering and Cipher technical "
        "team regarding transformer specifications for the Substation "
        "upgrade. Cipher requests 500 MVA capacity; Luminant proposes "
        "phased installation starting with 350 MVA.",
        "communications",
    ),
    # Category 3: Invoices & Financial Records
    (
        "INVOICE #LUM-2022-0847 from Luminant Generation Company to "
        "Cipher Mining Inc. Period: September 2022. Energy delivered: "
        "218,400 MWh. Amount due: $4,368,000.00. Payment terms: Net 30. "
        "Note: Credit of $127,500 applied for outage hours per Section 5.3.",
        "invoices",
    ),
    (
        "Wire transfer confirmation from Cipher Mining Inc. to Luminant "
        "Generation Company. Date: October 28, 2022. Amount: $4,240,500.00. "
        "Reference: INV-LUM-2022-0847. Bank: JPMorgan Chase.",
        "invoices",
    ),
    (
        "Credit memo from Luminant to Cipher: 'Per the dispute resolution "
        "under Section 9.2, Luminant agrees to credit Cipher $312,000 "
        "for the August 2022 billing period to account for documented "
        "power delivery shortfalls at the Substation.'",
        "invoices",
    ),
    (
        "Accounts receivable aging report — Luminant internal. Shows "
        "Cipher Mining Inc. with outstanding balance of $8,736,000 "
        "across invoices #LUM-2022-0901 through #LUM-2022-0904. "
        "Notes: 'Cipher disputes Q4 invoices; escalated to Legal.'",
        "invoices",
    ),
    # Category 4: ERCOT Approval
    (
        "ERCOT Interconnection Agreement approval letter dated April 22, "
        "2019. 'ERCOT hereby approves the energization of the Substation "
        "located at the Odessa generating facility for the purpose of "
        "serving the Cipher Mining load interconnection. Standard "
        "reliability requirements per ERCOT Protocols Section 6 apply.'",
        "ercot",
    ),
    (
        "ERCOT Planning Committee meeting notes, May 2019: 'Item 7 — "
        "Approval of energization request for the Luminant Odessa "
        "Substation to serve Cipher Mining data center load. Voted "
        "unanimously to approve. Compliance monitoring to begin upon "
        "commercial operation date.'",
        "ercot",
    ),
    (
        "Email from ERCOT Compliance to Luminant: 'Your annual "
        "Substation inspection report for the Cipher interconnection "
        "point has been received and is under review. Preliminary "
        "assessment indicates compliance with all applicable protocols.'",
        "ercot",
    ),
    # Category 5: Odessa Plant
    (
        "Odessa Plant daily operations log, July 15, 2022: 'Unit 2 "
        "tripped at 14:32 due to condenser tube leak. Cipher mining "
        "load shed to 180 MW from 300 MW. Estimated repair time: 8 "
        "hours. Notified Cipher operations team at 14:45.'",
        "odessa_plant",
    ),
    (
        "Engineering report: 'Assessment of Odessa Generating Station "
        "Capacity for Cipher Mining Operations.' Prepared by Burns & "
        "McDonnell. Evaluates the Plant's ability to sustain 300 MW "
        "continuous delivery to Cipher's data mining center through "
        "the dedicated Substation connection.",
        "odessa_plant",
    ),
    (
        "Maintenance schedule for Odessa Plant Q4 2022. Notes: 'Unit 1 "
        "overhaul Oct 10-24. During this period, Cipher load must be "
        "served from Unit 2 only. Maximum available capacity: 200 MW. "
        "Cipher has been notified per contractual requirements.'",
        "odessa_plant",
    ),
    # Category 6: Project References (Bitfury / Cipher / Cypher)
    (
        "Internal Vistra email: 'The Bitfury project is on track for "
        "Phase 2 expansion. Engineering has confirmed the Substation can "
        "handle the additional 150 MW. Need sign-off from the Odessa "
        "Plant manager before we proceed with the ERCOT filing.'",
        "project_ref",
    ),
    (
        "Luminant project tracker spreadsheet excerpt: 'Project: Bitfury "
        "/ Cipher Data Center. Status: Active. Location: Odessa, TX. "
        "Contracted capacity: 300 MW. Point of contact: T. Williams "
        "(Cipher), J. Roberts (Luminant).'",
        "project_ref",
    ),
    (
        "Email from Luminant engineer to plant operations: 'The Cypher "
        "team is sending technicians next week to inspect the metering "
        "equipment at the Substation. Please coordinate access with "
        "site security.' [Note: Cipher misspelled as Cypher]",
        "project_ref",
    ),
    (
        "Board presentation slide: 'Bitfury Partnership Update — The "
        "Cipher mining operation at Odessa is generating $4.2M/month in "
        "revenue for Luminant. Recommend extending the agreement through "
        "2026 with revised pricing per the attached term sheet.'",
        "project_ref",
    ),
]

NOT_RELEVANT_DOCS: list[tuple[str, str]] = [
    # ── Hard not-relevant: mention key terms but genuinely irrelevant ──
    # These test whether the classifier avoids false positives from
    # keyword matching alone.
    (
        "Cipher Security Solutions LLC (unrelated company) proposal to "
        "Luminant for cybersecurity penetration testing of corporate IT "
        "infrastructure. Scope: network vulnerability assessment, phishing "
        "simulation, and incident response tabletop exercise. $185K fixed "
        "fee. Different 'Cipher' entity — not Cipher Mining.",
        "hard_nr_cipher_name",
    ),
    (
        "Luminant internal email: 'The Odessa Plant will undergo its "
        "regular 10-year turbine inspection next quarter. This is a "
        "routine maintenance item that affects Unit 3 only. No customer "
        "impact expected as Unit 3 has been mothballed since 2018.' "
        "Mentions Odessa Plant but in a context unrelated to Cipher.",
        "hard_nr_odessa",
    ),
    (
        "ERCOT approval letter for a DIFFERENT substation: 'ERCOT hereby "
        "approves energization of the Luminant Midlothian Substation "
        "upgrade for expanded industrial load service.' Same approval "
        "type, same company, different facility entirely.",
        "hard_nr_ercot",
    ),
    (
        "Invoice from Bitfury Group Limited (the parent company, not "
        "Cipher Mining) to a European data center operator for mining "
        "hardware: 50x BlockBox AC units. Shipped to Iceland facility. "
        "Mentions 'Bitfury' but involves a different entity and geography.",
        "hard_nr_bitfury",
    ),
    (
        "News article: 'Vistra Corp. CEO discusses the company's "
        "cryptocurrency mining partnerships in an earnings call. \"We have "
        "several arrangements across our fleet to monetize excess "
        "generation capacity with mining operators.\" ' General strategic "
        "statement — no Cipher-specific details or Odessa reference.",
        "hard_nr_crypto_vistra",
    ),
    (
        "Luminant contract with Cipher Mining Inc. for the SALE of surplus "
        "office furniture from the Dallas headquarters. 200 desks and "
        "chairs at $50 each. While between the correct parties, this is "
        "an unrelated transaction with no connection to the Odessa Plant, "
        "Substation, or power agreements.",
        "hard_nr_wrong_contract",
    ),
    (
        "Internal Vistra strategy presentation: 'Cryptocurrency Mining "
        "Opportunities — Market Overview.' Discusses industry trends, "
        "Bitcoin hash rates, energy costs, and potential partnerships. "
        "Mentions Odessa as one of 15 potential sites. No reference to "
        "Cipher or any existing agreement. Pure market research.",
        "hard_nr_strategy",
    ),
    (
        "La Frontera Holdings LLC annual report showing all commercial "
        "tenants across their 40-property portfolio. Cipher Mining appears "
        "in a table as one line item among 40 tenants with no additional "
        "detail beyond address and lease start date. Routine landlord "
        "document, not communications between parties.",
        "hard_nr_la_frontera",
    ),
    (
        "Email from Luminant engineer to colleague: 'I heard the Cypher "
        "project might get cancelled. Anyway, completely different topic — "
        "can you review the attached specs for the Martin Lake scrubber "
        "upgrade?' The Cipher reference is hearsay in passing; the "
        "substantive content is about an unrelated plant upgrade.",
        "hard_nr_passing_mention",
    ),
    (
        "ERCOT compliance audit report for the Odessa Substation covering "
        "general electrical safety standards, vegetation management, and "
        "equipment inspection results. Standard regulatory filing that "
        "mentions the Substation but contains no information about Cipher, "
        "the interconnection agreement, or power delivery to the mining "
        "facility.",
        "hard_nr_compliance",
    ),

    # ── Standard not-relevant: no key terms ───────────────────────────
    # Tangential energy docs — no connection to Cipher/Luminant/Odessa
    (
        "ERCOT Seasonal Assessment of Resource Adequacy, Winter 2022-2023. "
        "Total installed generation capacity in ERCOT: 112 GW. Projected "
        "peak demand: 57 GW. Reserve margin: 22.8%. No reliability "
        "concerns identified for the upcoming season.",
        "generic_ercot",
    ),
    (
        "Press release: 'Vistra Corp. announces completion of the 400 MW "
        "Moss Landing battery storage facility in California. The project "
        "is the largest lithium-ion battery storage system in the world.'",
        "generic_vistra",
    ),
    (
        "Email from Vistra HR: 'Reminder — Open enrollment for 2023 "
        "benefits begins November 1. Please review your health plan "
        "options in Workday by November 15.'",
        "hr_admin",
    ),
    (
        "Luminant Generation Company safety incident report: 'Slip and "
        "fall at the Martin Lake power plant cafeteria on September 3. "
        "Employee treated for minor knee abrasion. Root cause: wet floor "
        "from recent mopping. Corrective action: additional signage.'",
        "safety_unrelated",
    ),
    (
        "ERCOT market notice: 'Real-time settlement point prices for "
        "the West zone exceeded $5,000/MWh between 14:00 and 16:00 on "
        "August 12, 2022 due to high demand and limited wind generation.'",
        "generic_ercot",
    ),
    (
        "Vistra IT department memo: 'Scheduled maintenance on the SAP "
        "ERP system this Saturday from 02:00 to 06:00 CT. All financial "
        "reporting modules will be unavailable during this window.'",
        "it_admin",
    ),
    (
        "Contract between Luminant and GE Energy for the supply of gas "
        "turbine replacement parts for the Midlothian Power Station. "
        "Total contract value: $12.4M. Delivery timeline: Q1 2023.",
        "unrelated_contract",
    ),
    (
        "Annual report excerpt — Vistra Corp. FY2022: 'Our retail "
        "electricity segment, TXU Energy, served approximately 2.8 "
        "million residential and commercial customers across Texas.'",
        "generic_vistra",
    ),
    (
        "Email from Luminant environmental compliance: 'The EPA has "
        "issued updated emission standards for coal-fired plants. Our "
        "Big Brown and Martin Lake facilities will need SCR upgrades "
        "by 2025. Estimated cost: $180M.'",
        "environmental",
    ),
    (
        "ERCOT Protocols Section 8 — Load Resource participation "
        "requirements. This document describes the technical "
        "requirements for loads participating in ERCOT ancillary "
        "services markets. General protocol document, not specific "
        "to any entity.",
        "generic_ercot",
    ),
    (
        "Board minutes — Vistra Corp. February 2022. Discussion of "
        "Q4 2021 earnings, dividend policy, and 2022 capital "
        "expenditure budget. No mention of Cipher, Odessa Plant, or "
        "the Substation project.",
        "generic_vistra",
    ),
    (
        "Luminant procurement RFP for janitorial services at the "
        "Dallas corporate headquarters. Scope: daily cleaning of "
        "office floors 8-14, including restrooms and break rooms. "
        "Contract term: 2 years.",
        "procurement_unrelated",
    ),
    (
        "Email from Vistra Legal to external counsel: 'Please review "
        "the attached lease renewal for our Collin County office "
        "space. The landlord is proposing a 12% rate increase which "
        "seems above market.'",
        "lease_unrelated",
    ),
    (
        "Technical report: 'Wind Resource Assessment for Proposed "
        "Upton County Wind Farm.' Prepared by DNV for Luminant "
        "Renewables. Evaluates annual energy production potential of "
        "a 250 MW wind project in West Texas.",
        "renewables_unrelated",
    ),
    (
        "Vistra investor relations FAQ: 'Q: What is Vistra's position "
        "on cryptocurrency mining? A: Vistra explores various "
        "opportunities to optimize our generation fleet utilization.' "
        "Generic public statement with no specifics about Cipher.",
        "generic_crypto",
    ),
    (
        "Luminant Generation Company quarterly financial summary, Q2 "
        "2022. Revenue breakdown by plant: Martin Lake ($84M), "
        "Comanche Peak ($192M), Midlothian ($47M). No line item for "
        "Odessa or Cipher-related revenue.",
        "financial_unrelated",
    ),
    (
        "ERCOT notice of proposed rulemaking: 'Amendments to Nodal "
        "Protocol Section 3.2 regarding congestion revenue rights "
        "allocation methodology. Public comment period closes "
        "March 31, 2023.'",
        "generic_ercot",
    ),
    (
        "Email between Luminant plant managers discussing shift "
        "scheduling for the Comanche Peak nuclear station during the "
        "upcoming holiday period. No mention of Odessa, Cipher, or "
        "any related entities.",
        "hr_admin",
    ),
    (
        "Insurance renewal proposal from AIG for Vistra Corp. "
        "property and casualty coverage. Covers all Vistra generation "
        "assets. Premium: $24.5M annually. General corporate "
        "insurance, not specific to any project.",
        "insurance_unrelated",
    ),
    (
        "Luminant safety training schedule for 2023. Mandatory OSHA "
        "refresher courses for all plant personnel. Includes dates "
        "for Martin Lake, Comanche Peak, and Midlothian facilities. "
        "Odessa Plant not mentioned.",
        "safety_unrelated",
    ),
    # Tricky near-misses — mention some keywords but are not relevant
    (
        "Email from Luminant to ERCOT: 'Submitting the annual "
        "generation capacity report for our fleet. Total registered "
        "capacity: 18,200 MW across 12 facilities.' Lists all plants "
        "including Odessa but in a routine fleet-wide filing with no "
        "Cipher or Substation specifics.",
        "near_miss_ercot",
    ),
    (
        "News article: 'Texas cryptocurrency miners face scrutiny as "
        "ERCOT warns of summer grid strain. Several large-scale mining "
        "operations across the state have agreed to curtail usage "
        "during peak periods.' General industry article mentioning "
        "no specific companies.",
        "near_miss_crypto",
    ),
    (
        "Vistra Corp. press release: 'Vistra announces strategic "
        "review of its generation portfolio, including potential "
        "retirement of aging natural gas units.' The Odessa Plant is "
        "mentioned in a list of 30 facilities under review, but no "
        "Cipher or Substation details are discussed.",
        "near_miss_odessa",
    ),
    (
        "Internal Luminant email discussing general power purchase "
        "agreement templates and standard contract terms. No reference "
        "to Cipher, Vistra, or the Odessa Plant. Purely template "
        "discussion among the legal team.",
        "near_miss_agreement",
    ),
    (
        "ERCOT market report showing real-time prices at the Odessa "
        "hub node. Standard hourly price data with no reference to "
        "the Substation, Cipher, or any specific interconnection. "
        "Routine market data published for all ERCOT nodes.",
        "near_miss_odessa",
    ),
]


def _build_split(
    relevant: list[tuple[str, str]],
    not_relevant: list[tuple[str, str]],
    out_dir: pathlib.Path,
) -> None:
    """Write input.jsonl and gold.jsonl for one split."""
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [(doc, 1) for doc, _ in relevant] + [
        (doc, 0) for doc, _ in not_relevant
    ]
    random.shuffle(rows)

    input_lines: list[str] = []
    gold_lines: list[str] = []
    for idx, (doc, label) in enumerate(rows):
        input_lines.append(
            json.dumps({"query": QUERY, "document": doc})
        )
        gold_lines.append(json.dumps({"idx": idx, "label": label}))

    (out_dir / "input.jsonl").write_text("\n".join(input_lines))
    (out_dir / "gold.jsonl").write_text("\n".join(gold_lines))
    print(
        f"  {out_dir.name}: {len(relevant)} relevant + "
        f"{len(not_relevant)} not relevant = {len(rows)} total"
    )


def main() -> None:
    print("Generating synthetic legal dataset...\n")

    # Shuffle all docs then split into dev / canary / test
    all_relevant = list(RELEVANT_DOCS)
    all_not_relevant = list(NOT_RELEVANT_DOCS)
    random.shuffle(all_relevant)
    random.shuffle(all_not_relevant)

    # Split ratios: dev 60%, canary 15%, test 25%
    def _split(
        items: list, ratios: tuple[float, float, float],
    ) -> tuple[list, list, list]:
        n = len(items)
        i = int(n * ratios[0])
        j = int(n * (ratios[0] + ratios[1]))
        return items[:i], items[i:j], items[j:]

    rel_dev, rel_canary, rel_test = _split(
        all_relevant, (0.60, 0.15, 0.25),
    )
    nr_dev, nr_canary, nr_test = _split(
        all_not_relevant, (0.60, 0.15, 0.25),
    )

    _build_split(rel_dev, nr_dev, ROOT / "dev")
    _build_split(rel_canary, nr_canary, ROOT / "canary")
    _build_split(rel_test, nr_test, ROOT / "test")

    total = len(all_relevant) + len(all_not_relevant)
    print(f"\nDone — {total} documents across 3 splits in {ROOT}/")


if __name__ == "__main__":
    main()
