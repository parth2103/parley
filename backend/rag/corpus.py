"""Fictional insurance policy documents with stable chunk IDs and numerical traps.

All policies and data are entirely fictional. No real insurer data or PII.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyChunk:
    chunk_id: str
    doc_id: str
    title: str
    content: str


CORPUS: list[PolicyChunk] = [
    # Document 1: Auto Policy (DOC-AUTO-401)
    PolicyChunk(
        chunk_id="CHUNK-AUTO-01",
        doc_id="DOC-AUTO-401",
        title="Comprehensive Auto Coverage & Standard Deductible",
        content=(
            "Policy POL-4401: Comprehensive auto physical damage coverage carries a standard "
            "deductible of $500 per incident for all covered non-collision perils including fire, "
            "vandalism, hail, and animal strikes. This $500 deductible applies before insurer "
            "reimbursement begins."
        ),
    ),
    PolicyChunk(
        chunk_id="CHUNK-AUTO-02",
        doc_id="DOC-AUTO-401",
        title="Safety Glass & Windshield Repair Sub-limit and Deductible",
        content=(
            "Policy POL-4401 Endorsement G: Windshield and safety glass replacement carries a "
            "reduced comprehensive deductible of $100. Chip repairs smaller than a dollar bill "
            "carry a $0 deductible and are fully covered without penalty."
        ),
    ),
    PolicyChunk(
        chunk_id="CHUNK-AUTO-03",
        doc_id="DOC-AUTO-401",
        title="Emergency Roadside Assistance and Towing Limits",
        content=(
            "Policy POL-4401 Endorsement R: Roadside towing assistance covers up to $150 per occurrence "
            "with a flat $50 deductible. Towing is limited to a maximum distance of 25 miles from disablement."
        ),
    ),
    PolicyChunk(
        chunk_id="CHUNK-AUTO-04",
        doc_id="DOC-AUTO-401",
        title="Rental Car Reimbursement Coverage",
        content=(
            "Policy POL-4401 Endorsement T: Rental car reimbursement provides up to $45 per day for a "
            "maximum of 30 days ($1,350 total per incident) while the covered automobile undergoes "
            "repairs for a covered comprehensive or collision loss."
        ),
    ),

    # Document 2: Homeowners Policy (DOC-HOME-302)
    PolicyChunk(
        chunk_id="CHUNK-HOME-01",
        doc_id="DOC-HOME-302",
        title="Sudden and Accidental Pipe Freeze & Plumbing Discharge",
        content=(
            "Policy HOM-302 Section A: Sudden and accidental discharge of water from plumbing systems, "
            "including frozen and burst indoor supply pipes, is covered up to a limit of $25,000 per occurrence. "
            "Premises must have maintained active heating at minimum 55 degrees Fahrenheit."
        ),
    ),
    PolicyChunk(
        chunk_id="CHUNK-HOME-02",
        doc_id="DOC-HOME-302",
        title="Sewer, Sump Pump, and Drain Water Backup Endorsement",
        content=(
            "Policy HOM-302 Endorsement WB: Water backup of sewers and sump pump overflow is an optional rider "
            "covering up to $5,000 in direct physical loss with a dedicated $1,000 water backup deductible."
        ),
    ),
    PolicyChunk(
        chunk_id="CHUNK-HOME-03",
        doc_id="DOC-HOME-302",
        title="Surface Flood, Rising Water, and Groundwater Exclusion",
        content=(
            "Policy HOM-302 Exclusion F: Damage resulting from surface floodwaters, storm surge, rising tides, "
            "or underground seepage carries a $0 coverage limit and is strictly excluded from standard "
            "homeowners coverage. Flood risks require a separate national flood insurance policy."
        ),
    ),
    PolicyChunk(
        chunk_id="CHUNK-HOME-04",
        doc_id="DOC-HOME-302",
        title="Earth Movement and Earthquake Exclusion",
        content=(
            "Policy HOM-302 Exclusion E: Direct or indirect damage caused by earthquake, tremor, landslide, "
            "or sinkholes is excluded with a $0 benefit under base coverage. Earthquake coverage requires a "
            "specialized earthquake endorsement."
        ),
    ),

    # Document 3: Personal Property Rider (DOC-PROP-205)
    PolicyChunk(
        chunk_id="CHUNK-PROP-01",
        doc_id="DOC-PROP-205",
        title="Off-Premises Theft of Personal Property in Vehicles",
        content=(
            "Rider PRP-205: Personal property stolen from inside an unattended locked vehicle is covered up "
            "to $1,500 total value, subject to a $250 property deductible. The auto policy itself does not cover "
            "stolen personal effects; claim must be filed under property coverage."
        ),
    ),
    PolicyChunk(
        chunk_id="CHUNK-PROP-02",
        doc_id="DOC-PROP-205",
        title="High-Value Portable Electronics and Computing Endorsement",
        content=(
            "Rider PRP-205 Endorsement EL: Portable electronics including laptops and mobile workstations are "
            "insured up to $3,000 aggregate with a $100 deductible when registered on the scheduled equipment list."
        ),
    ),
    PolicyChunk(
        chunk_id="CHUNK-PROP-03",
        doc_id="DOC-PROP-205",
        title="Jewelry, Watches, and Precious Metals Sub-limit",
        content=(
            "Rider PRP-205 Section J: Theft of unscheduled jewelry, watches, precious gems, and furs is limited "
            "to a strict maximum sub-limit of $1,000 aggregate per loss, subject to the $250 standard rider deductible."
        ),
    ),

    # Document 4: Claims Filing & Notification Guidelines (DOC-CLAIM-101)
    PolicyChunk(
        chunk_id="CHUNK-CLAIM-01",
        doc_id="DOC-CLAIM-101",
        title="General Property Loss Notice and Submission Deadlines",
        content=(
            "Claims Guide CLM-101: Insured parties must provide formal written or telephonic notice of property "
            "damage within 30 days of the triggering event. Failure to file within 30 days may forfeit coverage "
            "if prejudice to insurer inspection occurs."
        ),
    ),
    PolicyChunk(
        chunk_id="CHUNK-CLAIM-02",
        doc_id="DOC-CLAIM-101",
        title="Theft and Vandalism Mandatory Police Report Timing",
        content=(
            "Claims Guide CLM-101 Section T: For all theft, burglary, and hit-and-run claims, an official police report "
            "must be filed within 24 hours of discovery and the report number provided to the claims adjuster."
        ),
    ),
    PolicyChunk(
        chunk_id="CHUNK-CLAIM-03",
        doc_id="DOC-CLAIM-101",
        title="Medical Expense and Personal Injury Protection Deadlines",
        content=(
            "Claims Guide CLM-101 Section M: Claims for medical expenses, bodily injury, and personal injury protection "
            "(PIP) must be filed within 180 days of treatment with itemized medical statements."
        ),
    ),
    PolicyChunk(
        chunk_id="CHUNK-CLAIM-04",
        doc_id="DOC-CLAIM-101",
        title="Total Loss Valuation Criteria for Vehicles",
        content=(
            "Claims Guide CLM-101 Section V: A vehicle is declared a constructive total loss when estimated repair "
            "costs plus salvage value equal or exceed 75 percent of the actual cash value (ACV) immediately prior to loss."
        ),
    ),
]
