"""Synthetic payments-industry knowledge base.

The corpus models the internal knowledge base of a fictional payment processor,
"NovaPay". It has two parts:

* ANCHOR_DOCS - hand-written policy / runbook / reference documents. Each holds
  specific, checkable facts; the golden eval set is written against them.
  NovaPay-specific numbers are fictional. Industry facts (ISO 8583 MTIs,
  PCI DSS requirements, ISO 20022 message names...) are illustrative only.
* filler docs  - templated incident reports, merchant notes and tickets that
  share vocabulary with the anchors and act as realistic retrieval distractors.

Generation is fully deterministic for a given seed.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from payrag.config import CATEGORIES, SEED


@dataclass
class Document:
    doc_id: str
    title: str
    category: str
    doc_type: str
    region: str
    date: str
    text: str
    is_anchor: bool = False


def _a(doc_id, title, category, doc_type, text, region="Global", date="2026-03-01"):
    return Document(doc_id, title, category, doc_type, region, date, " ".join(text.split()), True)


ANCHOR_DOCS: list[Document] = [
    # ---------------------------------------------------------------- fraud
    _a("FR-001", "Velocity Rules for Card-Not-Present Fraud", "fraud_detection", "policy", """
        NovaPay's risk engine applies velocity rules to card-not-present (CNP) traffic.
        A card is flagged when more than 5 card-not-present authorization attempts occur within 10 minutes.
        A card is also flagged when it is used at more than 3 distinct merchants within one hour.
        Flagged transactions are routed to step-up authentication with 3-D Secure 2 (3DS2) instead of being declined outright.
        The Fraud Strategy team reviews velocity thresholds monthly."""),
    _a("FR-002", "Account Takeover Detection Playbook", "fraud_detection", "runbook", """
        Account takeover (ATO) is detected by combining login and profile-change signals.
        The strongest ATO pattern is a new device fingerprint, followed by a password reset and a change of payout bank account within 24 hours.
        When the ATO risk score is above 820 NovaPay automatically freezes merchant payouts.
        Scores between 650 and 820 go to the manual review queue, which has a 4-hour SLA.
        Analysts must confirm identity through a verified phone call before unfreezing payouts."""),
    _a("FR-003", "Card Testing Attack Response Runbook", "fraud_detection", "runbook", """
        Card testing, also called card enumeration, is when bots submit many small authorizations, usually under $1, to validate stolen card numbers.
        During a card testing attack, enable CAPTCHA on the hosted checkout and rate-limit authorization requests to 20 per minute per IP address.
        Decline zero-dollar account verification requests from BIN ranges with a decline rate above 30% in the last hour.
        A card testing attack is handled as a SEV-2 incident and the merchant is notified within 1 hour."""),
    _a("FR-004", "Fraud Model Monitoring and Drift", "fraud_detection", "reference", """
        NovaPay scores every transaction with the gradient-boosted fraud model Sentinel v3.
        Sentinel v3 is retrained every 6 weeks on the latest labelled chargeback and confirmed-fraud data.
        Model drift is monitored with the population stability index (PSI); an alert fires when PSI exceeds 0.2 on any top-20 feature.
        The operating target is 85% precision at 70% recall on the weekly holdout set."""),
    # ---------------------------------------------------------- chargebacks
    _a("CB-001", "Chargeback Representment Deadlines", "chargebacks_disputes", "policy", """
        Merchants must submit representment evidence to NovaPay within 20 calendar days of the chargeback notification.
        The card network deadline is 30 days, and NovaPay keeps the remaining 10 days to review and file the case.
        If the merchant misses the 20-day deadline the chargeback is automatically accepted and the funds are not recovered.
        Evidence is uploaded in the Disputes tab of the NovaPay dashboard as PDF or PNG files."""),
    _a("CB-002", "Visa Reason Code 10.4 Evidence Checklist", "chargebacks_disputes", "reference", """
        Visa reason code 10.4 is Other Fraud in a card-absent environment.
        Under Compelling Evidence 3.0 the merchant can provide two prior undisputed transactions on the same card that are between 120 and 365 days old.
        The prior transactions must share at least two data elements with the disputed one, and one of them must be the IP address or the device ID.
        Accepted Compelling Evidence 3.0 shifts liability back to the issuer."""),
    _a("CB-003", "Chargeback Ratio Monitoring", "chargebacks_disputes", "policy", """
        The monthly chargeback ratio is the number of chargebacks received in a month divided by the number of sales transactions in the same month.
        NovaPay sends a warning when a merchant's chargeback ratio reaches 0.65%.
        At 0.9% the merchant is placed on a mandatory remediation plan.
        Merchants in the excessive chargeback program pay a $50 fee per chargeback until the ratio is under 0.65% for three consecutive months."""),
    _a("CB-004", "Friendly Fraud and Pre-Dispute Deflection", "chargebacks_disputes", "guide", """
        Friendly fraud happens when a cardholder disputes a legitimate purchase, for example a forgotten subscription renewal.
        NovaPay deflects disputes before they become chargebacks by using pre-dispute alerts from issuers.
        Pre-dispute alerts under $25 are refunded automatically, which is cheaper than fighting the chargeback.
        The deflection rate target is 40% of all incoming dispute alerts."""),
    # ------------------------------------------------------------------ PCI
    _a("PCI-001", "PCI DSS v4.0 Scope and Validation", "pci_compliance", "policy", """
        NovaPay is a PCI DSS Level 1 service provider.
        Level 1 validation requires an annual Report on Compliance (ROC) signed by a Qualified Security Assessor (QSA).
        External vulnerability scans by an Approved Scanning Vendor (ASV) are run quarterly.
        The future-dated PCI DSS v4.0 requirements became mandatory on 31 March 2025."""),
    _a("PCI-002", "Tokenization and PAN Storage Standard", "pci_compliance", "standard", """
        The primary account number (PAN) is never stored in application databases.
        The PAN is replaced by a format-preserving token issued by the NovaPay token vault.
        The token vault runs in a segmented cardholder data environment and its encryption keys are rotated annually in a hardware security module (HSM).
        When a PAN is displayed it must be masked to at most the first six and last four digits."""),
    _a("PCI-003", "Cardholder Data Environment Access Control", "pci_compliance", "standard", """
        Multi-factor authentication (MFA) is required for all access into the cardholder data environment (CDE), in line with PCI DSS requirement 8.4.2.
        User access rights to the CDE are reviewed every 6 months by system owners.
        Break-glass emergency accounts are sealed, monitored, and their passwords are rotated after every use."""),
    _a("PCI-004", "Audit Logging and Security Monitoring", "pci_compliance", "standard", """
        Audit logs for systems in scope of PCI DSS are retained for at least 12 months.
        The most recent 3 months of audit logs must be immediately available for analysis, per requirement 10.5.1.
        Daily log review is automated in the SIEM, and critical alerts page the security on-call engineer."""),
    # -------------------------------------------------- auth & settlement
    _a("AS-001", "Authorization Messages and ISO 8583", "authorization_settlement", "reference", """
        Card authorizations are exchanged as ISO 8583 messages.
        Message type 0100 is the authorization request and 0110 is the authorization response.
        Message type 0400 is a reversal request, used when an authorized transaction must be cancelled.
        Common response codes are 00 approved, 05 do not honor and 51 insufficient funds."""),
    _a("AS-002", "Clearing and Settlement Timeline", "authorization_settlement", "policy", """
        NovaPay batches captured transactions for clearing at a daily cut-off of 23:00 UTC.
        Clearing files are submitted to the card networks after the cut-off.
        Merchants are paid out on a T+2 business day settlement schedule by default.
        Merchants on the Premium plan receive T+1 settlement."""),
    _a("AS-003", "Authorization Holds and Expiry", "authorization_settlement", "policy", """
        An authorization places a hold on the cardholder's available funds until the transaction is captured.
        NovaPay automatically voids uncaptured authorizations after 7 days to release the hold.
        Hotels and car rental merchants may use incremental authorizations to increase the held amount.
        Partial capture is allowed, and the remaining authorized amount is released."""),
    _a("AS-004", "Decline Handling and Retry Logic", "authorization_settlement", "guide", """
        Declines are either soft declines, which may succeed later, or hard declines, which will not.
        NovaPay retries soft declines such as response code 91 issuer unavailable up to 3 times over 72 hours.
        Hard declines such as 04 pick up card and 14 invalid card number are never retried.
        Excessive retries can trigger card network penalty fees."""),
    # ------------------------------------------------ interchange & pricing
    _a("IP-001", "Interchange++ Pricing Model", "interchange_pricing", "reference", """
        Under Interchange++ (IC++) pricing the merchant pays three components: the interchange fee, the card scheme fee, and the NovaPay markup.
        The standard NovaPay markup on IC++ is 0.25% plus $0.08 per transaction.
        Interchange is paid to the issuing bank and scheme fees are paid to the card network.
        IC++ statements itemize all three components for every transaction."""),
    _a("IP-002", "Regulated Debit Interchange", "interchange_pricing", "reference", """
        Under the Durbin Amendment and Regulation II, debit interchange for issuers with more than $10 billion in assets is capped.
        The regulated cap is 21 cents plus 0.05% of the transaction, plus a 1 cent fraud-prevention adjustment.
        Debit cards from smaller exempt issuers are not subject to the regulated interchange cap.
        NovaPay routes eligible debit transactions to the lowest-cost network."""),
    _a("IP-003", "Level 2 and Level 3 Data for B2B Payments", "interchange_pricing", "guide", """
        Commercial and purchasing cards qualify for lower interchange when enhanced data is sent.
        Level 2 data adds the sales tax amount and a customer code.
        Level 3 data adds line-item details such as product codes, quantities and unit prices.
        NovaPay automatically populates Level 2 and Level 3 fields from invoices for B2B merchants."""),
    _a("IP-004", "Pricing Plans: Flat-Rate, Tiered and IC++", "interchange_pricing", "policy", """
        Small merchants start on the NovaPay Starter flat-rate plan at 2.9% plus $0.30 per card transaction.
        Tiered pricing groups transactions into qualified, mid-qualified and non-qualified rates.
        Merchants are migrated to Interchange++ pricing when monthly card volume exceeds $250,000.
        Pricing changes require 30 days written notice to the merchant."""),
    # --------------------------------------------------------- cross-border
    _a("FX-001", "Dynamic Currency Conversion Policy", "cross_border_fx", "policy", """
        Dynamic currency conversion (DCC) lets a foreign cardholder pay in their home currency at the point of sale.
        The cardholder must actively opt in to DCC, and the exchange rate and markup must be disclosed before payment.
        The NovaPay DCC markup is capped at 3.5% over the wholesale exchange rate.
        DCC must never be the pre-selected default option."""),
    _a("FX-002", "Multi-Currency Settlement", "cross_border_fx", "reference", """
        NovaPay supports multi-currency settlement in 14 currencies.
        The FX rate is locked at capture time and includes a 0.4% conversion spread.
        Merchants can hold balances in each settlement currency to avoid repeated conversion.
        Settlement currency is configured per merchant account."""),
    _a("FX-003", "Cross-Border Fees and Assessments", "cross_border_fx", "reference", """
        A cross-border transaction is one where the card issuer country differs from the merchant's acquiring country.
        Card networks charge a cross-border assessment that NovaPay passes through at cost.
        NovaPay also adds a 1% international card fee on transactions with foreign-issued cards.
        Intra-EEA card transactions are treated as domestic for this fee."""),
    _a("FX-004", "Sanctions Screening for Cross-Border Payouts", "cross_border_fx", "policy", """
        All cross-border payouts are screened against the OFAC SDN list and the EU and UN consolidated sanctions lists.
        A fuzzy name match score of 0.92 or higher places the payout on hold.
        The compliance team resolves sanctions holds within 24 hours.
        Confirmed sanctions matches are blocked and reported to the regulator."""),
    # ------------------------------------------------------- onboarding/KYC
    _a("KYC-001", "Merchant Underwriting Risk Tiers", "merchant_onboarding_kyc", "policy", """
        Merchants are underwritten into low, medium or high risk tiers based on merchant category code (MCC), volume and business model.
        High-risk MCCs such as 5967 direct marketing and 7995 gambling require enhanced due diligence.
        High-risk merchants are subject to a 10% rolling reserve held for 180 days.
        Underwriting decisions for high-risk merchants need sign-off from a senior underwriter."""),
    _a("KYC-002", "Beneficial Ownership Verification", "merchant_onboarding_kyc", "standard", """
        During onboarding NovaPay identifies every beneficial owner who owns 25% or more of the business.
        NovaPay also identifies one control person with significant management responsibility.
        Each beneficial owner is verified with a government-issued ID and a selfie liveness check.
        Verification results are stored for five years after the relationship ends."""),
    _a("KYC-003", "Know Your Business Document Requirements", "merchant_onboarding_kyc", "checklist", """
        Know Your Business (KYB) checks require the business registration certificate and the tax ID or EIN.
        The settlement bank account is verified with micro-deposits or instant account verification.
        Merchants moving from another processor must provide processing statements for the last 3 months.
        Applications missing documents expire after 30 days."""),
    _a("KYC-004", "Ongoing Merchant Monitoring", "merchant_onboarding_kyc", "policy", """
        After onboarding, merchants are monitored for transaction laundering and prohibited products.
        NovaPay crawls each merchant website every 30 days to detect prohibited goods.
        Every new merchant and its principals are checked against the Mastercard MATCH list before approval.
        A MATCH hit requires escalation to the risk committee."""),
    # --------------------------------------------------- real-time payments
    _a("RTP-001", "Instant Payment Rails Overview", "real_time_payments", "reference", """
        In the United States, instant payments run on The Clearing House RTP network, launched in 2017, and the Federal Reserve FedNow Service, launched in July 2023.
        Both rails settle individual payments 24 hours a day, 7 days a week, 365 days a year.
        Instant payments are irrevocable once settled.
        NovaPay caps instant payouts at $100,000 per transaction."""),
    _a("RTP-002", "ISO 20022 Messages for Instant Payments", "real_time_payments", "reference", """
        Instant payment rails use ISO 20022 messages.
        pacs.008 is the customer credit transfer message that moves the funds.
        pacs.002 is the payment status report that confirms acceptance or rejection.
        camt.056 is used to request the cancellation or return of a payment, and pain.013 is the request for payment message."""),
    _a("RTP-003", "Instant Payout Fraud Controls", "real_time_payments", "runbook", """
        Because instant payouts are irrevocable they are a target for authorized push payment (APP) fraud.
        NovaPay runs confirmation of payee name matching before every instant payout.
        The first payout to a newly added beneficiary is held for a 30-minute cooling-off period.
        Name mismatches require the merchant to re-confirm the beneficiary."""),
    _a("RTP-004", "Request for Payment Use Cases", "real_time_payments", "guide", """
        Request for Payment (RfP) lets a biller send a pain.013 request that the payer approves in their banking app.
        Common uses are bill pay, invoice collection and insurance premiums.
        At NovaPay an unanswered request for payment expires after 48 hours by default.
        Approved requests are paid immediately over the instant payment rail."""),
]


# ---------------------------------------------------------------- fillers
_FILLER = {
    "fraud_detection": {
        "issues": ["a spike in card-not-present fraud", "suspicious login activity", "bot traffic at checkout",
                   "a rise in fraud alerts", "stolen card usage", "a surge of 3DS challenges"],
        "causes": ["a compromised merchant integration", "a credential stuffing campaign", "a leaked BIN list",
                   "a new mule account network", "an outdated device fingerprint library"],
        "actions": ["tightened risk rules", "added the merchant to the fraud watchlist", "escalated to the fraud analytics team",
                    "updated the blocklist of device IDs", "enabled stricter step-up authentication"],
    },
    "chargebacks_disputes": {
        "issues": ["a wave of chargebacks", "an increase in dispute alerts", "late representment submissions",
                   "subscription-related disputes", "item-not-received disputes"],
        "causes": ["unclear billing descriptors", "slow shipping", "missing delivery confirmation",
                   "a confusing cancellation flow", "recurring billing without reminders"],
        "actions": ["updated the billing descriptor", "coached the merchant on dispute evidence", "filed representment for open cases",
                    "enabled dispute alert notifications", "reviewed the refund policy"],
    },
    "pci_compliance": {
        "issues": ["a PCI audit finding", "an unencrypted log file", "an expired firewall rule review",
                   "an overdue vulnerability scan", "a QSA follow-up request"],
        "causes": ["a misconfigured logging library", "a missed quarterly review", "shadow IT in the CDE",
                   "an outdated network diagram", "a vendor without an attestation"],
        "actions": ["remediated the finding before the deadline", "updated the compliance evidence repository",
                    "rescanned the external perimeter", "rotated the affected credentials", "documented a compensating control"],
    },
    "authorization_settlement": {
        "issues": ["elevated authorization decline rates", "a delayed settlement batch", "duplicate captures",
                   "issuer timeouts", "a reconciliation mismatch in payouts"],
        "causes": ["an issuer processor outage", "a clearing file rejected by the network", "a retry loop in the merchant integration",
                   "a timezone bug in the batch scheduler", "a stale authorization reversal"],
        "actions": ["reprocessed the settlement batch", "reversed duplicate authorizations", "added idempotency keys",
                    "opened a ticket with the network", "reconciled the payout ledger"],
    },
    "interchange_pricing": {
        "issues": ["higher than expected processing fees", "a pricing dispute with a merchant", "downgraded interchange rates",
                   "a request for custom pricing", "questions about the monthly statement"],
        "causes": ["missing enhanced data fields", "transactions settling after the qualification window", "a mis-assigned MCC",
                   "a legacy tiered pricing contract", "unexpected scheme fee changes"],
        "actions": ["explained the statement line items", "migrated the merchant to a new pricing plan", "fixed the interchange qualification data",
                    "issued a fee credit", "scheduled a pricing review"],
    },
    "cross_border_fx": {
        "issues": ["FX rate complaints from cardholders", "a cross-border payout delay", "currency conversion errors",
                   "high international card fees", "a held international payout"],
        "causes": ["a rate feed outage", "a sanctions screening false positive", "wrong settlement currency configuration",
                   "foreign-issued cards on a domestic plan", "an intermediary bank delay"],
        "actions": ["corrected the settlement currency", "released the payout after review", "refunded the FX difference",
                    "explained the conversion spread", "switched the rate provider"],
    },
    "merchant_onboarding_kyc": {
        "issues": ["a stalled merchant application", "a failed identity verification", "a high-risk merchant review",
                   "a prohibited product on a merchant site", "missing beneficial owner details"],
        "causes": ["blurry ID documents", "a mismatch between the registered and trading name", "an unlisted business model",
                   "an expired business license", "incomplete ownership information"],
        "actions": ["requested additional documents", "approved the merchant with a reserve", "declined the application",
                    "escalated to the underwriting team", "scheduled enhanced due diligence"],
    },
    "real_time_payments": {
        "issues": ["failed instant payouts", "a payee name mismatch", "delayed request for payment responses",
                   "a suspected push payment scam", "instant payment rail downtime"],
        "causes": ["an incorrect routing number", "a beneficiary bank not on the instant rail", "a message format error",
                   "a social engineering scam", "planned network maintenance"],
        "actions": ["fell back to ACH", "contacted the beneficiary bank", "blocked the beneficiary",
                    "corrected the message mapping", "notified merchants of the downtime"],
    },
}
_MERCHANTS = ["Acme Outdoor", "BrightCart", "CloudMeals", "Delta Ticketing", "EverGreen Pharmacy", "FlexGym",
              "GlobeTravel", "HomeHub", "Indigo Games", "JetFuel Coffee", "Kappa Electronics", "Luma Fashion"]
_REGIONS = ["North America", "EMEA", "APAC", "LATAM"]
_TYPES = ["incident report", "merchant note", "support ticket"]


def _filler_docs(rng: random.Random, per_category: int) -> list[Document]:
    docs: list[Document] = []
    for cat in CATEGORIES:
        vocab = _FILLER[cat]
        for i in range(per_category):
            merchant = rng.choice(_MERCHANTS)
            region = rng.choice(_REGIONS)
            issue, cause, action = rng.choice(vocab["issues"]), rng.choice(vocab["causes"]), rng.choice(vocab["actions"])
            doc_type = rng.choice(_TYPES)
            date = f"2026-{rng.randint(1, 10):02d}-{rng.randint(1, 28):02d}"
            text = (
                f"{doc_type.capitalize()} for {merchant} ({region}). "
                f"On {date} the team observed {issue}. "
                f"Investigation found the root cause was {cause}. "
                f"The operations team {action}. "
                f"Follow-up review is scheduled in {rng.choice([7, 14, 30])} days."
            )
            docs.append(Document(
                doc_id=f"{cat.split('_')[0][:3].upper()}-N{i + 1:03d}",
                title=f"{issue.capitalize()} at {merchant}",
                category=cat, doc_type=doc_type, region=region, date=date, text=text,
            ))
    return docs


def generate_corpus(seed: int = SEED, filler_per_category: int = 10) -> list[Document]:
    rng = random.Random(seed)
    return list(ANCHOR_DOCS) + _filler_docs(rng, filler_per_category)


def write_corpus(docs: list[Document], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for d in docs:
            f.write(json.dumps(asdict(d)) + "\n")


def read_corpus(path: Path) -> list[Document]:
    with path.open() as f:
        return [Document(**json.loads(line)) for line in f if line.strip()]
