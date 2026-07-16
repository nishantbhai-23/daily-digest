#!/usr/bin/env python3
"""
Synthetic Data Generator for Daily Digest
==========================================
Generates four data sources for Avery Chen, co-founder & CEO of Tessera (a
12-person supply-chain visibility SaaS company), matching the operator
profile in data/persona.md:
  1. ~500 emails (.eml) over 30 days
  2. A calendar (.ics) with meetings, blocks, declines
  3. 10 markdown notes (meeting notes, drafts, todos)
  4. 5 tasks (JSON)

The content is deliberately built to exercise the specific blind spots the
persona calls out ("What I might miss without help"): a quiet investor
thread, a quietly-souring customer, an unlogged promise, a stalled hiring
loop, and a calendar collision with a family commitment.

Usage:
    python generate_data.py [--seed 42] [--output-dir data]

Zero external dependencies — stdlib only.
"""

import argparse
import glob
import json
import os
import random
import textwrap
import uuid
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from email.utils import format_datetime, make_msgid

# ─── Constants ────────────────────────────────────────────────────────────────

SEED = 42
OUTPUT_DIR = "data"

START_DATE = datetime(2026, 6, 16, tzinfo=timezone(timedelta(hours=-7)))  # PDT
END_DATE = datetime(2026, 7, 15, 23, 59, 59, tzinfo=timezone(timedelta(hours=-7)))
NUM_DAYS = 30

COMPANY = "Tessera"
DOMAIN = "tessera.io"
PERSONAL_DOMAIN = "gmail.com"

PROTAGONIST = {
    "name": "Avery Chen",
    "email": f"avery@{DOMAIN}",
    "title": "Co-founder & CEO",
    "team": "Leadership",
}

# ─── People ───────────────────────────────────────────────────────────────────
# 11 teammates + Avery = 12-person company, matching the persona.

TEAMMATES = [
    {"name": "Priya Iyer", "email": f"priya@{DOMAIN}", "role": "Co-founder & CTO", "relation": "cofounder"},
    {"name": "Jordan Liu", "email": f"jordan@{DOMAIN}", "role": "Head of Engineering", "relation": "eng-lead"},
    {"name": "Tomás Reyes", "email": f"tomas@{DOMAIN}", "role": "Head of GTM", "relation": "gtm-lead"},
    {"name": "Maya Fischer", "email": f"maya@{DOMAIN}", "role": "Senior Engineer", "relation": "report-eng"},
    {"name": "Devon Brooks", "email": f"devon@{DOMAIN}", "role": "Engineer", "relation": "report-eng"},
    {"name": "Kenji Watanabe", "email": f"kenji@{DOMAIN}", "role": "Data Engineer", "relation": "report-eng"},
    {"name": "Zoe Alvarez", "email": f"zoe@{DOMAIN}", "role": "Customer Success Manager", "relation": "report-gtm"},
    {"name": "Ravi Chandra", "email": f"ravi@{DOMAIN}", "role": "Account Executive", "relation": "report-gtm"},
    {"name": "Nadia Hassan", "email": f"nadia@{DOMAIN}", "role": "Product Designer", "relation": "report-product"},
    {"name": "Owen Fitzgerald", "email": f"owen@{DOMAIN}", "role": "Head of Ops & Finance", "relation": "ops-lead"},
    {"name": "Grace Lin", "email": f"grace@{DOMAIN}", "role": "People Ops (fractional)", "relation": "people-ops"},
]

EXTERNAL_CONTACTS = [
    {"name": "Marcus Webb", "email": "marcus.webb@inflectionpointvc.com", "company": "Inflection Point Ventures", "relation": "investor-lead"},
    {"name": "Diane Okafor", "email": "diane@bramblewoodvc.com", "company": "Bramblewood Capital", "relation": "board"},
    {"name": "Ben Schaffer", "email": "bschaffer@wsgr.com", "company": "Wilson Sonsini", "relation": "lawyer"},
    {"name": "Carla Whitfield", "email": "carla.whitfield@halberdmfg.com", "company": "Halberd Manufacturing", "relation": "customer"},
    {"name": "Derek Osei", "email": "derek.osei@northstarfoods.com", "company": "Northstar Foods", "relation": "customer"},
    {"name": "Lindsey Cho", "email": "lindsey.cho@veritascomponents.com", "company": "Veritas Components", "relation": "customer"},
    {"name": "Trent Bailey", "email": "trent@apextechrecruiting.com", "company": "Apex Technical Recruiting", "relation": "recruiter"},
    {"name": "Michelle Ford", "email": "michelle@bridgeworkstalent.com", "company": "Bridgeworks Talent", "relation": "recruiter"},
]

PERSONAL_CONTACTS = [
    {"name": "Sam Park", "email": f"sam.park@{PERSONAL_DOMAIN}", "relation": "partner"},
]

NOTIFICATION_SENDERS = [
    {"name": "Ashby", "email": "notifications@ashbyhq.com"},
    {"name": "Ramp", "email": "no-reply@ramp.com"},
    {"name": "Ironclad", "email": "notify@ironcladapp.com"},
    {"name": "Google Calendar", "email": "calendar-notification@google.com"},
    {"name": "Notion", "email": "team@makenotion.com"},
    {"name": "Linear", "email": "notifications@linear.app"},
]

NEWSLETTER_SENDERS = [
    {"name": "Stratechery", "email": "ben@stratechery.com"},
    {"name": "Lenny's Newsletter", "email": "news@lennysnewsletter.com"},
    {"name": "SaaStr", "email": "news@saastr.com"},
    {"name": "Ramp", "email": "growth@ramp.com"},
    {"name": "Notion", "email": "updates@makenotion.com"},
]

# ─── Projects & Topics ───────────────────────────────────────────────────────

PROJECTS = [
    "Series A",                      # fundraising process
    "Halberd Integration",           # reference customer's ERP integration
    "Inventory Forecasting Model",   # core product feature
    "SOC 2 Audit",                   # compliance, matters for enterprise + diligence
    "Customer Onboarding v2",        # reducing time-to-value
]

# ─── Email Templates ─────────────────────────────────────────────────────────

def _email_templates(rng):
    """Return a list of (category, weight, subject_fn, body_fn) tuples."""

    def _pick_project():
        return rng.choice(PROJECTS)

    templates = [
        # ── Investor / fundraising (ambient — Marcus Webb is handled via a
        #    dedicated planted thread so his silence is deliberate, not random) ──
        (
            "investor_fundraising", 8,
            lambda: rng.choice([
                "Re: Series A — diligence checklist",
                "Cap table question",
                "Board deck — Q2 numbers",
                "Re: reference call intro",
                "SAFE conversion — quick question",
            ]),
            lambda subj: rng.choice([
                f"Hi Avery,\n\n"
                f"Following up on the diligence checklist — could you send over "
                f"{rng.choice(['the updated cap table', 'June financials', 'the customer reference list', 'the SAFE conversion schedule'])} "
                f"when you get a chance? Trying to keep the process moving on our end.\n\n"
                f"No rush, just don't want it to slip.\n\n"
                f"Best,\n",

                f"Avery,\n\n"
                f"Quick one — can you intro me to {rng.choice(['Carla at Halberd', 'Derek at Northstar', 'Lindsey at Veritas'])} "
                f"for a reference call? Trying to close out diligence on the customer side this week.\n\n"
                f"Thanks,\n",
            ]),
        ),

        # ── Board (Diane specifically — sparse, but the "I owe her an update" tension) ──
        (
            "board_update", 3,
            lambda: rng.choice([
                "Following up — Q2 board update?",
                "Great catching up last week",
                "Board meeting — anything you need from me?",
            ]),
            lambda subj: rng.choice([
                f"Hi Avery,\n\n"
                f"No urgency at all, but I realized I haven't seen a written update since "
                f"{rng.choice(['early Q1', 'the last board meeting', 'March'])}. "
                f"Whenever you have 20 minutes — even bullet points are fine.\n\n"
                f"Hope the raise process is going well.\n\n"
                f"Diane\n",

                f"Avery — enjoyed the conversation last week about {rng.choice(['the Halberd expansion', 'the forecasting model roadmap', 'the Series A timeline'])}. "
                f"Let me know if there's anything useful I can do on the investor intro side.\n\n"
                f"Diane\n",
            ]),
        ),

        # ── Customer health (Halberd handled mostly via planted thread; Derek/Lindsey ambient) ──
        (
            "customer_health", 14,
            lambda: rng.choice([
                f"Re: {rng.choice(['onboarding status', 'API integration question', 'dashboard access for new users', 'renewal timeline', 'feature request'])}",
                f"Quick question about {rng.choice(['supplier risk alerts', 'the inventory forecast accuracy', 'SSO setup', 'data export'])}",
            ]),
            lambda subj: rng.choice([
                f"Hi Avery,\n\n"
                f"{rng.choice(['Loving the new dashboard so far.', 'Team has been getting good use out of the alerts.', 'Onboarding has gone smoothly on our end.'])} "
                f"One question — {rng.choice(['can we add 3 more seats for the ops team?', 'is there a way to export supplier scores to CSV?', 'when does the next forecasting model update ship?'])}\n\n"
                f"Thanks,\n",

                f"Hi Avery,\n\n"
                f"{rng.choice(['Following up on the SSO setup', 'Checking in on the API rate limit increase', 'Wanted to flag a small data discrepancy'])} — "
                f"{rng.choice(['our IT team is ready whenever you are.', 'nothing urgent, just want to close the loop.', 'noticed a mismatch on the recent shipment counts, might just be a timezone thing.'])}\n\n"
                f"Best,\n",
            ]),
        ),

        # ── GTM (Tomás — high volume, "look for the one thing that matters") ──
        (
            "gtm_updates", 16,
            lambda: rng.choice([
                "Pipeline update",
                "Re: Halberd expansion conversation",
                "Quick thought on pricing",
                "Competitor mention — worth a look",
                f"[{_pick_project()}] GTM notes",
                "Demo went great today",
            ]),
            lambda subj: rng.choice([
                f"Hey Avery,\n\n"
                f"Pipeline this week: {rng.randint(3,7)} demos, {rng.randint(1,3)} in late-stage. "
                f"{rng.choice(['Nothing urgent, just keeping you posted.', 'Ravi is closing in on the mid-market deal we talked about.', 'One prospect asked about SOC 2 — worth prioritizing given how often this comes up now.'])}\n\n"
                f"— Tomás\n",

                f"Avery — {rng.choice(['saw a competitor (SupplyLens) undercutting us on price with a mid-market prospect.', 'a prospect asked if we support multi-warehouse SKUs — do we?', 'had a great call with a 200-person manufacturer, they want a pilot.'])} "
                f"{rng.choice(['Thoughts?', 'Wanted your take before I respond.', 'Not blocking, just flagging.'])}\n\n"
                f"— Tomás\n",

                f"Quick one: {rng.choice(['Halberd mentioned they might want to expand to their Ohio plant next quarter — worth a call?', 'should we bump the starter tier price by $200/mo? Feels underpriced.', 'marketing site conversion is up 12% since the new homepage.'])}\n\n"
                f"Tomás\n",
            ]),
        ),

        # ── Eng updates (Jordan — infrequent, rare escalation) ──
        (
            "eng_updates", 7,
            lambda: rng.choice([
                f"[{_pick_project()}] Status update",
                "Sprint notes",
                "Architecture decision — need your input",
            ]),
            lambda subj: rng.choice([
                f"Hey Avery,\n\n"
                f"{rng.choice(['Forecasting model accuracy is up to 87% on the backtest.', 'Halberd integration is on track for next Friday.', 'SOC 2 control mapping is about 60% done.'])} "
                f"No blockers on my end.\n\n"
                f"— Jordan\n",

                f"Avery — need a quick call on {rng.choice(['whether we support multi-warehouse in the data model (Tomás is asking)', 'the SOC 2 timeline — auditor wants a decision on scope', 'a scaling question for the Halberd data volume'])}. "
                f"Nothing urgent, this week is fine.\n\n"
                f"— Jordan\n",
            ]),
        ),

        # ── Product/eng day-to-day chatter (Priya + reports) ──
        (
            "product_eng_chatter", 8,
            lambda: rng.choice([
                f"[{_pick_project()}] PR up for review",
                "Re: forecasting model accuracy",
                "Design feedback needed",
                "Quick sync notes",
            ]),
            lambda subj: rng.choice([
                f"Hey Avery,\n\n"
                f"{rng.choice(['Pushed a fix for the flaky inventory sync job.', 'Redesigned the onboarding checklist screen — mocks in Figma.', 'Found an edge case in the forecasting model for seasonal SKUs.'])}\n\n"
                f"Let me know if you want to take a look.\n\n"
                f"— {rng.choice(['Priya', 'Maya', 'Devon', 'Kenji', 'Nadia'])}\n",

                f"Avery,\n\n"
                f"We should talk through {rng.choice(['the data model for multi-warehouse support', 'whether to build or buy the anomaly detection piece', 'onboarding flow drop-off — 40% stall at step 3'])} "
                f"sometime this week.\n\n"
                f"— Priya\n",
            ]),
        ),

        # ── Hiring (Grace / Jordan — includes a stalled-candidate thread) ──
        (
            "hiring", 8,
            lambda: rng.choice([
                "Candidate pipeline update",
                "Feedback needed — onsite debrief",
                "Two eng roles + 1 design role — status",
            ]),
            lambda subj: rng.choice([
                f"Hi Avery,\n\n"
                f"Pipeline update: {rng.randint(2,5)} candidates in screening for the "
                f"{rng.choice(['senior engineer', 'product designer', 'backend engineer'])} role. "
                f"{rng.choice(['Nothing needs your input yet.', 'Will send onsite debriefs as they wrap.', 'One offer likely to go out this week.'])}\n\n"
                f"— Grace\n",

                f"Avery — {rng.choice(['candidate declined our offer, going with a bigger comp package elsewhere.', 'onsite for the design role is scheduled for Thursday.', 'behind on getting back to a couple candidates, will clean up this week.'])}\n\n"
                f"— Grace\n",
            ]),
        ),

        # ── HR / Ops / Finance (Owen) ──
        (
            "hr_ops_admin", 7,
            lambda: rng.choice([
                "Monthly burn update",
                "Payroll runs Friday",
                "Expense report — need receipts",
                "Benefits renewal — decision needed",
            ]),
            lambda subj: rng.choice([
                f"Hi Avery,\n\n"
                f"{rng.choice(['Burn was $210K last month, runway sits at about 11 months at current pace.', 'Payroll processes Friday as usual, no action needed.', 'Health insurance renewal is due — premiums going up 8%, need a decision by month end.'])}\n\n"
                f"— Owen\n",

                f"Avery — quick reminder, need receipts for the {rng.choice(['conference trip', 'team offsite', 'AWS re:Invent booth'])} expense report "
                f"by {rng.choice(['Friday', 'end of week', 'Monday'])} to close the books on time.\n\n"
                f"— Owen\n",
            ]),
        ),

        # ── Legal / SaaS vendor (Ben Schaffer routine + vendor renewals) ──
        (
            "legal_saas_vendor", 5,
            lambda: rng.choice([
                "Re: SAFE amendment redline",
                "Contract renewal — action needed",
                "[Ironclad] Your Tessera-Halberd MSA is ready for signature",
            ]),
            lambda subj: rng.choice([
                f"Hi Avery,\n\n"
                f"Sent over the redlined {rng.choice(['SAFE amendment', 'employment agreement template', 'Halberd MSA'])}. "
                f"Nothing major, mostly standard language. Let me know if you want to discuss.\n\n"
                f"Best,\nBen\n",

                f"This is an automated notice: your {rng.choice(['AWS', 'Datadog', 'Segment', 'Vercel'])} subscription renews on "
                f"{rng.choice(['August 1', 'August 12', 'August 20'])}. Current plan: ${rng.randint(200,900)}/mo.\n\n"
                f"Manage your subscription in the billing portal.\n",
            ]),
        ),

        # ── Personal (Sam Park) ──
        (
            "personal_sam", 4,
            lambda: rng.choice([
                "daycare pickup tomorrow?",
                "Wren's pediatrician follow-up",
                "grocery run?",
                "Wren said the funniest thing today",
            ]),
            lambda subj: rng.choice([
                f"hey — {rng.choice(['can you do daycare pickup tomorrow? I have a late meeting.', 'pediatrician follow-up got moved, put it on the calendar.', 'out of basically everything at home, can you swing by the store?'])}\n\nlove you\n",
                f"wren asked if the wifi 'needs water' today. thought you'd want to know. see you tonight ❤️\n",
            ]),
        ),

        # ── Recruiter cold-email (P4 noise) ──
        (
            "recruiter_coldemail", 5,
            lambda: rng.choice([
                "Exceptional Staff Engineer candidate available",
                "Quick intro — VP Sales candidate",
                "Placement opportunity — Series A companies",
            ]),
            lambda subj: (
                f"Hi Avery,\n\nI have a {rng.choice(['Staff Engineer', 'VP of Sales', 'Head of Marketing'])} candidate "
                f"who I think would be a great fit for Tessera given your stage. "
                f"Would you have 15 minutes this week to discuss?\n\nBest,\n"
            ),
        ),

        # ── Newsletters / SaaS marketing (should NOT be surfaced) ──
        (
            "newsletter_marketing", 8,
            lambda: rng.choice([
                "The Series A market is shifting",
                "5 pricing lessons from B2B SaaS leaders",
                "New: Ramp's AI-powered expense review",
                "This week in SaaStr: benchmarks you should know",
                "Notion 3.0 is here",
            ]),
            lambda subj: (
                f"{rng.choice(['This week we look at', 'In this issue:', 'A quick read on'])} "
                f"{rng.choice(['fundraising trends for early-stage SaaS.', 'how top founders think about pricing.', 'new product updates you might have missed.'])}\n\n"
                f"Read more on our site →\n"
            ),
        ),

        # ── Automated notifications ──
        (
            "notification_automated", 5,
            lambda: rng.choice([
                "[Ashby] New application: Senior Engineer",
                "[Linear] TESS-214 moved to In Progress",
                "[Ironclad] Contract fully executed",
                "[Notion] Weekly workspace digest",
                "[Google Calendar] Event updated: Board Meeting",
            ]),
            lambda subj: (
                f"Automated notification — no reply needed.\n\n---\n"
                f"{rng.choice(['A new application was submitted.', 'A ticket status changed.', 'A document was signed by all parties.', 'Your weekly workspace summary is ready.'])}\n"
            ),
        ),

        # ── Casual / social ──
        (
            "casual_social", 3,
            lambda: rng.choice([
                "lunch today?",
                "coffee chat?",
                "congrats on the writeup!",
            ]),
            lambda subj: rng.choice([
                f"Hey! {rng.choice(['Want to grab lunch? Found a new spot near the office.', 'Got 15 min for coffee later?', 'Saw the piece about Tessera in the newsletter — congrats!'])}\n\n— {rng.choice(TEAMMATES)['name'].split()[0]}\n",
            ]),
        ),
    ]

    return templates


# ─── Planted Scenarios ────────────────────────────────────────────────────────
# Deliberate, deterministic email threads exercising the persona's stated blind
# spots — not left to random sampling, so the digest can be evaluated against
# a known baseline instead of chance.

MARCUS_WEBB = next(c for c in EXTERNAL_CONTACTS if c["name"] == "Marcus Webb")
CARLA_WHITFIELD = next(c for c in EXTERNAL_CONTACTS if c["name"] == "Carla Whitfield")
ZOE_ALVAREZ = next(t for t in TEAMMATES if t["name"] == "Zoe Alvarez")
GRACE_LIN = next(t for t in TEAMMATES if t["name"] == "Grace Lin")


def _planted_emails():
    """Return (day_offset, sender, subject, body) tuples for scenarios the
    persona explicitly says she might miss without help. Day offsets are
    chosen so each thread goes quiet with enough runway before END_DATE
    (day_offset 29) to actually test the "N business days of silence" bar.
    """
    return [
        # Quiet investor thread: Marcus asks for data room access + metrics
        # on day 3, and never appears again — by day 29 that's 3+ weeks quiet.
        (
            3, MARCUS_WEBB,
            "Series A — data room access + latest metrics",
            "Hi Avery,\n\n"
            "Following up from our call — can you get me access to the data room and "
            "the latest MRR/logo retention numbers? Want to keep this moving on our "
            "partner meeting timeline.\n\n"
            "Best,\nMarcus\n",
        ),
        # Customer health signal buried in a routine-sounding thread: Halberd
        # goes from normal cadence to a soft warning, then silence.
        (
            5, CARLA_WHITFIELD,
            "Re: onboarding status",
            "Hi Avery,\n\n"
            "Onboarding is going well — the ops team is fully ramped on the dashboard. "
            "Appreciate the quick support turnaround.\n\n"
            "Best,\nCarla\n",
        ),
        (
            17, CARLA_WHITFIELD,
            "Re: Q3 renewal timeline",
            "Hi Avery,\n\n"
            "Wanted to flag — we're reassessing budget allocations this quarter given "
            "some belt-tightening upstream. Should still be fine for renewal but wanted "
            "to be upfront rather than surprise you in a few weeks.\n\n"
            "Best,\nCarla\n",
        ),
        # Unlogged promise: a teammate references something Avery said she'd
        # send, with no corresponding task in tasks.json.
        (
            9, ZOE_ALVAREZ,
            "Following up — Halberd SLA doc",
            "Hey Avery,\n\n"
            "Just confirming — you mentioned on Tuesday you'd send Halberd the updated "
            "SLA one-pager by Friday. Want me to draft it, or are you sending it "
            "yourself?\n\n"
            "— Zoe\n",
        ),
        # Stalled hiring loop: strong onsite signal on day 6, then nothing for
        # the rest of the window — by day 29 that's 23 days of silence.
        (
            6, GRACE_LIN,
            "Feedback needed — Elena Marsh onsite (Senior Eng)",
            "Hi Avery,\n\n"
            "Elena Marsh's onsite went really well — strong signal across the board, "
            "especially the systems design round. We should move fast here, good "
            "candidates like this don't stay open long. Can you take a look at the "
            "feedback and let me know if you want to move to offer?\n\n"
            "— Grace\n",
        ),
    ]


# ─── Email Generator ──────────────────────────────────────────────────────────

def generate_emails(rng, output_dir):
    """Generate ~500 .eml files with realistic distribution."""
    inbox_dir = os.path.join(output_dir, "inbox")
    os.makedirs(inbox_dir, exist_ok=True)
    for stale in glob.glob(os.path.join(inbox_dir, "*.eml")):
        os.remove(stale)

    templates = _email_templates(rng)
    categories = [t[0] for t in templates]
    weights = [t[1] for t in templates]
    subject_fns = [t[2] for t in templates]
    body_fns = [t[3] for t in templates]

    # Marcus and Carla are excluded from the general reply pool too, not just
    # new-email selection — otherwise either can get randomly picked to
    # "reply" to an unrelated thread, breaking their planted scenarios
    # (Marcus's only appearance must be the single planted email on day 3;
    # Carla's must be exactly the two planted emails on days 5 and 17).
    _SCRIPTED_ONLY = {"Marcus Webb", "Carla Whitfield"}
    reply_pool = [p for p in (TEAMMATES + EXTERNAL_CONTACTS) if p["name"] not in _SCRIPTED_ONLY]

    def _sender_for_category(category):
        if category == "notification_automated":
            return rng.choice(NOTIFICATION_SENDERS)
        if category == "newsletter_marketing":
            return rng.choice(NEWSLETTER_SENDERS)
        if category == "recruiter_coldemail":
            return rng.choice([c for c in EXTERNAL_CONTACTS if c["relation"] == "recruiter"])
        if category == "investor_fundraising":
            # Marcus is deliberately excluded — his one appearance is the planted quiet thread.
            return rng.choice([c for c in EXTERNAL_CONTACTS if c["name"] in ("Diane Okafor", "Ben Schaffer")])
        if category == "board_update":
            return next(c for c in EXTERNAL_CONTACTS if c["name"] == "Diane Okafor")
        if category == "customer_health":
            # Carla is handled via the planted thread; ambient traffic is Derek/Lindsey.
            return rng.choice([c for c in EXTERNAL_CONTACTS if c["name"] in ("Derek Osei", "Lindsey Cho")])
        if category == "gtm_updates":
            return next(t for t in TEAMMATES if t["name"] == "Tomás Reyes")
        if category == "eng_updates":
            return rng.choices(
                [t for t in TEAMMATES if t["name"] == "Jordan Liu"] * 3
                + [t for t in TEAMMATES if t["name"] in ("Maya Fischer", "Devon Brooks", "Kenji Watanabe")],
                k=1,
            )[0]
        if category == "product_eng_chatter":
            return rng.choice([t for t in TEAMMATES if t["name"] in ("Priya Iyer", "Maya Fischer", "Devon Brooks", "Kenji Watanabe", "Nadia Hassan")])
        if category == "hiring":
            return rng.choice([t for t in TEAMMATES if t["name"] in ("Grace Lin", "Jordan Liu")])
        if category == "hr_ops_admin":
            return next(t for t in TEAMMATES if t["name"] == "Owen Fitzgerald")
        if category == "legal_saas_vendor":
            return rng.choice([c for c in EXTERNAL_CONTACTS if c["name"] == "Ben Schaffer"] + NOTIFICATION_SENDERS[:3])
        if category == "personal_sam":
            return PERSONAL_CONTACTS[0]
        return rng.choice(TEAMMATES)

    # Track threads for reply generation
    threads = []  # list of (message_id, subject, sender_email)

    email_count = 0

    def _write_email(sender, subject, body_text, email_dt, in_reply_to=None, allow_cc=True):
        nonlocal email_count
        email_count += 1

        if not any(body_text.rstrip().endswith(s) for s in ["\n", sender["name"], sender["name"].split()[0]]):
            body_text += sender["name"].split()[0] + "\n"

        msg = MIMEText(body_text, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = f"{sender['name']} <{sender['email']}>"
        msg["To"] = f"{PROTAGONIST['name']} <{PROTAGONIST['email']}>"

        if allow_cc and rng.random() < 0.2 and len(TEAMMATES) > 1:
            cc_count = rng.randint(1, 2)
            cc_people = rng.sample([t for t in TEAMMATES if t["email"] != sender.get("email")], min(cc_count, len(TEAMMATES) - 1))
            msg["Cc"] = ", ".join(f"{p['name']} <{p['email']}>" for p in cc_people)

        msg["Date"] = format_datetime(email_dt)
        msg_id = make_msgid(domain=DOMAIN)
        msg["Message-ID"] = msg_id
        if in_reply_to:
            msg["In-Reply-To"] = in_reply_to
            msg["References"] = in_reply_to

        filename = f"{email_count:04d}.eml"
        with open(os.path.join(inbox_dir, filename), "w", encoding="utf-8") as f:
            f.write(msg.as_string())

        return msg_id

    # ── Planted scenario emails first, at fixed day offsets ──
    planted_by_day = {}
    for day_offset, sender, subject, body in _planted_emails():
        planted_by_day.setdefault(day_offset, []).append((sender, subject, body))

    for day_offset in range(NUM_DAYS):
        current_date = START_DATE + timedelta(days=day_offset)
        is_weekend = current_date.weekday() >= 5

        for sender, subject, body in planted_by_day.get(day_offset, []):
            email_dt = current_date.replace(hour=rng.choice([8, 9, 10]), minute=rng.randint(0, 59))
            _write_email(sender, subject, body, email_dt, allow_cc=False)
            threads.append((f"<planted-{day_offset}-{uuid.uuid4().hex[:6]}@{DOMAIN}>", subject, sender["email"]))

        # Weekday: 15-25 emails, Weekend: 3-8 emails
        num_emails = rng.randint(3, 8) if is_weekend else rng.randint(15, 25)

        for _ in range(num_emails):
            is_reply = len(threads) > 0 and rng.random() < 0.25

            if is_reply:
                orig_msg_id, orig_subject, orig_sender = rng.choice(threads)
                subject = f"Re: {orig_subject}" if not orig_subject.startswith("Re: ") else orig_subject
                cat_idx = rng.choices(range(len(categories)), weights=weights, k=1)[0]
                body_text = body_fns[cat_idx](subject)
                possible_senders = [s for s in reply_pool if s["email"] != orig_sender]
                sender = rng.choice(possible_senders) if possible_senders else rng.choice(TEAMMATES)
                in_reply_to = orig_msg_id
            else:
                cat_idx = rng.choices(range(len(categories)), weights=weights, k=1)[0]
                category = categories[cat_idx]
                subject = subject_fns[cat_idx]()
                body_text = body_fns[cat_idx](subject)
                sender = _sender_for_category(category)
                in_reply_to = None

            if is_weekend:
                hour = rng.choices(range(24), weights=[0]*8 + [2,3,4,3,2,2,1,1,1,0,0,0,0,0,0,0], k=1)[0]
            else:
                hour = rng.choices(range(24), weights=[0,0,0,0,0,0,0,1, 4,8,8,6, 3,4,6,6,4, 2,1,1,0,0,0,0], k=1)[0]
            minute = rng.randint(0, 59)
            second = rng.randint(0, 59)
            email_dt = current_date.replace(hour=hour, minute=minute, second=second)

            msg_id = _write_email(sender, subject, body_text, email_dt, in_reply_to=in_reply_to)

            if not is_reply:
                threads.append((msg_id, subject, sender["email"]))
                if len(threads) > 100:
                    threads = threads[-80:]

    return email_count


# ─── Calendar Generator ──────────────────────────────────────────────────────

def _ics_event(uid, summary, dtstart, dtend, description="", location="", status="CONFIRMED", attendees=None, rrule=None):
    """Format a single VEVENT block."""
    lines = [
        "BEGIN:VEVENT",
        f"UID:{uid}",
        f"DTSTAMP:{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        f"DTSTART:{dtstart.strftime('%Y%m%dT%H%M%S')}",
        f"DTEND:{dtend.strftime('%Y%m%dT%H%M%S')}",
        f"SUMMARY:{summary}",
        f"STATUS:{status}",
    ]
    if description:
        desc_escaped = description.replace("\n", "\\n")
        lines.append(f"DESCRIPTION:{desc_escaped}")
    if location:
        lines.append(f"LOCATION:{location}")
    if rrule:
        lines.append(f"RRULE:{rrule}")
    if attendees:
        for att in attendees:
            partstat = att.get("partstat", "ACCEPTED")
            lines.append(f"ATTENDEE;CN={att['name']};PARTSTAT={partstat}:mailto:{att['email']}")
    lines.append("END:VEVENT")
    return "\n".join(lines)


def generate_calendar(rng, output_dir):
    """Generate a .ics calendar file with diverse event types, including a
    deliberate deep-work-block override and a family-calendar collision.
    """
    cal_dir = os.path.join(output_dir, "calendar")
    os.makedirs(cal_dir, exist_ok=True)

    priya = next(t for t in TEAMMATES if t["name"] == "Priya Iyer")
    jordan = next(t for t in TEAMMATES if t["name"] == "Jordan Liu")
    tomas = next(t for t in TEAMMATES if t["name"] == "Tomás Reyes")
    zoe = next(t for t in TEAMMATES if t["name"] == "Zoe Alvarez")
    ravi = next(t for t in TEAMMATES if t["name"] == "Ravi Chandra")
    grace = next(t for t in TEAMMATES if t["name"] == "Grace Lin")
    sam = PERSONAL_CONTACTS[0]
    diane = next(c for c in EXTERNAL_CONTACTS if c["name"] == "Diane Okafor")
    marcus = next(c for c in EXTERNAL_CONTACTS if c["name"] == "Marcus Webb")
    carla = next(c for c in EXTERNAL_CONTACTS if c["name"] == "Carla Whitfield")
    derek = next(c for c in EXTERNAL_CONTACTS if c["name"] == "Derek Osei")
    lindsey = next(c for c in EXTERNAL_CONTACTS if c["name"] == "Lindsey Cho")

    events = []

    for day_offset in range(NUM_DAYS):
        current_date = START_DATE + timedelta(days=day_offset)
        is_weekend = current_date.weekday() >= 5
        if is_weekend:
            continue

        naive_date = current_date.replace(tzinfo=None)
        weekday = current_date.weekday()  # 0=Mon .. 4=Fri

        # ── Leadership standup (Mon/Wed/Fri, 9:00-9:15) ──
        if weekday in (0, 2, 4):
            events.append(_ics_event(
                uid=f"leadership-standup-{day_offset}@{DOMAIN}",
                summary="Leadership Standup",
                dtstart=naive_date.replace(hour=9, minute=0),
                dtend=naive_date.replace(hour=9, minute=15),
                description="Priorities, blockers, quick decisions",
                location="Zoom — Leadership",
                attendees=[{"name": p["name"], "email": p["email"]} for p in (priya, jordan, tomas)],
            ))

        # ── Protected deep-work block (Tue/Thu, 9:00-11:00) — per persona,
        #    nobody should schedule over this without asking. ──
        if weekday in (1, 3):
            events.append(_ics_event(
                uid=f"deep-work-{day_offset}@{DOMAIN}",
                summary="🔒 Deep Work Block",
                dtstart=naive_date.replace(hour=9, minute=0),
                dtend=naive_date.replace(hour=11, minute=0),
                description="Protected — no meetings.",
            ))

        # ── Deliberate violation: something gets scheduled over the deep-work
        #    block without asking, on the second Tuesday (day_offset 14 = 2026-06-30). ──
        if day_offset == 14:  # Tuesday
            events.append(_ics_event(
                uid=f"halberd-qbr-{day_offset}@{DOMAIN}",
                summary="Customer Call: Halberd Quarterly Review",
                dtstart=naive_date.replace(hour=9, minute=30),
                dtend=naive_date.replace(hour=10, minute=30),
                description="Quarterly business review with Halberd Manufacturing",
                location="Zoom — External",
                attendees=[
                    {"name": carla["name"], "email": carla["email"]},
                    {"name": zoe["name"], "email": zoe["email"]},
                ],
            ))

        # ── 1:1s: Priya 2x/week (co-founder), Jordan + Tomás weekly ──
        if weekday in (1, 3):
            events.append(_ics_event(
                uid=f"one-on-one-priya-{day_offset}@{DOMAIN}",
                summary="1:1 with Priya",
                dtstart=naive_date.replace(hour=11, minute=30),
                dtend=naive_date.replace(hour=12, minute=0),
                description="Co-founder sync — product, eng, whatever's on fire",
                location="Zoom — Private",
                attendees=[{"name": priya["name"], "email": priya["email"]}],
            ))
        if weekday == 2:
            events.append(_ics_event(
                uid=f"one-on-one-jordan-{day_offset}@{DOMAIN}",
                summary="1:1 with Jordan",
                dtstart=naive_date.replace(hour=13, minute=0),
                dtend=naive_date.replace(hour=13, minute=30),
                location="Zoom — Private",
                attendees=[{"name": jordan["name"], "email": jordan["email"]}],
            ))
        if weekday == 4:
            events.append(_ics_event(
                uid=f"one-on-one-tomas-{day_offset}@{DOMAIN}",
                summary="1:1 with Tomás",
                dtstart=naive_date.replace(hour=13, minute=0),
                dtend=naive_date.replace(hour=13, minute=30),
                location="Zoom — Private",
                attendees=[{"name": tomas["name"], "email": tomas["email"]}],
            ))

        # ── GTM pipeline review (weekly, Thursday 10:00-10:45) ──
        if weekday == 3:
            events.append(_ics_event(
                uid=f"gtm-pipeline-{day_offset}@{DOMAIN}",
                summary="GTM Pipeline Review",
                dtstart=naive_date.replace(hour=10, minute=0),
                dtend=naive_date.replace(hour=10, minute=45),
                description="Pipeline, deal status, expansion opportunities",
                location="Zoom — Leadership",
                attendees=[{"name": p["name"], "email": p["email"]} for p in (tomas, ravi, zoe)],
            ))

        # ── Family collision: Wren's pediatrician follow-up lands directly on
        #    top of the GTM Pipeline Review, on the same Thursday (day_offset 9 = 2026-06-25). ──
        if day_offset == 9:
            events.append(_ics_event(
                uid=f"wren-pediatrician-{day_offset}@{DOMAIN}",
                summary="Wren — Pediatrician Follow-up",
                dtstart=naive_date.replace(hour=10, minute=15),
                dtend=naive_date.replace(hour=11, minute=0),
                description="Added by Sam",
                location="Oakland Pediatrics",
                attendees=[{"name": sam["name"], "email": sam["email"]}],
            ))

        # ── Lunch block (~60% of days) ──
        if rng.random() < 0.6:
            events.append(_ics_event(
                uid=f"lunch-{day_offset}@{DOMAIN}",
                summary="Lunch Break",
                dtstart=naive_date.replace(hour=12, minute=0),
                dtend=naive_date.replace(hour=13, minute=0),
            ))

        # ── Biweekly all-hands (every other Friday, 16:00-16:30) ──
        if weekday == 4 and (day_offset // 7) % 2 == 0:
            events.append(_ics_event(
                uid=f"all-hands-{day_offset}@{DOMAIN}",
                summary="Tessera All-Hands",
                dtstart=naive_date.replace(hour=16, minute=0),
                dtend=naive_date.replace(hour=16, minute=30),
                description="Company update, metrics, Q&A",
                location="Office / Zoom",
            ))

        # ── Hiring interviews (scattered) ──
        if day_offset in (6, 13, 20):
            events.append(_ics_event(
                uid=f"interview-{day_offset}@{DOMAIN}",
                summary=[
                    "Onsite: Elena Marsh (Senior Engineer)",
                    "Interview: Product Designer candidate",
                    "Interview: Backend Engineer candidate",
                ][[6, 13, 20].index(day_offset)],
                dtstart=naive_date.replace(hour=14, minute=0),
                dtend=naive_date.replace(hour=16, minute=0),
                location="Zoom — Interview",
                attendees=[{"name": grace["name"], "email": grace["email"]}, {"name": jordan["name"], "email": jordan["email"]}],
            ))

        # ── Customer check-ins (scattered, one per reference customer) ──
        if day_offset in (10, 22, 24):
            cust = [derek, lindsey, carla][[10, 22, 24].index(day_offset)]
            events.append(_ics_event(
                uid=f"customer-checkin-{day_offset}@{DOMAIN}",
                summary=f"Customer Check-in: {cust['company']}",
                dtstart=naive_date.replace(hour=11, minute=0),
                dtend=naive_date.replace(hour=11, minute=30),
                location="Zoom — External",
                attendees=[{"name": cust["name"], "email": cust["email"]}, {"name": zoe["name"], "email": zoe["email"]}],
            ))

        # ── Investor / board (Marcus tapers off after day 4 to match the
        #    quiet-thread scenario in email; Diane 1:1 + one formal board mtg) ──
        if day_offset in (2, 8):
            events.append(_ics_event(
                uid=f"marcus-call-{day_offset}@{DOMAIN}",
                summary="Investor Call: Marcus Webb (Inflection Point)",
                dtstart=naive_date.replace(hour=15, minute=0),
                dtend=naive_date.replace(hour=15, minute=30),
                location="Zoom — External",
                attendees=[{"name": marcus["name"], "email": marcus["email"]}],
            ))
        if day_offset == 15:
            events.append(_ics_event(
                uid=f"board-meeting-{day_offset}@{DOMAIN}",
                summary="Board Meeting — Q2",
                dtstart=naive_date.replace(hour=14, minute=0),
                dtend=naive_date.replace(hour=15, minute=30),
                location="Zoom — Board",
                attendees=[{"name": diane["name"], "email": diane["email"]}],
            ))
        if day_offset == 21:
            events.append(_ics_event(
                uid=f"diane-1on1-{day_offset}@{DOMAIN}",
                summary="1:1 with Diane Okafor (Board)",
                dtstart=naive_date.replace(hour=16, minute=0),
                dtend=naive_date.replace(hour=16, minute=30),
                location="Zoom — External",
                attendees=[{"name": diane["name"], "email": diane["email"]}],
            ))

        # ── Declined events (scattered) ──
        if day_offset in (7, 17, 27):
            decline_subjects = [
                "Optional: SaaStr Annual — panel invite",
                "Brainstorm: pricing page redesign",
                "Webinar: Series A benchmarks 2026",
            ]
            idx = [7, 17, 27].index(day_offset)
            events.append(_ics_event(
                uid=f"declined-{day_offset}@{DOMAIN}",
                summary=decline_subjects[idx],
                dtstart=naive_date.replace(hour=rng.choice([10, 11, 14]), minute=0),
                dtend=naive_date.replace(hour=rng.choice([11, 12, 15]), minute=0),
                status="CANCELLED",
                attendees=[{"name": PROTAGONIST["name"], "email": PROTAGONIST["email"], "partstat": "DECLINED"}],
            ))

    ics_content = "\n".join([
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Tessera//Daily Digest Generator//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        f"X-WR-CALNAME:{PROTAGONIST['name']}'s Calendar",
        "X-WR-TIMEZONE:America/Los_Angeles",
    ] + events + [
        "END:VCALENDAR",
    ])

    filepath = os.path.join(cal_dir, "calendar.ics")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(ics_content)

    return len(events)


# ─── Notes Generator ─────────────────────────────────────────────────────────

def generate_notes(rng, output_dir):
    """Generate 10 markdown notes with diverse types."""
    notes_dir = os.path.join(output_dir, "notes")
    os.makedirs(notes_dir, exist_ok=True)
    for stale in glob.glob(os.path.join(notes_dir, "*.md")):
        os.remove(stale)

    notes = [
        (
            "2026-06-17-leadership-standup-notes.md",
            textwrap.dedent("""\
            # Leadership Standup — June 17, 2026

            **Attendees**: Avery, Priya, Jordan, Tomás

            ## Updates

            ### Priya
            - Forecasting model backtest accuracy up to 84%
            - Halberd integration on track for end of month
            - Blocked on: needs Jordan's call on multi-warehouse data model

            ### Jordan
            - SOC 2 control mapping underway with Owen
            - Two eng candidates in final rounds
            - No major blockers

            ### Tomás
            - Pipeline healthy — 3 late-stage deals
            - Halberd may want to expand to their Ohio plant next quarter
            - Competitor (SupplyLens) undercutting on price with one prospect

            ## Action Items
            - [ ] @avery — decide on multi-warehouse data model direction with Priya/Jordan
            - [ ] @tomas — loop Avery in before responding to SupplyLens pricing pressure
            - [ ] @jordan — send SOC 2 scope decision to auditor
            """),
        ),
        (
            "2026-06-23-1on1-priya.md",
            textwrap.dedent("""\
            # 1:1 with Priya — June 23, 2026

            ## Topics Discussed

            ### Halberd Integration
            - On track, but multi-warehouse support is a real open question —
              affects both Halberd and the Northstar Ohio conversation
            - Need to decide: build it properly now or ship a workaround for Halberd only

            ### Series A
            - Priya wants to stay heads-down on product through the raise, not join
              investor calls unless it's a deep technical diligence session
            - Flagged: SOC 2 timeline is the thing most likely to slip

            ### Team
            - Maya is ready for more ownership — consider giving her the forecasting
              model work end-to-end
            - Hiring bar discussion — agreed to hold the line even under pipeline pressure

            ## Action Items
            - [ ] Decide on multi-warehouse data model approach (this week)
            - [ ] Give Maya ownership of forecasting model v2
            - [ ] Avery to keep board updated on SOC 2 timeline risk
            """),
        ),
        (
            "2026-07-01-board-prep-q2.md",
            textwrap.dedent("""\
            # Board Update Prep — Q2 2026

            *Draft — not yet sent to Diane*

            ## Metrics
            - ARR: $3.2M (up from $2.7M end of Q1)
            - Net revenue retention: 118%
            - 3 reference customers (Halberd, Northstar, Veritas) — all healthy on paper,
              Halberd renewal conversation should be monitored
            - Burn: ~$210K/mo, runway ~11 months at current pace

            ## Series A
            - Marcus (Inflection Point) leading conversations, term sheet possible next month
            - Data room ~80% complete — still need updated cap table and reference list
            - Diane offered investor intros — haven't taken her up on it yet

            ## Product
            - Forecasting model accuracy improving (84% → target 90%+)
            - SOC 2 Type II underway, timeline is the biggest near-term risk

            ## Team
            - 12 people, hiring 2 engineers + 1 designer
            - Strong candidate (Elena Marsh) in pipeline for senior eng — need to move fast

            ## TODO before sending
            - [ ] Fill in final ARR number from Owen
            - [ ] Get Priya's SOC 2 timeline confidence level
            - [ ] Actually send this to Diane — overdue since early Q1
            """),
        ),
        (
            "2026-06-26-series-a-data-room-checklist.md",
            textwrap.dedent("""\
            # Series A — Data Room Checklist

            **Lead**: Ben Schaffer (Wilson Sonsini)
            **Target**: term sheet by end of July

            ## Financials
            - [x] P&L, balance sheet, cash flow (trailing 18mo)
            - [x] Cap table (current)
            - [ ] Cap table (fully diluted, post-round scenarios) — Owen working on it
            - [x] Burn / runway model

            ## Legal
            - [x] Certificate of incorporation, bylaws
            - [x] Prior SAFE agreements
            - [ ] IP assignment agreements for all employees — 2 missing (contractors)
            - [ ] SAFE amendment redline from Ben — reviewed, need to sign

            ## Commercial
            - [ ] Customer reference list for Marcus — Halberd, Northstar, Veritas
            - [ ] Reference calls scheduled — none booked yet
            - [x] Top 10 customer contracts

            ## Product / Team
            - [x] Product roadmap deck
            - [x] Org chart + hiring plan
            - [ ] SOC 2 status summary (investors keep asking)

            ## Notes
            - Marcus asked for data room access + latest metrics — need to grant access
            - Don't forget: Diane offered to make investor intros, worth following up
            """),
        ),
        (
            "2026-06-30-halberd-integration-design.md",
            textwrap.dedent("""\
            # Halberd Integration — Technical Design Notes

            **Author**: Priya Iyer
            **Status**: In progress
            **Reviewers**: Avery, Jordan, Kenji

            ## Summary
            Halberd's ERP exports shipment and inventory data nightly via SFTP. We need
            to ingest, normalize, and reconcile it against our own inventory model, then
            surface supplier risk alerts within 15 minutes of ingestion.

            ## Open Question: Multi-Warehouse
            Halberd operates 2 warehouses today, possibly 3 if the Ohio plant conversation
            (via Tomás) goes anywhere. Our current data model assumes single-warehouse per
            customer. Two options:

            1. **Workaround for Halberd only** — hacky per-customer flag, ships faster
            2. **Proper multi-warehouse support** — bigger lift, unblocks Northstar/Ohio too

            Leaning toward option 2 given Northstar is asking similar questions, but it
            pushes the Halberd deadline by ~1 week.

            ## Reconciliation Logic
            - Match on SKU + warehouse + timestamp window
            - Flag mismatches > 5% as anomalies for the supplier risk alert
            - Nightly batch for now, streaming is a v2 goal

            ## Timeline
            - Week 1: SFTP ingestion + normalization
            - Week 2: Reconciliation engine + anomaly detection
            - Week 3: Multi-warehouse support (if we go with option 2)
            - Week 4: Beta with Halberd ops team

            ## Open Questions
            - [ ] Multi-warehouse: option 1 or 2? (Avery to decide with Priya/Jordan)
            - [ ] Do we need SLA guarantees written into the Halberd MSA for ingestion latency?
            """),
        ),
        (
            "2026-07-07-weekly-priorities.md",
            textwrap.dedent("""\
            # Weekly Priorities — July 7–11, 2026

            ## 🔴 Must Do
            - [ ] Decide multi-warehouse approach with Priya/Jordan
            - [ ] Send Diane the Q2 board update (overdue)
            - [ ] Grant Marcus data room access + send latest metrics
            - [ ] Respond to Grace on Elena Marsh — move to offer or pass

            ## 🟡 Should Do
            - [ ] Finish cap table (fully diluted) with Owen
            - [ ] Book reference calls for Marcus (Halberd, Northstar, Veritas)
            - [ ] Sign the SAFE amendment redline from Ben
            - [ ] Follow up with Tomás on SupplyLens pricing pressure

            ## 🟢 Nice to Have
            - [ ] Take Diane up on investor intro offer
            - [ ] Read the SOC 2 auditor's control mapping doc
            - [ ] Draft Halberd SLA one-pager (or confirm Zoe is doing it)

            ## 📝 Notes
            - Tuesday/Thursday mornings are protected — don't let anything creep in
            - Halberd renewal conversation needs a closer look, tone shifted slightly
            """),
        ),
        (
            "2026-07-03-hiring-pipeline.md",
            textwrap.dedent("""\
            # Hiring Pipeline — Snapshot, July 3, 2026

            **Open roles**: Senior Engineer, Backend Engineer, Product Designer

            ## Senior Engineer
            - **Elena Marsh** — onsite completed June 22, strong signal across the board
              (especially systems design). Grace flagged this needs a fast decision.
              **Status: awaiting Avery's go/no-go on offer.**
            - 2 candidates in earlier screening

            ## Backend Engineer
            - 1 candidate in onsite loop (interview scheduled)
            - 3 in phone screen

            ## Product Designer
            - 1 candidate in onsite loop
            - Pipeline otherwise thin — Grace considering a new sourcing channel

            ## Notes
            - Hiring bar discussion with Priya: hold the line even under pipeline pressure
            - Comp bands need a refresh before we lose another candidate to a bigger offer
            """),
        ),
        (
            "2026-07-10-soc2-audit-checklist.md",
            textwrap.dedent("""\
            # SOC 2 Type II — Audit Prep Checklist

            **Target audit window**: September 2026
            **Owner**: Owen Fitzgerald (with Jordan on technical controls)

            ## Access Control
            - [x] SSO enforced for all internal tools
            - [x] Access review process documented
            - [ ] Quarterly access review — first one not yet run

            ## Change Management
            - [x] CI/CD pipeline requires review + approval
            - [ ] Formal change management policy doc

            ## Vendor Management
            - [ ] Security questionnaires sent to all subprocessors
            - [ ] Vendor risk assessment doc — 3 of 9 vendors done

            ## Incident Response
            - [x] Incident response plan drafted
            - [ ] Tabletop exercise scheduled

            ## Monitoring
            - [x] Centralized logging in place
            - [ ] Alerting on anomalous access patterns

            ## Notes
            - Auditor keeps asking for a scope decision on multi-warehouse data handling —
              blocked on the same product decision as the Halberd integration
            - This is the thing most likely to slip the Series A timeline if it drags
            """),
        ),
        (
            "2026-07-09-northstar-call-notes.md",
            textwrap.dedent("""\
            # Call Notes — Northstar Foods (Derek Osei), July 9, 2026

            ## Discussion
            - Derek asked about multi-warehouse support — they're considering consolidating
              two regional warehouses and want visibility across both during the transition
            - Generally happy with the product, no complaints
            - Mentioned Halberd by name as a peer reference during their own board
              discussions — good signal for us

            ## Action Items
            - [ ] Loop Derek in once multi-warehouse decision is made
            - [ ] Consider Derek as a reference call for Marcus's diligence
            """),
        ),
        (
            "2026-07-01-decision-log.md",
            textwrap.dedent("""\
            # Tessera — Decision Log

            A record of key product, technical, and company decisions.

            ---

            ## Decision 001: Hold the line on hiring bar (June 23)

            **Context**: Pipeline pressure tempting us to move faster on marginal candidates.
            **Decision**: No bar-lowering, even for open roles open 60+ days.
            **Decided by**: Avery, Priya
            **Status**: Approved ✅

            ---

            ## Decision 002: SOC 2 auditor selection (June 28)

            **Context**: Needed a Type II auditor ahead of enterprise deals + Series A diligence.
            **Decision**: Went with Vantage Assurance over two other quotes — best turnaround time.
            **Decided by**: Avery, Owen
            **Status**: Approved ✅

            ---

            ## Decision 003: Protect Tue/Thu 9-11am for deep work (July 1)

            **Context**: Too many things creeping into what used to be focus time.
            **Decision**: Nothing gets scheduled in that window without asking first.
            **Decided by**: Avery
            **Status**: Approved ✅ (compliance: mixed so far)

            ---

            ## Decision 004: Multi-warehouse data model — leaning toward proper support (July 2)

            **Context**: Both Halberd and Northstar are asking; quick workaround vs. real fix.
            **Decision**: Not yet final — leaning toward building it properly, adds ~1 week
            to Halberd timeline.
            **Decided by**: Pending — Avery/Priya/Jordan
            **Status**: Open ⏳
            """),
        ),
    ]

    for filename, content in notes:
        filepath = os.path.join(notes_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

    return len(notes)


# ─── Tasks Generator ─────────────────────────────────────────────────────────

def generate_tasks(rng, output_dir):
    """Generate 5 tasks in JSON format."""
    tasks_dir = os.path.join(output_dir, "tasks")
    os.makedirs(tasks_dir, exist_ok=True)

    tasks = [
        {
            "id": "TESS-201",
            "title": "Finalize Series A data room",
            "description": (
                "Close out remaining items in the data room checklist: fully-diluted cap "
                "table, IP assignment agreements for contractors, customer reference list "
                "for Marcus, and the signed SAFE amendment redline from Ben."
            ),
            "status": "in-progress",
            "priority": "P1",
            "assignee": "Avery Chen",
            "due_date": "2026-07-18",
            "created_at": "2026-06-26T09:00:00-07:00",
            "tags": ["series-a", "fundraising", "legal"],
            "subtasks": [
                {"title": "Financials, cap table (current), burn model", "done": True},
                {"title": "Fully diluted cap table with post-round scenarios", "done": False},
                {"title": "IP assignment agreements for contractors", "done": False},
                {"title": "Sign SAFE amendment redline from Ben", "done": False},
                {"title": "Customer reference list + book reference calls", "done": False},
            ],
        },
        {
            "id": "TESS-207",
            "title": "Close Halberd Manufacturing integration",
            "description": (
                "Ship the Halberd ERP integration: SFTP ingestion, reconciliation engine, "
                "anomaly detection for supplier risk alerts, and a decision on whether to "
                "build proper multi-warehouse support now or ship a workaround."
            ),
            "status": "in-progress",
            "priority": "P1",
            "assignee": "Avery Chen",
            "due_date": "2026-07-24",
            "created_at": "2026-06-30T11:00:00-07:00",
            "tags": ["engineering", "halberd-integration", "customer"],
            "subtasks": [
                {"title": "SFTP ingestion + normalization", "done": True},
                {"title": "Reconciliation engine + anomaly detection", "done": False},
                {"title": "Decide multi-warehouse approach (option 1 vs 2)", "done": False},
                {"title": "Beta with Halberd ops team", "done": False},
            ],
        },
        {
            "id": "TESS-212",
            "title": "Send Diane the Q2 board update",
            "description": (
                "Write and send the Q2 board update to Diane Okafor — metrics, Series A "
                "status, product progress, hiring. Draft exists in notes but has not been "
                "sent since early Q1."
            ),
            "status": "todo",
            "priority": "P2",
            "assignee": "Avery Chen",
            "due_date": "2026-07-15",
            "created_at": "2026-07-01T08:30:00-07:00",
            "tags": ["board", "investor-relations"],
            "subtasks": [
                {"title": "Fill in final ARR number from Owen", "done": False},
                {"title": "Get Priya's SOC 2 timeline confidence level", "done": False},
                {"title": "Send to Diane", "done": False},
            ],
        },
        {
            "id": "TESS-215",
            "title": "SOC 2 Type II audit prep",
            "description": (
                "Prepare for the September SOC 2 Type II audit: vendor security "
                "questionnaires, quarterly access review, change management policy, "
                "and incident response tabletop exercise."
            ),
            "status": "blocked",
            "priority": "P2",
            "assignee": "Avery Chen",
            "due_date": "2026-08-15",
            "created_at": "2026-06-28T10:00:00-07:00",
            "tags": ["compliance", "soc2", "series-a"],
            "blocked_by": "Waiting on Owen to finish vendor security questionnaires (3 of 9 vendors done); also blocked on multi-warehouse data handling scope decision.",
            "subtasks": [
                {"title": "SSO + access review process", "done": True},
                {"title": "Run first quarterly access review", "done": False},
                {"title": "Vendor security questionnaires (9 vendors)", "done": False},
                {"title": "Change management policy doc", "done": False},
                {"title": "Incident response tabletop exercise", "done": False},
            ],
        },
        {
            "id": "TESS-219",
            "title": "Decide on Elena Marsh offer (Senior Engineer)",
            "description": (
                "Elena Marsh completed her onsite with strong signal across the board. "
                "Grace flagged this needs a fast decision — good candidates like this "
                "don't stay open long. Review feedback and decide: extend offer or pass."
            ),
            "status": "todo",
            "priority": "P1",
            "assignee": "Avery Chen",
            "due_date": "2026-06-26",
            "created_at": "2026-06-22T15:00:00-07:00",
            "tags": ["hiring", "engineering"],
            "subtasks": [
                {"title": "Review onsite feedback", "done": False},
                {"title": "Decide: offer or pass", "done": False},
                {"title": "If offer: align on comp with Owen", "done": False},
            ],
        },
    ]

    filepath = os.path.join(tasks_dir, "tasks.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)

    return len(tasks)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data for Daily Digest")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for reproducibility")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory (default: data)")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print(f"🌱 Seed: {args.seed}")
    print(f"📁 Output: {args.output_dir}/")
    print(f"📅 Date range: {START_DATE.date()} → {END_DATE.date()} ({NUM_DAYS} days)")
    print()

    print("📧 Generating emails...")
    email_count = generate_emails(rng, args.output_dir)
    print(f"   ✅ {email_count} emails → {args.output_dir}/inbox/")

    print("📅 Generating calendar...")
    event_count = generate_calendar(rng, args.output_dir)
    print(f"   ✅ {event_count} events → {args.output_dir}/calendar/calendar.ics")

    print("📝 Generating notes...")
    note_count = generate_notes(rng, args.output_dir)
    print(f"   ✅ {note_count} notes → {args.output_dir}/notes/")

    print("✅ Generating tasks...")
    task_count = generate_tasks(rng, args.output_dir)
    print(f"   ✅ {task_count} tasks → {args.output_dir}/tasks/tasks.json")

    print()
    print("🎉 Done! All synthetic data generated successfully.")
    print()
    print("Summary:")
    print(f"  Emails:   {email_count:>4} files  ({args.output_dir}/inbox/)")
    print(f"  Calendar: {event_count:>4} events ({args.output_dir}/calendar/calendar.ics)")
    print(f"  Notes:    {note_count:>4} files  ({args.output_dir}/notes/)")
    print(f"  Tasks:    {task_count:>4} items  ({args.output_dir}/tasks/tasks.json)")


if __name__ == "__main__":
    main()
