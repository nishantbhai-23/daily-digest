#!/usr/bin/env python3
"""
Synthetic Data Generator for Daily Digest
==========================================
Generates four data sources for a fictional employee Avery Chen at Meridian Labs:
  1. ~500 emails (.eml) over 30 days
  2. A calendar (.ics) with meetings, blocks, declines
  3. 10 markdown notes (meeting notes, drafts, todos)
  4. 5 tasks (JSON)

Usage:
    python generate_data.py [--seed 42] [--output-dir data]

Zero external dependencies — stdlib only.
"""

import argparse
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

COMPANY = "Meridian Labs"
DOMAIN = "meridianlabs.io"
PROTAGONIST = {
    "name": "Avery Chen",
    "email": f"avery.chen@{DOMAIN}",
    "title": "Senior Product Engineer",
    "team": "Platform",
}

# ─── People ───────────────────────────────────────────────────────────────────

TEAMMATES = [
    {"name": "Jordan Reeves", "email": f"jordan.reeves@{DOMAIN}", "role": "Engineering Manager", "relation": "manager"},
    {"name": "Priya Sharma", "email": f"priya.sharma@{DOMAIN}", "role": "Staff Engineer", "relation": "peer"},
    {"name": "Marcus Kim", "email": f"marcus.kim@{DOMAIN}", "role": "Product Manager", "relation": "peer"},
    {"name": "Elena Vasquez", "email": f"elena.vasquez@{DOMAIN}", "role": "Designer", "relation": "peer"},
    {"name": "Tomás Herrera", "email": f"tomas.herrera@{DOMAIN}", "role": "Backend Engineer", "relation": "report"},
    {"name": "Aisha Patel", "email": f"aisha.patel@{DOMAIN}", "role": "Frontend Engineer", "relation": "report"},
    {"name": "Sam O'Brien", "email": f"sam.obrien@{DOMAIN}", "role": "DevOps Engineer", "relation": "peer"},
    {"name": "Lina Zhang", "email": f"lina.zhang@{DOMAIN}", "role": "Data Engineer", "relation": "peer"},
    {"name": "Derek Okafor", "email": f"derek.okafor@{DOMAIN}", "role": "QA Lead", "relation": "peer"},
    {"name": "Rachel Nakamura", "email": f"rachel.nakamura@{DOMAIN}", "role": "VP Engineering", "relation": "skip-level"},
    {"name": "Ben Torres", "email": f"ben.torres@{DOMAIN}", "role": "Solutions Architect", "relation": "peer"},
    {"name": "Nina Kowalski", "email": f"nina.kowalski@{DOMAIN}", "role": "Technical Writer", "relation": "peer"},
    {"name": "Chris Huang", "email": f"chris.huang@{DOMAIN}", "role": "Security Engineer", "relation": "peer"},
    {"name": "Fatima Al-Rashid", "email": f"fatima.alrashid@{DOMAIN}", "role": "HR Business Partner", "relation": "hr"},
    {"name": "Oliver Grant", "email": f"oliver.grant@{DOMAIN}", "role": "CEO", "relation": "exec"},
]

EXTERNAL_CONTACTS = [
    {"name": "David Lawson", "email": "david.lawson@stripebridge.com", "company": "StripeBridge"},
    {"name": "Sarah Mitchell", "email": "sarah.mitchell@cloudnova.io", "company": "CloudNova"},
    {"name": "James Park", "email": "james.park@vendorworks.com", "company": "VendorWorks"},
    {"name": "Amy Tran", "email": "amy.tran@talentflow.co", "company": "TalentFlow (Recruiting)"},
    {"name": "Raj Mehta", "email": "raj.mehta@cybershield.io", "company": "CyberShield"},
]

NOTIFICATION_SENDERS = [
    {"name": "Jira", "email": "noreply@jira.meridianlabs.io"},
    {"name": "GitHub", "email": "notifications@github.com"},
    {"name": "Slack", "email": "notification@slack.com"},
    {"name": "Google Calendar", "email": "calendar-notification@google.com"},
    {"name": "Datadog", "email": "alert@datadoghq.com"},
    {"name": "PagerDuty", "email": "noreply@pagerduty.com"},
]

# ─── Projects & Topics ───────────────────────────────────────────────────────

PROJECTS = [
    "Project Aurora",       # API redesign initiative
    "Lighthouse",           # Observability/monitoring overhaul
    "Atlas Migration",      # Database migration
    "Customer Portal v2",   # Customer-facing dashboard revamp
    "Platform SDK",         # Internal developer SDK
]

MEETING_NAMES = [
    "Platform Standup",
    "Sprint Planning",
    "Sprint Retro",
    "All Hands",
    "1:1 with Jordan",
    "1:1 with Tomás",
    "1:1 with Aisha",
    "Design Review: Customer Portal v2",
    "Architecture Review: Project Aurora",
    "Security Review",
    "Demo Day",
    "Incident Postmortem",
    "Quarterly OKR Check-in",
]

# ─── Email Templates ─────────────────────────────────────────────────────────

def _email_templates(rng):
    """Return a list of (category, weight, subject_fn, body_fn) tuples."""

    def _pick_project():
        return rng.choice(PROJECTS)

    def _pick_teammate():
        return rng.choice(TEAMMATES)

    templates = [
        # ── Project Updates (25%) ──
        (
            "project_update", 25,
            lambda: f"[{_pick_project()}] {rng.choice(['Status update', 'Weekly progress', 'Milestone reached', 'Blockers update', 'Sprint update'])}",
            lambda subj: rng.choice([
                f"Hi team,\n\nQuick update on where we stand.\n\n"
                f"We've completed the {rng.choice(['schema migration', 'API endpoint refactor', 'load testing', 'design review', 'security audit'])} "
                f"and are on track for the {rng.choice(['end-of-sprint', 'mid-July', 'Q3', 'next milestone'])} deadline.\n\n"
                f"Remaining items:\n- {rng.choice(['Finalize caching strategy', 'Update API docs', 'Set up monitoring dashboards'])}\n"
                f"- {rng.choice(['Run regression tests', 'Get sign-off from security', 'Deploy to staging'])}\n"
                f"- {rng.choice(['Write runbook', 'Update changelog', 'Sync with design on edge cases'])}\n\n"
                f"Let me know if you have questions.\n\nBest,\n",

                f"Hey all,\n\n"
                f"Wanted to flag that we hit a snag with {rng.choice(['the third-party API rate limits', 'flaky integration tests', 'a dependency conflict', 'the auth token rotation'])}. "
                f"I have created a ticket and {rng.choice(['Tomás is picking it up', 'we will discuss in standup', 'I will pair with Priya on it'])}.\n\n"
                f"ETA for resolution: {rng.choice(['EOD today', 'tomorrow morning', 'by end of sprint'])}.\n\n"
                f"— ",

                f"Team,\n\n"
                f"Great news — we shipped the {rng.choice(['new dashboard widgets', 'v2 API endpoints', 'batch processing pipeline', 'SSO integration'])} "
                f"to {rng.choice(['staging', 'canary', 'beta customers'])} today. 🎉\n\n"
                f"Early metrics look {rng.choice(['promising', 'solid', 'encouraging'])}. "
                f"P95 latency is down {rng.randint(15, 40)}% compared to the previous version.\n\n"
                f"Next steps: {rng.choice(['GA rollout next week', 'collect feedback from beta users', 'write the announcement blog post'])}.\n\n"
                f"Cheers,\n",
            ]),
        ),

        # ── 1:1 / Follow-ups (15%) ──
        (
            "one_on_one", 15,
            lambda: rng.choice([
                "Re: 1:1 Follow-up",
                "Action items from our chat",
                "Following up on our conversation",
                "Re: Career growth discussion",
                "Notes from our sync",
            ]),
            lambda subj: rng.choice([
                f"Hey Avery,\n\n"
                f"Thanks for the great conversation today. To recap what we discussed:\n\n"
                f"1. {rng.choice(['You will draft the RFC for the caching layer', 'We will revisit the promotion criteria next month', 'I will loop in Rachel on the headcount ask'])}\n"
                f"2. {rng.choice(['Target the tech talk for early August', 'You will mentor Tomás on the API design patterns', 'We will set up a shadow on-call rotation'])}\n"
                f"3. {rng.choice(['I will share the leadership reading list', 'Schedule a skip-level with Rachel', 'Review the perf feedback before next cycle'])}\n\n"
                f"Let me know if I missed anything.\n\n"
                f"Best,\nJordan\n",

                f"Hi Avery,\n\n"
                f"Just wanted to follow up on {rng.choice(['the scope question for Aurora', 'the on-call rotation changes', 'the interview panel assignments'])}.\n\n"
                f"I think we should {rng.choice(['go with option B', 'timebox this to 2 weeks', 'get input from Priya before deciding'])}. "
                f"Does that work for you?\n\n"
                f"— Jordan\n",
            ]),
        ),

        # ── Code Reviews / PRs (12%) ──
        (
            "code_review", 12,
            lambda: f"[PR #{rng.randint(1200, 1500)}] {rng.choice(['feat:', 'fix:', 'refactor:', 'chore:'])} {rng.choice(['Add rate limiting to API gateway', 'Fix N+1 query in dashboard endpoint', 'Refactor auth middleware', 'Update SDK client retry logic', 'Add pagination to list endpoints', 'Fix timezone handling in scheduler', 'Migrate to new ORM syntax'])}",
            lambda subj: rng.choice([
                f"Hey Avery,\n\n"
                f"Can you take a look at this PR when you get a chance? "
                f"It's {rng.choice(['pretty small (~150 lines)', 'a medium-sized refactor', 'a larger change but well-scoped'])}.\n\n"
                f"Key changes:\n"
                f"- {rng.choice(['Added input validation', 'Refactored the service layer', 'Updated error handling'])}\n"
                f"- {rng.choice(['Added unit tests', 'Updated integration tests', 'Added a migration script'])}\n\n"
                f"I've tested locally and CI is green. Let me know if you have questions.\n\n"
                f"Thanks!\n",

                f"Avery — left a few comments on your PR.\n\n"
                f"Overall looks good! A couple of suggestions:\n"
                f"- Consider using {rng.choice(['a builder pattern', 'dependency injection', 'a factory method'])} for the {rng.choice(['client initialization', 'config setup', 'test fixtures'])}\n"
                f"- The {rng.choice(['error message could be more descriptive', 'variable naming could be clearer', 'test coverage could be better for edge cases'])}\n\n"
                f"Nothing blocking — approve once addressed.\n",
            ]),
        ),

        # ── Bug Reports / Incidents (8%) ──
        (
            "bug_report", 8,
            lambda: f"[{rng.choice(['BUG', 'INCIDENT', 'P1', 'SEV-2'])}] {rng.choice(['Elevated error rates on /api/v2/orders', 'Dashboard loading timeout for enterprise accounts', 'Memory leak in background worker', 'Auth tokens not refreshing after rotation', 'Webhook delivery failures to customer endpoints', 'Search indexer stuck in retry loop'])}",
            lambda subj: rng.choice([
                f"Team,\n\n"
                f"We're seeing {rng.choice(['elevated 500s', 'increased latency', 'timeout errors', 'memory pressure'])} "
                f"on {rng.choice(['the orders service', 'the analytics pipeline', 'the notification worker', 'the search cluster'])}.\n\n"
                f"**Impact**: {rng.choice(['~5% of API requests affected', 'Enterprise dashboard degraded', 'Webhook deliveries delayed by 10+ min'])}\n"
                f"**Started**: ~{rng.randint(1, 4)} hours ago\n"
                f"**Current status**: {rng.choice(['Investigating', 'Mitigation deployed, monitoring', 'Root cause identified, fix in progress'])}\n\n"
                f"I've created {rng.choice(['a war room in Slack #incident-channel', 'a PagerDuty incident', 'a Jira ticket'])}. "
                f"Will update in 30 min.\n\n"
                f"— ",

                f"Quick update on the {rng.choice(['memory leak', 'timeout issue', 'error spike'])}:\n\n"
                f"Root cause was {rng.choice(['a missing connection pool limit', 'a regex backtracking issue', 'an unbounded cache growth', 'a deadlock in the job queue'])}. "
                f"Fix has been deployed to {rng.choice(['production', 'staging for verification'])}.\n\n"
                f"Postmortem scheduled for {rng.choice(['Thursday', 'Friday', 'next Monday'])}.\n",
            ]),
        ),

        # ── HR / Company-wide (8%) ──
        (
            "hr_company", 8,
            lambda: rng.choice([
                f"[All Hands] {rng.choice(['July All Hands Agenda', 'Q3 Kickoff Details', 'Company Update'])}",
                f"[HR] {rng.choice(['Benefits enrollment reminder', 'PTO policy update', 'Performance cycle timeline', 'New hire announcement'])}",
                f"[People Ops] {rng.choice(['Office hours this week', 'Team outing planning', 'Wellness program launch'])}",
                f"[Facilities] {rng.choice(['Office renovation update', 'Parking changes', 'Kitchen restock schedule'])}",
            ]),
            lambda subj: rng.choice([
                f"Hi everyone,\n\n"
                f"A few updates from the People team:\n\n"
                f"📅 **{rng.choice(['Benefits enrollment', 'Performance reviews', 'PTO requests'])}** — "
                f"deadline is {rng.choice(['July 31', 'end of month', 'next Friday'])}. "
                f"Please make sure to {rng.choice(['submit your selections', 'complete your self-review', 'log any remaining PTO'])} in Workday.\n\n"
                f"🎉 **New hire**: Please welcome {rng.choice(['Alex Rivera', 'Dana Kowalski', 'Morgan Lee'])} "
                f"who is joining the {rng.choice(['Platform', 'Growth', 'Infrastructure'])} team as a {rng.choice(['Software Engineer', 'Product Designer', 'Data Analyst'])}!\n\n"
                f"Questions? Reach out to the People team on Slack (#people-ops).\n\n"
                f"Best,\nFatima\n",

                f"Team,\n\n"
                f"Reminder: the {rng.choice(['Q3 All Hands', 'July town hall', 'mid-year check-in'])} is "
                f"{rng.choice(['this Thursday at 4pm PT', 'next Tuesday at 11am PT', 'Friday at 3pm PT'])}.\n\n"
                f"Agenda:\n"
                f"- Company metrics & financials (Oliver)\n"
                f"- Product roadmap update (Marcus)\n"
                f"- Engineering highlights (Rachel)\n"
                f"- Q&A\n\n"
                f"Please submit questions via the anonymous form.\n\n"
                f"See you there!\n",
            ]),
        ),

        # ── Customer Escalations (7%) ──
        (
            "customer_escalation", 7,
            lambda: f"[Customer] {rng.choice(['Acme Corp', 'GlobalTech', 'Pinnacle Inc', 'NovaStar', 'Quantum Dynamics'])} — {rng.choice(['API integration issue', 'Data export request', 'SLA concern', 'Feature request escalation', 'Billing discrepancy'])}",
            lambda subj: rng.choice([
                f"Hi Avery,\n\n"
                f"Looping you in on this customer issue. "
                f"{rng.choice(['Acme Corp', 'GlobalTech', 'Pinnacle Inc'])} is experiencing "
                f"{rng.choice(['intermittent 429s on our API', 'data inconsistencies in their export', 'SSO login failures', 'slow webhook deliveries'])}.\n\n"
                f"They're on our {rng.choice(['Enterprise', 'Growth', 'Scale'])} plan and this is "
                f"{rng.choice(['affecting their production workflow', 'blocking their Q3 launch', 'causing concern about their renewal'])}.\n\n"
                f"Can you {rng.choice(['take a look at their account logs', 'check if this is related to the recent deploy', 'hop on a call with their engineering team'])}? "
                f"Their TAM ({rng.choice(['Ben', 'David', 'Sarah'])}) is also cc'd.\n\n"
                f"Thanks,\n",

                f"Update on the {rng.choice(['Acme Corp', 'GlobalTech'])} situation:\n\n"
                f"I spoke with their team and the issue is "
                f"{rng.choice(['a misconfigured webhook URL on their end', 'related to our rate limit changes last week', 'a known bug fixed in the next release'])}.\n\n"
                f"Proposed resolution: {rng.choice(['We will bump their rate limits temporarily', 'Deploying the hotfix tonight', 'Sending them updated SDK docs'])}.\n\n"
                f"Customer seems {rng.choice(['satisfied with the plan', 'still concerned — may need exec attention', 'happy and appreciative of the quick response'])}.\n",
            ]),
        ),

        # ── Vendor / External (5%) ──
        (
            "vendor_external", 5,
            lambda: rng.choice([
                f"Re: {rng.choice(['Contract renewal', 'Integration partnership', 'Security assessment'])} — {rng.choice(['StripeBridge', 'CloudNova', 'VendorWorks'])}",
                f"[{rng.choice(['StripeBridge', 'CloudNova'])}] {rng.choice(['API deprecation notice', 'New feature announcement', 'Scheduled maintenance'])}",
                f"Meeting request: {rng.choice(['Quarterly business review', 'Technical deep-dive', 'Partnership sync'])}",
            ]),
            lambda subj: rng.choice([
                f"Hi Avery,\n\n"
                f"Following up on our discussion about the {rng.choice(['API integration', 'data pipeline', 'security audit'])} with {rng.choice(['StripeBridge', 'CloudNova'])}.\n\n"
                f"I've attached the {rng.choice(['updated SOW', 'technical spec', 'compliance questionnaire'])}. "
                f"Key points:\n"
                f"- {rng.choice(['New pricing tier starts at $X/mo', 'They support our SSO requirements', 'Migration timeline is ~2 weeks'])}\n"
                f"- {rng.choice(['They need our DPA signed by EOW', 'Their SDK supports Python 3.10+', 'SLA guarantees 99.95% uptime'])}\n\n"
                f"Let me know if you want to schedule a follow-up call.\n\n"
                f"Best regards,\n",

                f"Hello,\n\n"
                f"This is a notice that {rng.choice(['our v1 API', 'the legacy webhook format', 'the current authentication method'])} "
                f"will be deprecated on {rng.choice(['September 1, 2026', 'August 15, 2026', 'October 1, 2026'])}.\n\n"
                f"Please migrate to {rng.choice(['v2 API', 'the new OAuth 2.0 flow', 'the updated SDK'])} before the deadline. "
                f"Documentation: https://docs.example.com/migration-guide\n\n"
                f"If you need assistance, please contact your account manager.\n\n"
                f"Regards,\nThe {rng.choice(['StripeBridge', 'CloudNova'])} Team\n",
            ]),
        ),

        # ── Notifications / Automated (10%) ──
        (
            "notification", 10,
            lambda: rng.choice([
                f"[Jira] {rng.choice(['PLAT-' + str(rng.randint(100, 999)), 'AUR-' + str(rng.randint(100, 999))])} — {rng.choice(['Moved to In Progress', 'Comment added', 'Status changed to Done', 'Assigned to you', 'Due date approaching'])}",
                f"[GitHub] {rng.choice(['PR #' + str(rng.randint(1200, 1500)) + ' approved', 'CI failed on main', 'New issue: ' + str(rng.randint(400, 600)), 'Dependabot alert: critical vulnerability'])}",
                f"[Slack] {rng.choice(['New message in #platform-eng', 'Thread reply in #incidents', 'DM from ' + rng.choice(TEAMMATES)['name']])}",
                f"[Datadog] {rng.choice(['Alert: High CPU usage on api-prod-3', 'Monitor recovered: Orders latency', 'Alert: Error rate > 1% on /api/v2/users'])}",
            ]),
            lambda subj: rng.choice([
                f"Automated notification — no reply needed.\n\n"
                f"---\n"
                f"Ticket {rng.choice(['PLAT', 'AUR'])}-{rng.randint(100, 999)} has been updated.\n"
                f"Status: {rng.choice(['To Do → In Progress', 'In Progress → In Review', 'In Review → Done'])}\n"
                f"Assignee: {rng.choice(TEAMMATES)['name']}\n"
                f"Priority: {rng.choice(['P0 - Critical', 'P1 - High', 'P2 - Medium', 'P3 - Low'])}\n",

                f"Build {rng.choice(['passed ✅', 'failed ❌'])} for commit {uuid.uuid4().hex[:7]} on branch {rng.choice(['main', 'feature/aurora-api', 'fix/rate-limiter', 'refactor/auth-middleware'])}.\n\n"
                f"Duration: {rng.randint(2, 15)} min {rng.randint(0, 59)} sec\n"
                f"Tests: {rng.randint(200, 400)} passed, {rng.randint(0, 3)} failed, {rng.randint(0, 2)} skipped\n"
                f"Coverage: {rng.randint(78, 95)}%\n",
            ]),
        ),

        # ── Expense / Admin (5%) ──
        (
            "expense_admin", 5,
            lambda: rng.choice([
                f"[Expense] {rng.choice(['Your expense report has been approved', 'Receipt needed for transaction', 'Monthly corporate card statement'])}",
                f"[IT] {rng.choice(['Password expiration reminder', 'New laptop provisioning', 'VPN configuration update', 'Software license renewal'])}",
                f"[Admin] {rng.choice(['Conference room booking confirmed', 'Visitor badge request', 'Equipment return reminder'])}",
            ]),
            lambda subj: rng.choice([
                f"Hi Avery,\n\n"
                f"Your expense report for {rng.choice(['June 2026', 'the SF team offsite', 'conference travel'])} "
                f"(${rng.randint(100, 2500):.2f}) has been {rng.choice(['approved', 'submitted for review', 'processed — reimbursement in 3-5 business days'])}.\n\n"
                f"If you have questions, reply to this email or contact finance@{DOMAIN}.\n\n"
                f"— Finance Team\n",

                f"Avery,\n\n"
                f"This is a reminder that your {rng.choice(['VPN certificate', 'SSO password', 'MFA token'])} "
                f"expires on {rng.choice(['July 20', 'July 25', 'August 1'])}.\n\n"
                f"Please {rng.choice(['renew it via the IT portal', 'contact IT support', 'update it in your security settings'])} "
                f"before the deadline to avoid access disruption.\n\n"
                f"— IT Support\n",
            ]),
        ),

        # ── Casual / Social (5%) ──
        (
            "casual_social", 5,
            lambda: rng.choice([
                f"Re: {rng.choice(['Lunch plans?', 'Team dinner this Friday', 'Trivia night signup', 'Birthday celebration for ' + rng.choice(TEAMMATES)['name'].split()[0]])}",
                f"{rng.choice(['Coffee chat?', 'Quick question', 'Saw this article, thought of you', 'Book recommendation'])}",
            ]),
            lambda subj: rng.choice([
                f"Hey Avery!\n\n"
                f"{rng.choice(['Want to grab lunch today? Thinking the new ramen place.', 'Anyone up for coffee at 3? I need to stretch my legs.', 'Are you coming to trivia night on Thursday? We need you on the team!'])}\n\n"
                f"— {rng.choice(TEAMMATES)['name'].split()[0]}\n",

                f"Hey!\n\n"
                f"Saw this {rng.choice(['article on distributed systems', 'talk by Kelsey Hightower', 'paper on consensus algorithms', 'thread on API design'])} "
                f"and thought you'd find it interesting: https://example.com/{uuid.uuid4().hex[:8]}\n\n"
                f"Worth a read if you have 10 min.\n\n"
                f"Cheers,\n{rng.choice(TEAMMATES)['name'].split()[0]}\n",
            ]),
        ),
    ]

    return templates


# ─── Email Generator ──────────────────────────────────────────────────────────

def generate_emails(rng, output_dir):
    """Generate ~500 .eml files with realistic distribution."""
    inbox_dir = os.path.join(output_dir, "inbox")
    os.makedirs(inbox_dir, exist_ok=True)

    templates = _email_templates(rng)
    categories = [t[0] for t in templates]
    weights = [t[1] for t in templates]
    subject_fns = [t[2] for t in templates]
    body_fns = [t[3] for t in templates]

    all_senders = TEAMMATES + EXTERNAL_CONTACTS + NOTIFICATION_SENDERS

    # Track threads for reply generation
    threads = []  # list of (message_id, subject, sender)

    email_count = 0
    for day_offset in range(NUM_DAYS):
        current_date = START_DATE + timedelta(days=day_offset)
        is_weekend = current_date.weekday() >= 5

        # Weekday: 15-25 emails, Weekend: 3-8 emails
        if is_weekend:
            num_emails = rng.randint(3, 8)
        else:
            num_emails = rng.randint(15, 25)

        for _ in range(num_emails):
            email_count += 1

            # Determine if this is a reply (~30% chance, if threads exist)
            is_reply = len(threads) > 0 and rng.random() < 0.30

            if is_reply:
                # Reply to an existing thread
                thread = rng.choice(threads)
                orig_msg_id, orig_subject, orig_sender = thread
                subject = f"Re: {orig_subject}" if not orig_subject.startswith("Re: ") else orig_subject

                # Pick category for body
                cat_idx = rng.choices(range(len(categories)), weights=weights, k=1)[0]
                body_text = body_fns[cat_idx](subject)

                # Reply comes from someone other than original sender
                possible_senders = [s for s in (TEAMMATES + EXTERNAL_CONTACTS) if s["email"] != orig_sender]
                if possible_senders:
                    sender = rng.choice(possible_senders)
                else:
                    sender = rng.choice(TEAMMATES)

                in_reply_to = orig_msg_id
            else:
                # New email
                cat_idx = rng.choices(range(len(categories)), weights=weights, k=1)[0]
                category = categories[cat_idx]
                subject = subject_fns[cat_idx]()
                body_text = body_fns[cat_idx](subject)

                # Pick sender based on category
                if category == "notification":
                    sender = rng.choice(NOTIFICATION_SENDERS)
                elif category == "vendor_external":
                    sender = rng.choice(EXTERNAL_CONTACTS)
                elif category == "customer_escalation":
                    sender = rng.choice(TEAMMATES[:5] + EXTERNAL_CONTACTS[:2])
                elif category == "hr_company":
                    sender = rng.choice([t for t in TEAMMATES if t.get("relation") in ("hr", "exec")] or TEAMMATES)
                elif category == "one_on_one":
                    sender = rng.choice([t for t in TEAMMATES if t.get("relation") == "manager"] or TEAMMATES[:3])
                else:
                    sender = rng.choice(TEAMMATES)

                in_reply_to = None

            # Add sender signature if body doesn't end with a name
            if not any(body_text.rstrip().endswith(s) for s in ["\n", sender["name"], sender["name"].split()[0]]):
                body_text += sender["name"].split()[0] + "\n"

            # Generate timestamp with realistic distribution
            if is_weekend:
                hour = rng.choices(
                    range(24),
                    weights=[0]*8 + [2,3,4,3,2,2,1,1,1,0,0,0,0,0,0,0],
                    k=1
                )[0]
            else:
                # Morning spike, afternoon cluster, evening trickle
                hour = rng.choices(
                    range(24),
                    weights=[0,0,0,0,0,0,0,1, 4,8,8,6, 3,4,6,6,4, 2,1,1,0,0,0,0],
                    k=1
                )[0]
            minute = rng.randint(0, 59)
            second = rng.randint(0, 59)

            email_dt = current_date.replace(hour=hour, minute=minute, second=second)

            # Build the email
            msg = MIMEText(body_text, "plain", "utf-8")
            msg["Subject"] = subject
            msg["From"] = f"{sender['name']} <{sender['email']}>"
            msg["To"] = f"{PROTAGONIST['name']} <{PROTAGONIST['email']}>"

            # Occasionally add Cc
            if rng.random() < 0.25 and len(TEAMMATES) > 1:
                cc_count = rng.randint(1, 3)
                cc_people = rng.sample([t for t in TEAMMATES if t["email"] != sender.get("email")], min(cc_count, len(TEAMMATES) - 1))
                msg["Cc"] = ", ".join(f"{p['name']} <{p['email']}>" for p in cc_people)

            msg["Date"] = format_datetime(email_dt)
            msg_id = make_msgid(domain=DOMAIN)
            msg["Message-ID"] = msg_id

            if in_reply_to:
                msg["In-Reply-To"] = in_reply_to
                msg["References"] = in_reply_to

            # Save to thread pool
            if not is_reply:
                threads.append((msg_id, subject, sender["email"]))
                # Keep thread pool manageable
                if len(threads) > 100:
                    threads = threads[-80:]

            # Write .eml file
            filename = f"{email_count:04d}.eml"
            filepath = os.path.join(inbox_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(msg.as_string())

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
        # Escape newlines for ICS
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
    """Generate a .ics calendar file with diverse event types."""
    cal_dir = os.path.join(output_dir, "calendar")
    os.makedirs(cal_dir, exist_ok=True)

    events = []

    for day_offset in range(NUM_DAYS):
        current_date = START_DATE + timedelta(days=day_offset)
        is_weekend = current_date.weekday() >= 5

        if is_weekend:
            continue

        naive_date = current_date.replace(tzinfo=None)

        # ── Daily standup (9:00–9:15) ──
        events.append(_ics_event(
            uid=f"standup-{day_offset}@{DOMAIN}",
            summary="Platform Standup",
            dtstart=naive_date.replace(hour=9, minute=0),
            dtend=naive_date.replace(hour=9, minute=15),
            description="Daily sync — blockers, progress, priorities",
            location="Zoom — Platform Room",
            attendees=[
                {"name": p["name"], "email": p["email"]}
                for p in TEAMMATES[:6]
            ],
        ))

        # ── Monday: Sprint Planning (10:00–11:00) ──
        if current_date.weekday() == 0:
            events.append(_ics_event(
                uid=f"sprint-planning-{day_offset}@{DOMAIN}",
                summary="Sprint Planning",
                dtstart=naive_date.replace(hour=10, minute=0),
                dtend=naive_date.replace(hour=11, minute=0),
                description="Review backlog, assign stories, set sprint goals",
                location="Conference Room — Atlas",
                attendees=[
                    {"name": p["name"], "email": p["email"]}
                    for p in TEAMMATES[:8]
                ],
            ))

        # ── Friday: Sprint Retro (15:00–16:00) ──
        if current_date.weekday() == 4:
            events.append(_ics_event(
                uid=f"sprint-retro-{day_offset}@{DOMAIN}",
                summary="Sprint Retro",
                dtstart=naive_date.replace(hour=15, minute=0),
                dtend=naive_date.replace(hour=16, minute=0),
                description="What went well, what didn't, action items",
                location="Conference Room — Atlas",
                attendees=[
                    {"name": p["name"], "email": p["email"]}
                    for p in TEAMMATES[:8]
                ],
            ))

        # ── 1:1 with Manager (Tuesday, 11:00–11:30) ──
        if current_date.weekday() == 1:
            events.append(_ics_event(
                uid=f"one-on-one-jordan-{day_offset}@{DOMAIN}",
                summary="1:1 with Jordan Reeves",
                dtstart=naive_date.replace(hour=11, minute=0),
                dtend=naive_date.replace(hour=11, minute=30),
                description="Career growth, project updates, blockers",
                location="Zoom — Private",
                attendees=[
                    {"name": "Jordan Reeves", "email": f"jordan.reeves@{DOMAIN}"},
                ],
            ))

        # ── 1:1 with reports (Wednesday: Tomás, Thursday: Aisha) ──
        if current_date.weekday() == 2:
            events.append(_ics_event(
                uid=f"one-on-one-tomas-{day_offset}@{DOMAIN}",
                summary="1:1 with Tomás Herrera",
                dtstart=naive_date.replace(hour=14, minute=0),
                dtend=naive_date.replace(hour=14, minute=30),
                description="Check-in, code review feedback, growth areas",
                location="Zoom — Private",
                attendees=[
                    {"name": "Tomás Herrera", "email": f"tomas.herrera@{DOMAIN}"},
                ],
            ))

        if current_date.weekday() == 3:
            events.append(_ics_event(
                uid=f"one-on-one-aisha-{day_offset}@{DOMAIN}",
                summary="1:1 with Aisha Patel",
                dtstart=naive_date.replace(hour=14, minute=0),
                dtend=naive_date.replace(hour=14, minute=30),
                description="Check-in, frontend architecture, project priorities",
                location="Zoom — Private",
                attendees=[
                    {"name": "Aisha Patel", "email": f"aisha.patel@{DOMAIN}"},
                ],
            ))

        # ── Focus blocks (3-4 per week, on Tue/Wed/Thu) ──
        if current_date.weekday() in (1, 2, 3) and rng.random() < 0.85:
            focus_start = rng.choice([13, 15, 16])
            events.append(_ics_event(
                uid=f"focus-{day_offset}@{DOMAIN}",
                summary="🔒 Focus Time — Deep Work",
                dtstart=naive_date.replace(hour=focus_start, minute=0),
                dtend=naive_date.replace(hour=focus_start + 2, minute=0),
                description="Protected time for coding, design docs, or research. No meetings.",
                status="CONFIRMED",
            ))

        # ── Lunch block (12:00–13:00, ~60% of days) ──
        if rng.random() < 0.6:
            events.append(_ics_event(
                uid=f"lunch-{day_offset}@{DOMAIN}",
                summary="Lunch Break",
                dtstart=naive_date.replace(hour=12, minute=0),
                dtend=naive_date.replace(hour=13, minute=0),
                status="CONFIRMED",
            ))

        # ── Biweekly All Hands (every other Thursday, 16:00–17:00) ──
        if current_date.weekday() == 3 and (day_offset // 7) % 2 == 0:
            events.append(_ics_event(
                uid=f"all-hands-{day_offset}@{DOMAIN}",
                summary="Meridian Labs All Hands",
                dtstart=naive_date.replace(hour=16, minute=0),
                dtend=naive_date.replace(hour=17, minute=0),
                description="Company updates, product demos, Q&A with leadership",
                location="Main Stage / Zoom",
                attendees=[
                    {"name": "Oliver Grant", "email": f"oliver.grant@{DOMAIN}"},
                    {"name": "Rachel Nakamura", "email": f"rachel.nakamura@{DOMAIN}"},
                ],
            ))

        # ── Declined events (2-3 total, scattered) ──
        if day_offset in (3, 12, 21):
            decline_subjects = [
                "Cross-functional Sync: Growth x Platform",
                "Brainstorm: New Onboarding Flow",
                "Optional: Tech Talks — Rust in Production",
            ]
            idx = [3, 12, 21].index(day_offset)
            events.append(_ics_event(
                uid=f"declined-{day_offset}@{DOMAIN}",
                summary=decline_subjects[idx],
                dtstart=naive_date.replace(hour=rng.choice([10, 11, 14]), minute=0),
                dtend=naive_date.replace(hour=rng.choice([11, 12, 15]), minute=0),
                status="CANCELLED",
                attendees=[
                    {"name": PROTAGONIST["name"], "email": PROTAGONIST["email"], "partstat": "DECLINED"},
                ],
            ))

        # ── External calls / demos (scattered) ──
        if day_offset in (5, 14, 22):
            external_events = [
                ("Demo: CloudNova Integration", "Sarah Mitchell", "sarah.mitchell@cloudnova.io"),
                ("Vendor Call: StripeBridge API Review", "David Lawson", "david.lawson@stripebridge.com"),
                ("Security Assessment: CyberShield", "Raj Mehta", "raj.mehta@cybershield.io"),
            ]
            idx = [5, 14, 22].index(day_offset)
            name, att_name, att_email = external_events[idx]
            events.append(_ics_event(
                uid=f"external-{day_offset}@{DOMAIN}",
                summary=name,
                dtstart=naive_date.replace(hour=rng.choice([10, 11, 14, 15]), minute=0),
                dtend=naive_date.replace(hour=rng.choice([11, 12, 15, 16]), minute=0),
                location="Zoom — External",
                attendees=[
                    {"name": att_name, "email": att_email},
                ],
            ))

        # ── Design / Architecture Reviews (scattered) ──
        if day_offset in (7, 16, 25):
            review_events = [
                "Design Review: Customer Portal v2",
                "Architecture Review: Project Aurora",
                "Security Review: Q3 Audit Prep",
            ]
            idx = [7, 16, 25].index(day_offset)
            events.append(_ics_event(
                uid=f"review-{day_offset}@{DOMAIN}",
                summary=review_events[idx],
                dtstart=naive_date.replace(hour=11, minute=0),
                dtend=naive_date.replace(hour=12, minute=0),
                description="Cross-functional review session",
                location="Conference Room — Horizon",
                attendees=[
                    {"name": p["name"], "email": p["email"]}
                    for p in rng.sample(TEAMMATES, 5)
                ],
            ))

    # ── Assemble ICS file ──
    ics_content = "\n".join([
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Meridian Labs//Daily Digest Generator//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        f"X-WR-CALNAME:{PROTAGONIST['name']}'s Calendar",
        f"X-WR-TIMEZONE:America/Los_Angeles",
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

    notes = [
        # ── 1. Standup Notes ──
        (
            "2026-06-18-standup-notes.md",
            textwrap.dedent("""\
            # Platform Standup — June 18, 2026

            **Attendees**: Avery, Jordan, Priya, Tomás, Aisha, Sam

            ## Updates

            ### Avery
            - Finished the rate limiter prototype for Project Aurora
            - PR #1287 is up for review — need eyes from @priya
            - Blocked on: waiting for DevOps to provision staging Redis cluster

            ### Tomás
            - Completed batch API endpoint (PLAT-412)
            - Starting on error handling improvements today
            - No blockers

            ### Aisha
            - Customer Portal v2 — dashboard layout is 80% done
            - Need design feedback from @elena on the chart components
            - PTO tomorrow (Friday)

            ### Priya
            - Atlas Migration — schema validation scripts are passing
            - Will pair with Avery on caching strategy after lunch
            - FYI: found a potential N+1 in the orders query

            ## Action Items
            - [ ] @avery — Follow up with Sam on Redis cluster ETA
            - [ ] @priya — Review PR #1287
            - [ ] @aisha — Share dashboard mockups in Slack
            - [ ] @tomas — Update PLAT-412 ticket with API docs link
            """),
        ),

        # ── 2. 1:1 Notes with Jordan ──
        (
            "2026-06-23-1on1-jordan.md",
            textwrap.dedent("""\
            # 1:1 with Jordan — June 23, 2026

            ## Topics Discussed

            ### Project Aurora Progress
            - API redesign is on track for mid-July milestone
            - Jordan suggested presenting at Demo Day (July 10)
            - Need to write the RFC for the caching layer by EOW

            ### Career Growth
            - Jordan shared feedback from the skip-level with Rachel
              - Positive: technical leadership, mentoring Tomás
              - Area to grow: cross-functional communication (more proactive updates to PM/Design)
            - Promotion timeline: targeting Q4 review cycle
            - Jordan will send the Staff Engineer rubric

            ### Team Health
            - Aisha feeling overwhelmed with Customer Portal scope — need to reprioritize
            - Consider bringing in a contractor for frontend polish work
            - Tomás is doing great — ready for more ownership

            ## Action Items
            - [ ] Write caching layer RFC by Friday
            - [ ] Prep Demo Day slides (15 min slot)
            - [ ] Schedule scope review with Aisha + Marcus (PM)
            - [ ] Read "Staff Engineer" by Will Larson (Jordan's rec)
            """),
        ),

        # ── 3. Sprint Retro ──
        (
            "2026-07-03-sprint-retro.md",
            textwrap.dedent("""\
            # Sprint 14 Retrospective — July 3, 2026

            **Sprint dates**: June 23 – July 3
            **Velocity**: 34 points (target: 32) ✅

            ## What Went Well 🎉
            - Shipped Aurora API v2 to staging — no P0 bugs!
            - Pair programming sessions (Avery + Priya) were super productive
            - Customer Portal v2 dashboard got positive feedback from beta testers
            - On-call rotation was smooth — only 1 off-hours page (Sam handled it)

            ## What Didn't Go Well 😕
            - Atlas Migration hit unexpected schema conflicts — 2 stories carried over
            - Code review turnaround was slow (avg 2 days vs target 1 day)
            - Too many meetings on Wednesday — no focus time at all
            - Flaky tests in CI blocked deploys twice

            ## Action Items
            - [ ] @sam — Fix the 3 flakiest tests (PLAT-501, PLAT-502, PLAT-503)
            - [ ] @avery — Propose "No Meeting Wednesday" to Jordan
            - [ ] @derek — Set up code review SLA dashboard
            - [ ] @priya — Document Atlas schema migration steps for handoff
            - [x] @jordan — Book retro room for next sprint ✓

            ## Shout-outs 🌟
            - Tomás for jumping in on the incident Friday night
            - Elena for the gorgeous Customer Portal designs
            - Priya for the incredibly thorough PR reviews
            """),
        ),

        # ── 4. RFC / Proposal Draft ──
        (
            "2026-06-25-aurora-caching-rfc.md",
            textwrap.dedent("""\
            # RFC: Caching Strategy for Project Aurora API

            **Author**: Avery Chen
            **Status**: Draft
            **Created**: June 25, 2026
            **Reviewers**: Priya Sharma, Sam O'Brien, Jordan Reeves

            ## Summary

            This RFC proposes a multi-layer caching strategy for the Aurora API to reduce
            latency and database load. The goal is to achieve P95 latency < 100ms for
            read-heavy endpoints (currently ~350ms).

            ## Background

            The Aurora API serves ~2M requests/day with a read:write ratio of 85:15.
            Current architecture queries PostgreSQL directly for every request. With the
            upcoming Enterprise launch, we expect 5x traffic growth by Q4.

            ## Proposed Architecture

            ### Layer 1: Application-level cache (in-process)
            - **Technology**: LRU cache with TTL (Python `cachetools`)
            - **Scope**: Per-instance, hot data only
            - **TTL**: 30 seconds
            - **Invalidation**: TTL-based (eventual consistency acceptable)

            ```python
            from cachetools import TTLCache

            cache = TTLCache(maxsize=1000, ttl=30)

            def get_order(order_id: str) -> Order:
                if order_id in cache:
                    return cache[order_id]
                order = db.query(Order).get(order_id)
                cache[order_id] = order
                return order
            ```

            ### Layer 2: Distributed cache (Redis)
            - **Technology**: Redis 7.x cluster
            - **Scope**: Shared across all API instances
            - **TTL**: 5 minutes
            - **Invalidation**: Write-through on mutations + pub/sub for cross-instance

            ### Layer 3: CDN edge caching (future)
            - For public/semi-public endpoints only
            - Deferred to Phase 2

            ## Metrics & Success Criteria
            | Metric | Current | Target |
            |--------|---------|--------|
            | P50 latency | 180ms | < 50ms |
            | P95 latency | 350ms | < 100ms |
            | P99 latency | 800ms | < 250ms |
            | DB queries/sec | 4,200 | < 1,000 |
            | Cache hit rate | N/A | > 85% |

            ## Risks & Mitigations
            1. **Stale data** — Mitigated by short TTLs and write-through invalidation
            2. **Cache stampede** — Use probabilistic early expiration
            3. **Redis failure** — Graceful fallback to DB-only mode

            ## Timeline
            - Week 1: Redis cluster setup + L2 cache implementation
            - Week 2: L1 cache + invalidation logic
            - Week 3: Load testing + monitoring dashboards
            - Week 4: Staged rollout (canary → 10% → 50% → 100%)

            ## Open Questions
            - [ ] Should we use Redis Cluster or Redis Sentinel?
            - [ ] Do we need cache warming on deploy?
            - [ ] What's the budget for Redis infrastructure?
            """),
        ),

        # ── 5. Blog Post Draft ──
        (
            "2026-07-08-blog-draft-api-rate-limiting.md",
            textwrap.dedent("""\
            # Building a Fair Rate Limiter for Multi-Tenant APIs

            *Draft — not for publication*
            *Author: Avery Chen | Last edited: July 8, 2026*

            ---

            > **TL;DR**: We built a sliding-window rate limiter that treats enterprise and
            > startup customers fairly, without penalizing bursty but legitimate traffic.
            > Here's how we designed it, the tradeoffs we made, and what we learned.

            ## The Problem

            At Meridian Labs, our API serves customers ranging from 50-person startups
            making 100 requests/day to enterprise accounts pushing 10M+ requests/day.
            Our old rate limiter used a fixed-window counter — simple, but it had two
            major problems:

            1. **Boundary burst**: A customer could send 2x their limit by timing
               requests across window boundaries
            2. **One size fits all**: The same 1,000 req/min limit for everyone meant
               enterprise customers were constantly hitting limits while small accounts
               had unused capacity

            ## Our Approach: Adaptive Sliding Windows

            We landed on a hybrid approach:

            ```
            effective_rate = base_rate × tier_multiplier × burst_allowance(recent_history)
            ```

            The key insight: instead of hard limits, we compute a per-customer effective
            rate that adapts based on their plan tier AND their recent usage pattern.

            ### Why not token bucket?

            We considered token bucket (and actually prototyped it), but found that:
            - It requires persistent state per customer (memory pressure at scale)
            - Burst handling is less intuitive to explain to customers
            - Our Redis-based sliding window was already battle-tested

            ## Results

            After rolling this out over 2 weeks:
            - Rate limit violations dropped 73% for enterprise customers
            - Zero increase in abuse or system overload
            - Customer satisfaction (NPS) for API experience went from 34 → 52

            ## Lessons Learned

            1. **Talk to your customers first** — We interviewed 8 customers before
               writing a single line of code. Three of them had workarounds that were
               more complex than our entire rate limiter.
            2. **Monitor the monitors** — Our rate limiter itself became a reliability
               concern. We added circuit breakers to fall back to a simple fixed window
               if Redis latency spikes.
            3. **Docs matter more than code** — The biggest impact came from clearly
               documenting the rate limits in our API docs with examples.

            ---

            *TODO: Add architecture diagram*
            *TODO: Get review from Priya and Marcus before publishing*
            *TODO: Add code samples in Python and JavaScript*
            """),
        ),

        # ── 6. Weekly Priorities TODO ──
        (
            "2026-07-07-weekly-priorities.md",
            textwrap.dedent("""\
            # Weekly Priorities — July 7–11, 2026

            ## 🔴 Must Do
            - [x] Finalize Aurora API caching RFC and send for review
            - [ ] Fix P1 bug: auth token rotation failing for SSO users (PLAT-489)
            - [ ] Prep Demo Day presentation (Thursday)
            - [ ] Review Tomás's PR for batch endpoint error handling

            ## 🟡 Should Do
            - [ ] Pair with Priya on Atlas Migration schema conflicts
            - [ ] Write unit tests for rate limiter edge cases
            - [ ] Update API docs with new rate limit behavior
            - [ ] Respond to CloudNova integration questions (David's email)

            ## 🟢 Nice to Have
            - [ ] Prototype L1 in-process cache for hot endpoints
            - [ ] Read the distributed systems paper Lina shared
            - [ ] Start drafting the rate limiting blog post

            ## 📝 Notes
            - Aisha is on PTO Monday — pick up any urgent Customer Portal bugs
            - Wednesday: try to protect focus time (no meetings after standup)
            - Jordan mentioned skip-level with Rachel might happen this week
            """),
        ),

        # ── 7. Project Checklist ──
        (
            "2026-06-30-aurora-launch-checklist.md",
            textwrap.dedent("""\
            # Project Aurora — Launch Checklist

            **Target launch**: July 21, 2026
            **Owner**: Avery Chen
            **Stakeholders**: Jordan (Eng), Marcus (PM), Elena (Design), Derek (QA)

            ## Pre-Launch

            ### Engineering
            - [x] API v2 endpoints implemented and tested
            - [x] Rate limiter deployed to staging
            - [ ] Caching layer (L1 + L2) implemented
            - [ ] Load testing complete (target: 10K req/sec)
            - [ ] Runbook written for on-call team
            - [ ] Feature flags configured for staged rollout
            - [x] Database migrations tested on staging

            ### Security
            - [x] Security review completed with Chris
            - [ ] Penetration test scheduled
            - [ ] API keys rotation mechanism verified

            ### Documentation
            - [ ] API reference updated (Swagger/OpenAPI)
            - [x] Internal architecture docs in Notion
            - [ ] Customer-facing migration guide
            - [ ] Changelog entry drafted

            ### QA
            - [x] Integration tests passing (142/142)
            - [ ] Performance regression suite green
            - [ ] Manual exploratory testing by Derek's team
            - [ ] Beta customer feedback addressed

            ## Launch Day
            - [ ] Feature flag: enable for 10% of traffic
            - [ ] Monitor error rates, latency, CPU for 1 hour
            - [ ] If green: ramp to 50%, then 100%
            - [ ] Send customer announcement email
            - [ ] Post in #engineering Slack channel

            ## Post-Launch
            - [ ] Monitor for 48 hours
            - [ ] Collect customer feedback
            - [ ] Schedule postmortem if any incidents
            - [ ] Plan v2.1 iteration based on feedback
            """),
        ),

        # ── 8. Brainstorm: SDK Design ──
        (
            "2026-07-02-sdk-brainstorm.md",
            textwrap.dedent("""\
            # Platform SDK — Design Brainstorm

            *Scratch notes from brainstorm session, July 2, 2026*
            *Attendees: Avery, Priya, Nina, Ben*

            ## Goal
            Build an internal SDK that other teams at Meridian can use to integrate with
            the Platform API without dealing with raw HTTP, auth, retries, etc.

            ## Key Questions
            - What languages? Python first, then TypeScript?
            - Sync vs async? Both? Async-first with sync wrapper?
            - How do we handle versioning? Semver? API version pinning?
            - Should the SDK be open-source eventually?

            ## Ideas

            ### Developer Experience First
            - Make the "hello world" < 5 lines of code
            - Auto-discovery of API endpoints from OpenAPI spec
            - Built-in retry with exponential backoff + jitter
            - Rich error types (not just HTTP status codes)

            ```python
            # Dream API:
            from meridian import Client

            client = Client()  # auto-discovers credentials
            orders = client.orders.list(status="active", limit=50)

            for order in orders:
                print(order.id, order.customer.name)
            ```

            ### Observability Built In
            - Structured logging with request IDs
            - OpenTelemetry traces out of the box
            - Metrics: request count, latency histograms, error rates

            ### Testing Support
            - Mock client for unit tests
            - Record/replay mode for integration tests
            - Fixtures generator from OpenAPI spec

            ## Architecture Options
            1. **Code-gen from OpenAPI** — pros: always up-to-date; cons: generated code can be ugly
            2. **Hand-written with spec validation** — pros: beautiful DX; cons: maintenance burden
            3. **Hybrid** — generate the transport layer, hand-write the public API

            → Leaning toward option 3. Let's prototype next week.

            ## Next Steps
            - [ ] Avery: prototype the hybrid approach (1 endpoint)
            - [ ] Nina: draft SDK documentation structure
            - [ ] Ben: survey how customers currently integrate (API patterns)
            - [ ] Priya: evaluate code-gen tools (openapi-generator vs custom)
            """),
        ),

        # ── 9. Incident Scratch Notes ──
        (
            "2026-07-10-incident-notes.md",
            textwrap.dedent("""\
            # Incident: Elevated 500s on Orders API

            **Date**: July 10, 2026, 14:32 PDT
            **Severity**: SEV-2
            **Duration**: ~45 minutes
            **Incident Commander**: Avery Chen

            ## Timeline

            | Time | Event |
            |------|-------|
            | 14:32 | Datadog alert: Error rate > 2% on `/api/v2/orders` |
            | 14:35 | Avery acknowledged, started investigating |
            | 14:38 | Identified: connection pool exhaustion on `orders-db-primary` |
            | 14:42 | Root cause: a long-running analytics query holding connections |
            | 14:45 | Killed the runaway query, connections started recovering |
            | 14:50 | Sam scaled up connection pool from 20 → 50 as interim fix |
            | 15:00 | Error rate back to normal (< 0.1%) |
            | 15:17 | All-clear posted in #incidents |

            ## Root Cause

            A scheduled analytics job (Lina's team) ran a full table scan on the orders
            table without a statement timeout. It acquired 18 of 20 available connections
            and held them for ~12 minutes, starving the API.

            ## Contributing Factors
            - No statement timeout configured on the analytics role
            - Connection pool was sized for normal load, no headroom
            - No alerting on connection pool saturation (only on error rate)

            ## Action Items
            - [ ] Add statement timeout (30s) for analytics DB role — @lina
            - [ ] Increase default connection pool to 50 — @sam
            - [ ] Add Datadog monitor for connection pool utilization > 80% — @avery
            - [ ] Move analytics queries to read replica — @lina (Q3 goal)
            - [ ] Add circuit breaker for connection acquisition — @priya

            ## Lessons Learned
            - We need better isolation between OLTP and analytics workloads
            - Connection pool sizing should account for 3x normal load
            - The analytics team should run heavy queries on the read replica
            """),
        ),

        # ── 10. Decision Log ──
        (
            "2026-07-01-decision-log.md",
            textwrap.dedent("""\
            # Platform Team — Decision Log

            A record of key technical and process decisions.

            ---

            ## Decision 001: Use Redis 7 for Aurora Caching (June 25)

            **Context**: Need a distributed cache for Aurora API. Options: Redis, Memcached, DynamoDB DAX.
            **Decision**: Redis 7 Cluster
            **Rationale**:
            - Team already has Redis operational expertise
            - Need pub/sub for cache invalidation (Memcached doesn't support this)
            - Redis 7 Functions allow server-side scripting for complex invalidation logic
            - DAX is too coupled to AWS — we want to stay cloud-agnostic

            **Decided by**: Avery, Priya, Sam
            **Status**: Approved ✅

            ---

            ## Decision 002: Async-first SDK Design (July 2)

            **Context**: Internal SDK needs to support both sync and async callers.
            **Decision**: Build async-first with sync wrappers
            **Rationale**:
            - Most internal services are async (FastAPI, async workers)
            - Sync wrapper is straightforward (`asyncio.run()` or `loop.run_until_complete()`)
            - Going the other direction (sync-first, async wrapper) is much harder
            - Matches industry trend (httpx, aiohttp, etc.)

            **Decided by**: Avery, Priya
            **Status**: Approved ✅

            ---

            ## Decision 003: No-Meeting Wednesdays (July 7)

            **Context**: Team feedback in retro — too many meetings, not enough focus time.
            **Decision**: No recurring meetings on Wednesdays (except incidents)
            **Rationale**:
            - Developers need at least one guaranteed deep-work day per week
            - Research shows context-switching costs ~23 minutes per interruption
            - Trial for 4 sprints, then evaluate

            **Decided by**: Jordan, Avery
            **Status**: Trial ⏳

            ---

            ## Decision 004: Feature Flags for Aurora Rollout (July 8)

            **Context**: Need safe rollout mechanism for Aurora API v2.
            **Decision**: Use LaunchDarkly for feature flags
            **Rationale**:
            - Already have LaunchDarkly license (Growth team uses it)
            - Supports percentage rollouts, user targeting, kill switches
            - Better than our homegrown config flags (no audit trail, no gradual rollout)
            - Cost: $0 incremental (existing license covers our usage)

            **Decided by**: Avery, Sam, Marcus
            **Status**: Approved ✅

            ---

            ## Decision 005: Migrate to Read Replicas for Analytics (July 10)

            **Context**: Analytics queries caused a SEV-2 incident by exhausting the primary DB connection pool.
            **Decision**: Route all analytics queries to a dedicated read replica
            **Rationale**:
            - Complete workload isolation between OLTP and analytics
            - Read replica can be scaled independently
            - Already have a replica running (just not routed to)
            - Prevents future incidents of this class

            **Decided by**: Avery, Lina, Sam
            **Status**: Planned (Q3) 📋
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
            "id": "PLAT-489",
            "title": "Fix auth token rotation for SSO users",
            "description": (
                "SSO users are experiencing auth failures after token rotation. "
                "The refresh token flow doesn't account for the SAML assertion expiry, "
                "causing a 401 loop. Need to update the token refresh logic to re-initiate "
                "SAML flow when the assertion is expired."
            ),
            "status": "in-progress",
            "priority": "P1",
            "assignee": "Avery Chen",
            "due_date": "2026-07-11",
            "created_at": "2026-07-07T09:15:00-07:00",
            "tags": ["bug", "auth", "sso", "project-aurora"],
            "subtasks": [
                {"title": "Reproduce the issue with test SSO provider", "done": True},
                {"title": "Update token refresh logic for SAML assertions", "done": False},
                {"title": "Add integration tests for SSO token rotation", "done": False},
                {"title": "Deploy fix to staging and verify", "done": False},
            ],
        },
        {
            "id": "PLAT-501",
            "title": "Implement L2 Redis cache for Aurora API",
            "description": (
                "As outlined in the caching RFC, implement the distributed Redis cache layer "
                "(L2) for Aurora API read endpoints. This includes setting up the Redis cluster "
                "connection, implementing write-through invalidation, and adding cache hit/miss "
                "metrics to Datadog."
            ),
            "status": "in-progress",
            "priority": "P2",
            "assignee": "Avery Chen",
            "due_date": "2026-07-18",
            "created_at": "2026-06-26T14:30:00-07:00",
            "tags": ["feature", "caching", "redis", "project-aurora"],
            "subtasks": [
                {"title": "Set up Redis 7 cluster configuration", "done": True},
                {"title": "Implement cache read/write for /orders endpoint", "done": True},
                {"title": "Add write-through invalidation on mutations", "done": False},
                {"title": "Add Datadog metrics (hit rate, latency, evictions)", "done": False},
                {"title": "Load test with cache enabled vs disabled", "done": False},
            ],
        },
        {
            "id": "PLAT-510",
            "title": "Write Aurora API customer migration guide",
            "description": (
                "Write a customer-facing migration guide for transitioning from Aurora API v1 "
                "to v2. Include endpoint mapping, breaking changes, new authentication flow, "
                "code examples in Python and JavaScript, and a FAQ section. Coordinate with "
                "Nina (tech writer) and Marcus (PM) for review."
            ),
            "status": "todo",
            "priority": "P2",
            "assignee": "Avery Chen",
            "due_date": "2026-07-21",
            "created_at": "2026-07-08T10:00:00-07:00",
            "tags": ["docs", "migration", "project-aurora", "customer-facing"],
            "subtasks": [
                {"title": "Outline migration guide structure", "done": False},
                {"title": "Document breaking changes and endpoint mapping", "done": False},
                {"title": "Write code examples (Python + JS)", "done": False},
                {"title": "Review with Nina and Marcus", "done": False},
            ],
        },
        {
            "id": "PLAT-515",
            "title": "Add connection pool saturation monitor",
            "description": (
                "Post-incident action item: add a Datadog monitor that alerts when database "
                "connection pool utilization exceeds 80%. This should trigger a PagerDuty "
                "notification to the on-call engineer. Also add a dashboard panel showing "
                "connection pool usage over time."
            ),
            "status": "blocked",
            "priority": "P1",
            "assignee": "Avery Chen",
            "due_date": "2026-07-14",
            "created_at": "2026-07-10T16:00:00-07:00",
            "tags": ["observability", "incident-followup", "datadog", "reliability"],
            "blocked_by": "Waiting for Sam to provision Datadog API key with write access to monitors",
            "subtasks": [
                {"title": "Define alert thresholds (warning: 70%, critical: 85%)", "done": True},
                {"title": "Create Datadog monitor via Terraform", "done": False},
                {"title": "Configure PagerDuty integration", "done": False},
                {"title": "Add dashboard panel for connection pool metrics", "done": False},
                {"title": "Test alert with synthetic load", "done": False},
            ],
        },
        {
            "id": "PLAT-520",
            "title": "Prototype Platform SDK (hybrid approach)",
            "description": (
                "Build a proof-of-concept for the internal Platform SDK using the hybrid "
                "approach (auto-generated transport layer + hand-written public API). "
                "Implement one endpoint (/orders) end-to-end including auth, retries, "
                "error handling, and type hints. Present findings in next week's standup."
            ),
            "status": "todo",
            "priority": "P3",
            "assignee": "Avery Chen",
            "due_date": "2026-07-25",
            "created_at": "2026-07-02T11:00:00-07:00",
            "tags": ["feature", "sdk", "prototype", "developer-experience"],
            "subtasks": [
                {"title": "Set up SDK project structure and build tooling", "done": False},
                {"title": "Generate transport layer from OpenAPI spec", "done": False},
                {"title": "Hand-write public Client API for /orders", "done": False},
                {"title": "Add retry logic with exponential backoff", "done": False},
                {"title": "Write example usage and README", "done": False},
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

    # Generate all data sources
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
