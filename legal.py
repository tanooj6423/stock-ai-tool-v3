"""
Terms of Service and Privacy Policy for Equitex.

Required for Razorpay KYC and generally for charging users.
Shown as expanders on the Plans tab. Review with a lawyer
before launch — this is a solid starting draft, not legal advice.
"""

import streamlit as st

from config import APP_NAME

EFFECTIVE_DATE = "19 July 2026"

TERMS_MD = f"""
**Effective date: {EFFECTIVE_DATE}**

**1. What Equitex is.** {APP_NAME} ("Equitex", "we") is a data
analytics and research platform for Indian equity markets. All
scores, probabilities, levels, scenarios and commentary are
statistical outputs of quantitative models applied to historical
market data.

**2. Not investment advice.** Equitex is not a SEBI-registered
Research Analyst or Investment Adviser. Nothing on this platform
is a recommendation, tip, or advice to buy, sell, or hold any
security. You are solely responsible for your investment
decisions. Consult a SEBI-registered professional before acting
on any information shown here.

**3. Accounts.** You must provide a valid email address and keep
your password secure. You are responsible for activity under
your account. One account per person; accounts are
non-transferable.

**4. Subscriptions and billing.** Pro subscriptions are billed
through Razorpay at the prices shown on the Plans page. Monthly
plans grant 31 days of access per payment; yearly plans grant
366 days. Payments already made are non-refundable except where
required by law; if the service materially fails to function for
an extended period, contact us and we will make it right.

**5. Fair use.** No scraping, redistributing, or reselling of
platform data or model outputs. No automated access without
written permission. We may suspend accounts that abuse the
service.

**6. Data and availability.** Market data comes from third-party
sources and may be delayed, incomplete, or wrong. The service is
provided "as is" without warranties of any kind. We do not
guarantee uptime, accuracy, or profitability of any model
output. Past model performance does not guarantee future results.

**7. Limitation of liability.** To the maximum extent permitted
by law, our total liability for any claim relating to the
service is limited to the subscription fees you paid in the 3
months before the claim arose. We are not liable for trading
losses.

**8. Termination.** You can stop using the service and delete
your account at any time by contacting support. We may terminate
accounts that violate these terms.

**9. Changes.** We may update these terms; material changes will
be announced in-app. Continued use after changes means
acceptance.

**10. Governing law.** These terms are governed by the laws of
India; courts at the operator's registered place of business
have exclusive jurisdiction.

**Contact:** support@equitex.example (update to your real
support email).
"""

PRIVACY_MD = f"""
**Effective date: {EFFECTIVE_DATE}**

**1. What we collect.** Email address and a salted hash of your
password (we never store the password itself); your watchlist,
journal entries and app settings; usage counters (e.g., analyses
per day); payment status from Razorpay (we never see or store
your card/UPI details — Razorpay processes payments).

**2. What we use it for.** Operating your account, enforcing
plan limits, processing subscriptions, and improving the
product. We do not sell your data. We do not send marketing
email without your consent.

**3. Third parties.** Razorpay (payments), our hosting provider
(infrastructure), market-data providers (no personal data is
shared with them), and Google Gemini for AI commentary (only
market data is sent — never your personal information).

**4. Storage and security.** Data is stored on our servers with
industry-standard protections. Passwords are hashed with
PBKDF2-SHA256. Sessions expire automatically.

**5. Your rights.** You may request a copy of your data or
deletion of your account and all associated data at any time by
emailing support. We comply with the Digital Personal Data
Protection Act, 2023 (India).

**6. Cookies.** We use only functional session tokens required
for login — no advertising trackers.

**Contact:** support@equitex.example (update to your real
support email).
"""


def render_legal_expanders(t):
    st.markdown(
        '<div class="section-label">Legal</div>',
        unsafe_allow_html=True
    )
    with st.expander("Terms of Service"):
        st.markdown(TERMS_MD)
    with st.expander("Privacy Policy"):
        st.markdown(PRIVACY_MD)
