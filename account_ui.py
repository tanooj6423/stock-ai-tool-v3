"""
Account UI: login gate, pricing & upgrade, account panel, admin panel.

Everything here renders with the app's theme dict `t` so it matches
the rest of the product.
"""

import streamlit as st

import auth
import billing
from config import APP_NAME, DISCLAIMER_SHORT


# ---------------------------------------------------------
# Session helpers
# ---------------------------------------------------------
def _login_user(user):
    token = auth.create_session(user["id"])
    st.session_state["auth_token"] = token
    st.query_params["s"] = token   # survives browser refresh
    st.session_state["user_email"] = user["email"]


def current_user():
    """Resolve the logged-in user from session state or URL token."""
    token = st.session_state.get("auth_token") or st.query_params.get("s")
    if token:
        user = auth.get_session_user(token)
        if user:
            st.session_state["auth_token"] = token
            st.session_state["user_email"] = user["email"]
            return user
    return None


def logout():
    token = st.session_state.get("auth_token")
    if token:
        auth.destroy_session(token)
    st.session_state.pop("auth_token", None)
    st.query_params.clear()
    st.rerun()


# ---------------------------------------------------------
# Login gate
# ---------------------------------------------------------
def render_login_gate(t):
    """
    Returns the logged-in user dict, or renders the auth screen and
    stops the script. Set AUTH_DISABLED=1 to bypass in local dev.
    """
    import os
    if os.getenv("AUTH_DISABLED", "") == "1":
        return {"id": 0, "email": "dev@local", "tier": "pro"}

    user = current_user()
    if user:
        return user

    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        st.markdown(f"""
        <div style="text-align:center;padding:48px 0 8px 0;">
          <div style="font-size:30px;font-weight:700;
               color:{t['text']};letter-spacing:-0.5px;">
            equitex<span style="color:{t['accent']};">.</span>
          </div>
          <div style="font-size:13px;color:{t['text2']};
               margin-top:6px;">
            Quantitative analytics for NSE markets
          </div>
        </div>
        """, unsafe_allow_html=True)

        tab_in, tab_up = st.tabs(["Sign in", "Create account"])

        with tab_in:
            with st.form("login_form"):
                email = st.text_input("Email", key="li_email")
                pw = st.text_input("Password", type="password",
                                   key="li_pw")
                ok = st.form_submit_button("Sign in",
                                           use_container_width=True)
            if ok:
                u = auth.verify_login(email, pw)
                if u:
                    _login_user(u)
                    st.rerun()
                else:
                    st.error("Incorrect email or password.")

        with tab_up:
            with st.form("signup_form"):
                email2 = st.text_input("Email", key="su_email")
                pw2 = st.text_input("Password (min 8 chars)",
                                    type="password", key="su_pw")
                agree = st.checkbox(
                    "I agree to the Terms of Service and understand "
                    "this is a research tool, not investment advice.",
                    key="su_agree"
                )
                ok2 = st.form_submit_button("Create free account",
                                            use_container_width=True)
            if ok2:
                if not agree:
                    st.error("Please accept the Terms of Service.")
                else:
                    created, msg = auth.create_user(email2, pw2)
                    if created:
                        _login_user(auth.get_user(email2))
                        st.rerun()
                    else:
                        st.error(msg)

        st.markdown(f"""
        <div style="text-align:center;font-size:11px;
             color:{t['text2']};margin-top:24px;
             line-height:1.6;">{DISCLAIMER_SHORT}</div>
        """, unsafe_allow_html=True)

    st.stop()


# ---------------------------------------------------------
# Pricing / upgrade
# ---------------------------------------------------------
def render_pricing(t, user):
    pro = auth.is_pro(user)

    st.markdown(
        '<div class="section-label">Plans</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)
    free_features = [
        f"{auth.FREE_ANALYSES_PER_DAY} stock analyses per day",
        f"Top {auth.FREE_PICKS_VISIBLE} daily pick",
        f"Watchlist up to {auth.FREE_WATCHLIST_MAX} stocks",
        "Track record (full transparency)",
    ]
    pro_features = [
        "Unlimited stock analyses",
        "Full daily scan — every ranked pick",
        "Unlimited watchlist + journal",
        "Morning check & price scenarios",
        "AI commentary on every analysis",
        "Priority support",
    ]

    def plan_card(col, name, price, sub, features, highlight):
        border = t['accent'] if highlight else t['border']
        badge = ("<span style='background:rgba(%s,0.15);"
                 "color:%s;font-size:10px;font-weight:700;"
                 "padding:2px 10px;border-radius:99px;"
                 "letter-spacing:1px;'>CURRENT PLAN</span>"
                 % (t['accent_rgb'], t['accent']))
        is_current = (highlight and pro) or \
                     (not highlight and not pro)
        items = "".join(
            f"<div style='font-size:12.5px;color:{t['text']};"
            f"padding:4px 0;'>&#10003;&nbsp; {f}</div>"
            for f in features
        )
        col.markdown(f"""
        <div style="border:1px solid {border};border-radius:14px;
             padding:22px;background:{t['card']};min-height:330px;">
          <div style="display:flex;justify-content:space-between;
               align-items:center;">
            <div style="font-size:14px;font-weight:700;
                 color:{t['text']};">{name}</div>
            {badge if is_current else ""}
          </div>
          <div style="font-size:26px;font-weight:700;
               color:{t['text']};margin:10px 0 2px 0;">{price}</div>
          <div style="font-size:11px;color:{t['text2']};
               margin-bottom:14px;">{sub}</div>
          {items}
        </div>
        """, unsafe_allow_html=True)

    plan_card(c1, "Free", "₹0",
              "For getting a feel of the platform",
              free_features, highlight=False)
    plan_card(c2, "Pro", "₹399/mo",
              "or ₹2,999/yr — 2 months free",
              pro_features, highlight=True)

    if pro:
        st.success("You're on Pro. Thanks for supporting Equitex.")
        return

    st.markdown("")
    b1, b2 = st.columns(2)
    if b1.button("Upgrade — ₹399 / month",
                 use_container_width=True, type="primary"):
        url, err = billing.create_payment_link(
            "pro_monthly", user["email"])
        if url:
            st.markdown(
                f"[Complete payment on Razorpay →]({url})")
            st.caption("Your account upgrades automatically "
                       "after payment.")
        else:
            st.warning(err)
    if b2.button("Upgrade — ₹2,999 / year",
                 use_container_width=True):
        url, err = billing.create_payment_link(
            "pro_yearly", user["email"])
        if url:
            st.markdown(
                f"[Complete payment on Razorpay →]({url})")
            st.caption("Your account upgrades automatically "
                       "after payment.")
        else:
            st.warning(err)

    if not billing.billing_configured():
        st.info(
            "Owner note (only you see the setup docs): add "
            "`RAZORPAY_KEY_ID`/`RAZORPAY_KEY_SECRET` or payment-page "
            "links to `.env` to activate checkout. See README."
        )


# ---------------------------------------------------------
# Account + admin panels (Settings tab)
# ---------------------------------------------------------
def render_account_panel(t, user):
    st.markdown(
        '<div class="section-label">Account</div>',
        unsafe_allow_html=True
    )
    tier = "Pro" if auth.is_pro(user) else "Free"
    used = auth.analyses_used_today(user)
    quota = ("Unlimited" if auth.is_pro(user)
             else f"{used}/{auth.FREE_ANALYSES_PER_DAY} used today")
    st.markdown(f"""
    <div style="border:1px solid {t['border']};border-radius:12px;
         padding:16px 18px;background:{t['card']};font-size:13px;
         color:{t['text']};line-height:2;">
      <b>{user['email']}</b><br>
      Plan: <b>{tier}</b> · Analyses: {quota}
    </div>
    """, unsafe_allow_html=True)
    if st.button("Sign out"):
        logout()

    if auth.is_admin(user):
        st.markdown(
            '<div class="section-label">Admin</div>',
            unsafe_allow_html=True
        )
        with st.expander("User management"):
            users = auth.list_users()
            if users:
                import pandas as pd
                st.dataframe(pd.DataFrame(users),
                             use_container_width=True,
                             hide_index=True)
            em = st.text_input("User email")
            c1, c2 = st.columns(2)
            if c1.button("Grant Pro (1 year)"):
                from datetime import datetime, timedelta
                auth.set_tier(
                    em, "pro",
                    (datetime.utcnow() +
                     timedelta(days=366)).isoformat())
                st.success(f"{em} → Pro")
            if c2.button("Set Free"):
                auth.set_tier(em, "free", None)
                st.success(f"{em} → Free")


# ---------------------------------------------------------
# Gating widgets
# ---------------------------------------------------------
def render_upgrade_nudge(t, message):
    st.markdown(f"""
    <div style="border:1px solid rgba({t['accent_rgb']},0.4);
         border-radius:12px;padding:18px 20px;
         background:rgba({t['accent_rgb']},0.07);
         font-size:13px;color:{t['text']};">
      <b>Pro feature</b><br>
      <span style="color:{t['text2']};">{message}</span>
    </div>
    """, unsafe_allow_html=True)
