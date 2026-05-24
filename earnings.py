st.markdown("---")
        from earnings import get_earnings_status, get_nse_earnings_calendar
        earnings_map = get_nse_earnings_calendar()
        earnings = get_earnings_status(ticker, earnings_map)
        if earnings["has_upcoming"]:
            if earnings["risk_level"] == "high":
                st.error(
                    f"🚨 EARNINGS ALERT: {earnings['message']} — "
                    f"Consider waiting until after results before entering."
                )
            elif earnings["risk_level"] == "medium":
                st.warning(
                    f"⚠️ EARNINGS NOTICE: {earnings['message']} — "
                    f"Elevated volatility expected. Reduce position size."
                )
            else:
                st.info(f"📅 {earnings['message']}")