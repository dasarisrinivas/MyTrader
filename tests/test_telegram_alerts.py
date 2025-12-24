from mytrader.utils.telegram_notifier import TelegramNotifier


def test_format_trade_alert_includes_protection_levels():
    message = TelegramNotifier.format_trade_alert(
        symbol="ES",
        side="BUY",
        quantity=2,
        fill_price=4800.25,
        stop_loss=4790.75,
        take_profit=4820.50,
        protection_note="protection from order_targets (order_id=123)",
    )
    assert "Stop Loss: $4790.75" in message
    assert "Take Profit: $4820.50" in message
    assert "protection from order_targets" in message


def test_format_trade_alert_notes_missing_protection():
    note = "protection missing (order_id=1, parent_id=2)"
    message = TelegramNotifier.format_trade_alert(
        symbol="ES",
        side="SELL",
        quantity=1,
        fill_price=4795.00,
        protection_note=note,
    )
    assert "Risk Management" in message
    assert note in message
